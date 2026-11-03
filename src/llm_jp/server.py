#!/usr/bin/env python3
"""Experimental local OpenAI-compatible chat endpoint for Strata's LLM-jp decoder."""
import argparse
import datetime
import json
import os
import pathlib
import re
import selectors
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from jinja2 import Environment
from tokenizers import Tokenizer

MODEL = 'LLM-jp-4.1-32b-a3b-thinking-Q4_K_M-gfx906'
FINAL_RE = re.compile(r'<\|channel\|>\s*final<\|message\|>\s*(.*?)(?:<\|return\|>|$)', re.S)
MAX_BODY = 1024 * 1024

class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status

class App:
    def __init__(self, engine, model, tokenizer_dir, max_context, default_tokens):
        self.engine = str(pathlib.Path(engine).resolve(strict=True))
        self.model_path = str(pathlib.Path(model).resolve(strict=True))
        path = pathlib.Path(tokenizer_dir)
        self.tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
        self.template = Environment().from_string((path / 'chat_template.jinja').read_text())
        self.max_context = max_context
        self.default_tokens = default_tokens
        self.lock = threading.Lock()

    def prepare(self, payload):
        if not isinstance(payload, dict):
            raise RequestError('The JSON body must be an object.')
        if payload.get('model', MODEL) != MODEL:
            raise RequestError('Unknown model.', 404)
        if payload.get('tools') or payload.get('functions') or payload.get('tool_choice') not in (None, 'none'):
            raise RequestError('Tool calling is not supported by this experimental server.')
        if payload.get('stream_options') not in (None, {}):
            raise RequestError('stream_options is not supported.')
        if payload.get('n', 1) != 1:
            raise RequestError('Only n=1 is supported.')
        # The native decoder is greedy; never silently accept a sampling request.
        if payload.get('temperature') not in (None, 0, 0.0):
            raise RequestError('Only greedy decoding (temperature=0) is supported.')
        neutral = {'top_p': (None, 1, 1.0), 'top_k': (None,), 'presence_penalty': (None, 0),
                   'frequency_penalty': (None, 0), 'stop': (None, []), 'response_format': (None, {'type': 'text'}),
                   'logprobs': (None, False), 'seed': (None,)}
        for key, accepted in neutral.items():
            if payload.get(key) not in accepted:
                raise RequestError(f'{key} is not supported by this experimental server.')
        messages = payload.get('messages')
        if not isinstance(messages, list) or not messages:
            raise RequestError('messages must be a nonempty array.')
        clean = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get('role') not in ('system', 'developer', 'user', 'assistant') or not isinstance(msg.get('content'), str):
                raise RequestError('Each message needs a supported role and text content; images and tools are unsupported.')
            if msg['role'] in ('system', 'developer') and i != 0:
                raise RequestError('A system or developer message is supported only at the start.')
            if msg.get('tool_calls') or msg.get('function_call'):
                raise RequestError('Tool calls in history are not supported.')
            clean.append({'role': msg['role'], 'content': msg['content']})
        if clean[-1]['role'] != 'user':
            raise RequestError('The final message must have role user.')
        count = payload.get('max_completion_tokens', payload.get('max_tokens', self.default_tokens))
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 1024:
            raise RequestError('max_tokens must be an integer from 1 to 1024.')
        stream = payload.get('stream', False)
        if not isinstance(stream, bool):
            raise RequestError('stream must be a boolean.')
        try:
            prompt = self.template.render(messages=clean, add_generation_prompt=True,
                conversation_start_date=datetime.date.today().isoformat())
            ids = [1] + self.tokenizer.encode(prompt, add_special_tokens=False).ids
        except Exception as e:
            raise RequestError(f'Could not render chat messages: {e}') from e
        if len(ids) + count > self.max_context:
            raise RequestError(f'Prompt ({len(ids)} tokens) plus max_tokens ({count}) exceeds the configured context limit ({self.max_context}).')
        return ids, count, stream

    def output(self, ids):
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        match = FINAL_RE.search(text)
        if not match:
            return None
        return match.group(1)

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'StrataLLMjp/0.1'

    @property
    def app(self):
        return self.server.app

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def error_json(self, status, message):
        self.send_json(status, {'error': {'message': message, 'type': 'invalid_request_error' if status < 500 else 'server_error', 'code': status}})

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ('/health', '/v1/health'):
            return self.send_json(200, {'status': 'ok', 'model': MODEL, 'busy': self.app.lock.locked(), 'engine': 'strata-llm-jp', 'context_limit': self.app.max_context})
        if path == '/v1/models':
            return self.send_json(200, {'object': 'list', 'data': [{'id': MODEL, 'object': 'model', 'created': 0, 'owned_by': 'llm-jp'}]})
        if path == '/':
            body = (pathlib.Path(__file__).with_name('chat.html')).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.error_json(404, 'Not found.')

    def do_POST(self):
        if urlsplit(self.path).path != '/v1/chat/completions':
            return self.error_json(404, 'Not found.')
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 1 or length > MAX_BODY:
                raise RequestError('Request body is missing or too large.')
            try:
                payload = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeError) as e:
                raise RequestError('Invalid JSON.') from e
            ids, count, stream = self.app.prepare(payload)
        except (RequestError, ValueError) as e:
            return self.error_json(getattr(e, 'status', 400), str(e))
        if not self.app.lock.acquire(blocking=False):
            return self.error_json(429, 'The LLM-jp engine is busy; try again after the current request finishes.')
        try:
            self.generate(ids, count, stream)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            self.app.lock.release()

    def generate(self, prompt_ids, max_tokens, stream):
        ident = 'chatcmpl-' + uuid.uuid4().hex
        created = int(time.time())
        common = {'id': ident, 'created': created, 'model': MODEL}
        def chunk(delta, finish=None):
            return {**common, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
        def sse(value):
            self.wfile.write(b'data: ' + json.dumps(value, ensure_ascii=False).encode('utf-8') + b'\n\n')
            self.wfile.flush()
        generated = []
        emitted = ''
        previous = ''
        ended = False
        with tempfile.TemporaryFile(mode='w+t', encoding='utf-8') as errors:
            try:
                proc = subprocess.Popen([self.app.engine, self.app.model_path, str(max_tokens), *map(str, prompt_ids)],
                                        stdout=subprocess.PIPE, stderr=errors, bufsize=0)
            except OSError as e:
                self.error_json(500, f'Could not start engine: {e}')
                return
            try:
                if stream:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    sse(chunk({'role': 'assistant'}))
                selector = selectors.DefaultSelector()
                selector.register(proc.stdout, selectors.EVENT_READ)
                pending = b''
                last_heartbeat = time.monotonic()
                while True:
                    events = selector.select(timeout=5)
                    if not events:
                        if stream:
                            self.wfile.write(f': processing {len(generated)}\n\n'.encode())
                            self.wfile.flush()
                            last_heartbeat = time.monotonic()
                        continue
                    data = os.read(proc.stdout.fileno(), 65536)
                    if not data:
                        break
                    pending += data
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        try:
                            token = int(line)
                        except ValueError as e:
                            raise RuntimeError('Engine returned an invalid token ID.') from e
                        generated.append(token)
                        ended = token == 2
                        if stream:
                            current = self.app.output(generated)
                            if current is not None:
                                # Delay one decode step: partial byte sequences can change on the next token.
                                stable = os.path.commonprefix((previous, current))
                                if not stable.startswith(emitted):
                                    raise RuntimeError('Tokenizer revised already streamed text.')
                                if len(stable) > len(emitted):
                                    sse(chunk({'content': stable[len(emitted):]}))
                                    emitted = stable
                                previous = current
                    if stream and time.monotonic() - last_heartbeat >= 5:
                        # The engine may emit thinking tokens continuously without a visible answer.
                        self.wfile.write(f': processing {len(generated)}\n\n'.encode())
                        self.wfile.flush()
                        last_heartbeat = time.monotonic()
                selector.close()
                if pending.strip():
                    raise RuntimeError('Engine returned an incomplete token line.')
                code = proc.wait()
                if code:
                    errors.seek(0, os.SEEK_END)
                    size = errors.tell()
                    errors.seek(max(0, size - 4096))
                    raise RuntimeError(f'Engine exited with code {code}: {errors.read().strip()}')
                answer = self.app.output(generated)
                if answer is None:
                    answer = ''
                finish = 'stop' if ended else 'length'
                if stream:
                    if not answer.startswith(emitted):
                        raise RuntimeError('Tokenizer revised already streamed text.')
                    if len(answer) > len(emitted):
                        sse(chunk({'content': answer[len(emitted):]}))
                    sse(chunk({}, finish))
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                else:
                    usage = {'prompt_tokens': len(prompt_ids), 'completion_tokens': len(generated), 'total_tokens': len(prompt_ids) + len(generated)}
                    self.send_json(200, {**common, 'object': 'chat.completion', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer}, 'finish_reason': finish}], 'usage': usage})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                raise
            except Exception as e:
                if stream:
                    sse({'error': {'message': str(e), 'type': 'server_error'}})
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                else:
                    self.error_json(500, str(e))
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                proc.stdout.close()

class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer-dir', required=True)
    parser.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1', 'localhost'])
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--max-context', type=int, default=1024)
    parser.add_argument('--default-max-tokens', type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.max_context <= 65536 or not 1 <= args.default_max_tokens <= min(1024, args.max_context):
        parser.error('Invalid context or default max tokens.')
    app = App(args.engine, args.model, args.tokenizer_dir, args.max_context, args.default_max_tokens)
    server = Server((args.host, args.port), Handler)
    server.app = app
    print(f'LLM-jp chat: http://{args.host}:{server.server_port}/  API: http://{args.host}:{server.server_port}/v1', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
