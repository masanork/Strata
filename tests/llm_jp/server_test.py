"""HTTP contract tests using a fake engine; no model, GPU or network required."""
import importlib.util
import http.client
import json
import pathlib
import socket
import time
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

source = pathlib.Path(__file__).resolve().parents[2] / 'src/llm_jp/server.py'
spec = importlib.util.spec_from_file_location('llm_jp_server', source)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class Tokenizer:
    def encode(self, prompt, add_special_tokens=False):
        return type('Encoding', (), {'ids': [10, 11]})()
    def decode(self, ids, skip_special_tokens=False):
        parts = {101: '<|channel|>final<|message|>こ�',
                 102: '<|channel|>final<|message|>こん',
                 103: '<|channel|>final<|message|>こんにちは！',
                 2: '<|channel|>final<|message|>こんにちは！<|return|>'}
        return parts.get(ids[-1], '') if ids else ''

class Template:
    def render(self, **kwargs):
        return str(kwargs['messages'])

class QuietHandler(mod.Handler):
    def log_message(self, *args):
        pass

class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        p = pathlib.Path(cls.tmp.name)
        engine = p / 'engine'
        engine.write_text('#!/bin/sh\nprintf "101\\n102\\n103\\n2\\n"\n')
        engine.chmod(0o755)
        cls.app = object.__new__(mod.App)
        cls.app.engine = str(engine)
        cls.app.model_path = str(p / 'model')
        cls.app.tokenizer = Tokenizer()
        cls.app.template = Template()
        cls.app.max_context = 256
        cls.app.default_tokens = 32
        cls.app.lock = threading.Lock()
        cls.server = mod.Server(('127.0.0.1', 0), QuietHandler)
        cls.server.app = cls.app
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.tmp.cleanup()

    def post(self, payload):
        req = urllib.request.Request(self.base + '/v1/chat/completions', json.dumps(payload).encode(), {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as e:
            try:
                return e.code, e.read().decode()
            finally:
                e.close()

    def payload(self):
        return {'messages': [{'role': 'user', 'content': 'こんにちは'}], 'max_tokens': 8}

    def test_nonstream_and_history(self):
        body = self.payload()
        body['messages'] += [{'role': 'assistant', 'content': 'こんにちは！'}, {'role': 'user', 'content': 'もう一度'}]
        status, data = self.post(body)
        response = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual(response['choices'][0]['message']['content'], 'こんにちは！')
        self.assertEqual(response['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(response['usage']['completion_tokens'], 4)

    def test_stream_preserves_unicode_when_partial_bytes_change(self):
        body = self.payload()
        body['stream'] = True
        status, data = self.post(body)
        chunks = [json.loads(x[6:]) for x in data.split('\n\n') if x.startswith('data: {')]
        answer = ''.join(x['choices'][0]['delta'].get('content', '') for x in chunks)
        self.assertEqual(status, 200)
        self.assertEqual(answer, 'こんにちは！')
        self.assertIn('data: [DONE]', data)
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'stop')

    def test_disconnect_during_continuous_thinking_stops_decoder(self):
        engine = pathlib.Path(self.app.engine)
        original = engine.read_text()
        engine.write_text('#!/bin/sh\nwhile :; do echo 110; sleep 0.05; done\n')
        sock = socket.create_connection(('127.0.0.1', self.server.server_port), timeout=5)
        try:
            body = json.dumps({**self.payload(), 'stream': True}).encode()
            request = (b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n'
                       b'Content-Type: application/json\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
            sock.sendall(request)
            self.assertIn(b'200', sock.recv(4096))
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            deadline = time.monotonic() + 13
            while self.app.lock.locked() and time.monotonic() < deadline:
                time.sleep(.1)
            self.assertFalse(self.app.lock.locked(), 'disconnected client left the decoder running')
        finally:
            sock.close()
            engine.write_text(original)

    def test_stale_worker_and_unread_post_body(self):
        with urllib.request.urlopen(self.base + '/sw.js', timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b'registration.unregister()', response.read())
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        conn.request('POST', '/tools', body='{"tool":"unsupported"}', headers={'Content-Type':'application/json'})
        response = conn.getresponse()
        self.assertEqual(response.status, 404)
        response.read()
        conn.request('GET', '/health')
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read())['status'], 'ok')
        conn.close()

    def test_invalid_and_busy_requests(self):
        body = self.payload()
        body['tools'] = [{'type': 'function'}]
        self.assertEqual(self.post(body)[0], 400)
        body.pop('tools')
        body['temperature'] = 0.7
        self.assertEqual(self.post(body)[0], 400)
        body.pop('temperature')
        body['max_tokens'] = 256
        self.assertEqual(self.post(body)[0], 400)
        body['max_tokens'] = 8
        self.app.lock.acquire()
        try:
            self.assertEqual(self.post(body)[0], 429)
        finally:
            self.app.lock.release()

if __name__ == '__main__':
    unittest.main()
