#!/usr/bin/env python3
"""Single-turn experimental chat; requires tokenizers and jinja2, not llama.cpp."""
import argparse
import datetime
import pathlib
import subprocess
import sys
from jinja2 import Environment
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--engine', required=True, help='llm_jp_token_probe executable')
parser.add_argument('--model', required=True, help='LLM-jp GGUF')
parser.add_argument('--tokenizer-dir', required=True, help='official tokenizer.json and chat_template.jinja')
parser.add_argument('--max-tokens', type=int, default=32)
parser.add_argument('--final-only', action='store_true', help='start directly in the final channel')
parser.add_argument('prompt')
a = parser.parse_args()
path = pathlib.Path(a.tokenizer_dir)
tok = Tokenizer.from_file(str(path / 'tokenizer.json'))
template = Environment().from_string((path / 'chat_template.jinja').read_text())
text = template.render(messages=[{'role':'user','content':a.prompt}], add_generation_prompt=True,
                       conversation_start_date=datetime.date.today().isoformat())
if a.final_only:
    text += "<|channel|>final<|message|>"
ids = [1] + tok.encode(text, add_special_tokens=False).ids
print(f'prompt: {len(ids)} tokens; generating up to {a.max_tokens} tokens', file=sys.stderr, flush=True)
proc = subprocess.Popen([a.engine, a.model, str(a.max_tokens), *map(str, ids)], stdout=subprocess.PIPE, text=True)
generated = []
try:
    for line in proc.stdout:
        generated.append(int(line))
        print(tok.decode(generated, skip_special_tokens=False), file=sys.stderr, end='\r', flush=True)
    code = proc.wait()
finally:
    if proc.poll() is None:
        proc.kill()
        proc.wait()
print(file=sys.stderr)
if code:
    raise SystemExit(code)
print(tok.decode(generated, skip_special_tokens=False))
