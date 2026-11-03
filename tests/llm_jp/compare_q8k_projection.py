"""Compare real layer-0 V projection against float and Q8_K emulation.

Usage: STRATA_GGUF_PY=... python compare_q8k_projection.py MODEL NATIVE_DIR REFERENCE_DIR
Both directories must contain position-0 traces from the same BOS input.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.environ['STRATA_GGUF_PY'])
import gguf

reader = gguf.GGUFReader(sys.argv[1])
tensor = next(t for t in reader.tensors if t.name == 'blk.0.attn_v.weight')
w = gguf.quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
def read(root, name):
    return np.fromfile(os.path.join(root, '0_0_'+name+'.bin'), dtype=np.float32)
x = read(sys.argv[2], 'attn_norm')
native = read(sys.argv[2], 'Vcur')
oracle = read(sys.argv[3], 'Vcur')
assert x.shape == (2560,) and native.shape == oracle.shape == (512,)
def rel(a,b):
    return float(np.sqrt(np.mean((a-b)**2)/np.mean(b*b)))
blocks=x.reshape(-1,256)
maxima=blocks[np.arange(len(blocks)),np.argmax(np.abs(blocks),axis=1)]
scale=np.divide(-127.0,maxima,out=np.zeros_like(maxima),where=maxima!=0)
q=np.minimum(127,np.rint(blocks*scale[:,None]))
qx=np.divide(q,scale[:,None],out=np.zeros_like(q),where=scale[:,None]!=0)
float_ref=w@x
q8k_ref=w@qx.reshape(-1)
print(f'native vs float: {rel(native,float_ref):.9g}')
print(f'oracle vs float: {rel(oracle,float_ref):.9g}')
print(f'oracle vs Q8_K emulation: {rel(oracle,q8k_ref):.9g}')
assert rel(oracle,q8k_ref)<1e-5
