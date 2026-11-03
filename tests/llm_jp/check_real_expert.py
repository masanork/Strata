"""Compare a full real expert against dequantized weights and float activations."""
import os
import sys
import numpy as np
sys.path.insert(0, os.environ['STRATA_GGUF_PY'])
import gguf

reader = gguf.GGUFReader(sys.argv[1])
tensors = {t.name: t for t in reader.tensors}
def weight(name):
    t = tensors['blk.' + sys.argv[2] + '.ffn_' + name + '_exps.weight']
    return gguf.quants.dequantize(t.data[int(sys.argv[3])], t.tensor_type).astype(np.float32)
x = np.sin(np.arange(2560, dtype=np.float32) * np.float32(.013)) * np.float32(.1)
gate = weight('gate') @ x
up = weight('up') @ x
hidden = gate / (1 + np.exp(-gate)) * up
reference = weight('down') @ hidden
gpu = np.fromfile(sys.argv[4], dtype=np.float32)
assert gpu.shape == reference.shape == (2560,) and np.isfinite(gpu).all()
relative_rmse = np.sqrt(np.mean((gpu - reference) ** 2) / np.mean(reference ** 2))
print(f'max_abs={np.max(np.abs(gpu-reference)):.9g} relative_rmse={relative_rmse:.9g}')
assert relative_rmse < .03
