"""Compare real_mmvq.hip output with dequantized GGUF weights and float inputs.

Usage: STRATA_GGUF_PY=/path/to/gguf-py python check_real_mmvq.py MODEL.gguf TENSOR OUTPUT.bin
The GPU path quantizes activations to Q8_1; this is a bounded-error comparison, not bit parity.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.environ['STRATA_GGUF_PY'])
import gguf

reader = gguf.GGUFReader(sys.argv[1])
tensor = next(t for t in reader.tensors if t.name == sys.argv[2])
assert tensor.tensor_type == gguf.GGMLQuantizationType.Q4_K
rows = tensor.data[:32] if tensor.data.ndim == 2 else tensor.data[0, :32]
weights = gguf.quants.dequantize(rows, tensor.tensor_type).reshape(32, 2560).astype(np.float64)
x = (np.sin(np.arange(2560, dtype=np.float32) * np.float32(.013)) * np.float32(.1)).astype(np.float64)
reference = weights @ x
gpu = np.fromfile(sys.argv[3], dtype=np.float32)
assert gpu.shape == (32,) and np.isfinite(gpu).all()
error = gpu - reference
rmse = float(np.sqrt(np.mean(error ** 2)))
ref_rms = float(np.sqrt(np.mean(reference ** 2)))
relative_rmse = rmse / ref_rms
print(f'max_abs={np.max(np.abs(error)):.9g} relative_rmse={relative_rmse:.9g}')
assert relative_rmse < 0.02
