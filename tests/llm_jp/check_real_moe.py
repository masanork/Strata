"""Independent real-weight comparison for one complete routed FFN."""
import os
import sys
import numpy as np
sys.path.insert(0, os.environ['STRATA_GGUF_PY'])
import gguf

reader = gguf.GGUFReader(sys.argv[1])
tensors = {t.name: t for t in reader.tensors}
prefix = 'blk.' + sys.argv[2] + '.ffn_'
x = np.sin(np.arange(2560, dtype=np.float32)*np.float32(.013))*np.float32(.1)
router = tensors[prefix+'gate_inp.weight'].data
logits = router @ x
ids = np.argsort(-logits, kind='stable')[:8]
w = np.exp(logits[ids]-logits[ids].max()); w /= w.sum()
with open(sys.argv[3], 'rb') as f:
    got_ids = np.fromfile(f, dtype=np.int32, count=8)
    got_w = np.fromfile(f, dtype=np.float32, count=8)
    got = np.fromfile(f, dtype=np.float32)
assert np.array_equal(ids, got_ids), (ids, got_ids)
assert np.allclose(w, got_w, rtol=2e-5, atol=1e-6), (w, got_w)
ref = np.zeros(2560, dtype=np.float32)
for idx, weight in zip(ids, w):
    def expert(name):
        t = tensors[prefix+name+'_exps.weight']
        return gguf.quants.dequantize(t.data[idx], t.tensor_type).astype(np.float32)
    gate = expert('gate') @ x
    up = expert('up') @ x
    hidden = gate / (1 + np.exp(-gate)) * up
    ref += weight * (expert('down') @ hidden)
assert got.shape == ref.shape and np.isfinite(got).all()
rmse = np.sqrt(np.mean((got-ref)**2)/np.mean(ref**2))
print(f'ids={ids.tolist()} max_abs={np.max(np.abs(got-ref)):.9g} relative_rmse={rmse:.9g}')
assert rmse < .04
