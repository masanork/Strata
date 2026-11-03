"""Compare two sequential real-weight attention positions with NumPy."""
import os
import sys
import numpy as np
sys.path.insert(0, os.environ['STRATA_GGUF_PY'])
import gguf
r=gguf.GGUFReader(sys.argv[1]); tensors={t.name:t for t in r.tensors}
prefix='blk.'+sys.argv[2]+'.attn_'
def weights(name):
    t=tensors[prefix+name+'.weight']
    return gguf.quants.dequantize(t.data,t.tensor_type).astype(np.float32)
qweight,kweight,vweight,oweight=(weights(n) for n in ('q','k','v','output'))
qg,kg=(weights(n) for n in ('q_norm','k_norm'))
def norm_rope(v,g,pos):
    v=v.reshape(-1,128)
    v=v/np.sqrt(np.mean(v*v,axis=1,keepdims=True)+np.float32(1e-6))*g
    angles=np.float32(pos)*np.power(np.float32(500000),-np.arange(64,dtype=np.float32)/np.float32(64))
    c,s=np.cos(angles),np.sin(angles)
    return np.concatenate((v[:,:64]*c-v[:,64:]*s,v[:,64:]*c+v[:,:64]*s),axis=1)
keys=[];values=[];results=[]
for pos in range(2):
    x=np.sin(np.arange(2560,dtype=np.float32)*np.float32(.013)+np.float32(pos)*np.float32(.31))*np.float32(.1)
    q=norm_rope(qweight@x,qg,pos);keys.append(norm_rope(kweight@x,kg,pos));values.append((vweight@x).reshape(4,128))
    outputs=[]
    for h in range(40):
        kv=h//10
        scores=np.array([q[h]@k[kv]/np.sqrt(128) for k in keys])
        probs=np.exp(scores-scores.max());probs/=probs.sum()
        outputs.append(sum(p*v[kv] for p,v in zip(probs,values)))
    results.append(oweight@np.array(outputs).reshape(-1))
ref=np.array(results);gpu=np.fromfile(sys.argv[3],dtype=np.float32).reshape(2,2560)
assert np.isfinite(gpu).all()
for pos in range(2):
    rmse=np.sqrt(np.mean((gpu[pos]-ref[pos])**2)/np.mean(ref[pos]**2))
    print(f'position={pos} max_abs={np.max(np.abs(gpu[pos]-ref[pos])):.9g} relative_rmse={rmse:.9g}')
    assert rmse < .04
