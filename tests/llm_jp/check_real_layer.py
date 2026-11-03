"""Compare two complete sequential decoder-layer outputs with NumPy."""
import os
import sys
import numpy as np
sys.path.insert(0, os.environ['STRATA_GGUF_PY'])
import gguf
r=gguf.GGUFReader(sys.argv[1]); tensors={t.name:t for t in r.tensors}
prefix='blk.'+sys.argv[2]+'.attn_'
with open(sys.argv[3], 'rb') as f:
    gpu=np.fromfile(f,dtype=np.float32,count=5120).reshape(2,2560)
    gpu_ids=np.fromfile(f,dtype=np.int32,count=16).reshape(2,8)
def weights(name):
    t=tensors[prefix+name+'.weight']
    return gguf.quants.dequantize(t.data,t.tensor_type).astype(np.float32)
qweight,kweight,vweight,oweight=(weights(n) for n in ('q','k','v','output'))
qg,kg=(weights(n) for n in ('q_norm','k_norm'))
base='blk.'+sys.argv[2]+'.'
attn_norm=tensors[base+'attn_norm.weight'].data
ffn_norm=tensors[base+'ffn_norm.weight'].data
router=tensors[base+'ffn_gate_inp.weight'].data
def rms(x,g): return x/np.sqrt(np.mean(x*x)+np.float32(1e-6))*g
def expert(name,idx):
    t=tensors[base+'ffn_'+name+'_exps.weight']
    return gguf.quants.dequantize(t.data[idx],t.tensor_type).astype(np.float32)
def norm_rope(v,g,pos):
    v=v.reshape(-1,128)
    v=v/np.sqrt(np.mean(v*v,axis=1,keepdims=True)+np.float32(1e-6))*g
    angles=np.float32(pos)*np.power(np.float32(500000),-np.arange(64,dtype=np.float32)/np.float32(64))
    c,s=np.cos(angles),np.sin(angles)
    return np.concatenate((v[:,:64]*c-v[:,64:]*s,v[:,64:]*c+v[:,:64]*s),axis=1)
keys=[];values=[];results=[]
for pos in range(2):
    x=np.sin(np.arange(2560,dtype=np.float32)*np.float32(.013)+np.float32(pos)*np.float32(.31))*np.float32(.1)
    xn=rms(x,attn_norm)
    q=norm_rope(qweight@xn,qg,pos);keys.append(norm_rope(kweight@xn,kg,pos));values.append((vweight@xn).reshape(4,128))
    outputs=[]
    for h in range(40):
        kv=h//10
        scores=np.array([q[h]@k[kv]/np.sqrt(128) for k in keys])
        probs=np.exp(scores-scores.max());probs/=probs.sum()
        outputs.append(sum(p*v[kv] for p,v in zip(probs,values)))
    residual=x+oweight@np.array(outputs).reshape(-1)
    xn=rms(residual,ffn_norm)
    logits=router@xn;ids=np.argsort(-logits,kind='stable')[:8]
    print('position',pos,'float route',ids.tolist(),'GPU route',gpu_ids[pos].tolist(),flush=True)
    ids=gpu_ids[pos]
    w=np.exp(logits[ids]-logits[ids].max());w/=w.sum()
    ffn=np.zeros(2560,dtype=np.float32)
    for idx,weight in zip(ids,w):
        gate=expert('gate',idx)@xn;up=expert('up',idx)@xn
        ffn+=weight*(expert('down',idx)@(gate/(1+np.exp(-gate))*up))
    results.append(residual+ffn)
ref=np.array(results)
assert np.isfinite(gpu).all()
for pos in range(2):
    rmse=np.sqrt(np.mean((gpu[pos]-ref[pos])**2)/np.mean(ref[pos]**2))
    print(f'position={pos} max_abs={np.max(np.abs(gpu[pos]-ref[pos])):.9g} relative_rmse={rmse:.9g}')
    assert rmse < .06
