
from __future__ import annotations
import numpy as np, torch
from .core import generate_instance,greedy_sequence,exact_repair_block
from .features import candidate_windows,block_features
from .model import BlockScorer

def build_dataset(n_instances=40,seed=42,n_jobs=4,n_machines=4,block_length=5):
    X=[]; y=[]
    for i in range(n_instances):
        inst=generate_instance(seed+1009*i,n_jobs,n_machines)
        seq=greedy_sequence(inst,"mwkr")
        base=__import__("nlns_jssp.core",fromlist=["decode"]).decode(inst,seq)[0]
        for s,l in candidate_windows(seq,block_length):
            _,best,_=exact_repair_block(inst,seq,s,l)
            X.append(block_features(inst,seq,s,l))
            y.append((base-best)/max(base,1))
    return np.stack(X),np.asarray(y,dtype=np.float32)

def train_scorer(X,y,seed=42,epochs=80):
    torch.manual_seed(seed); np.random.seed(seed)
    X=torch.tensor(X,dtype=torch.float32); y=torch.tensor(y,dtype=torch.float32)
    model=BlockScorer(X.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=2e-3)
    for _ in range(epochs):
        pred=model(X); loss=torch.nn.functional.mse_loss(pred,y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model,float(loss.item())
