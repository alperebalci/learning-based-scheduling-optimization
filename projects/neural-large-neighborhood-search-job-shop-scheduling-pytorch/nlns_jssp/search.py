
from __future__ import annotations
import numpy as np, torch
from .core import decode, exact_repair_block
from .features import candidate_windows,block_features

def learned_lns(instance,initial,model,iterations=8,block_length=5):
    seq=tuple(initial); history=[decode(instance,seq)[0]]
    for _ in range(iterations):
        wins=candidate_windows(seq,block_length)
        X=torch.tensor(np.stack([block_features(instance,seq,s,l) for s,l in wins]),dtype=torch.float32)
        with torch.no_grad(): scores=model(X).numpy()
        order=np.argsort(-scores)
        improved=False
        for idx in order[:min(3,len(order))]:
            s,l=wins[int(idx)]
            cand,obj,_=exact_repair_block(instance,seq,s,l)
            if obj<history[-1]:
                seq=cand; improved=True; break
        history.append(decode(instance,seq)[0])
        if not improved: break
    return seq,tuple(history)

def random_lns(instance,initial,seed=0,iterations=8,block_length=5):
    rng=np.random.default_rng(seed); seq=tuple(initial); hist=[decode(instance,seq)[0]]
    for _ in range(iterations):
        wins=candidate_windows(seq,block_length)
        s,l=wins[int(rng.integers(len(wins)))]
        cand,obj,_=exact_repair_block(instance,seq,s,l)
        if obj<hist[-1]: seq=cand
        hist.append(decode(instance,seq)[0])
    return seq,tuple(hist)

def oracle_lns(instance,initial,iterations=8,block_length=5):
    seq=tuple(initial); hist=[decode(instance,seq)[0]]
    for _ in range(iterations):
        bestseq=seq; best=hist[-1]
        for s,l in candidate_windows(seq,block_length):
            cand,obj,_=exact_repair_block(instance,seq,s,l)
            if obj<best: best,bestseq=obj,cand
        seq=bestseq; hist.append(best)
        if hist[-1]==hist[-2]: break
    return seq,tuple(hist)
