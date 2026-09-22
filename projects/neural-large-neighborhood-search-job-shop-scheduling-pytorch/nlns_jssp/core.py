
from __future__ import annotations
from dataclasses import dataclass
import itertools, numpy as np

@dataclass(frozen=True)
class JSSP:
    machines: np.ndarray
    durations: np.ndarray
    def __post_init__(self):
        m=np.asarray(self.machines,int); p=np.asarray(self.durations,int)
        if m.ndim!=2 or p.shape!=m.shape: raise ValueError("shape mismatch")
        if np.any(p<=0): raise ValueError("positive durations required")
        M=m.shape[1]
        for row in m:
            if sorted(row.tolist())!=list(range(M)): raise ValueError("each job must visit each machine once")
    @property
    def n_jobs(self): return int(self.machines.shape[0])
    @property
    def n_machines(self): return int(self.machines.shape[1])
    @property
    def n_ops(self): return self.n_jobs*self.n_machines

def generate_instance(seed=42,n_jobs=4,n_machines=4):
    rng=np.random.default_rng(seed)
    m=np.stack([rng.permutation(n_machines) for _ in range(n_jobs)])
    p=rng.integers(1,10,size=(n_jobs,n_machines))
    return JSSP(m,p)

@dataclass(frozen=True)
class Operation:
    job:int; op:int; machine:int; start:int; end:int

def decode(instance:JSSP, sequence):
    seq=tuple(map(int,sequence))
    if len(seq)!=instance.n_ops: raise ValueError("sequence length")
    counts=np.bincount(seq,minlength=instance.n_jobs)
    if not np.all(counts==instance.n_machines): raise ValueError("each job must appear M times")
    nextop=np.zeros(instance.n_jobs,int); jr=np.zeros(instance.n_jobs,int); mr=np.zeros(instance.n_machines,int)
    ops=[]
    for j in seq:
        k=int(nextop[j]); mach=int(instance.machines[j,k]); dur=int(instance.durations[j,k])
        st=int(max(jr[j],mr[mach])); en=st+dur
        ops.append(Operation(j,k,mach,st,en))
        jr[j]=en; mr[mach]=en; nextop[j]+=1
    return int(max(jr)),tuple(ops)

def greedy_sequence(instance:JSSP, rule="mwkr"):
    remaining=[list(range(instance.n_machines)) for _ in range(instance.n_jobs)]
    seq=[]
    while len(seq)<instance.n_ops:
        candidates=[j for j in range(instance.n_jobs) if remaining[j]]
        if rule=="spt":
            j=min(candidates,key=lambda a:(instance.durations[a,remaining[a][0]],a))
        else:
            j=max(candidates,key=lambda a:(instance.durations[a,remaining[a]].sum(),-a))
        seq.append(j); remaining[j].pop(0)
    return tuple(seq)

def audit(instance,sequence,ops):
    if len(ops)!=instance.n_ops: return False
    for j in range(instance.n_jobs):
        js=sorted([o for o in ops if o.job==j],key=lambda x:x.op)
        if len(js)!=instance.n_machines:return False
        for a,b in zip(js,js[1:]):
            if a.end>b.start:return False
    for m in range(instance.n_machines):
        ms=sorted([o for o in ops if o.machine==m],key=lambda x:x.start)
        for a,b in zip(ms,ms[1:]):
            if a.end>b.start:return False
    return True

def unique_permutations(values):
    return set(itertools.permutations(values))

def exact_repair_block(instance,sequence,start,length):
    seq=list(sequence); block=seq[start:start+length]
    best=tuple(seq); best_obj=decode(instance,best)[0]
    explored=0
    for perm in unique_permutations(block):
        cand=tuple(seq[:start]+list(perm)+seq[start+length:])
        obj=decode(instance,cand)[0]; explored+=1
        if obj<best_obj:
            best_obj=obj; best=cand
    return best,best_obj,explored
