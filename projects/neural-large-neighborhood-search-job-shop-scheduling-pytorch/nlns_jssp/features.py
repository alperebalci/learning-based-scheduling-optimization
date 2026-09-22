
from __future__ import annotations
import numpy as np
from .core import decode

def candidate_windows(sequence,length):
    return [(s,length) for s in range(0,len(sequence)-length+1)]

def block_features(instance,sequence,start,length):
    obj,ops=decode(instance,sequence)
    block=sequence[start:start+length]
    jobs=np.asarray(block,int)
    counts=np.bincount(jobs,minlength=instance.n_jobs)
    block_ops=[ops[i] for i in range(start,start+length)]
    dur=np.array([instance.durations[o.job,o.op] for o in block_ops],float)
    machine_counts=np.bincount([o.machine for o in block_ops],minlength=instance.n_machines)
    feat=[
        obj/max(instance.durations.sum(),1),
        start/max(len(sequence)-1,1),
        length/len(sequence),
        len(np.unique(jobs))/instance.n_jobs,
        counts.max()/length,
        dur.mean()/max(instance.durations.max(),1),
        dur.std()/max(instance.durations.max(),1),
        dur.max()/max(instance.durations.max(),1),
        machine_counts.max()/length,
        np.mean([o.end for o in block_ops])/max(obj,1),
        np.max([o.end for o in block_ops])/max(obj,1),
    ]
    return np.asarray(feat,dtype=np.float32)
