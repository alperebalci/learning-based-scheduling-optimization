
from __future__ import annotations
import argparse, numpy as np
from nlns_jssp import (
    generate_instance,greedy_sequence,decode,audit,
    build_dataset,train_scorer,learned_lns,random_lns,oracle_lns
)

def self_test():
    inst=generate_instance(1,3,3)
    seq=greedy_sequence(inst)
    obj,ops=decode(inst,seq)
    assert audit(inst,seq,ops)
    X,y=build_dataset(3,seed=2,n_jobs=3,n_machines=3,block_length=4)
    model,loss=train_scorer(X,y,epochs=3)
    assert np.isfinite(loss)
    print("Neural LNS JSSP self-test: OK")

def main(args):
    X,y=build_dataset(args.train_instances,seed=args.seed,n_jobs=args.jobs,n_machines=args.machines,block_length=args.block_length)
    model,loss=train_scorer(X,y,seed=args.seed,epochs=args.epochs)
    print(f"training examples={len(y)} final MSE={loss:.6f} positive-label-rate={(y>0).mean():.3f}")
    rows=[]
    for i in range(args.test_instances):
        inst=generate_instance(args.seed+2_000_000+1009*i,args.jobs,args.machines)
        init=greedy_sequence(inst,"mwkr")
        base=decode(inst,init)[0]
        _,lh=learned_lns(inst,init,model,args.iterations,args.block_length)
        _,rh=random_lns(inst,init,args.seed+i,args.iterations,args.block_length)
        _,oh=oracle_lns(inst,init,args.iterations,args.block_length)
        rows.append((base,lh[-1],rh[-1],oh[-1]))
    a=np.asarray(rows,float)
    print(f"{'method':<18}{'mean Cmax':>12}{'mean improve':>15}")
    for name,col in [("Initial MWKR",0),("Learned LNS",1),("Random LNS",2),("Oracle-block LNS",3)]:
        improve=np.mean((a[:,0]-a[:,col])/a[:,0])*100
        print(f"{name:<18}{a[:,col].mean():12.3f}{improve:14.2f}%")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--jobs",type=int,default=4)
    p.add_argument("--machines",type=int,default=4)
    p.add_argument("--block-length",type=int,default=5)
    p.add_argument("--train-instances",type=int,default=35)
    p.add_argument("--epochs",type=int,default=70)
    p.add_argument("--test-instances",type=int,default=20)
    p.add_argument("--iterations",type=int,default=7)
    return p.parse_args()

if __name__=="__main__":
    a=parse_args(); self_test() if a.self_test else main(a)
