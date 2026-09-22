
import unittest, numpy as np
from nlns_jssp import *
from nlns_jssp.core import unique_permutations

class Tests(unittest.TestCase):
    def test_decode_feasible(self):
        inst=generate_instance(1,3,3); seq=greedy_sequence(inst)
        obj,ops=decode(inst,seq); self.assertTrue(audit(inst,seq,ops)); self.assertGreater(obj,0)

    def test_exact_repair_matches_explicit_block_enumeration(self):
        inst=generate_instance(2,3,3); seq=greedy_sequence(inst)
        s,l=2,4; bestseq,best,_=exact_repair_block(inst,seq,s,l)
        vals=[]
        block=seq[s:s+l]
        for perm in unique_permutations(block):
            cand=seq[:s]+perm+seq[s+l:]
            vals.append(decode(inst,cand)[0])
        self.assertEqual(best,min(vals))

    def test_features_finite(self):
        inst=generate_instance(3,4,4); seq=greedy_sequence(inst)
        x=block_features(inst,seq,0,5)
        self.assertEqual(x.shape,(11,)); self.assertTrue(np.isfinite(x).all())

    def test_training_real_gradient(self):
        X,y=build_dataset(5,seed=4,n_jobs=3,n_machines=3,block_length=4)
        m,loss=train_scorer(X,y,epochs=5)
        self.assertTrue(np.isfinite(loss))

    def test_lns_never_worsens_incumbent(self):
        X,y=build_dataset(6,seed=5,n_jobs=3,n_machines=3,block_length=4)
        m,_=train_scorer(X,y,epochs=5)
        inst=generate_instance(999,3,3); seq=greedy_sequence(inst)
        for fn in [
            lambda: learned_lns(inst,seq,m,5,4),
            lambda: random_lns(inst,seq,1,5,4),
            lambda: oracle_lns(inst,seq,5,4),
        ]:
            _,hist=fn()
            self.assertTrue(all(b<=a for a,b in zip(hist,hist[1:])))

if __name__=="__main__": unittest.main()
