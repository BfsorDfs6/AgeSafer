"""Evaluate a single augmented backbone with original-history exclusions."""
import argparse
import numpy as np
from common import Inputs,dump,evaluate
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--scores',required=True);p.add_argument('--output',required=True);p.add_argument('--backbone',required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 d=Inputs(a.config);x=np.load(a.scores,mmap_mode='r');assert x.shape==(d.nu,d.ni)
 lists=d.rank(x,'test',20);assert all(len(v)==20 for v in lists.values())
 dump(a.output,{'dataset':d.ds,'backbone':a.backbone,'seed':a.seed,'method':'LLM4IDRec+AS-single','metrics':evaluate(lists,d)})
if __name__=='__main__':main()
