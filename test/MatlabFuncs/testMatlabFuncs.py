import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from MatlabFuncs import *;

'''
DiagExt
'''
in0 = [[1,2], [3, 4]]
in1 = np.asarray(in0)
in2 = pt.tensor(in0)
d = [1, 4]
assert(sum(DiagExt(in0) == d) == 2)
assert(sum(DiagExt(in1) == d) == 2)
assert(sum(DiagExt(in2) == pt.tensor(d)) == 2)
# batch
in0 = [in0, in0]
in1 = np.asarray(in0)
in2 = pt.tensor(in0)
d = [d, d]
assert(np.sum(DiagExt(in0) == d) == 4)
assert(np.sum(DiagExt(in1) == d) == 4)
assert(pt.sum(DiagExt(in2) == pt.tensor(d)) == 4)

'''
DiagGen
'''
in0 = [1,2]
in1 = np.asarray(in0)
in2 = pt.tensor(in0)
d = [[1, 0], [0, 2]]
assert(np.sum(DiagGen(in0) == d) == 4)
assert(np.sum(DiagGen(in1) == d) == 4)
assert(pt.sum(DiagGen(in2) == pt.tensor(d)) == 4)
# batch
in0 = [in0, in0]
in1 = np.asarray(in0)
in2 = pt.tensor(in0)
d = [d, d]
assert(np.sum(DiagGen(in0) == d) == 8)
assert(np.sum(DiagGen(in1) == d) == 8)
assert(pt.sum(DiagGen(in2) == pt.tensor(d)) == 8)