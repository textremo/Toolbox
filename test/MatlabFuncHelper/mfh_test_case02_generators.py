import numpy as np
import torch as pt
import tensorflow as tf
from MatlabFuncHelper import MatlabFuncHelper

bs = 5;
mfh = MatlabFuncHelper();

# ones
# ones - np
mfh.batch_size = None;
rtn_ones = mfh.ones(4, order="F");
assert(rtn_ones.shape == (4,4));
rtn_ones = mfh.ones(4, order="C");
assert(rtn_ones.shape == (4,));
mfh.batch_size = 18;
rtn_ones = mfh.ones(4, order="F");
assert(rtn_ones.shape == (18,4,4));
rtn_ones = mfh.ones(4, order="C");
assert(rtn_ones.shape == (18,4));
# ones - pt
mfh.toPT();
mfh.batch_size = None;
rtn_ones = mfh.ones(4, order="F");
assert(rtn_ones.shape == (4,4));
rtn_ones = mfh.ones(4, order="C");
assert(rtn_ones.shape == (4,));
mfh.batch_size = 18;
rtn_ones = mfh.ones(4, order="F");
assert(rtn_ones.shape == (18,4,4));
rtn_ones = mfh.ones(4, order="C");
assert(rtn_ones.shape == (18,4));
# ones - tf
mfh.toTF();
mfh.batch_size = None;
rtn_ones = mfh.ones(4, order="F");
assert(rtn_ones.shape == (4,4));
rtn_ones = mfh.ones(4, order="C");
assert(rtn_ones.shape == (4,));
mfh.batch_size = 18;
rtn_ones = mfh.ones(4, order="F");
assert(rtn_ones.shape == (18,4,4));
rtn_ones = mfh.ones(4, order="C");
assert(rtn_ones.shape == (18,4));

# eye
mfh.batch_size = None;
rtn_eye = mfh.eye(4);
assert(rtn_eye.shape == (4,4));
rtn_eye = mfh.eye(4, 8);
assert(rtn_eye.shape == (4,8));
mfh.batch_size = 18;
rtn_eye = mfh.eye(4);
assert(rtn_eye.shape == (18,4,4));
rtn_eye = mfh.eye(4, 8);
assert(rtn_eye.shape == (18,4,8));



# eye
