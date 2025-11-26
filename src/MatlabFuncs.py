import numpy as np
import torch as pt
try:
    import tensorflow as tf
    if not tf.__file__:
        tf = None
except ImportError:
    tf = None


'''
extract the diagonal elements
@in1: the input data
@dims: a two elements list for the dimension to extract (if not given, use the last two dimesnion)
@keepdims: whether keep the dimension
'''
def DiagExt(in1, *args, keepdims=False):
    if not isinstance(in1, (np.ndarray, pt.Tensor)):
        if tf:
            if not tf.is_tensor(in1):
                in1 = np.asarray(in1)
        else:
            in1 = np.asarray(in1)
    
    dim0 = -1
    dim1 = -2
    if len(args) > 0:
        axes = args[0]
        if len(axes) != 2:
            raise Exception("axes must be two elements")
        else:
            dim0 = axes[0]
            dim1 = axes[1]
    
    if isinstance(in1, np.ndarray):
        diag = np.diagonal(in1, axis1=dim0, axis2=dim1)
    if isinstance(in1, pt.Tensor):
        diag = pt.diagonal(in1, dim1=dim0, dim2=dim1)
    if tf:
        if tf.is_tensor(in1):
            pass
    if keepdims:
        diag = diag[..., None]
    return diag

'''
generate a diagonal matrix
@in1: the input data
'''
def DiagGen(in1):
    if not isinstance(in1, (np.ndarray, pt.Tensor)):
        if tf:
            if not tf.is_tensor(in1):
                in1 = np.asarray(in1)
        else:
            in1 = np.asarray(in1)
    if isinstance(in1, np.ndarray):
        return in1[..., None]*np.eye(in1.shape[-1])
    if isinstance(in1, pt.Tensor):
        return pt.diag_embed(in1)
    if tf:
        if tf.is_tensor(in1):
            pass