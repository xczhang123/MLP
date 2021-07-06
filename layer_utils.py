from layers import *

# Ref: cs231
def affine_norm_sigmoid_forward(x, w, b, gamma, beta, bn_param, normalization, dropout, do_param):
    bn_cache, do_cache = None, None
    out, fc_cache = affine_forward(x,w,b)
    
    if normalization == 'batchnorm':
       out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
    
    out, sigmoid_cache = sigmoid_forward(out)
    
    if dropout:
       out, do_cache = dropout_forward(out, do_param)
    
    return out, (fc_cache, bn_cache, sigmoid_cache, do_cache)

def affine_norm_sigmoid_backward(dout, cache, normalization, dropout):
    fc_cache, bn_cache, sigmoid_cache, do_cache = cache
    
    if dropout:
       dout = dropout_backward(dout, do_cache)
    
    dout = sigmoid_backward(dout, sigmoid_cache)
    
    dgamma, dbeta = None, None
    if normalization == 'batchnorm':
       dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)   
    
    dx, dw, db = affine_backward(dout, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_norm_tanh_forward(x, w, b, gamma, beta, bn_param, normalization, dropout, do_param):
    bn_cache, do_cache = None, None
    out, fc_cache = affine_forward(x,w,b)
    
    if normalization == 'batchnorm':
       out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
    
    out, tanh_cache = tanh_forward(out)
    
    if dropout:
       out, do_cache = dropout_forward(out, do_param)
    
    return out, (fc_cache, bn_cache, tanh_cache, do_cache)

def affine_norm_tanh_backward(dout, cache, normalization, dropout):
    fc_cache, bn_cache, tanh_cache, do_cache = cache
    
    if dropout:
       dout = dropout_backward(dout, do_cache)
    
    dout = tanh_backward(dout, tanh_cache)
    
    dgamma, dbeta = None, None
    if normalization == 'batchnorm':
       dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)   
    
    dx, dw, db = affine_backward(dout, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_norm_lrelu_forward(x, w, b, a, gamma, beta, bn_param, normalization, dropout, do_param):
    bn_cache, do_cache = None, None
    out, fc_cache = affine_forward(x,w,b)
    
    if normalization == 'batchnorm':
       out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
    
    out, lrelu_cache = lrelu_forward(out, a)
    
    if dropout:
       out, do_cache = dropout_forward(out, do_param)
    
    return out, (fc_cache, bn_cache, lrelu_cache, do_cache)

def affine_norm_lrelu_backward(dout, cache, normalization, dropout):
    fc_cache, bn_cache, lrelu_cache, do_cache = cache
    
    if dropout:
       dout = dropout_backward(dout, do_cache)
    
    dout, da = lrelu_backward(dout, lrelu_cache)
    
    dgamma, dbeta = None, None
    if normalization == 'batchnorm':
       dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)   
    
    dx, dw, db = affine_backward(dout, fc_cache)
    return dx, dw, db, da, dgamma, dbeta


def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param, normalization, dropout, do_param):
    bn_cache, do_cache = None, None
    out, fc_cache = affine_forward(x,w,b)
    
    if normalization == 'batchnorm':
       out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
    
    out, relu_cache = relu_forward(out)
    
    if dropout:
       out, do_cache = dropout_forward(out, do_param)
    
    return out, (fc_cache, bn_cache, relu_cache, do_cache)

def affine_norm_relu_backward(dout, cache, normalization, dropout):
    fc_cache, bn_cache, relu_cache, do_cache = cache
    
    if dropout:
       dout = dropout_backward(dout, do_cache)
    
    dout = relu_backward(dout, relu_cache)
    
    dgamma, dbeta = None, None
    if normalization == 'batchnorm':
       dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)   
    
    dx, dw, db = affine_backward(dout, fc_cache)
    return dx, dw, db, dgamma, dbeta