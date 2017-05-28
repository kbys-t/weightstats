# coding:utf-8

import sys
import numpy as np

def mean(a, axis, weights, keepdims=False):
    if a.shape[axis] == len(weights):
        w = _govern_Weight(a, axis, weights)
        r = a*w
        return r[np.isfinite(r)].sum(axis=axis, keepdims=keepdims)
    else:
        sys.stderr.write("size is different in weightstats.mean\n")
        sys.exit(-1)

def var(a, axis, weights, keepdims=False):
    if a.shape[axis] == len(weights):
        m = mean(a, axis, weights, keepdims)
        w = _govern_Weight(a, axis, weights)
        r = np.power(a-m, 2)*w
        return r[np.isfinite(r)].sum(axis=axis, keepdims=keepdims)
    else:
        sys.stderr.write("size is different in weightstats.var\n")
        sys.exit(-1)

def std(a, axis, weights, keepdims=False):
    if a.shape[axis] == len(weights):
        return np.sqrt(var(a, axis, weights, keepdims))
    else:
        sys.stderr.write("size is different in weightstats.std\n")
        sys.exit(-1)

def percentile(a, q, axis, weights, keepdims=False):
    if a.shape[axis] == len(weights):
        r = np.apply_along_axis(_percentile_1D, axis, a, q, weights)
        if keepdims:
            k = tuple([ 1 if i == axis else x for i,x in enumerate(a.shape) ])
            return r.reshape(k)
        else:
            return r
    else:
        sys.stderr.write("size is different in weightstats.percentile\n")
        sys.exit(-1)

def median(a, axis, weights, keepdims=False):
    if a.shape[axis] == len(weights):
        return percentile(a, 50, axis, weights, keepdims)
    else:
        sys.stderr.write("size is different in weightstats.median\n")
        sys.exit(-1)

def _percentile_1D(a, q, weights, old=True):
    idx = np.argsort( a[ (np.isfinite(a))&(np.isfinite(weights)) ] )
    w = weights[idx].cumsum()
    if old:
        w -= w[0]
    s = np.maximum( w[-1] , 1e-12 )
    w *= 100.0 / s
    return np.interp(q, w, a[idx])

def _govern_Weight(a, axis, weights):
    k = tuple([len(weights) if i == axis else 1 for i in range(len(a.shape))])
    s = np.maximum( weights.sum() , 1e-12 )
    return weights.reshape(k) / s

if __name__ == '__main__':
    # test
    a = np.random.rand(6, 3, 2)
    w = np.random.rand(6)
    print("sample: ", a)
    print("weight: ", w)

    # mean
    print("mean: ", mean(a, 0, w, True), np.mean(a, axis=0, keepdims=True))
    # var
    print("var: ", var(a, 0, w, True), np.var(a, axis=0, keepdims=True))
    # std
    print("std: ", std(a, 0, w, True), np.std(a, axis=0, keepdims=True))
    # percentile
    print("percentile: ", percentile(a, 10, 0, w, True), np.percentile(a, 10, axis=0, keepdims=True))
    # median
    print("median: ", median(a, 0, w, True), np.median(a, axis=0, keepdims=True))
