import numpy as np

def df(node, weights, x):
    fp_2 = node(weights, x=(x + np.pi / 2.0))
    fm_2 = node(weights, x=(x - np.pi / 2.0))
    return (fp_2 + fm_2) / 2.0


def d2f(node, weights, x):
    f = node(weights, x=x)
    fp = node(weights, x=(x + np.pi))
    fm = node(weights, x=(x - np.pi))
    return (fm + fp - 2.0 * f) / 4.0