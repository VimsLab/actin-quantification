import numpy as np

def least_squares(x, y):
    x_ = x.mean()
    y_ = y.mean()

    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)

    for i in range(len(x)):
        k = (x[i] - x_) * (y[i] - y_)
        m += k
        p = np.square(x[i] - x_)
        n = n + p

    a = m/(n+0.00000000000001)
    b = y_ - a * x_

    return a,b