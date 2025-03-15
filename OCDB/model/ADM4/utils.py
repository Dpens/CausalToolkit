import numpy as np

class Parameters(object):
    def __init__(self, decay=3, max_iter=5, em_max_iter=50, K = 6, rho=.1, threshold=.01, sparse=True, low_rank= False):
        self.decay = decay
        self.max_iter = max_iter
        self.em_max_iter = em_max_iter
        self.K = K  # num of nodes
        self.rho = rho
        self.threshold = threshold

        self.sparse=sparse
        self.low_rank = low_rank
        self.alpha = 0.1


def kernel_integration(dt, decay):
    # exp kernel
    G = (np.exp(-decay * dt) - 1) / - 1
    G[np.where(G < 0)] = 0
    return G


def soft_thres_S(A, threshold):
    tmp = A.copy()
    s = np.sign(tmp)
    tmp = abs(tmp) - threshold
    tmp[np.where(tmp <= 0)] = 0
    return s * tmp

