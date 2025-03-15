import numpy as np


def _reshape_wa(wa_vec: np.ndarray, d_vars: int, p_orders: int):
    """
    Helper function for `_learn_dynamic_structure`. Transform adjacency vector to matrix form

    Args:
        wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights
        d_vars (int): number of variables in the model
        p_orders (int): number of past indexes we to use
    Returns:
        intra- and inter-slice adjacency matrices
    """

    w_tilde = wa_vec.reshape([2 * (p_orders + 1) * d_vars, d_vars])
    w_plus = w_tilde[:d_vars, :]
    w_minus = w_tilde[d_vars : 2 * d_vars, :]
    w_mat = w_plus - w_minus
    a_plus = (
        w_tilde[2 * d_vars :]
        .reshape(2 * p_orders, d_vars**2)[::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_minus = (
        w_tilde[2 * d_vars :]
        .reshape(2 * p_orders, d_vars**2)[1::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_mat = a_plus - a_minus
    return w_mat, a_mat
