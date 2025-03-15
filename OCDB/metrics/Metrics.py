import networkx as nx
import numpy as np
import cdt.metrics as cm
from sklearn.metrics import confusion_matrix


def f1_score(G_true, G_pred):
        label = G_true.flatten().tolist()
        pred = G_pred.flatten().tolist()
        tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        return 2 * recall * precision / (recall + precision)

def CSD(G_true, G_pred):
    return np.sum(np.abs(G_true-G_pred))

def SID(G_true, G_pred):
    CPDAG = False
    for i in range(G_pred.shape[0]):
        for j in range(i, G_pred.shape[0]):
            if G_pred[i][j] == 1 and G_pred[j][i] == 1:
                CPDAG = True
                break
        if CPDAG:
            break
    if CPDAG:
        print("The final Causal Graph is a CPDAG.")
    else:
        print("The final Causal Graph is a DAG.")
    if not CPDAG:
        return cm.SID(G_true, G_pred)
    else:
        return cm.SID_CPDAG(G_true, G_pred)

def CED(G, A, k):
    """Compute Causal Effect Distance(CED).

    Args:
        G: Ground truth graph
        A: Predicted graph
        k: Maximum length of reachable path

    Returns:
        CED
    """
    def Reachable_Matrix(adj, k):
        # compute k-hop Reachable Matrix
        adj = adj + np.eye(k)
        adj = adj > 0

        adj = np.linalg.matrix_power(adj, k - 1)
        return adj
    
    G_hat = Reachable_Matrix(G, k)
    A_hat = Reachable_Matrix(A, k)
    # False Descendant Relationship 
    FDR = np.sum(np.abs(G_hat != A_hat))
    # The node pair with the true descendant relationship
    E = np.argwhere(G_hat == A_hat)
    TDR = 0
    for edge in E:
        i, j = edge
        if i == j or G_hat[i, j] == False:
            continue
        
        A_copy = A.copy()
        A_copy[j, i] = 0
        A_copy[i, i] = 0

        Z = np.argwhere(A_copy[:, i] > 0)
        if len(Z) > 0:
            H = G.copy()
            for z in Z:
                PA = np.argwhere(H[:, z] == 1)
                H[z, PA] = 1
            H[j, :] = 0
            H_hat = Reachable_Matrix(H, k)
            # By controlling z whether or not opens part of the path that was originally blocked
            error1 = np.sum(H_hat[i, Z] * H_hat[Z, j])
            if error1 > 0:
                TDR += 1
                continue

        T = G.copy()
        T[:, Z] = 0
        T[Z, :] = 0
        T[i, :] = 0
        T_hat = Reachable_Matrix(T, k)
        # By controlling z whether or not there are still unblocked paths.
        error2 = np.sum(T_hat[:, i] * T_hat[:, j])
        if error2 > 0:
            TDR += 1
            continue
        
        # whether the valid adjustment sets include any descendants of intermediate nodes on the directed path
        if len(Z) > 0:
            M = G.copy()
            M[j, :] = 0
            M_hat = Reachable_Matrix(M, k)
            M_hat[i, i] = 0
            M_hat[j, j] = 0
            descendant_i = M_hat[i, :]
            ancester_z = np.sum(np.reshape(M_hat[:, Z], newshape=(M.shape[0], -1)), axis=1)
            ancestor_j = M_hat[:, j]
            error3 = np.sum(descendant_i * ancestor_j * ancester_z)
            if error3 > 0:
                TDR += 1
                continue
    return FDR + TDR


def SHD_CPDAG(G_true, G_pred):
    return cm.SHD_CPDAG(G_true, G_pred)