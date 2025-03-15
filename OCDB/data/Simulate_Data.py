import networkx as nx
import numpy as np

class Simulate_Data():
    """
    Parameters
    ----------
    n: int, default: 5000,
        the first dimension of feature.
    num_variable: int, default: 10, 
        the number of variables.
    degree: int, default: 2,
        expected node degree.
    -----------
    self.G: numpy.array
    self.X: numpy.array
    """
    def __init__(self, n=5000, num_varibale=10, degree=2, graph_type='erdos-renyi', 
                 x_dim=1, sem_type='linear-gauss', linear_type='linear') -> None:
        self.G = self.simulate_random_dag(num_varibale, degree, graph_type)
        self.X = self.simulate_sem(self.G, n, x_dim, sem_type, linear_type)
        
    def data(self):
        return self.X, self.G

    def simulate_random_dag(self, num_varibale: int,
                        degree: float,
                        graph_type: str,
                        w_range: tuple = (0.5, 2.0)) -> np.ndarray:
        """Simulate random DAG with some expected degree.

        Args:
            d: number of nodes
            degree: expected node degree, in + out
            graph_type: {erdos-renyi, barabasi-albert, full}
            w_range: weight range +/- (low, high)

        Returns:
            G: weighted DAG
        """
        if graph_type == 'erdos-renyi':
            prob = float(degree) / (num_varibale - 1)
            B = np.tril((np.random.rand(num_varibale, num_varibale) < prob).astype(float), k=-1)
        elif graph_type == 'barabasi-albert':
            m = int(round(degree / 2))
            B = np.zeros([num_varibale, num_varibale])
            bag = [0]
            for ii in range(1, num_varibale):
                dest = np.random.choice(bag, size=m)
                for jj in dest:
                    B[ii, jj] = 1
                bag.append(ii)
                bag.extend(dest)
        elif graph_type == 'full':  # ignore degree, only for experimental use
            B = np.tril(np.ones([num_varibale, num_varibale]), k=-1)
        else:
            raise ValueError('unknown graph type')
        # random permutation
        P = np.random.permutation(np.eye(num_varibale, num_varibale))  # permutes first axis only
        B_perm = P.T.dot(B).dot(P)
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[num_varibale, num_varibale])
        U[np.random.rand(num_varibale, num_varibale) < 0.5] *= -1
        G_adj = (B_perm != 0).astype(float) * U
        return G_adj


    def simulate_sem(self, G: np.array,
                 n: int, x_dim: int,
                 sem_type: str,
                 linear_type: str,
                 noise_scale: float = 1.0) -> np.ndarray:
        """Simulate samples from SEM with specified type of noise.

        Args:
            G: weigthed DAG
            n: number of samples
            sem_type: {linear-gauss,linear-exp,linear-gumbel}
            noise_scale: scale parameter of noise distribution in linear SEM

        Returns:
            X: [n,d] sample matrix
        """
        W = G
        G = nx.DiGraph(G)
        d = W.shape[0]
        X = np.zeros([n, d, x_dim])
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            if linear_type == 'linear':
                eta = X[:, parents, 0].dot(W[parents, j])
            elif linear_type == 'nonlinear_1':
                eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j])
            elif linear_type == 'nonlinear_2':
                eta = (X[:, parents, 0]+0.5).dot(W[parents, j])
            else:
                raise ValueError('unknown linear data type')

            if sem_type == 'linear-gauss':
                if linear_type == 'linear':
                    X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
                elif linear_type == 'nonlinear_1':
                    X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
                elif linear_type == 'nonlinear_2':
                    X[:, j, 0] = 2.*np.sin(eta) + eta + np.random.normal(scale=noise_scale, size=n)
            elif sem_type == 'linear-exp':
                X[:, j, 0] = eta + np.random.exponential(scale=noise_scale, size=n)
            elif sem_type == 'linear-gumbel':
                X[:, j, 0] = eta + np.random.gumbel(scale=noise_scale, size=n)
            else:
                raise ValueError('unknown sem type')
        if x_dim > 1 :
            for i in range(x_dim-1):
                X[:, :, i+1] = np.random.normal(scale=noise_scale, size=1)*X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
            X[:, :, 0] = np.random.normal(scale=noise_scale, size=1) * X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
        return X

