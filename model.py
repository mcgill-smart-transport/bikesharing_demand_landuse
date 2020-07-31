from math import radians, cos, sin, asin, sqrt
import numpy as np
import cvxpy as cp
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# %% Functions for building the graph topology
def dist(lat1, lon1, lat2, lon2):
    """Calculate the distance between two stations"""

    # convert from degree to radians
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    # radius of earth in kms (6371) and miles (3956)
    r = 6371
    return c * r


def get_edges(dm, num):
    """
    Based on the number of neighbors.
    Get a list of **undirected** edges from distance matrix.

    Parameters
    ----------
    dm: distance matrix
    num: the number of neighbors

    Returns
    -------
    edges: ndarray, row i represents an edge from `edges[i,0]` to `edges[i,1]`.
    """
    order = np.argsort(dm, axis=1)
    edge_dict = {}  # A dictionary of edges like {start1: {end1, end2, ...}, start2...}
    for start in range(order.shape[0]):
        end_candidates = set(order[start, 1:num + 1])
        end_set = set()
        # The graph is undirected, so only add new edges
        for end in end_candidates:
            if end not in edge_dict:
                end_set.add(end)
            elif start not in edge_dict[end]:
                end_set.add(end)

        if end_set:
            edge_dict[start] = end_set

    # Transform the edge_dict to edges
    edges = []
    for start, end_set in edge_dict.items():
        edges += [(start, end) for end in end_set]
    return np.array(edges, dtype=np.int)


# %% Functions for the regression with the graph regularization
def loss_fun(beta, features, target):
    # Loss function for N independent linear regressions.
    f0 = cp.sum(cp.multiply(beta, features), axis=1)
    f1 = cp.sum_squares(f0 - target)
    return f1


def graph_reg(beta, edges, weight):
    # Graph regularization (squared).
    return weight @ cp.square(cp.pnorm(beta[edges[:, 0], :] - beta[edges[:, 1], :], axis=1, keepdims=True))


def objective(beta, features, target, edges, weight, lam):
    # Loss function for the linear regression with a graph regularization.
    return loss_fun(beta, features, target) + lam * graph_reg(beta, edges, weight)


# %% Functions for estimating the coefficients for new stations.
def loss_fun2(beta, neighbor_beta, weight):
    """
    Objective function to determine the coefficient of a new station.

    Parameters
    ----------
    beta: cp.Variable
        The coefficients of the new station.
    neighbor_beta: ndarray
        The coefficients of neighbor stations, shape=(number of neighbors, number of coefficients)
    weight: ndarray
    """
    y = 0
    for i in range(neighbor_beta.shape[0]):
        y = weight[i] * cp.square(cp.pnorm(beta - neighbor_beta[i, :]))
    return y


#%%
class RegressionGraph:
    """
    A linear regression with graph regularization.

    Key parameters
    ----------------
    dm: a N by N distance matrix for all stations (both training and test).
    features: a N by m feature array.
    target: a target array with N elements.
    num_nei: the number of neighbors for each station.
    decay: str, the type of weight decay function, 'power' or 'exp'.
    lam: float, controlling strength of the graph regularization.
    train_idx: optional, a index for training stations (with known targets).
    test_idx: optional, a index for test stations (assuming does not know the targets).

    Key methods
    ---------------
    fit: estimate the coefficients for training stations.
    predict: estimate the coefficients for test stations by interpolating the
        coefficients of training stations.
    update_train_test: update the training and test set and the corresponding graph structure.
    """
    def __init__(self, dm, features, target, num_nei=5, decay='power', alpha=1, lam=1, train_idx=None, test_idx=None):
        self.dm = dm
        self.features = features
        self.target = target
        self.num_nei = num_nei
        self.decay = decay
        self.alpha = alpha
        self.lam = lam
        self.N = dm.shape[0]
        if train_idx is None and test_idx is None:
            train_idx, test_idx = np.arange(0, self.N), np.arange(0, self.N)
        self.update_train_test(test_idx, test_idx)

    def update_train_test(self, train_idx, test_idx):
        self.train_num = len(train_idx)
        self.test_num = len(test_idx)
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.train_features = self.features[train_idx, :]
        self.train_target = self.target[train_idx]
        self.test_features = self.features[test_idx, :]
        self.test_target = self.target[test_idx]

        self.train_dm = self.dm[train_idx.reshape((-1, 1)), train_idx.reshape(1, -1)]
        self.dm_test_train = self.dm[test_idx.reshape((-1, 1)), self.train_idx.reshape((1, -1))]
        self.build_edges()

    def build_edges(self, num_nei=None):
        if num_nei:
            self.num_nei = num_nei
        self.train_edges = get_edges(self.train_dm, self.num_nei)
        self.test_neighbors = np.argsort(self.dm_test_train)[:, 1:self.num_nei + 1]  # In the reference of training set
        self.build_weight()

    def build_weight(self, alpha=None, decay=None):
        if alpha:
            self.alpha = alpha
        if decay:
            self.decay = decay
        if self.decay == 'power':
            self.train_weight = self.train_dm[self.train_edges[:, 0], self.train_edges[:, 1]] ** (-self.alpha)
            self.test_weight = self.dm_test_train[np.arange(self.test_num).reshape((self.test_num, 1)),
                                                  self.test_neighbors] ** (-self.alpha)
        elif self.decay == 'exp':
            self.train_weight = np.exp(-self.alpha * self.train_dm[self.train_edges[:, 0], self.train_edges[:, 1]])
            self.test_weight = np.exp(-self.alpha * self.dm_test_train[np.arange(self.test_num).reshape((-1, 1)),
                                                                       self.test_neighbors])

    def fit(self):
        beta = cp.Variable(self.train_features.shape)
        self.problem = cp.Problem(cp.Minimize(objective(beta, self.train_features, self.train_target,
                                                        self.train_edges, self.train_weight,
                                                        self.lam)))
        self.problem.solve()
        self.fitted_beta = beta.value
        self.fitted_Y = np.sum(self.fitted_beta * self.train_features, axis=1)

    def predict(self):
        predicted_beta = np.zeros(self.test_features.shape)
        for i in range(self.test_num):
            x = cp.Variable(self.test_features.shape[1])
            neighbors = self.test_neighbors[i, :]
            weights = self.test_weight[i, :]
            problem2 = cp.Problem(cp.Minimize(loss_fun2(x, self.fitted_beta[neighbors, :], weights)))
            problem2.solve()
            predicted_beta[i, :] = x.value
        self.predicted_beta = predicted_beta
        self.predicted_Y = np.sum(self.predicted_beta * self.test_features, axis=1)

    def fit_lam(self, lam):
        self.lam = lam
        self.fit()
        self.predict()
        return mse(self.test_target, self.predicted_Y)

    def optimize_lam(self, bounds=(0.01, 10), tol=0.01):
        "Optimize lambda based on mse at test set"
        res = minimize_scalar(self.fit_lam, bounds=bounds, tol=tol)
        return res.x, res.fun

    def cross_validate(self, train_idx_list, test_idx_list, **kwargs):
        mse_list = []
        for train_idx, test_idx in zip(train_idx_list, test_idx_list):
            self.update_train_test(train_idx, test_idx)
            _, mse = self.optimize_lam(**kwargs)
            mse_list.append(mse)
        return np.mean(mse_list)
