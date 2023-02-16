import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils_data

from . import base


class SliceInfominLayer(base.ParametricInfoEstimator):
    '''
        sub-network used in infomin, based on slicing
    '''
    def __init__(self, architecture, hyperparams={}):
        super().__init__(hyperparams=hyperparams)
        self.architecture = architecture
        self.n_slice = hyperparams.get('n_slice', 200)
        self.solver_mode = hyperparams.get('solver_mode', 'CCA')
        self.slice_mode = hyperparams.get('slice_mode', 'sphere')

        self.bs = 500
        self.lr = 1e-3
        self.wd = 0e-3

        self.init_modules()

    def init_modules(self):
        D, order = self.n_slice, 3
        arch = self.architecture
        self.dim_weight = D * order + 1
        self.slice = Slice(arch[0], arch[-1], D, self.slice_mode)
        self.cca = linear_cca()
        self.m1, self.m2, self.w1, self.w2 = None, None, None, None

    def forward(self, z):
        zz = self.features(z)  # n*(3S+1), S = n_slice
        out = self.out(zz)
        return out             # n*K, K = output dim

    def features(self, z, mode=1):
        n, d = z.size()
        z = self.slice.forward(z, mode)
        z = torch.tanh(z)
        ones = torch.zeros(n, 1).to(z.device)
        z = torch.cat([ones, z, z**2, z**3], dim=1)
        return z

    def solve_CCA(self, z1, z2):
        z1_train, z2_train, z1_val, z2_val = utils_data.divide_train_val(z1, z2)
        # [1]. compute slices
        Z, Y = utils_data.standardize_(z1_train), utils_data.standardize_(z2_train)
        Z = self.features(Z, mode=1).detach().cpu().numpy()
        Y = self.features(Y, mode=2).detach().cpu().numpy()
        # [2]. run CCA
        cca = self.cca
        cca.fit(Z, Y, outdim_size=1)
        cca.to_torch(z1.device)
        self.m1, self.m2 = cca.m[0].clone().requires_grad_(True), cca.m[1].clone().requires_grad_(True)
        self.w1, self.w2 = cca.w[0].clone().requires_grad_(True), cca.w[1].clone().requires_grad_(True)
        # [3]. check train & val objective values
        mi_train = self.objective_func(z1_train, z2_train).item()
        mi_val = self.objective_func(z1_val, z2_val).item()
        return mi_train, mi_val

    def solve_SGD(self, z1, z2):
        return super().learn(z1, z2)

    def objective_func(self, z1, z2):
        z1, z2 = utils_data.standardize_(z1), utils_data.standardize_(z2)
        z = self.features(z1, mode=1)
        y = self.features(z2, mode=2)
        ZZ, YY = torch.matmul(z - self.m1, self.w1), torch.matmul(y - self.m2, self.w2)
        ZZ, YY = utils_data.standardize_(ZZ), utils_data.standardize_(YY)
        ret = (ZZ * YY).mean(dim=0).abs()
        return ret

    def learn(self, z1, z2):
        self.to(z1.device)
        z1, z2 = z1.float(), z2.float()
        if self.solver_mode == 'CCA':
            self.init_modules()
            self.to(z1.device)
            return self.solve_CCA(z1, z2)
        if self.solver_mode == 'SGD':
            self.to(z1.device)
            return self.solve_SGD(z1, z2)


class Slice(nn.Module):
    def __init__(self, d1, d2, k, style='sphere'):
        super(Slice, self).__init__()
        self.w1 = nn.Linear(d1, k, bias=True)
        self.w2 = nn.Linear(d2, k, bias=True)
        self.style = style
        self.init_weights(self.w1)
        self.init_weights(self.w2)

    def init_weights(self, w):
        k, d1 = w.weight.data.size()
        if self.style == 'orth':
            v1 = torch.Tensor(ortho_group.rvs(dim=d1))   # v1*v1^T = I, row i orth to row j
            w.weight.data = v1[0:k, :]
        if self.style == 'sphere':
            v1 = torch.randn(k, d1)
            w.weight.data = (v1 / torch.norm(v1, dim=1, keepdim=True))[:k, :]
        if self.style == 'gaussian':
            w.weight.data = torch.randn(k, d1)/(d1 ** 0.5)
            w.bias.data = torch.randn(self.w.bias.data.size()) * 0.1

    def _forward(self, x, linear):
        if self.style != 'gaussian':
            w = linear.weight
            w_norm = torch.norm(w, dim=1, keepdim=True)
            w = w / w_norm
            n, d = x.size()
            z = F.linear(x, w)
        else:
            w, b = linear.weight, linear.bias
            z = F.linear(x, w) + b
        return z

    def forward(self, x, mode=1):
        return self._forward(x, self.w1) if mode==1 else self._forward(x, self.w2)


# Copied from https://github.com/Michaelvll/DeepCCA
class linear_cca():
    def __init__(self):
        self.w = [None, None]
        self.m = [None, None]

    def fit(self, H1, H2, outdim_size):
        """
        An implementation of linear CCA
        # Arguments:
            H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
            outdim_size: specifies the number of new features
        # Returns
            A and B: the linear transformation matrices
            mean1 and mean2: the means of data for both views
        """
        r1 = 1e-3
        r2 = 1e-3

        m = H1.shape[0]
        o1 = H1.shape[1]
        o2 = H2.shape[1]

        self.m[0] = np.mean(H1, axis=0)
        self.m[1] = np.mean(H2, axis=0)
        H1bar = H1 - np.tile(self.m[0], (m, 1))
        H2bar = H2 - np.tile(self.m[1], (m, 1))

        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o1)
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o2)

        [D1, V1] = np.linalg.eigh(SigmaHat11)
        [D2, V2] = np.linalg.eigh(SigmaHat22)
        SigmaHat11RootInv = np.dot( np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = np.dot( np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

        Tval = np.dot(np.dot(SigmaHat11RootInv,
                                   SigmaHat12), SigmaHat22RootInv)

        [U, D, V] = np.linalg.svd(Tval)
        V = V.T
        self.w[0] = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
        self.w[1] = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
        D = D[0:outdim_size]

    def _get_result(self, x, idx):
        result = x - self.m[idx].reshape([1, -1]).repeat(len(x), axis=0)
        result = numpy.dot(result, self.w[idx])
        return result

    def test(self, H1, H2):
        return self._get_result(H1, 0), self._get_result(H2, 1)

    def to_torch(self, device):
        self.m[0] = torch.Tensor(self.m[0]).to(device)
        self.m[1] = torch.Tensor(self.m[1]).to(device)
        self.w[0] = torch.Tensor(self.w[0]).to(device)
        self.w[1] = torch.Tensor(self.w[1]).to(device)

