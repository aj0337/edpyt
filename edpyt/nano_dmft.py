import numpy as np
from edpyt.integrate_gf import matsum_gf as integrate_gf
from edpyt.observs import get_occupation

# from edpyt.dmft import _DMFT, adjust_mu


def _get_sigma_method(comm):
    if comm is not None:

        def wrap(self, z):
            # Collect sigmas and expand to non equivalent indices.
            comm = self.comm
            sigma_loc = self.Sigma(z)
            sigma = np.empty(comm.size * sigma_loc.size, sigma_loc.dtype)
            comm.Allgather(
                [sigma_loc, sigma_loc.size],
                [sigma, sigma_loc.size],
            )
            shape = list(sigma_loc.shape)
            shape[0] *= comm.size
            return sigma.reshape(shape)
    else:

        def wrap(self, z):
            return self.Sigma(z)

    return wrap


def _get_occps_method(comm):
    if comm is not None:

        def wrap(self, mu):
            occps_loc = integrate_gf(self, mu)
            occps = np.empty(occps_loc.size * self.comm.size, occps_loc.dtype)
            self.comm.Allgather(
                [occps_loc, occps_loc.size],
                [occps, occps_loc.size],
            )
            shape = list(occps_loc.shape)
            shape[0] *= self.comm.size
            return np.squeeze(occps.reshape(shape)[self.idx_inv, ...])
    else:

        def wrap(self, mu):
            occps = integrate_gf(self, mu)
            return np.squeeze(occps[self.idx_inv, ...])

    return wrap


def _get_idx_world(comm, n):
    if comm is None:
        return slice(None)
    else:
        stride = n // comm.size
        return slice(comm.rank * stride, (comm.rank + 1) * stride)


class Gfloc:
    """nano Local lattice green's function.

    H : Hamiltonian matrix
    S : overlap matrix
    Hybrid : (callable) hybridization funciton, must
        return a matrix of the same dimensions of H(S).
    (below, see also np.unique)
    idx_neq : the indices of the input array that give the unique values
    idx_inv : the indices of the unique array that reconstruct the input array
    """

    def __init__(self, H, S, Hybrid, idx_neq, idx_inv, comm=None) -> None:
        self.n = H.shape[-1]
        self.H = H
        self.S = S
        self.Hybrid = Hybrid
        self.idx_world = _get_idx_world(comm, len(idx_neq))
        self.idx_neq = idx_neq[self.idx_world]  # _get_idx_world(comm, len(idx_neq))]
        self.idx_inv = idx_inv
        self.comm = comm
        self.get_sigma = _get_sigma_method(comm).__get__(self)
        self.get_occps = _get_occps_method(comm).__get__(self)

    @property
    def ed(self):
        return self.H.diagonal()[self.idx_neq]

    def __call__(self, z):
        """Interacting Green's function."""
        z = np.atleast_1d(z)
        sigma = self.get_sigma(z)
        it = np.nditer([sigma, z, None], flags=["external_loop"], order="F")
        with it:
            for sigma_, z_, out in it:
                x = self.free(z_[0], inverse=True)
                x.flat[:: (self.n + 1)] -= sigma_[self.idx_inv]
                out[self.idx_neq, ...] = np.linalg.inv(x).diagonal()[self.idx_neq]
            return it.operands[2][self.idx_neq, ...]
            # out[self.idx_world,...] = np.linalg.inv(x).diagonal()[self.idx_neq]
            # return it.operands[2][self.idx_world,...]

    def update(self, mu):
        """Update chemical potential."""
        self.mu = mu

    def set_local(self, Sigma):
        """Set impurity self-energy to diagonal elements of local self-energy!"""
        #
        # TODO : actually compute and store sigma.
        #
        self.Sigma = Sigma

    def Delta(self, z):
        """Hybridization."""
        #                                       -1
        # Delta(z) = z+mu - Sigma(z) - ( G (z) )
        #                                 ii
        z = np.atleast_1d(z)
        weiss = self.Weiss(z)
        ndim = weiss.ndim - 1
        it = (
            np.nditer(
                [self.ed, weiss, z, None],
                op_axes=[[0] + [-1] * ndim, None, [-1] * ndim + [0], None],
                flags=["external_loop"],
                order="F",
            ),
        )
        with it:
            for ed, weiss, z, out in it:
                out[...] = z + self.mu - ed - weiss
            return it.operands[3]

    def Weiss(self, z):
        """Weiss field."""
        #  -1                             -1
        # G     (z) = Sigma(z) + ( G (z) )
        #  0,ii                     ii
        gloc_inv = np.reciprocal(self(z))
        return gloc_inv + self.Sigma(z)

    def free(self, z, inverse=False):
        """Non-interacting green's function."""
        #                                       -1
        #  g (z) = ((z + mu)*S - H - Hybrid(z))
        #   0
        g0_inv = (z + self.mu) * self.S - self.H - self.Hybrid(z)
        if inverse:
            return g0_inv
        return np.linalg.inv(g0_inv)

    # Helper

    def integrate(self, mu=0.0):
        occps = self.get_occps(mu)
        # occps_loc = integrate_gf(self, mu)
        # if self.comm is not None:
        #     occps = np.empty(occps_loc.size*self.comm.size, occps_loc.dtype)
        #     self.comm.Allgather([occps_loc, occps_loc.size], [occps, occps_loc.size])
        #     shape = list(occps_loc.shape)
        #     shape[0] *= self.comm.size
        #     occps.shape = shape
        # else:
        #     occps = occps_loc
        # occps = np.squeeze(occps[self.idx_inv,...])
        if occps.ndim < 2:
            return 2.0 * occps  # .sum()
        return occps.sum(1)  # .sum()


class Gfimp:
    def __init__(self, gfimp) -> None:
        self.gfimp = gfimp

    @property
    def nmats(self):
        if hasattr(self, "gfimp") and self.gfimp:
            return self.gfimp[0].nmats

    @property
    def beta(self):
        if hasattr(self, "gfimp") and self.gfimp:
            return self.gfimp[0].beta

    @property
    def x(self):
        if hasattr(self, "gfimp") and self.gfimp:
            return self.gfimp[0].x

    def reset_bath(self):
        for gf in self:
            gf.reset_bath()

    def update(self, mu):
        """Updated chemical potential."""
        mu = np.broadcast_to(mu, len(self))
        for i, gf in enumerate(self):
            gf.update(mu[i])

    @property
    def up(self):
        return Gfimp([gf.up for gf in self])

    @property
    def dw(self):
        return Gfimp([gf.dw for gf in self])

    def fit(self, delta):
        """Fit discrete bath."""
        for i, gf in enumerate(self):
            gf.fit(delta[i])

    def Sigma(self, z):
        """Correlated self-energy."""
        return np.stack([gf.Sigma(z) for gf in self])

    def solve(self):
        for gf in self:
            gf.solve()

    def spin_symmetrize(self):
        for gf in self:
            gf.spin_symmetrize()

    def get_local_moments(self):
        nup = np.zeros(len(self))
        ndw = np.zeros(len(self))
        for i, gf in enumerate(self.gfimp):
            nup[i], ndw[i] = map(
                lambda m: m[0], get_occupation(gf.espace, gf.egs, self.beta, self.n)
            )
        return nup - ndw

    def __getitem__(self, i):
        return self.gfimp[i]

    def __len__(self):
        return len(self.gfimp)

    def __iter__(self):
        yield from iter(self.gfimp)
