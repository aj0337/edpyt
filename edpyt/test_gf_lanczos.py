import numpy as np

from gf_lanczos import (
    build_gf_lanczos
)

from espace import (
    build_espace
)

from lookup import (
    get_sector_index
)

from sector import (
    get_sector_dim
)


"""Hubbard dimer atomic limit.

"""

t = 0.1
ed = 2.
U = 5*t

n = 7
nup = 4
ndw = 3

beta = 10

H_ = lambda ed, t, n: np.diag([ed]*n) + np.diag([-t]*(n-1),k=1) + np.diag([-t]*(n-1),k=-1)

V_ = lambda U, n: np.diag([U]*n)

H = H_(ed,t,n)
V = V_(U,n)

def test_build_gf_lanczos():
    from matplotlib import pyplot as plt
    from shared import params

    params['hfmode'] = True
    params['mu'] = ed

    neig_sector = np.zeros((n+1)*(n+1), int)

    neig_sector[
        get_sector_index(nup, ndw, n)
        ] = 5 #get_sector_dim(n, nup) * get_sector_dim(n, ndw)
    neig_sector[
        get_sector_index(ndw, nup, n)
        ] = 5

    espace, egs = build_espace(H, V, neig_sector)
    gf = build_gf_lanczos(H, V, espace, beta=beta, mu=params['mu'])

    eta = 1e-3
    energies = np.arange(-2,2,1e-3)

    # https://www.cond-mat.de/events/correl20/manuscripts/pavarini.pdf
    plt.plot(energies, -1/np.pi*gf(energies, eta).imag)
    plt.savefig('gf.png')
    plt.close()
