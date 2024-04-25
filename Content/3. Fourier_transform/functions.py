import numpy as np


def fourier_series(x, a0, an, bn):
    '''
    Compute the Fourier series expansion (in the sine-cosine form)
    of a given function with period 2pi.

    Parameters
    ----------
    x : 1D array
        Coordinates where the Fourier expansion will be computed.
    a0 : scalar
        Coefficient a0 of the expansion.
    an, bn : 1D arrays or None
        Cosine and sine coefficients of the expansion. If not None, must
        have n-1 elements, where n is the maximum degree of the expansion.
        If an and bn are not None, they must have the same number of elements.

    Returns
    -------
    fourier_series : 1D array
        Fourier expansion computed at the points x up to degree n.
    '''
    assert isinstance(x, np.ndarray), 'x must be an array'
    assert x.ndim == 1, 'x must be an 1D array'
    assert np.isscalar(a0), 'a0 must be a scalar'
    if an is not None:
        assert isinstance(an, np.ndarray), 'an must be an array'
        assert an.ndim == 1, 'an must be a 1D array'
    if bn is not None:
        assert isinstance(bn, np.ndarray), 'bn must be an array'
        assert bn.ndim == 1, 'bn must be a 1D array'
    if (an is not None) and (bn is not None):
        assert an.size == bn.size, 'an and bn must have the same number of elements'

    fourier_series = np.zeros_like(x) + a0/2

    if an is not None:
        #for ani in an:
        # for ni in range(an.size):
        #     ani = an[ni]
        for ni, ani in enumerate(an):
            fourier_series += ani*np.cos((ni+1)*x)
            #fourier_series += ani*np.cos(2*np.pi*(ni+1)*f0*y)
    if bn is not None:
        for ni, bni in enumerate(bn):
            fourier_series += bni*np.sin((ni+1)*x)

    return fourier_series

def sawtooth_bn(n):
    '''
    Compute the sine coefficient bn up to degree n
    of the upward sawtooth function:
    s(x) = x/pi for -pi < x < pi
    s(x + 2pi*k) for k integer

    Parameters
    ----------
    n : int
        Degree.
    Returns
    -------
    bn : 1D array
        Sine coefficients up to degree n.
    '''
    assert isinstance(n, int), 'n must be an integer'
    assert n >= 1, 'n must be greater than or equal to 1'
    N = np.arange(1, n+1)
    bn = (2*(-1)**(N + 1))/(np.pi*N)

    return bn

def square_bn(n):
    '''
    Compute the sine coefficients bn up to degree n
    of the odd square function:

    s(x) = 0 for -pi < x < 0
    s(x) = 1 for 0 < x < pi
    s(x + 2pi*k) for k integer

    Parameters
    ----------
    n : int
        Degree.

    Returns
    -------
    bn : 1D array
        Sine coefficients up to degree n.
    '''
    assert isinstance(n, int), 'n must be an integer'
    assert n >= 1, 'n must be greater than or equal to 1'

    N = np.arange(1, n+1)
    bn = (1 - (-1)**N)/(np.pi*N)

    return bn
