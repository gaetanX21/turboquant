from typing import Callable

import torch

# for readability
T = torch.Tensor


def sample_rotation_matrix(d: int) -> T:
    """Sample (d,d) rotation matrix from uniform distribution."""
    A = torch.randn(d, d)
    Q, R = torch.linalg.qr(A)
    # required because signs of diagonal elements of R are arbitrary
    Q = Q * torch.sign(torch.diag(R))
    return Q


def dist_unit_sphere(d: int) -> Callable[[T], T]:
    """Probability density of coordinates on unit sphere in dimension d."""
    d = torch.tensor([d])
    frac = torch.pi**-0.5 * torch.exp(torch.lgamma(d / 2) - torch.lgamma((d - 1) / 2))
    return lambda x: frac * torch.pow(1 - x**2, (d - 3) / 2)


def lloyd_max(
    dist: Callable[[T], T],
    num_bins: int,
    b_init: T,
    max_iter: int = 1_000,
    eps: float = 1e-6,
    integration_steps: int = 100,
) -> tuple[T, T]:
    """Lloyd-Max algorithm.

    For a given distribution, finds optimal centroids to minimize quantization error.
    """
    b = b_init
    min_, max_ = b[0], b[-1]

    for _ in range(max_iter):
        # 1. compute optimal centroids
        q = compute_q(dist, b, integration_steps)

        # 2. compute optimal boundaries
        midpoints = (q[:-1] + q[1:]) / 2.0
        b_new = torch.cat((min_[None, ...], midpoints, max_[None, ...]))

        # check for convergence
        if torch.max((b - b_new) ** 2) < eps:
            break

        b = b_new

    return b, q


def compute_q(
    dist: Callable[[T], T],
    b: T,
    integration_steps: int,
) -> T:
    """Compute center of mass q[i] for each bin (b[i], b[i+1]) using trapezoidal integration."""
    q = torch.empty(len(b) - 1)

    for i in range(len(q)):
        x = torch.linspace(b[i].item(), b[i + 1].item(), integration_steps)
        pdf_vals = dist(x)
        numerator = torch.trapezoid(x * pdf_vals, x)
        denominator = torch.trapezoid(pdf_vals, x)
        q[i] = numerator / denominator

    return q


class TurboQuant:
    """Implement the TurboQuant MSE quantization algorithm."""

    def __init__(self, *, d: int, b: int) -> "TurboQuant":
        self.d = d
        self.b = b
        self._dist = dist_unit_sphere(d)
        b, q = lloyd_max(self._dist, 2**b, self.b_init)
        self._boundaries = b
        self._centroids = q
        self._PI = sample_rotation_matrix(d)

    @property
    def b_init(self):
        """Compute smart initial values for boundaries vector b."""
        # for point on sphere, coordinates are between -1 and 1
        min_, max_ = torch.tensor([-1.0]), torch.tensor([1.0])

        # rather than evenly spaced initial centroids (which create numerical issues),
        # we initialize using quantiles of N(0,1/d), which is the asymptotic distribution of self._dist
        p = torch.linspace(0.0, 1.0, steps=2**self.b + 1)[1:-1]
        quantiles = torch.distributions.Normal(loc=0.0, scale=self.d**-0.5).icdf(p)

        b_init = torch.cat((min_, quantiles, max_))

        return b_init

    def quantize(self, X: T) -> tuple[T, T]:
        """Quantize a real-valued matrix of shape (B, d) into matrix (B, d) filled with b-bits codes."""
        norms = torch.linalg.vector_norm(X, dim=-1, keepdim=True)
        X_normalized = X / norms

        # NOTE: Fast Walsh-Hadamard Transform turns O(d^2) into O(d * log d)
        y = X_normalized @ self._PI
        # torch.bucketize replaces naive O(K) search with O(log K) since boundaries are sorted
        idx = torch.bucketize(y, self._boundaries) - 1  # -1 to start at 0
        idx = torch.clamp(idx, 0, len(self._centroids) - 1)  # safety
        return idx, norms

    def dequantize(self, idx: T, norms: T) -> tuple[T, T]:
        """Dequantize a matrix of (B, d) b-bits codes into a real-valued matrix of shape (B,d)."""
        y_hat = self._centroids[idx]
        x_hat_normalized = y_hat @ self._PI.T
        x_hat = x_hat_normalized * norms
        return x_hat
