"""Paired frame bootstrap for KID, shared by collect_chapter6.py and rescore_ablation_kid.py.

The ``kid_std`` the metrics stage reports is a spread over subsets of one fixed sample: it
understates uncertainty, and it cannot pair two curves. This module resamples FRAMES instead,
with one set of draws shared by every curve of a scene. Rendered frames correspond across shift
levels, rounds and variants (same cameras, same timestamps), so differences between any two of
them get paired intervals: a 3 m penalty, or the change in that penalty between two rounds.

The estimator is the unbiased MMD^2 under KID's kernel, computed on per-frame counts. Copies of
one original frame are excluded from the within-set sums, as the estimator's i != j exclusion
intends; plain resampling would count them and bias MMD^2 upward.
"""
import numpy as np
import torch


def poly_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """KID's kernel: (x.y / d + 1)^3, as torchmetrics with gamma=None, coef=1, degree=3."""
    return (a @ b.T / a.shape[1] + 1.0) ** 3


def _within(K: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Within-set term for a batch of count vectors C (B x N): pairs of distinct originals."""
    n = C.sum(1)
    sq = (C * C).sum(1)
    return (((C @ K) * C).sum(1) - (C * C) @ K.diagonal()) / (n * n - sq)


def mmd2_batch(Krr, Kff, Krf, Cr, Cf) -> torch.Tensor:
    """Unbiased MMD^2 for B paired draws given as count matrices (B x Nr, B x Nf)."""
    cross = ((Cr @ Krf) * Cf).sum(1) / (Cr.sum(1) * Cf.sum(1))
    return _within(Krr, Cr) + _within(Kff, Cf) - 2.0 * cross


class SceneBootstrap:
    """One scene's real features plus frame draws shared by every rendered set scored against them.

    ``score(f_fake)`` returns the full-sample point estimate and one MMD^2 per draw. Two sets
    scored here are paired draw by draw, provided they hold the same frames in the same order,
    which is how the renders are enumerated (camera by camera, frame by frame).
    """

    def __init__(self, f_real: torch.Tensor, draws: int = 1000, seed: int = 0,
                 device: str = "cuda", chunk: int = 250):
        self.device, self.draws, self.seed, self.chunk = device, draws, seed, chunk
        self.fr = f_real.to(device, torch.float64)
        self.Krr = poly_kernel(self.fr, self.fr)
        self.nr = self.fr.shape[0]
        self.Cr = self._counts(self.nr, seed)
        self._cf = {}

    def _counts(self, n: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randint(n, (self.draws, n), generator=g)
        C = torch.zeros(self.draws, n, dtype=torch.float64)
        C.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float64))
        return C.to(self.device)

    def fake_counts(self, n: int) -> torch.Tensor:
        # Same seed for every set of a given size, so equal-sized sets share their draws.
        if n not in self._cf:
            self._cf[n] = self._counts(n, self.seed + 1)
        return self._cf[n]

    @torch.no_grad()
    def score(self, f_fake: torch.Tensor):
        ff = f_fake.to(self.device, torch.float64)
        Kff, Krf = poly_kernel(ff, ff), poly_kernel(self.fr, ff)
        nf = ff.shape[0]
        one_r = torch.ones(1, self.nr, dtype=torch.float64, device=self.device)
        one_f = torch.ones(1, nf, dtype=torch.float64, device=self.device)
        point = float(mmd2_batch(self.Krr, Kff, Krf, one_r, one_f)[0])
        Cf = self.fake_counts(nf)
        out = [mmd2_batch(self.Krr, Kff, Krf, self.Cr[i:i + self.chunk], Cf[i:i + self.chunk])
               for i in range(0, self.draws, self.chunk)]
        return point, torch.cat(out).cpu().numpy()


def ci(samples: np.ndarray, level: float = 95.0):
    """Percentile interval of bootstrap samples."""
    lo = (100.0 - level) / 2.0
    return float(np.percentile(samples, lo)), float(np.percentile(samples, 100.0 - lo))
