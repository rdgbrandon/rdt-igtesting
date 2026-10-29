"""Method-agnostic faithfulness metrics for image attributions of RDT.

All perturbations happen in PIXEL space on the SigLIP input (both external
history frames, same patches), and are re-encoded by SigLIP - so no method is
favoured by sharing the metric's perturbation (as embedding-space metrics
would favour embedding-space baselines). Removing every patch gives RDT's
own "missing camera" image (SigLIP image_mean, normalized pixel ~0).

Effect size is the L2 deviation of the executed, normalized 8-joint plan from
the unperturbed plan, averaged over EVALUATION noise seeds that are disjoint
from the seeds used to compute the explanation, and normalized by the
deviation when the whole view is removed, D_full.

Removed patches are inpainted from the remaining pixels by default (fill.py);
fill="gray" pastes the "missing camera" colour instead (the pilot 2 metric).
D(f)/D_full is not bounded by 1: the deletion curve peaks at ~2 on average (pilot 3,
every method, under either fill), i.e. removing a few top patches moves the plan more
than removing the whole view (without a camera RDT falls
back on joint state; with the object hidden it plans for a different scene). D_full is
shared by all methods within a moment, so per-moment comparisons do not depend on it,
but means over moments can be dominated by moments with a tiny D_full (report.py
summarizes with medians and per-moment ranks).

  deletion (MoRF)   remove most-relevant patches first; AUC of D(f)/D_full      higher = better
  deletion (LeRF)   remove least-relevant first;                                lower  = better
  insertion         start from all-removed, insert most-relevant first;
                    AUC of 1 - D(f)/D_full                                      higher = better

The endpoints (nothing / everything removed) are shared by all methods and
curves and are computed once.
"""
import numpy as np
import torch
from scipy.stats import spearmanr

from .fill import remove_pixels

FRACS_FULL = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
FRACS_FAST = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
PATCH = 14
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")       # numpy 2 renamed it


def _auc(y, x):
    return float(_trapz(y, x) / (x[-1] - x[0]))


class Faithfulness:
    def __init__(self, policy, eval_seeds=(101, 202), fracs=FRACS_FULL, curves=("morf", "lerf", "ins"), batch=10,
                 fill="inpaint"):
        self.p = policy
        self.fill = fill
        self.eval_seeds = list(eval_seeds)
        self.fracs = np.asarray(fracs, dtype=np.float64)
        self.curves = tuple(curves)
        self.batch = batch
        with torch.no_grad():
            self.bg_pix = policy.preprocess([policy.bg_image])[0]          # (3, 384, 384)

    @classmethod
    def fast(cls, policy, batch=10, fill="inpaint"):
        return cls(policy, eval_seeds=(101,), fracs=FRACS_FAST, curves=("morf", "ins"), batch=batch, fill=fill)

    def _patch_mask(self, order, k):
        """Boolean (384,384) pixel mask covering the first k patches in `order`."""
        G = self.p.grid
        m = torch.zeros(G * G, dtype=torch.bool)
        m[order[:k]] = True
        m = m.reshape(G, G).repeat_interleave(PATCH, 0).repeat_interleave(PATCH, 1)
        full = torch.zeros(self.bg_pix.shape[-2:], dtype=torch.bool)
        full[:m.shape[0], :m.shape[1]] = m
        return full.to(self.p.device)

    @torch.no_grad()
    def _plans(self, cond, pix_prev, pix_curr):
        """pix_*: (N, 3, 384, 384) -> executed normalized plans (N, S, 16, 8)."""
        p = self.p
        out = []
        for i in range(0, pix_curr.shape[0], self.batch):
            pc, pp = pix_curr[i:i + self.batch], pix_prev[i:i + self.batch]
            emb = p.vision(pixel_values=torch.cat([pp, pc])).last_hidden_state
            ep, ec = emb[:pp.shape[0]], emb[pp.shape[0]:]
            per_seed = []
            for s in self.eval_seeds:
                noise = p.noise(s).expand(ec.shape[0], -1, -1)
                a = p.sample(cond, emb_curr=ec, emb_prev=ep, noise=noise)
                per_seed.append(a[:, 0:64:4][:, :, p.idx])
            out.append(torch.stack(per_seed, 1))
        return torch.cat(out).float()

    def evaluate(self, cond, img_prev, img_curr, maps):
        """maps: {method: (27,27) array}. Returns {method: metrics}."""
        p, n = self.p, self.p.n_patch
        x_prev = p.preprocess([img_prev])[0]
        x_curr = p.preprocess([img_curr])[0]
        ks = [int(round(f * n)) for f in self.fracs]
        inner = [k for k in ks if 0 < k < n]

        # shared endpoints: reference plan and fully removed view
        end = self._plans(cond, torch.stack([x_prev, self.bg_pix]), torch.stack([x_curr, self.bg_pix]))
        ref = end[0]
        dev = lambda P: (P - ref[None]).flatten(2).norm(dim=-1).mean(1).cpu().numpy()
        d_full = max(float(dev(end[1:2])[0]), 1e-8)

        # every (method, curve, interior k) perturbation in one batched pass
        jobs, masks = [], []
        for name, m in maps.items():
            score = torch.as_tensor(np.asarray(m, dtype=np.float32).reshape(-1))
            orders = {"morf": torch.argsort(score, descending=True), "lerf": torch.argsort(score)}
            orders["ins"] = orders["morf"]
            for cv in self.curves:
                for k in inner:
                    M = self._patch_mask(orders[cv], k)
                    masks.append(~M if cv == "ins" else M)
                    jobs.append((name, cv))
        d = []
        if jobs:
            M = torch.stack(masks)[:, None]
            pp = remove_pixels(x_prev, M, self.bg_pix, self.fill)
            pc = remove_pixels(x_curr, M, self.bg_pix, self.fill)
            d = dev(self._plans(cond, pp, pc)) / d_full

        res = {name: dict(d_full=d_full) for name in maps}
        it = iter(d)
        for name in maps:
            for cv in self.curves:
                mid = [next(it) for _ in inner]
                if cv == "ins":
                    curve = [1.0 - 1.0] + [1.0 - x for x in mid] + [1.0]            # 0 inserted -> all removed
                    res[name]["insertion"] = _auc(curve, self.fracs)
                else:
                    curve = [0.0] + mid + [1.0]
                    res[name]["deletion_" + cv] = _auc(curve, self.fracs)
                res[name]["curve_" + cv] = [float(x) for x in curve]
            if "deletion_lerf" in res[name]:
                res[name]["morf_minus_lerf"] = res[name]["deletion_morf"] - res[name]["deletion_lerf"]
        return res


def rank_agreement(a, b):
    """Spearman rank correlation of two patch maps."""
    return float(spearmanr(np.asarray(a).ravel(), np.asarray(b).ravel()).correlation)


def topk_iou(a, b, frac=0.1):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    k = max(1, int(round(frac * a.size)))
    A, B = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(A & B) / len(A | B)
