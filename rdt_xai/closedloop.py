"""Closed-loop causal test: at every policy call, explain the current inputs,
remove a fraction of patches from what the policy sees (both history frames,
filled with RDT's "missing camera" colour), and record task success.

If an attribution method is faithful, removing its top patches should hurt
success far more than removing random patches, and removing its bottom
patches should hurt least. The simulator itself is never modified.

Ranking: 'top' / 'bottom' order patches by the signed score, so for signed
methods (IG family) 'bottom' removes the most *negative* patches, which matter
as much as the most positive ones. 'top_abs' / 'bottom_abs' order by |score|:
'bottom_abs' is the "least relevant patches" control.
"""
import numpy as np
import torch
from PIL import Image

from .rollout import rollout

PATCH, SIG = 14, 384


def patch_mask_image(sel, size, grid=27):
    """Boolean patch selection (729,) -> PIL 'L' mask at the frame's size.
    Patch (r, c) covers SigLIP input pixels [14r, 14r+14) x [14c, 14c+14) of
    the 384x384 resize of the (square) frame."""
    m = np.zeros((SIG, SIG), dtype=np.uint8)
    g = np.asarray(sel, dtype=bool).reshape(grid, grid)
    m[:grid * PATCH, :grid * PATCH] = np.kron(g, np.ones((PATCH, PATCH), dtype=np.uint8)) * 255
    return Image.fromarray(m).resize(size, Image.NEAREST)


def apply_mask(img, mask_img, color):
    if img is None:
        return None
    out = img.convert("RGB").copy()
    out.paste(Image.new("RGB", out.size, color), mask=mask_img.resize(out.size, Image.NEAREST))
    return out


class MaskHook:
    """obs_hook for rollout(): mode in {'none', 'top', 'bottom', 'top_abs', 'bottom_abs', 'random'}."""

    def __init__(self, policy, explainer, method="eig", mode="top", frac=0.1, n_evals=8, n_seeds=4, seed=0):
        self.p, self.ex = policy, explainer
        self.method, self.mode, self.frac = method, mode, frac
        self.n_evals, self.n_seeds, self.seed = n_evals, n_seeds, seed
        self.k = int(round(frac * policy.n_patch))
        self.log = []

    def __call__(self, prev_img, curr_img, ctx):
        if self.mode == "none" or self.k == 0:
            return prev_img, curr_img
        n = self.p.n_patch
        if self.mode == "random":
            rng = np.random.default_rng(self.seed * 100003 + ctx["call"])
            order = rng.permutation(n)
        else:
            embs = self.p.encode([prev_img, curr_img])
            cond = self.p.conditions(ctx["task"], embs[0:1], embs[1:2], ctx["qpos8"])
            r = self.ex.explain(cond, self.method, n_evals=self.n_evals, n_seeds=self.n_seeds,
                                seed=self.seed * 1000 + ctx["call"])
            score = r["map"].reshape(-1)
            if self.mode.endswith("_abs"):
                score = np.abs(score)
            order = np.argsort(-score) if self.mode.startswith("top") else np.argsort(score)
        sel = np.zeros(n, dtype=bool)
        sel[order[:self.k]] = True
        self.log.append(dict(call=ctx["call"], step=ctx["step"], sel=np.flatnonzero(sel).tolist()))
        mask = patch_mask_image(sel, curr_img.size)
        return apply_mask(prev_img, mask, self.p.bg_color), apply_mask(curr_img, mask, self.p.bg_color)


def run_condition(policy, explainer, task, episodes, method, mode, frac, env=None, n_evals=8, verbose=False):
    rows = []
    for ep in episodes:
        hook = MaskHook(policy, explainer, method=method, mode=mode, frac=frac, n_evals=n_evals, seed=ep)
        r = rollout(policy, task, ep, env=env, obs_hook=hook)
        rows.append(dict(task=task, episode=ep, method=method, mode=mode, frac=frac,
                         success=bool(r["success"]), steps=r["steps"], wall=r["wall"]))
        if verbose:
            print(rows[-1], flush=True)
    return rows
