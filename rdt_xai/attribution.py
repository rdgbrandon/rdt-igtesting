"""Attribution methods for RDT image inputs, all on a common footing.

Input space: SigLIP patch embeddings of the external camera, both history
slots (t-1, t) jointly; the 4 wrist slots stay at the background ("missing
camera") embedding the policy actually receives on ManiSkill.

Target: RDT outputs a 64-step chunk of which every 4th step is executed. For a
sampler noise z we explain the scalar

    F_z(E) = < A(E; z)[exec, joints],  u_z >,
    u_z    = (A(E; z) - A(E'; z)) / ||A(E; z) - A(E'; z)||    (restricted to exec, joints)

i.e. how far the executed plan moves from the plan the policy would make
without visual evidence (E' = background embedding), along that very
direction. Completeness then reads sum(attr) = ||A(E;z) - A(E';z)||. A joint
subset gives per-joint explanations with the same semantics.

Methods (n_evals = gradient evaluations; forward-only methods get ~3x as many
forwards, since a backward pass costs ~2 forwards):

  gradient-based
    grad       |dF/dE| (L2 over channels)                                   1 eval
    gxi        dF/dE . (E - E')                                             1 eval
    ig_zero    IG, zero baseline, fixed noise [Sundararajan'17]             n_evals
    ig_bg      IG, background baseline, fixed noise                        n_evals
    blurig     BlurIG path in embedding space, fixed noise [Xu'20]          n_evals
               (the original pipeline of this repo, on the correct input)
    eig        expected IG over sampler noise: background baseline, noise   n_evals
               resampled per evaluation, stratified alphas  [proposed]
  attention-based
    daam       cross-attention mass on each patch, aggregated over          1 forward
               denoising steps, blocks, heads, action tokens [Tang'23 DAAM]
    gae        gradient-weighted cross-attention E_h[(dF/dA . A)^+]         1 eval
               [Chefer'21 generic attention explainability]
  perturbation-based
    occlusion  kxk-patch block replacement by background, dF [Zeiler'14]   81 (k=3) / 49 (k=4) fwd
    iss        random token masks (keep p), E[||dA||^2 | patch masked]     3*n_evals fwd
               [Embodied Interpretability, ICML'26; RISE, Petsiuk'18]
  control
    random     iid uniform map                                              0

Analyses (not ranked as maps):
    eig_steps  exact split of expected IG by denoising step: each step sees its
               own copy of the input, so sum over steps == eig
    knockout   plan deviation when vision is removed from one denoising step
               (or given only at one step)
"""
import math
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF

EXEC_ROWS = torch.arange(0, 64, 4)
METHODS = ["random", "daam", "grad", "gxi", "gae", "occlusion", "iss", "ig_zero", "ig_bg", "blurig", "eig"]


def _blur(E, sigma, grid=27):
    if sigma < 0.05:
        return E
    B, N, D = E.shape
    img = E.reshape(B, grid, grid, D).permute(0, 3, 1, 2)
    ks = min(2 * int(3 * sigma + 0.5) + 1, 2 * grid - 1)       # reflect pad must stay < grid
    out = TF.gaussian_blur(img.float(), kernel_size=[ks, ks], sigma=float(sigma)).to(E.dtype)
    return out.permute(0, 2, 3, 1).reshape(B, N, D)


def _eig_schedule(n_evals, n_seeds, seed):
    """(alphas, seeds): n_seeds sampler noises, each with stratified alphas
    offset by a fraction of a bin so the union covers [0,1] evenly."""
    S = max(1, min(n_seeds, n_evals))
    per = n_evals // S
    alphas, seeds = [], []
    for si in range(S):
        s = seed * 7919 + si
        for k in range(per):
            alphas.append((k + (si + 0.5) / S) / per)
            seeds.append(s)
    return alphas, seeds


class Explainer:
    def __init__(self, policy, joints=None, batch=8):
        self.p = policy
        self.batch = batch
        self.set_joints(joints)

    def set_joints(self, joints):
        self.joints = list(range(8)) if joints is None else list(joints)
        self.jidx = self.p.idx[self.joints]

    # ── target ───────────────────────────────────────────────────────────────
    def _exec(self, A):
        return A[:, EXEC_ROWS.to(A.device)][:, :, self.jidx]             # (B, 16, J)

    def _noise(self, seeds):
        return torch.cat([self.p.noise(x) for x in seeds])

    @torch.no_grad()
    def actions(self, cond, Ep, Ec, seeds, **kw):
        """Executed normalized actions for a batch of (embedding, seed) pairs."""
        out = []
        for i in range(0, len(seeds), self.batch):
            s = seeds[i:i + self.batch]
            b = len(s)
            pick = lambda E: E[i:i + b] if E.shape[0] > 1 else E.expand(b, -1, -1)
            out.append(self._exec(self.p.sample(cond, emb_curr=pick(Ec), emb_prev=pick(Ep),
                                                noise=self._noise(s), **kw)))
        return torch.cat(out)

    def directions(self, cond, base_p, base_c, seeds):
        a = self.actions(cond, cond["emb_prev"], cond["emb_curr"], seeds)
        a0 = self.actions(cond, base_p, base_c, seeds)
        d = a - a0
        n = d.flatten(1).norm(dim=1).clamp_min(1e-8)
        return d / n[:, None, None], n, a, a0

    def _dirs_for(self, cond, base_p, base_c, seeds):
        uniq = sorted(set(seeds))
        u, n, _, _ = self.directions(cond, base_p, base_c, uniq)
        lut = {s: k for k, s in enumerate(uniq)}
        return torch.stack([u[lut[s]] for s in seeds]), torch.stack([n[lut[s]] for s in seeds])

    # ── path-integral engine ─────────────────────────────────────────────────
    def _path_integral(self, cond, pts_p, pts_c, deltas_p, deltas_c, seeds, dirs, step_split=False):
        """sum_k grad F_{z_k}(P_k) . dP_k for points P_k with increments dP_k.
        pts_*: (K, 729, D); seeds: K ints; dirs: (K, 16, J) unit directions.
        step_split: return per-denoising-step attributions (n_steps, 729, D)."""
        K = len(seeds)
        T = self.p.rdt.num_inference_timesteps if step_split else 1
        shape = (T,) + tuple(cond["emb_prev"].shape[1:])
        attr_p = torch.zeros(shape, device=self.p.device, dtype=torch.float32)
        attr_c = torch.zeros_like(attr_p)
        for i in range(0, K, self.batch):
            sl = slice(i, min(K, i + self.batch))
            with torch.enable_grad():
                if step_split:
                    leaves = [(pts_p[sl].clone().requires_grad_(True), pts_c[sl].clone().requires_grad_(True))
                              for _ in range(T)]
                    A = self._exec(self.p.sample(cond, step_embs=leaves, noise=self._noise(seeds[sl])))
                    flat = [x for pc in leaves for x in pc]
                else:
                    Ep = pts_p[sl].clone().requires_grad_(True)
                    Ec = pts_c[sl].clone().requires_grad_(True)
                    A = self._exec(self.p.sample(cond, emb_curr=Ec, emb_prev=Ep, noise=self._noise(seeds[sl])))
                    flat = [Ep, Ec]
                F = (A * dirs[sl]).sum()
                grads = torch.autograd.grad(F, flat)
            for t in range(T):
                attr_p[t] += (grads[2 * t] * deltas_p[sl]).float().sum(0)
                attr_c[t] += (grads[2 * t + 1] * deltas_c[sl]).float().sum(0)
        if not step_split:
            attr_p, attr_c = attr_p[0], attr_c[0]
        return attr_p, attr_c

    def _straight(self, cond, base, alphas, seeds, step_split=False):
        dirs, dn = self._dirs_for(cond, base, base, seeds)
        a = torch.tensor(alphas, device=self.p.device, dtype=self.p.dtype)[:, None, None]
        Ep, Ec = cond["emb_prev"], cond["emb_curr"]
        K = len(alphas)
        pts_p, pts_c = base + a * (Ep - base), base + a * (Ec - base)
        dp = ((Ep - base) / K).expand(K, -1, -1)
        dc = ((Ec - base) / K).expand(K, -1, -1)
        ap, ac = self._path_integral(cond, pts_p, pts_c, dp, dc, seeds, dirs, step_split)
        # each evaluation estimates (F_z(E) - F_z(E')) / K, so the total should
        # match the mean of ||d_z|| over evaluations
        return ap, ac, float(dn.mean())

    # ── methods ──────────────────────────────────────────────────────────────
    def explain(self, cond, method, n_evals=16, seed=0, n_seeds=4, sigma_max=None, occ_block=3, iss_keep=0.3):
        """Returns dict(map=(27,27) patch attribution summed over both external
        slots, prev/curr maps where defined, completeness info, seconds)."""
        t0 = time.time()
        p, G = self.p, self.p.grid
        bg = p.bg_emb
        Ep, Ec = cond["emb_prev"], cond["emb_curr"]
        info = {}
        if method in ("ig_bg", "ig_zero", "eig", "eig_steps"):
            base = torch.zeros_like(bg) if method == "ig_zero" else bg
            if method in ("eig", "eig_steps"):
                alphas, seeds = _eig_schedule(n_evals, n_seeds, seed)
            else:
                alphas, seeds = [(k + 0.5) / n_evals for k in range(n_evals)], [seed] * n_evals
            ap, ac, target = self._straight(cond, base, alphas, seeds, step_split=(method == "eig_steps"))
            info["completeness_target"] = target
            if method == "eig_steps":
                steps = (ap.sum(-1) + ac.sum(-1)).cpu().numpy().reshape(-1, G, G)        # (T, 27, 27)
                out = self._pack(torch.from_numpy(steps.sum(0)), None, None, info, t0)
                out["steps"] = steps
                return out
        elif method == "blurig":
            smax = sigma_max or float(G) / 4                       # ~featureless at grid scale
            sig = np.linspace(smax, 0.0, n_evals + 1)
            mids = [(sig[k] + sig[k + 1]) / 2 for k in range(n_evals)]
            dirs, dn, _, _ = self.directions(cond, _blur(Ep, smax, G), _blur(Ec, smax, G), [seed])
            P = lambda E, s: torch.cat([_blur(E, x, G) for x in s])
            dp = torch.cat([_blur(Ep, sig[k + 1], G) - _blur(Ep, sig[k], G) for k in range(n_evals)])
            dc = torch.cat([_blur(Ec, sig[k + 1], G) - _blur(Ec, sig[k], G) for k in range(n_evals)])
            ap, ac = self._path_integral(cond, P(Ep, mids), P(Ec, mids), dp, dc, [seed] * n_evals,
                                         dirs.expand(n_evals, -1, -1))
            info["completeness_target"] = float(dn[0])
        elif method in ("grad", "gxi"):
            dirs, _, _, _ = self.directions(cond, bg, bg, [seed])
            one = torch.ones_like(Ep)
            gp, gc = self._path_integral(cond, Ep, Ec, one, one, [seed], dirs)
            if method == "grad":
                return self._pack(gp.norm(dim=-1) + gc.norm(dim=-1), None, None, info, t0, unsigned=True)
            ap, ac = gp * (Ep[0] - bg[0]).float(), gc * (Ec[0] - bg[0]).float()
        elif method == "daam":
            with torch.no_grad():
                _, log = p.sample(cond, noise=p.noise(seed), return_attn=True)
            w = torch.stack(log).float()[:, 0, -p.horizon:].mean((0, 1))            # (L,)
            return self._pack(self._ext_patches(w), None, None, info, t0, unsigned=True)
        elif method == "gae":
            return self._gae(cond, seed, t0)
        elif method == "occlusion":
            return self._occlusion(cond, seed, t0, block=occ_block)
        elif method == "iss":
            return self._iss(cond, seed, t0, n_masks=3 * n_evals, keep=iss_keep)
        elif method == "random":
            g = torch.Generator().manual_seed(seed)
            return self._pack(torch.rand(p.n_patch, generator=g), None, None, info, t0, unsigned=True)
        else:
            raise ValueError(method)
        return self._pack(ap.sum(-1) + ac.sum(-1), ap.sum(-1), ac.sum(-1), info, t0)

    def _ext_patches(self, w):
        """Collapse a per-condition-token vector (6*729,) onto the 27x27 grid
        by summing the two external-camera slots (t-1 and t)."""
        n = self.p.n_patch
        return w[0:n] + w[3 * n:4 * n]

    def _gae(self, cond, seed, t0):
        """Chefer et al. 2021 (ICCV), cross-attention rule: relevance of image
        tokens = sum over steps and blocks of E_h[(dF/dA * A)^+], summed over
        action-token queries. RDT's image tokens do not self-attend, so no
        rollout across layers is needed on the image side."""
        p = self.p
        dirs, _, _, _ = self.directions(cond, p.bg_emb, p.bg_emb, [seed])
        Ep = cond["emb_prev"].clone().requires_grad_(True)
        Ec = cond["emb_curr"].clone().requires_grad_(True)
        with torch.enable_grad():
            A, log = p.sample(cond, emb_curr=Ec, emb_prev=Ep, noise=p.noise(seed), return_attn=True, attn_grad=True)
            F = (self._exec(A) * dirs).sum()
            F.backward()
        rel = torch.zeros(log[0].shape[-1], device=p.device)
        for w in log:
            rel += (w.grad * w).clamp_min(0).mean(1)[0, -p.horizon:].sum(0).float()
        return self._pack(self._ext_patches(rel.detach()), None, None, {}, t0, unsigned=True)

    def _occlusion(self, cond, seed, t0, block=3):
        p, G = self.p, self.p.grid
        nb = math.ceil(G / block)
        dirs, _, a_full, _ = self.directions(cond, p.bg_emb, p.bg_emb, [seed])
        F_full = (a_full * dirs).sum().item()
        masks = []
        for by in range(nb):
            for bx in range(nb):
                m = torch.zeros(G, G, dtype=torch.bool)
                m[by * block:(by + 1) * block, bx * block:(bx + 1) * block] = True
                masks.append(m.flatten())
        M = torch.stack(masks).to(p.device)[:, :, None]                       # (R, 729, 1)
        Ep = torch.where(M, p.bg_emb, cond["emb_prev"])
        Ec = torch.where(M, p.bg_emb, cond["emb_curr"])
        A = self.actions(cond, Ep, Ec, [seed] * len(masks))
        drop = F_full - (A * dirs).flatten(1).sum(1)                         # (R,)
        m = (M[:, :, 0].float() * drop[:, None].float()).sum(0) / block ** 2
        return self._pack(m, None, None, {}, t0)

    def _iss(self, cond, seed, t0, n_masks=48, keep=0.3):
        """Interventional significance (ICML'26, embedding-space variant with
        our background replacement): random Bernoulli(keep) token masks shared
        by both slots, score = squared plan deviation, credited to the masked
        patches and normalized by the expected number of times each is masked."""
        p = self.p
        g = torch.Generator().manual_seed(seed + 12345)
        keep_m = (torch.rand(n_masks, p.n_patch, generator=g) < keep).to(p.device)
        ref = self.actions(cond, cond["emb_prev"], cond["emb_curr"], [seed])
        M = (~keep_m)[:, :, None]
        A = self.actions(cond, torch.where(M, p.bg_emb, cond["emb_prev"]), torch.where(M, p.bg_emb, cond["emb_curr"]),
                         [seed] * n_masks)
        score = (A - ref).flatten(1).pow(2).sum(1).float()                   # (N,)
        sal = ((~keep_m).float() * score[:, None]).sum(0) / (n_masks * (1 - keep))
        return self._pack(sal, None, None, {}, t0, unsigned=True)

    @torch.no_grad()
    def knockout(self, cond, seed=0, n_seeds=2):
        """Per denoising step s: deviation of the executed plan when the view is
        removed only at step s ('drop'), and when it is given only at step s
        ('only', relative to no vision at all). Normalized by the no-vision
        deviation. Returns dict of (T,) arrays."""
        p = self.p
        T = p.rdt.num_inference_timesteps
        seeds = [seed * 7919 + s for s in range(n_seeds)]
        Ep, Ec = cond["emb_prev"], cond["emb_curr"]
        full = self.actions(cond, Ep, Ec, seeds)
        none = self.actions(cond, Ep, Ec, seeds, vision_steps=[])
        d_none = (full - none).flatten(1).norm(dim=1)
        drop, only = [], []
        for s in range(T):
            a_drop = self.actions(cond, Ep, Ec, seeds, vision_steps=[x for x in range(T) if x != s])
            a_only = self.actions(cond, Ep, Ec, seeds, vision_steps=[s])
            drop.append(((full - a_drop).flatten(1).norm(dim=1) / d_none).mean().item())
            only.append((1 - (full - a_only).flatten(1).norm(dim=1) / d_none).mean().item())
        return dict(drop=np.array(drop), only=np.array(only), d_none=float(d_none.mean()))

    def _pack(self, m, mp, mc, info, t0, unsigned=False):
        G = self.p.grid
        to = lambda x: None if x is None else x.detach().float().cpu().numpy().reshape(G, G)
        out = dict(map=to(m), prev=to(mp), curr=to(mc), seconds=time.time() - t0, unsigned=unsigned, **info)
        if "completeness_target" in info:
            s = float(out["map"].sum())
            out["attr_sum"] = s
            out["completeness_rel_err"] = abs(s - info["completeness_target"]) / (abs(info["completeness_target"]) + 1e-8)
        if mp is not None and mc is not None:
            a, b = np.abs(out["prev"]).sum(), np.abs(out["curr"]).sum()
            out["prev_share"] = float(a / (a + b + 1e-12))
        return out
