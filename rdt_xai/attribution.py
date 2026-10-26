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

Methods (n_evals = number of gradient evaluations, matched across methods):
  grad        |dF/dE| (L2 over channels)                         1 eval
  gxi         dF/dE . (E - E')                                   1 eval
  ig_zero     IG, zero baseline, fixed noise                     n_evals
  ig_bg       IG, background baseline, fixed noise               n_evals
  blurig      BlurIG path in embedding space, fixed noise        n_evals
  eig         expected IG: background baseline, noise z resampled n_evals
              per evaluation (stratified alphas per seed)  [proposed]
  attn        cross-attention mass on each patch                 0 (1 forward)
  occlusion   kxk-patch block replacement by background, dF     81 (k=3) / 49 (k=4) forwards
  random      iid noise map                                      0
"""
import math
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF

EXEC_ROWS = torch.arange(0, 64, 4)


def _blur(E, sigma, grid=27):
    if sigma < 0.05:
        return E
    B, N, D = E.shape
    img = E.reshape(B, grid, grid, D).permute(0, 3, 1, 2)
    ks = 2 * int(3 * sigma + 0.5) + 1
    out = TF.gaussian_blur(img.float(), kernel_size=[ks, ks], sigma=float(sigma)).to(E.dtype)
    return out.permute(0, 2, 3, 1).reshape(B, N, D)


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

    @torch.no_grad()
    def actions(self, cond, Ep, Ec, seeds):
        """Normalized executed actions for a batch of (embedding, seed) pairs."""
        out = []
        for i in range(0, len(seeds), self.batch):
            s = seeds[i:i + self.batch]
            noise = torch.cat([self.p.noise(x) for x in s])
            b = len(s)
            out.append(self._exec(self.p.sample(cond, emb_curr=Ec[i:i + b] if Ec.shape[0] > 1 else Ec.expand(b, -1, -1),
                                                emb_prev=Ep[i:i + b] if Ep.shape[0] > 1 else Ep.expand(b, -1, -1),
                                                noise=noise)))
        return torch.cat(out)

    def directions(self, cond, base_p, base_c, seeds):
        a = self.actions(cond, cond["emb_prev"], cond["emb_curr"], seeds)
        a0 = self.actions(cond, base_p, base_c, seeds)
        d = a - a0
        n = d.flatten(1).norm(dim=1).clamp_min(1e-8)
        return d / n[:, None, None], n, a, a0

    # ── path-integral engine ─────────────────────────────────────────────────
    def _path_integral(self, cond, pts_p, pts_c, deltas_p, deltas_c, seeds, dirs):
        """sum_k grad F_{z_k}(P_k) . dP_k for points P_k with increments dP_k.
        pts_*: (K, 729, D); seeds: K ints; dirs: (K, 16, J) unit directions."""
        K = len(seeds)
        attr_p = torch.zeros_like(cond["emb_prev"][0], dtype=torch.float32)
        attr_c = torch.zeros_like(attr_p)
        for i in range(0, K, self.batch):
            sl = slice(i, min(K, i + self.batch))
            Ep = pts_p[sl].clone().requires_grad_(True)
            Ec = pts_c[sl].clone().requires_grad_(True)
            noise = torch.cat([self.p.noise(x) for x in seeds[sl]])
            with torch.enable_grad():
                A = self._exec(self.p.sample(cond, emb_curr=Ec, emb_prev=Ep, noise=noise))
                F = (A * dirs[sl]).sum()
                gp, gc = torch.autograd.grad(F, [Ep, Ec])
            attr_p += (gp * deltas_p[sl]).float().sum(0)
            attr_c += (gc * deltas_c[sl]).float().sum(0)
        return attr_p, attr_c

    def _straight(self, cond, base_p, base_c, alphas, seeds, dir_seeds):
        dirs_u, dnorm, _, _ = self.directions(cond, base_p, base_c, sorted(set(dir_seeds)))
        lut = {s: k for k, s in enumerate(sorted(set(dir_seeds)))}
        dirs = torch.stack([dirs_u[lut[s]] for s in dir_seeds])
        a = torch.tensor(alphas, device=self.p.device, dtype=self.p.dtype)[:, None, None]
        Ep, Ec = cond["emb_prev"], cond["emb_curr"]
        pts_p, pts_c = base_p + a * (Ep - base_p), base_c + a * (Ec - base_c)
        K = len(alphas)
        dp = ((Ep - base_p) / K).expand(K, -1, -1)
        dc = ((Ec - base_c) / K).expand(K, -1, -1)
        ap, ac = self._path_integral(cond, pts_p, pts_c, dp, dc, seeds, dirs)
        # completeness: each evaluation estimates F_z(E) - F_z(E') = ||d_z|| / K
        # of the total, so sum(attr) should match the mean of ||d_z|| over evals
        target = float(sum(dnorm[lut[s]] for s in dir_seeds) / K)
        return ap, ac, target

    # ── methods ──────────────────────────────────────────────────────────────
    def explain(self, cond, method, n_evals=16, seed=0, n_seeds=4, sigma_max=None, occ_block=3):
        """Returns dict(map=(27,27) signed patch attribution summed over both
        slots, prev/curr maps, completeness error, seconds)."""
        t0 = time.time()
        p, G = self.p, self.p.grid
        bg = p.bg_emb
        Ep, Ec = cond["emb_prev"], cond["emb_curr"]
        info = {}
        if method in ("ig_bg", "ig_zero", "eig"):
            base = torch.zeros_like(bg) if method == "ig_zero" else bg
            if method == "eig":
                S = max(1, min(n_seeds, n_evals))
                per = n_evals // S
                seeds_ = [seed * 7919 + s for s in range(S)]
                # stratified alpha: seed s gets offsets shifted by s/S of a bin
                alphas, seeds = [], []
                for si, s in enumerate(seeds_):
                    for k in range(per):
                        alphas.append((k + (si + 0.5) / S) / per)
                        seeds.append(s)
            else:
                alphas = [(k + 0.5) / n_evals for k in range(n_evals)]
                seeds = [seed] * n_evals
            ap, ac, target = self._straight(cond, base, base, alphas, seeds, seeds)
            info["completeness_target"] = target
        elif method == "blurig":
            smax = sigma_max or float(G) / 2                       # fully blurred at grid scale
            sig = np.linspace(smax, 0.0, n_evals + 1)
            mids = [(sig[k] + sig[k + 1]) / 2 for k in range(n_evals)]
            # blur path endpoints define the "baseline" for the direction
            dirs, dn, _, _ = self.directions(cond, _blur(Ep, smax, G), _blur(Ec, smax, G), [seed])
            P = lambda E, s: torch.cat([_blur(E, x, G) for x in s])
            pts_p, pts_c = P(Ep, mids), P(Ec, mids)
            dp = torch.cat([_blur(Ep, sig[k + 1], G) - _blur(Ep, sig[k], G) for k in range(n_evals)])
            dc = torch.cat([_blur(Ec, sig[k + 1], G) - _blur(Ec, sig[k], G) for k in range(n_evals)])
            ap, ac = self._path_integral(cond, pts_p, pts_c, dp, dc, [seed] * n_evals, dirs.expand(n_evals, -1, -1))
            info["completeness_target"] = float(dn[0])
        elif method in ("grad", "gxi"):
            dirs, _, _, _ = self.directions(cond, bg, bg, [seed])
            one = torch.ones(1, device=p.device, dtype=p.dtype)[:, None, None]
            gp, gc = self._path_integral(cond, Ep, Ec, one.expand(1, *Ep.shape[1:]), one.expand(1, *Ec.shape[1:]),
                                         [seed], dirs)
            if method == "grad":
                m = (gp.norm(dim=-1) + gc.norm(dim=-1))
                return self._pack(m, None, None, info, t0, unsigned=True)
            ap, ac = gp * (Ep[0] - bg[0]).float(), gc * (Ec[0] - bg[0]).float()
        elif method == "attn":
            with torch.no_grad():
                _, log = p.sample(cond, noise=p.noise(seed), return_attn=True)
            w = torch.stack(log).float()                            # (steps*blocks, B, N, L)
            w = w[:, 0, -p.horizon:].mean((0, 1))                   # mean over steps, blocks, action tokens
            n = p.n_patch
            m = w[0:n] + w[3 * n:4 * n]                             # prev + curr external slots
            return self._pack(m, None, None, info, t0, unsigned=True)
        elif method == "occlusion":
            return self._occlusion(cond, seed, t0, block=occ_block)
        elif method == "random":
            g = torch.Generator().manual_seed(seed)
            return self._pack(torch.rand(p.n_patch, generator=g), None, None, info, t0, unsigned=True)
        else:
            raise ValueError(method)
        return self._pack(ap.sum(-1) + ac.sum(-1), ap.sum(-1), ac.sum(-1), info, t0)

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

    def _pack(self, m, mp, mc, info, t0, unsigned=False):
        G = self.p.grid
        to = lambda x: None if x is None else x.detach().float().cpu().numpy().reshape(G, G)
        out = dict(map=to(m), prev=to(mp), curr=to(mc), seconds=time.time() - t0, unsigned=unsigned, **info)
        if "completeness_target" in info:
            s = float(out["map"].sum())
            out["attr_sum"] = s
            out["completeness_rel_err"] = abs(s - info["completeness_target"]) / (abs(info["completeness_target"]) + 1e-8)
        return out


METHODS = ["random", "attn", "grad", "gxi", "occlusion", "ig_zero", "ig_bg", "blurig", "eig"]
