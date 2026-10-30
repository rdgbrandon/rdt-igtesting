"""ManiSkill's Diffusion Policy baseline (train_rgbd.py) on the same attribution footing as RDT.

Policy: 2 frames of the 128x128 base camera -> PlainConv (shared over frames) -> features, concatenated
with the 2 proprioceptive states -> FiLM conditioning of a 1D U-Net that denoises a 16-step chunk of
pd_ee_delta_pos actions with 100 DDPM steps; steps 1..8 of the chunk are executed.

Differences from RDT and how each analysis is carried over:
  * input space   RGB pixels of both frames (RDT: SigLIP patch embeddings). Maps are summed over
                  channels, both frames and 8x8-pixel patches onto a 16x16 grid (RDT: 27x27).
  * no vision     the policy was never trained without a camera, so the baseline image is the mean
                  frame of the evaluation episodes (stage A), the counterpart of RDT's "missing camera"
                  image; removed patches are inpainted relative to it (fill.py).
  * sampler noise DDPM draws fresh noise at every step, so a "seed" fixes the initial noise and all
                  100 step noises (the DDPM update is written out to make that explicit and batched).
  * time          the 100 denoising steps are split into 5 equal bins (RDT: 5 DPMSolver steps).
  * attention     the visual encoder is a CNN; DAAM / GAE are replaced by Grad-CAM on its last
                  feature map (4x4 per frame).
  * state order   gymnasium >= 1.0 sorts the keys of the batched observation space, which is the
                  order the training data packs the state in (colab.DP_PATCHES); `state_vector` does
                  the same.

Target, as for RDT: F_z(x) = < A(x; z)[exec], u_z >, u_z = unit(A(x; z) - A(x'; z)) with x' the no-vision
input, so completeness reads sum(attr) = ||A(x; z) - A(x'; z)||.
"""
import json
import math
import os
import sys
import time
import types
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .fill import remove_pixels
from .grounding import claims, group_index, groups, score, to_groups
from .metrics import FRACS_FAST, FRACS_FULL, _auc

IMG, GRID, PATCH = 128, 16, 8
N_BINS = 5
OBS_H, ACT_H = 2, 8
EXEC = slice(OBS_H - 1, OBS_H - 1 + ACT_H)              # executed rows of the 16-step chunk
MAX_STEPS = {"PickCube-v1": 100, "PushCube-v1": 100, "StackCube-v1": 200}
METHODS = ["random", "gradcam", "grad", "gxi", "occlusion", "iss", "ig_zero", "ig_bg", "blurig", "eig"]
DP_CODE = os.environ.get("DP_CODE", "/content/ManiSkill/examples/baselines/diffusion_policy")


# ── loading ────────────────────────────────────────────────────────────────────
def load_agent(ckpt, code=DP_CODE, device=None):
    """The baseline's Agent with a checkpoint's EMA weights. Dimensions are read from the weights."""
    from gymnasium import spaces
    if code not in sys.path:
        sys.path.insert(0, code)
    import train_rgbd as T
    sd = torch.load(ckpt, map_location="cpu")["ema_agent"]
    args = T.Args()
    cond = next(v.shape[1] for k, v in sd.items() if k.endswith("cond_encoder.1.weight"))
    state_dim = (cond - args.diffusion_step_embed_dim) // args.obs_horizon - 256       # 256: PlainConv out_dim
    act_dim = sd["noise_pred_net.final_conv.1.weight"].shape[0]
    box = lambda *s: spaces.Box(-np.inf, np.inf, s, np.float32)
    env = types.SimpleNamespace(
        single_observation_space=spaces.Dict(dict(state=box(args.obs_horizon, state_dim),
                                                  rgb=spaces.Box(0, 255, (args.obs_horizon, IMG, IMG, 3), np.uint8))),
        single_action_space=spaces.Box(-1, 1, (act_dim,), np.float32))
    agent = T.Agent(env, args)
    agent.load_state_dict(sd)
    return agent.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval().requires_grad_(False)


def state_vector(obs, space):
    """Proprioceptive state as train_rgbd.py packs it: agent then extra values, in the key order of the
    env's batched observation space. obs: raw ManiSkill obs (batch 1). Returns (D,) float32."""
    parts = []
    for group in ("agent", "extra"):
        for k in space[group].keys():
            v = obs[group][k]
            v = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
            parts.append(v.astype(np.float32).reshape(-1))
    return np.concatenate(parts)


# ── policy ─────────────────────────────────────────────────────────────────────
class DPPolicy:
    def __init__(self, agent, bg=None, state_mean=None):
        self.agent = agent
        self.device = next(agent.parameters()).device
        self.dtype = torch.float32
        self.grid, self.n_patch, self.patch = GRID, GRID * GRID, PATCH
        self.horizon, self.act_dim = agent.pred_horizon, agent.act_dim
        self.sched = agent.noise_scheduler
        self.T = self.sched.config.num_train_timesteps
        self.ts = torch.arange(self.T - 1, -1, -1)                     # 99 .. 0, as the baseline samples
        self.bins = [i * N_BINS // self.T for i in range(self.T)]       # denoising step index -> bin
        ac = self.sched.alphas_cumprod.to(self.device, torch.float32)
        prev = torch.cat([torch.ones(1, device=self.device), ac[:-1]])
        beta = 1 - ac / prev
        self._c = dict(sa=ac.sqrt(), s1a=(1 - ac).sqrt(), cx0=prev.sqrt() * beta / (1 - ac),
                       cxt=(ac / prev).sqrt() * (1 - prev) / (1 - ac),
                       sd=((1 - prev) / (1 - ac) * beta).clamp_min(1e-20).sqrt())
        self.set_background(bg, state_mean)

    def set_background(self, bg=None, state_mean=None):
        """bg: (H, W, 3) uint8 or (3, H, W) float no-vision image (default mid-grey)."""
        if bg is None:
            bg = np.full((IMG, IMG, 3), 127, np.uint8)
        bg = torch.as_tensor(np.asarray(bg))
        if bg.dtype == torch.uint8:
            bg = bg.permute(2, 0, 1).float() / 255
        self.bg = bg.to(self.device, self.dtype)                          # (3, H, W)
        self.state_mean = None if state_mean is None else torch.as_tensor(
            np.asarray(state_mean, np.float32), device=self.device)

    # inputs
    def pixels(self, frames):
        """(2, H, W, 3) uint8 frames [t-1, t] -> (1, 2, 3, H, W) float in [0, 1]."""
        x = torch.as_tensor(np.asarray(frames)).to(self.device)
        return (x.permute(0, 3, 1, 2).float() / 255)[None]

    def conditions(self, frames, states):
        return dict(x=self.pixels(frames), state=torch.as_tensor(np.asarray(states, np.float32),
                                                                 device=self.device)[None])

    def no_vision(self, x):
        return self.bg.expand_as(x)

    def features(self, x, fmap=False):
        """x: (B, 2, 3, H, W) in [0, 1] -> (B, 2, 256) visual features (and the CNN's last feature map)."""
        enc = self.agent.visual_encoder
        B = x.shape[0]
        m = enc.cnn(x.flatten(0, 1))                                      # (B*2, 128, 4, 4)
        f = enc.fc((enc.pool(m) if enc.pool is not None else m).flatten(1)).reshape(B, OBS_H, -1)
        return (f, m) if fmap else f

    def _cond(self, feat, state):
        return torch.cat([feat, state.expand(feat.shape[0], -1, -1)], -1).flatten(1)

    def noise(self, seed, batch=1):
        """Initial noise and the 100 DDPM step noises of one seed: ((B, 16, A), (T, B, 16, A))."""
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        z0 = torch.randn(1, self.horizon, self.act_dim, generator=g)
        zs = torch.randn(self.T, 1, self.horizon, self.act_dim, generator=g)
        to = lambda z: z.expand(*z.shape[:-3], batch, *z.shape[-2:]).to(self.device, self.dtype)
        return to(z0), to(zs)

    def noises(self, seeds):
        n = [self.noise(s) for s in seeds]
        return torch.cat([a for a, _ in n]), torch.cat([b for _, b in n], 1)

    def sample(self, cond, x=None, state=None, noise=None, seed=0, vision_bins=None, bin_x=None,
               return_fmap=False):
        """Denoise an action chunk; returns normalized actions (B, 16, A).
        x: override the pixels (B, 2, 3, H, W), may require grad.  state: override (B or 1, 2, D).
        vision_bins: bins (of 5) whose steps see the real view; the others see the no-vision image.
        bin_x: list of 5 pixel tensors, one per bin (separate leaves: one backward splits the gradient
               by denoising time)."""
        x = cond["x"] if x is None else x
        state = cond["state"] if state is None else state
        B = bin_x[0].shape[0] if bin_x is not None else x.shape[0]
        z0, zs = self.noise(seed, B) if noise is None else noise
        fmap = None
        if bin_x is not None:
            conds = [self._cond(self.features(b), state) for b in bin_x]
        else:
            if return_fmap:
                feat, fmap = self.features(x, fmap=True)
            else:
                feat = self.features(x)
            real = self._cond(feat, state)
            conds = [real] * N_BINS
            if vision_bins is not None:
                off = self._cond(self.features(self.no_vision(x[:1])).expand(B, -1, -1), state)
                conds = [real if b in set(vision_bins) else off for b in range(N_BINS)]
        c, net = self._c, self.agent.noise_pred_net
        a = z0
        for i, t in enumerate(self.ts.tolist()):
            eps = net(a, torch.full((B,), t, device=self.device, dtype=torch.long), global_cond=conds[self.bins[i]])
            x0 = ((a - c["s1a"][t] * eps) / c["sa"][t]).clamp(-1, 1)            # clip_sample=True
            a = c["cx0"][t] * x0 + c["cxt"][t] * a
            if t > 0:
                a = a + c["sd"][t] * zs[i]
        return (a, fmap) if return_fmap else a

    @torch.no_grad()
    def act(self, frames, states, seed=0):
        """One policy call as the baseline's evaluate(): the 8 executed actions (8, A)."""
        return self.sample(self.conditions(frames, states), seed=seed)[0, EXEC].float().cpu().numpy()


# ── attribution ────────────────────────────────────────────────────────────────
def _patches(pix):
    """(..., H, W) pixel map -> (..., 16, 16) sums over 8x8 patches."""
    return pix.reshape(*pix.shape[:-2], GRID, PATCH, GRID, PATCH).sum((-3, -1))


def _blur(x, sigma):
    if sigma < 0.05:
        return x
    ks = min(2 * int(3 * sigma + 0.5) + 1, IMG - 1)                    # reflect pad must stay < size
    return TF.gaussian_blur(x.flatten(0, 1), kernel_size=[ks, ks], sigma=float(sigma)).reshape(x.shape)


def _eig_schedule(n_evals, n_seeds, seed):
    S = max(1, min(n_seeds, n_evals))
    per = n_evals // S
    alphas, seeds = [], []
    for si in range(S):
        for k in range(per):
            alphas.append((k + (si + 0.5) / S) / per)
            seeds.append(seed * 7919 + si)
    return alphas, seeds


class DPExplainer:
    def __init__(self, policy, batch=8):
        self.p = policy
        self.batch = batch

    def _exec(self, A):
        return A[:, EXEC]

    @torch.no_grad()
    def actions(self, cond, X, seeds, **kw):
        """Executed actions for (pixels, seed) pairs; X: (K or 1, 2, 3, H, W)."""
        out = []
        for i in range(0, len(seeds), self.batch):
            s = seeds[i:i + self.batch]
            xb = X[i:i + len(s)] if X.shape[0] > 1 else X.expand(len(s), -1, -1, -1, -1)
            out.append(self._exec(self.p.sample(cond, x=xb, noise=self.p.noises(s), **kw)))
        return torch.cat(out)

    def directions(self, cond, base, seeds):
        a = self.actions(cond, cond["x"], seeds)
        a0 = self.actions(cond, base, seeds)
        d = a - a0
        n = d.flatten(1).norm(dim=1).clamp_min(1e-8)
        return d / n[:, None, None], n, a, a0

    def _path_integral(self, cond, pts, deltas, seeds, dirs, bin_split=False):
        """sum_k grad F_{z_k}(P_k) * dP_k (per pixel, channels kept): (2, 3, H, W), or per bin (5, ...)."""
        nb = N_BINS if bin_split else 1
        attr = torch.zeros((nb,) + tuple(cond["x"].shape[1:]), device=self.p.device)
        for i in range(0, len(seeds), self.batch):
            sl = slice(i, min(len(seeds), i + self.batch))
            with torch.enable_grad():
                if bin_split:
                    leaves = [pts[sl].clone().requires_grad_(True) for _ in range(N_BINS)]
                    A = self._exec(self.p.sample(cond, bin_x=leaves, noise=self.p.noises(seeds[sl])))
                else:
                    leaves = [pts[sl].clone().requires_grad_(True)]
                    A = self._exec(self.p.sample(cond, x=leaves[0], noise=self.p.noises(seeds[sl])))
                grads = torch.autograd.grad((A * dirs[sl]).sum(), leaves)
            for b in range(nb):
                attr[b] += (grads[b] * deltas[sl]).sum(0)
        return attr if bin_split else attr[0]

    def _straight(self, cond, base, alphas, seeds, bin_split=False):
        uniq = sorted(set(seeds))
        u, n, _, _ = self.directions(cond, base, uniq)
        lut = {s: k for k, s in enumerate(uniq)}
        dirs = torch.stack([u[lut[s]] for s in seeds])
        a = torch.tensor(alphas, device=self.p.device)[:, None, None, None, None]
        x = cond["x"]
        pts = base + a * (x - base)
        d = ((x - base) / len(alphas)).expand(len(alphas), -1, -1, -1, -1)
        return self._path_integral(cond, pts, d, seeds, dirs, bin_split), float(n[[lut[s] for s in seeds]].mean())

    def explain(self, cond, method, n_evals=16, seed=0, n_seeds=4, occ_block=2, iss_keep=0.3):
        """dict(map=(16, 16), prev/curr maps where defined, completeness info, seconds)."""
        t0 = time.time()
        p, x = self.p, cond["x"]
        bg = p.no_vision(x)
        info = {}
        if method in ("ig_bg", "ig_zero", "eig", "eig_steps"):
            base = torch.zeros_like(x) if method == "ig_zero" else bg
            if method in ("eig", "eig_steps"):
                alphas, seeds = _eig_schedule(n_evals, n_seeds, seed)
            else:
                alphas, seeds = [(k + 0.5) / n_evals for k in range(n_evals)], [seed] * n_evals
            attr, info["completeness_target"] = self._straight(cond, base, alphas, seeds, method == "eig_steps")
            if method == "eig_steps":
                steps = _patches(attr.sum(2)).sum(1).cpu().numpy()                    # (5, 16, 16)
                out = self._pack(torch.from_numpy(steps.sum(0)), None, info, t0)
                out["steps"] = steps
                return out
        elif method == "blurig":
            smax = IMG / 4
            sig = np.linspace(smax, 0.0, n_evals + 1)
            dirs, dn, _, _ = self.directions(cond, _blur(x, smax), [seed])
            pts = torch.cat([_blur(x, (sig[k] + sig[k + 1]) / 2) for k in range(n_evals)])
            d = torch.cat([_blur(x, sig[k + 1]) - _blur(x, sig[k]) for k in range(n_evals)])
            attr = self._path_integral(cond, pts, d, [seed] * n_evals, dirs.expand(n_evals, -1, -1))
            info["completeness_target"] = float(dn[0])
        elif method in ("grad", "gxi"):
            dirs, _, _, _ = self.directions(cond, bg, [seed])
            g = self._path_integral(cond, x, torch.ones_like(x), [seed], dirs)
            if method == "grad":
                return self._pack(_patches(g.norm(dim=1)).sum(0), None, info, t0, unsigned=True)
            attr = g * (x[0] - bg[0])
        elif method == "gradcam":
            return self._gradcam(cond, seed, t0)
        elif method == "occlusion":
            return self._occlusion(cond, seed, t0, occ_block)
        elif method == "iss":
            return self._iss(cond, seed, t0, 3 * n_evals, iss_keep)
        elif method == "random":
            g = torch.Generator().manual_seed(seed)
            return self._pack(torch.rand(GRID, GRID, generator=g), None, info, t0, unsigned=True)
        else:
            raise ValueError(method)
        return self._pack(_patches(attr.sum(1)).sum(0), _patches(attr.sum(1)), info, t0)

    def _gradcam(self, cond, seed, t0):
        """Grad-CAM (Selvaraju'17) on PlainConv's last feature map, per frame, upsampled to the grid."""
        dirs, _, _, _ = self.directions(cond, self.p.no_vision(cond["x"]), [seed])
        with torch.enable_grad():
            A, fmap = self.p.sample(cond, x=cond["x"].clone().requires_grad_(True), noise=self.p.noises([seed]),
                                    return_fmap=True)
            g, = torch.autograd.grad((self._exec(A) * dirs).sum(), fmap)
        cam = (g.mean((-2, -1), keepdim=True) * fmap).sum(1).clamp_min(0)          # (2, 4, 4)
        cam = F.interpolate(cam[:, None], size=(GRID, GRID), mode="bilinear", align_corners=False)[:, 0]
        return self._pack(cam.sum(0), None, {}, t0, unsigned=True)

    def _mask_pixels(self, sel):
        """(R, 256) bool patch selections -> (R, 1, 1, 1, H, W) pixel masks (both frames)."""
        m = sel.reshape(-1, GRID, GRID).repeat_interleave(PATCH, 1).repeat_interleave(PATCH, 2)
        return m[:, None, None, None].to(self.p.device)

    def _occlusion(self, cond, seed, t0, block=2):
        nb = math.ceil(GRID / block)
        dirs, _, a_full, _ = self.directions(cond, self.p.no_vision(cond["x"]), [seed])
        F_full = (a_full * dirs).sum().item()
        sel = torch.zeros(nb * nb, GRID, GRID, dtype=torch.bool)
        for by in range(nb):
            for bx in range(nb):
                sel[by * nb + bx, by * block:(by + 1) * block, bx * block:(bx + 1) * block] = True
        M = self._mask_pixels(sel.reshape(nb * nb, -1))[:, 0]
        X = torch.where(M, self.p.bg, cond["x"])
        A = self.actions(cond, X, [seed] * len(sel))
        drop = F_full - (A * dirs).flatten(1).sum(1)
        return self._pack((sel.to(drop.device).float() * drop[:, None, None]).sum(0) / block ** 2, None, {}, t0)

    def _iss(self, cond, seed, t0, n_masks=48, keep=0.3):
        g = torch.Generator().manual_seed(seed + 12345)
        removed = torch.rand(n_masks, self.p.n_patch, generator=g) >= keep
        ref = self.actions(cond, cond["x"], [seed])
        X = torch.where(self._mask_pixels(removed)[:, 0], self.p.bg, cond["x"])
        A = self.actions(cond, X, [seed] * n_masks)
        sc = (A - ref).flatten(1).pow(2).sum(1)
        sal = (removed.to(sc.device).float() * sc[:, None]).sum(0) / (n_masks * (1 - keep))
        return self._pack(sal.reshape(GRID, GRID), None, {}, t0, unsigned=True)

    @torch.no_grad()
    def knockout(self, cond, seed=0, n_seeds=2):
        """Per bin of denoising steps: plan deviation with the view removed only there ('drop'), and the
        share of the vision effect recovered with the view only there ('only'). Normalized by the
        no-vision deviation."""
        seeds = [seed * 7919 + s for s in range(n_seeds)]
        x = cond["x"]
        full = self.actions(cond, x, seeds)
        none = self.actions(cond, x, seeds, vision_bins=[])
        d_none = (full - none).flatten(1).norm(dim=1)
        drop, only = [], []
        for b in range(N_BINS):
            a_drop = self.actions(cond, x, seeds, vision_bins=[i for i in range(N_BINS) if i != b])
            a_only = self.actions(cond, x, seeds, vision_bins=[b])
            drop.append(((full - a_drop).flatten(1).norm(dim=1) / d_none).mean().item())
            only.append((1 - (full - a_only).flatten(1).norm(dim=1) / d_none).mean().item())
        return dict(drop=np.array(drop), only=np.array(only), d_none=float(d_none.mean()))

    @torch.no_grad()
    def modality(self, cond, seeds=(0, 1)):
        """Two-player Shapley split of the plan between vision and proprioception: v(S) = deviation of the
        plan with only S present from the plan with neither (no-vision image, mean state)."""
        p, seeds = self.p, list(seeds)
        x0, s0 = p.no_vision(cond["x"]), p.state_mean[None, None].expand_as(cond["state"])
        plan = lambda x, s: self._exec(p.sample(cond, x=x.expand(len(seeds), -1, -1, -1, -1), state=s,
                                                noise=p.noises(seeds)))
        a = {k: plan(x, s) for k, (x, s) in dict(none=(x0, s0), v=(cond["x"], s0), s=(x0, cond["state"]),
                                                     vs=(cond["x"], cond["state"])).items()}
        v = {k: float((a[k] - a["none"]).flatten(1).norm(dim=1).mean()) for k in a}
        phi_v = 0.5 * (v["v"] + v["vs"] - v["s"])
        phi_s = 0.5 * (v["s"] + v["vs"] - v["v"])
        tot = abs(phi_v) + abs(phi_s) + 1e-12
        return dict(vision=abs(phi_v) / tot, state=abs(phi_s) / tot, v=v)

    def _pack(self, m, frames, info, t0, unsigned=False):
        to = lambda t: t.detach().float().cpu().numpy()
        out = dict(map=to(m).reshape(GRID, GRID), seconds=time.time() - t0, unsigned=unsigned, **info)
        if frames is not None:
            out["prev"], out["curr"] = to(frames[0]), to(frames[1])
            a, b = np.abs(out["prev"]).sum(), np.abs(out["curr"]).sum()
            out["prev_share"] = float(a / (a + b + 1e-12))
        if "completeness_target" in info:
            out["attr_sum"] = float(out["map"].sum())
            out["completeness_rel_err"] = abs(out["attr_sum"] - info["completeness_target"]) / (
                abs(info["completeness_target"]) + 1e-8)
        return out


# ── faithfulness (pixel space, same metrics as metrics.Faithfulness) ──────────
def patch_pixels(sel):
    """(256,) or (R, 256) bool patches -> (R, 1, H, W) bool pixels."""
    sel = torch.as_tensor(np.asarray(sel)).reshape(-1, GRID, GRID)
    return sel.repeat_interleave(PATCH, 1).repeat_interleave(PATCH, 2)[:, None]


class DPFaithfulness:
    def __init__(self, policy, eval_seeds=(101, 202), fracs=FRACS_FULL, curves=("morf", "lerf", "ins"), batch=10,
                 fill="inpaint"):
        self.p, self.fill = policy, fill
        self.eval_seeds = list(eval_seeds)
        self.fracs = np.asarray(fracs, dtype=np.float64)
        self.curves, self.batch = tuple(curves), batch

    @classmethod
    def fast(cls, policy, batch=10, fill="inpaint"):
        return cls(policy, eval_seeds=(101,), fracs=FRACS_FAST, curves=("morf", "ins"), batch=batch, fill=fill)

    @torch.no_grad()
    def plans(self, cond, X):
        """X: (N, 2, 3, H, W) -> executed plans (N, S, 8, A)."""
        out = []
        for i in range(0, X.shape[0], self.batch):
            xb = X[i:i + self.batch]
            out.append(torch.stack([self.p.sample(cond, x=xb, noise=self.p.noises([s] * xb.shape[0]))[:, EXEC]
                                    for s in self.eval_seeds], 1))
        return torch.cat(out).float()

    def remove(self, x, M):
        """x: (1, 2, 3, H, W); M: (J, 1, H, W) or per frame (J, 2, 1, H, W) bool -> (J, 2, 3, H, W)."""
        M = M.to(self.p.device)
        Mf = [M[:, f] if M.dim() == 5 else M for f in range(OBS_H)]
        return torch.stack([remove_pixels(x[0, f], Mf[f], self.p.bg, self.fill) for f in range(OBS_H)], 1)

    def endpoints(self, cond):
        """Reference plans and d_full (whole view removed) shared by every curve of a moment."""
        end = self.plans(cond, torch.cat([cond["x"], self.p.no_vision(cond["x"])]))
        ref = end[0]
        dev = lambda P: (P - ref[None]).flatten(2).norm(dim=-1).mean(1).cpu().numpy()
        return dev, max(float(dev(end[1:2])[0]), 1e-8)

    def evaluate(self, cond, maps):
        n = self.p.n_patch
        ks = [int(round(f * n)) for f in self.fracs]
        inner = [k for k in ks if 0 < k < n]
        dev, d_full = self.endpoints(cond)
        jobs, sels = [], []
        for name, m in maps.items():
            s = torch.as_tensor(np.asarray(m, dtype=np.float32).reshape(-1))
            orders = {"morf": torch.argsort(s, descending=True), "lerf": torch.argsort(s)}
            orders["ins"] = orders["morf"]
            for cv in self.curves:
                for k in inner:
                    sel = torch.zeros(n, dtype=torch.bool)
                    sel[orders[cv][:k]] = True
                    sels.append(~sel if cv == "ins" else sel)
                    jobs.append((name, cv))
        d = dev(self.plans(cond, self.remove(cond["x"], patch_pixels(torch.stack(sels))))) / d_full if jobs else []
        res = {name: dict(d_full=d_full) for name in maps}
        it = iter(d)
        for name in maps:
            for cv in self.curves:
                mid = [float(next(it)) for _ in inner]
                if cv == "ins":
                    curve = [0.0] + [1.0 - x for x in mid] + [1.0]
                    res[name]["insertion"] = _auc(curve, self.fracs)
                else:
                    curve = [0.0] + mid + [1.0]
                    res[name]["deletion_" + cv] = _auc(curve, self.fracs)
                res[name]["curve_" + cv] = curve
            if "deletion_lerf" in res[name]:
                res[name]["morf_minus_lerf"] = res[name]["deletion_morf"] - res[name]["deletion_lerf"]
        return res


# ── grounding on the sensor camera's own segmentation ──────────────────────────
class DPGrounding:
    """grounding.Grounding for DP: the segmentation comes with every observation (no replay)."""

    def __init__(self, policy, fill="inpaint", seed=0, eval_seeds=(101, 202)):
        self.p, self.fm = policy, DPFaithfulness(policy, eval_seeds=eval_seeds, fill=fill)
        self.rng = np.random.default_rng(seed)

    def evaluate(self, task, cond, gmaps, maps):
        """gmaps: (2, H, W) group index of each pixel of both frames."""
        gs = groups(task)
        G = len(gs)
        oh = np.eye(G)[gmaps].reshape(OBS_H, GRID, PATCH, GRID, PATCH, G)
        frac = oh.mean((2, 4)).mean(0).reshape(self.p.n_patch, G)
        dev, d_full = self.fm.endpoints(cond)
        obj = list(range(G - 1))
        masks, jobs = [], []
        for gi in obj:
            m = gmaps == gi
            area = int(m.reshape(OBS_H, -1).sum(1).max())
            if area == 0:
                continue
            masks.append(torch.from_numpy(m)[:, None]), jobs.append((gi, "object"))
            cand = np.flatnonzero(frac[:, -1] >= 0.9)
            sel = np.zeros(self.p.n_patch, bool)
            sel[self.rng.choice(cand, size=min(int(np.ceil(area / PATCH ** 2)), len(cand)), replace=False)] = True
            masks.append(patch_pixels(sel).expand(OBS_H, -1, -1, -1)), jobs.append((gi, "control"))
        effect, control = np.full(G, np.nan), np.full(G, np.nan)
        if jobs:
            d = dev(self.fm.plans(cond, self.fm.remove(cond["x"], torch.stack(masks)))) / d_full
            for (gi, kind), v in zip(jobs, d):
                (effect if kind == "object" else control)[gi] = v
        cl = claims(maps, frac)
        sc = {m: score(c, effect, obj) for m, c in cl.items()}
        area = frac.mean(0)
        return dict(groups=gs, area_share=area.tolist(), effect=effect.tolist(), control=control.tolist(),
                    d_full=d_full, claim={m: c.tolist() for m, c in cl.items()},
                    enrichment={m: (c / np.maximum(area, 1e-6)).tolist() for m, c in cl.items()},
                    spearman={m: s[0] for m, s in sc.items()}, top1={m: s[1] for m, s in sc.items()})


# ── closed loop ────────────────────────────────────────────────────────────────
def make_env(task, segmentation=True, max_steps=None):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    return gym.make(task, obs_mode="rgb+segmentation" if segmentation else "rgb", control_mode="pd_ee_delta_pos",
                    render_mode="rgb_array", reward_mode="sparse", sim_backend="physx_cpu",
                    max_episode_steps=max_steps or MAX_STEPS.get(task, 100))


def _np(v):
    return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)


def _scalars(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            continue
        v = _np(v)
        if v.size == 1:
            out[k] = float(v.reshape(-1)[0])
    return out


class MaskHook:
    """Closed-loop intervention: at every call remove a fraction of the patches (both frames) chosen by
    a method's map. mode: none / top / bottom / top_abs / bottom_abs / random (as closedloop.MaskHook)."""

    def __init__(self, policy, explainer, method="eig", mode="top", frac=0.05, n_evals=8, n_seeds=4, seed=0,
                 fill="inpaint"):
        self.p, self.ex, self.method, self.mode, self.fill = policy, explainer, method, mode, fill
        self.n_evals, self.n_seeds, self.seed = n_evals, n_seeds, seed
        self.k = int(round(frac * policy.n_patch))
        self.fm = DPFaithfulness(policy, fill=fill)
        self.log = []

    def __call__(self, frames, states, ctx):
        if self.mode == "none" or self.k == 0:
            return frames
        n = self.p.n_patch
        cond = self.p.conditions(frames, states)
        if self.mode == "random":
            order = np.random.default_rng(self.seed * 100003 + ctx["call"]).permutation(n)
        else:
            r = self.ex.explain(cond, self.method, n_evals=self.n_evals, n_seeds=self.n_seeds,
                                seed=self.seed * 1000 + ctx["call"])
            s = r["map"].reshape(-1)
            s = np.abs(s) if self.mode.endswith("_abs") else s
            order = np.argsort(-s) if self.mode.startswith("top") else np.argsort(s)
        sel = np.zeros(n, bool)
        sel[order[:self.k]] = True
        self.log.append(dict(call=ctx["call"], step=ctx["step"], sel=np.flatnonzero(sel).tolist()))
        x = self.fm.remove(cond["x"], patch_pixels(sel))[0]                                  # (2, 3, H, W)
        return (x.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().byte().cpu().numpy()


def rollout(policy, task, episode, env=None, obs_hook=None, base_seed=20241201, save_dir=None, max_steps=None):
    """One episode as the baseline evaluates it (2-frame stack of consecutive env steps, 8 actions per
    call, success_once), stopping at the first success. Records every frame, its segmentation and
    state, and every call's inputs, so moments can be explained and grounded later."""
    own = env is None
    env = env or make_env(task, max_steps=max_steps)
    max_steps = max_steps or MAX_STEPS.get(task, 100)
    space = env.unwrapped.observation_space
    seed = base_seed + episode
    obs, _ = env.reset(seed=seed)
    cam = next(iter(obs["sensor_data"]))

    def frame(o):
        d = o["sensor_data"][cam]
        seg = _np(d["segmentation"])[0, ..., 0] if "segmentation" in d else None
        return _np(d["rgb"])[0].astype(np.uint8), seg, state_vector(o, space)

    rgbs, segs, states = [list(v) for v in zip(frame(obs))]
    win = deque([0, 0], maxlen=OBS_H)
    calls, step, success, info, t0 = [], 0, False, {}, time.time()
    while step < max_steps and not success:
        idx = list(win)
        frames, st = np.stack([rgbs[i] for i in idx]), np.stack([states[i] for i in idx])
        ctx = dict(step=step, call=len(calls), task=task)
        seen = obs_hook(frames, st, ctx) if obs_hook is not None else frames
        call_seed = seed * 1000 + len(calls)
        acts = policy.act(seen, st, seed=call_seed)
        calls.append(dict(step=step, frames=idx, seed=call_seed, actions=acts.tolist(), info=_scalars(info)))
        for a in acts:
            obs, _, _, _, info = env.step(a)
            for lst, v in zip((rgbs, segs, states), frame(obs)):
                lst.append(v)
            win.append(len(rgbs) - 1)
            step += 1
            success = bool(_np(info.get("success", False)).reshape(-1)[0])
            if success or step >= max_steps:
                break
    names = {int(i): o.name for i, o in getattr(env.unwrapped, "segmentation_id_map", {}).items()}
    if own:
        env.close()
    res = dict(task=task, episode=episode, seed=seed, success=success, steps=step, n_calls=len(calls),
               wall=time.time() - t0, final_info=_scalars(info))
    if save_dir:
        d = os.path.join(save_dir, f"{task}_ep{episode:03d}")
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "frames.npz"), rgb=np.stack(rgbs), state=np.stack(states),
                            **({"seg": np.stack(segs).astype(np.int16)} if segs[0] is not None else {}))
        with open(os.path.join(d, "episode.json"), "w") as f:
            json.dump(dict(res, calls=calls, seg_names=names), f)
    res.update(calls=calls, rgb=rgbs, seg=segs, state=states, seg_names=names)
    return res


def load_episode(ep_dir):
    with open(os.path.join(ep_dir, "episode.json")) as f:
        meta = json.load(f)
    z = np.load(os.path.join(ep_dir, "frames.npz"))
    meta.update(rgb=z["rgb"], state=z["state"], seg=z["seg"] if "seg" in z else None)
    meta["seg_names"] = {int(k): v for k, v in meta["seg_names"].items()}
    return meta


def moment(meta, call):
    """(frames (2, H, W, 3), states (2, D), segmentation (2, H, W) or None) of one call."""
    idx = meta["calls"][call]["frames"]
    seg = None if meta["seg"] is None else meta["seg"][idx]
    return meta["rgb"][idx], meta["state"][idx], seg


def group_maps(task, meta, seg):
    lut = group_index(task, meta["seg_names"])
    return np.stack([to_groups(s, lut, len(groups(task))) for s in seg])
