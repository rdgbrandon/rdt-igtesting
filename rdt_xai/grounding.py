"""Object-level grounding of attribution maps, checked by intervention (a feedback loop).

An attribution map claims which pixels the policy used. The simulator knows which object
every pixel shows, so each claim can be restated as "the policy relies on object X" and
then tested on the policy itself:

  1. ground  replay the saved episode (same seed, recorded actions) and render the
             actor/link id of every pixel of the frames the policy saw; the replayed RGB is
             compared with the saved frame. Per patch: the fraction covered by each object
             group (task objects, gripper, arm, background).
  2. claim   per method, the share of |attribution| on each object group, and its
             enrichment (share / area share).
  3. test    remove one object group from both frames (inpainted, fill.py) and measure how
             far the executed plan moves (normalized by removing the whole view, D_full);
             control: an equally large random set of background patches.
  4. score   per method and moment, Spearman between claimed share and measured effect
             across the non-background groups, and whether the most-claimed group is the
             most-effective one.

Steps 3-4 are model-agnostic given a policy that exposes preprocess / sample, so the same
loop scores any attribution method on any policy. Caveat: SigLIP patch tokens mix
information across the image, so "patch (r, c) -> object under it" is the standard
positional reading, not a guarantee.
"""
import os

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

from .fill import remove_pixels
from .metrics import Faithfulness

SIG, PATCH = 384, 14

# task objects by actor name (ManiSkill 3 tabletop tasks); then gripper, arm, background
TASK_OBJECTS = {
    "PickCube-v1": (("cube", ("cube",)), ("goal", ("goal_site",))),
    "PushCube-v1": (("cube", ("cube",)), ("goal", ("goal_region",))),
    "StackCube-v1": (("red cube", ("cubeA",)), ("green cube", ("cubeB",))),
}
GRIPPER = ("hand", "finger", "tcp")
BACKGROUND = "background"


def groups(task):
    return [g for g, _ in TASK_OBJECTS.get(task, ())] + ["gripper", "arm", BACKGROUND]


def group_index(task, names):
    """{segmentation id: group index}; ids not listed (and 0) are background."""
    gs = groups(task)
    out = {}
    for i, name in names.items():
        n = name.lower()
        g = BACKGROUND
        for label, keys in TASK_OBJECTS.get(task, ()):
            if any(n == k.lower() or n.endswith("_" + k.lower()) for k in keys):
                g = label
                break
        else:
            if any(k in n for k in GRIPPER):
                g = "gripper"
            elif "panda" in n or "link" in n:
                g = "arm"
        out[int(i)] = gs.index(g)
    return out


def to_groups(seg, lut, n_groups):
    """(H, W) segmentation ids -> (H, W) group index."""
    g = np.full(seg.shape, n_groups - 1, dtype=np.int16)
    for i, gi in lut.items():
        g[seg == i] = gi
    return g


def _sig(gmap):
    """Group map at frame size -> SigLIP input size (frames are square, as RDT's camera)."""
    return np.asarray(Image.fromarray(gmap.astype(np.uint8)).resize((SIG, SIG), Image.NEAREST))


def patch_fractions(gmaps, n_groups, grid=27):
    """Group maps of the frames (None = missing camera) -> (grid*grid, G): per patch, the mean over
    frames of the fraction of its pixels in each group."""
    fr = []
    for g in gmaps:
        if g is None:
            continue
        s = _sig(g)[:grid * PATCH, :grid * PATCH]
        oh = np.eye(n_groups)[s].reshape(grid, PATCH, grid, PATCH, n_groups)
        fr.append(oh.mean((1, 3)).reshape(grid * grid, n_groups))
    return np.mean(fr, 0)


def claims(maps, frac):
    """{method: (G,) share of |attribution| per group}."""
    out = {}
    for m, a in maps.items():
        a = np.abs(np.asarray(a, dtype=np.float64)).reshape(-1)
        out[m] = (a @ frac) / (a.sum() + 1e-12)
    return out


def score(claim, effect, obj):
    """Spearman of claimed share vs measured effect over groups `obj`, and top-1 agreement."""
    c, e = np.asarray(claim)[obj], np.asarray(effect)[obj]
    ok = np.isfinite(e)
    c, e = c[ok], e[ok]
    if len(e) < 2:
        return np.nan, np.nan
    rho = spearmanr(c, e).correlation if np.ptp(c) > 0 and np.ptp(e) > 0 else np.nan
    return float(rho), float(np.argmax(c) == np.argmax(e))


class Grounding:
    def __init__(self, policy, eval_seeds=(101, 202), fill="inpaint", seed=0):
        self.p = policy
        self.fm = Faithfulness(policy, eval_seeds=eval_seeds, fill=fill)
        self.fill = fill
        self.rng = np.random.default_rng(seed)

    def _bg_patches(self, frac, k):
        """k random patches that are >= 90% background (fewer if not enough exist)."""
        cand = np.flatnonzero(frac[:, -1] >= 0.9)
        return self.rng.choice(cand, size=min(k, len(cand)), replace=False)

    def _patch_pixels(self, idx, grid):
        m = np.zeros(grid * grid, dtype=bool)
        m[idx] = True
        m = np.kron(m.reshape(grid, grid), np.ones((PATCH, PATCH), dtype=bool))
        out = np.zeros((SIG, SIG), dtype=bool)
        out[:m.shape[0], :m.shape[1]] = m
        return out

    def evaluate(self, task, cond, img_prev, img_curr, g_prev, g_curr, maps):
        """g_*: (H, W) group maps of the two frames (None where the frame is the missing camera).
        Returns per-group area share, causal effect, control effect, and per-method claims and scores."""
        p, fm = self.p, self.fm
        gs = groups(task)
        G, grid = len(gs), p.grid
        frac = patch_fractions([g_prev, g_curr], G, grid)
        x_prev, x_curr = p.preprocess([img_prev])[0], p.preprocess([img_curr])[0]
        end = fm._plans(cond, torch.stack([x_prev, fm.bg_pix]), torch.stack([x_curr, fm.bg_pix]))
        ref = end[0]
        dev = lambda P: (P - ref[None]).flatten(2).norm(dim=-1).mean(1).cpu().numpy()
        d_full = max(float(dev(end[1:2])[0]), 1e-8)

        sig = [None if g is None else _sig(g) for g in (g_prev, g_curr)]
        obj = list(range(G - 1))                                        # every group but background
        mp, mc, jobs = [], [], []
        for gi in obj:
            masks = [np.zeros((SIG, SIG), bool) if s is None else s == gi for s in sig]
            area = max(m.sum() for m in masks)
            if area == 0:
                continue
            mp.append(masks[0]), mc.append(masks[1]), jobs.append((gi, "object"))
            ctrl = self._patch_pixels(self._bg_patches(frac, int(np.ceil(area / PATCH ** 2))), grid)
            mp.append(ctrl & (sig[0] is not None)), mc.append(ctrl), jobs.append((gi, "control"))
        effect, control = np.full(G, np.nan), np.full(G, np.nan)
        if jobs:
            to = lambda ms: torch.from_numpy(np.stack(ms))[:, None].to(x_curr.device)
            pp = remove_pixels(x_prev, to(mp), fm.bg_pix, self.fill)
            pc = remove_pixels(x_curr, to(mc), fm.bg_pix, self.fill)
            d = dev(fm._plans(cond, pp, pc)) / d_full
            for (gi, kind), v in zip(jobs, d):
                (effect if kind == "object" else control)[gi] = v
        cl = claims(maps, frac)
        sc = {m: score(c, effect, obj) for m, c in cl.items()}
        area = frac.mean(0)
        return dict(groups=gs, area_share=area.tolist(), effect=effect.tolist(), control=control.tolist(),
                    d_full=d_full, claim={m: c.tolist() for m, c in cl.items()},
                    enrichment={m: (c / np.maximum(area, 1e-6)).tolist() for m, c in cl.items()},
                    spearman={m: s[0] for m, s in sc.items()}, top1={m: s[1] for m, s in sc.items()})


# ── replaying a saved episode for segmentation ─────────────────────────────────
def segmentation(env, camera="render_camera"):
    """(H, W) actor/link id of every pixel of the render camera (the frames RDT sees)."""
    cam = env.unwrapped.scene.human_render_cameras[camera]
    cam.capture()
    s = cam.get_obs(rgb=False, depth=False, position=False, segmentation=True)["segmentation"]
    s = s.cpu().numpy() if hasattr(s, "cpu") else np.asarray(s)
    return s.squeeze().astype(np.int32)


def _scalars(d):
    out = {}
    for k, v in (d or {}).items():
        v = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        if v.size == 1:
            out[k] = float(v.reshape(-1)[0])
    return out


def replay_segmentation(meta, ep_dir, env, render, seg_fn=segmentation):
    """Re-run a saved episode with its seed and recorded actions; at every decision, render the
    segmentation of the frames the policy saw and the task state (env.evaluate()). Returns
    ({'callKKK_prev'/'callKKK_curr': (H, W) ids}, {id: name}, {call: state}, [mean |RGB diff|
    against the saved frames])."""
    env.reset(seed=meta["seed"])
    decs = meta["decisions"]
    need = {}
    for k, dp in enumerate(decs):
        need.setdefault(dp["step"], []).append((k, "curr"))
        if dp["prev"] is not None:
            need.setdefault(dp["step"] - 1, []).append((k, "prev"))
    segs, state, diffs = {}, {}, []

    def grab(step):
        if step not in need:
            return
        rgb = np.asarray(render(env), dtype=np.int16)
        seg = seg_fn(env)
        for k, name in need[step]:
            segs[f"call{k:03d}_{name}"] = seg
            saved = decs[k][name]
            if isinstance(saved, str):
                saved = Image.open(os.path.join(ep_dir, saved))
            diffs.append(float(np.abs(rgb - np.asarray(saved.convert("RGB"), dtype=np.int16)).mean()))
            if name == "curr":
                state[k] = _scalars(env.unwrapped.evaluate())

    step = 0
    grab(0)
    for k, dp in enumerate(decs):
        if dp["step"] != step:
            raise RuntimeError(f"replay out of sync: decision {k} at step {dp['step']}, replay at {step}")
        end = decs[k + 1]["step"] if k + 1 < len(decs) else meta["steps"]
        for a in dp["actions"][:end - step]:
            env.step(np.asarray(a))
            step += 1
            grab(step)
    names = {int(i): o.name for i, o in env.unwrapped.segmentation_id_map.items()}
    return segs, names, state, diffs
