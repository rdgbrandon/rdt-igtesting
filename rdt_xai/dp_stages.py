"""The RDT study's stages, run on the Diffusion Policy baseline (dp.py). Same letters as experiments.py:

  A  rollouts         episodes with every frame, segmentation and call recorded; success rate; the
                      mean frame / state of these episodes become the no-vision inputs
  B  benchmark        maps of every method + deletion / insertion faithfulness, per moment
  G  grounding        (with B: the segmentation is recorded live) claimed vs measured object reliance
  C  denoise          expected-IG share and vision knockout per bin of denoising steps
  D  closed_loop      success with a method's top / bottom / random patches removed at every call
  E  sanity           cascading randomization of the U-Net, then the visual encoder
  F  modality         vision vs proprioception Shapley shares over whole episodes

Every unit of work is one .json written last (atomically), so an interrupted stage resumes.
"""
import copy
import glob
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from . import dp as D

_dump = lambda path, obj: (json.dump(obj, open(path + ".tmp", "w")), os.replace(path + ".tmp", path))


def _done(path):
    return os.path.exists(path)


# ── A ─────────────────────────────────────────────────────────────────────────
def rollouts(pol, task, episodes, out, verbose=True):
    os.makedirs(out, exist_ok=True)
    env = D.make_env(task)
    try:
        for ep in episodes:
            path = os.path.join(out, f"{task}_ep{ep:03d}.json")
            if _done(path):
                continue
            r = D.rollout(pol, task, ep, env=env, save_dir=out)
            row = {k: r[k] for k in ("task", "episode", "seed", "success", "steps", "n_calls", "wall")}
            _dump(path, row)
            if verbose:
                print(f"A {task} ep{ep:03d}: success={r['success']} steps={r['steps']} ({r['wall']:.0f}s)", flush=True)
    finally:
        env.close()
    background(out, task)


def background(rollout_dir, task):
    """Mean frame and mean state over the stage-A episodes of a task (the no-vision inputs)."""
    path = os.path.join(rollout_dir, f"{task}_background.npz")
    if not _done(path):
        rgb, st = [], []
        for d in sorted(glob.glob(os.path.join(rollout_dir, f"{task}_ep*/"))):
            z = np.load(os.path.join(d, "frames.npz"))
            rgb.append(z["rgb"].astype(np.float64).mean(0)), st.append(z["state"].mean(0))
        np.savez(path, rgb=np.mean(rgb, 0).round().astype(np.uint8), state=np.mean(st, 0).astype(np.float32))
    z = np.load(path)
    return z["rgb"], z["state"]


def moments(rollout_dir, task, every=3, max_per_episode=4, only_success=True, n=None):
    out = []
    for d in sorted(glob.glob(os.path.join(rollout_dir, f"{task}_ep*/"))):
        meta = json.load(open(os.path.join(d, "episode.json")))
        if only_success and not meta["success"]:
            continue
        calls = list(range(0, meta["n_calls"], every))[:max_per_episode]
        out += [(d.rstrip("/\\"), c) for c in calls]
    return out[:n] if n else out


def _key(d, c):
    return f"{os.path.basename(d)}_call{c:03d}"


# ── B + G ─────────────────────────────────────────────────────────────────────
def benchmark(pol, task, moms, methods, out, n_evals=16, n_seeds=4, fast=False, fill="inpaint", verbose=True):
    os.makedirs(out, exist_ok=True)
    ex = D.DPExplainer(pol)
    fm = D.DPFaithfulness.fast(pol, fill=fill) if fast else D.DPFaithfulness(pol, fill=fill)
    gr = D.DPGrounding(pol, fill=fill)
    for d, c in moms:
        path = os.path.join(out, _key(d, c) + ".json")
        if _done(path):
            continue
        meta = D.load_episode(d)
        frames, states, seg = D.moment(meta, c)
        cond = pol.conditions(frames, states)
        res = {m: ex.explain(cond, m, n_evals=n_evals, n_seeds=n_seeds, seed=1) for m in methods}
        maps = {m: r["map"] for m, r in res.items()}
        faith = fm.evaluate(cond, maps)
        rec = dict(task=task, episode=os.path.basename(d), call=c, step=meta["calls"][c]["step"],
                   progress=c / max(1, meta["n_calls"] - 1), info=meta["calls"][c]["info"], fill=fill,
                   methods={m: dict(faith[m], seconds=r["seconds"], unsigned=r["unsigned"],
                                    completeness_rel_err=r.get("completeness_rel_err"),
                                    prev_share=r.get("prev_share")) for m, r in res.items()})
        if seg is not None:
            rec["grounding"] = gr.evaluate(task, cond, D.group_maps(task, meta, seg), maps)
        np.savez_compressed(path[:-5] + "_maps.npz", **maps)
        _dump(path, rec)
        if verbose:
            g = rec.get("grounding", {})
            print(f"B {_key(d, c)}: " + " ".join(f"{m}={v['deletion_morf']:.2f}/{v['insertion']:.2f}"
                                                 for m, v in rec["methods"].items())
                  + (f" | effect {np.round(g['effect'], 2).tolist()}" if g else ""), flush=True)


# ── C ─────────────────────────────────────────────────────────────────────────
def denoise(pol, moms, out, n_evals=16, n_seeds=4, verbose=True):
    os.makedirs(out, exist_ok=True)
    ex = D.DPExplainer(pol)
    for d, c in moms:
        path = os.path.join(out, _key(d, c) + ".json")
        if _done(path):
            continue
        frames, states, _ = D.moment(D.load_episode(d), c)
        cond = pol.conditions(frames, states)
        r = ex.explain(cond, "eig_steps", n_evals=n_evals, n_seeds=n_seeds, seed=1)
        mass = np.abs(r["steps"]).reshape(D.N_BINS, -1).sum(1)
        ko = ex.knockout(cond, seed=1)
        rec = dict(share=(mass / mass.sum()).tolist(), drop=ko["drop"].tolist(), only=ko["only"].tolist(),
                   d_none=ko["d_none"], completeness_rel_err=r.get("completeness_rel_err"))
        _dump(path, rec)
        if verbose:
            print(f"C {_key(d, c)}: share {np.round(rec['share'], 2).tolist()} drop {np.round(rec['drop'], 2).tolist()}",
                  flush=True)


# ── D ─────────────────────────────────────────────────────────────────────────
def closed_loop(pol, task, episodes, conditions, out, frac=0.05, n_evals=8, fill="inpaint", verbose=True):
    """conditions: list of 'method:mode' (e.g. 'none:none', 'random:random', 'eig:top_abs')."""
    os.makedirs(out, exist_ok=True)
    ex = D.DPExplainer(pol)
    env = D.make_env(task, segmentation=False)
    try:
        for cnd in conditions:
            method, mode = cnd.split(":")
            for ep in episodes:
                path = os.path.join(out, f"{task}_{method}_{mode}_{frac:g}_ep{ep:03d}.json")
                if _done(path):
                    continue
                hook = D.MaskHook(pol, ex, method=method, mode=mode, frac=frac, n_evals=n_evals, seed=ep, fill=fill)
                r = D.rollout(pol, task, 1000 + ep, env=env, obs_hook=hook)
                _dump(path, dict(task=task, episode=ep, method=method, mode=mode, frac=frac, fill=fill,
                                 success=r["success"], steps=r["steps"], wall=r["wall"]))
                if verbose:
                    print(f"D {task} {cnd} ep{ep:03d}: success={r['success']} ({r['wall']:.0f}s)", flush=True)
    finally:
        env.close()


# ── E ─────────────────────────────────────────────────────────────────────────
def _cascade(agent):
    """Module groups from the output backwards (Adebayo et al. 2018)."""
    net, enc = agent.noise_pred_net, agent.visual_encoder
    return [("final_conv", [net.final_conv]), ("up", list(net.up_modules)), ("mid", list(net.mid_modules)),
            ("down", list(net.down_modules) + [net.diffusion_step_encoder]), ("encoder", [enc])]


def _randomize(modules, g):
    for mod in modules:
        for name, prm in mod.named_parameters():
            if prm.dim() >= 2:                                 # weights: redraw at the trained scale
                prm.data = torch.randn(prm.shape, generator=g).to(prm) * (prm.float().std().item() or 0.02)
            elif name.endswith("weight"):                      # GroupNorm scales: reset to init
                prm.data.fill_(1.0)
            else:                                              # biases
                prm.data.zero_()


def sanity(pol, moms, methods, out, n_evals=8, verbose=True):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "sanity.json")
    if _done(path):
        return
    conds = [pol.conditions(*D.moment(D.load_episode(d), c)[:2]) for d, c in moms]
    ex = D.DPExplainer(pol)
    base = [{m: ex.explain(cd, m, n_evals=n_evals, seed=1)["map"] for m in methods} for cd in conds]
    agent = copy.deepcopy(pol.agent)
    g = torch.Generator().manual_seed(0)
    rows, upto = [], []
    for level, mods in _cascade(agent):
        _randomize(mods, g)
        upto.append(level)
        rp = D.DPPolicy(agent, bg=pol.bg, state_mean=pol.state_mean)
        rx = D.DPExplainer(rp)
        for i, cd in enumerate(conds):
            for m in methods:
                a = rx.explain(cd, m, n_evals=n_evals, seed=1)["map"]
                rho = spearmanr(np.abs(base[i][m]).ravel(), np.abs(a).ravel()).correlation
                rows.append(dict(level=level, upto="+".join(upto), moment=i, method=m, spearman_abs=float(rho)))
        if verbose:
            print(f"E {level}: " + " ".join(f"{m}={np.nanmean([r['spearman_abs'] for r in rows if r['level'] == level and r['method'] == m]):.2f}"
                                           for m in methods), flush=True)
    _dump(path, rows)


# ── F ─────────────────────────────────────────────────────────────────────────
def modality(pol, rollout_dir, task, n_episodes, out, verbose=True):
    os.makedirs(out, exist_ok=True)
    ex = D.DPExplainer(pol)
    eps = [d for d in sorted(glob.glob(os.path.join(rollout_dir, f"{task}_ep*/")))][:n_episodes]
    for d in eps:
        path = os.path.join(out, os.path.basename(d.rstrip("/\\")) + ".json")
        if _done(path):
            continue
        meta = D.load_episode(d)
        rows = []
        for c in range(meta["n_calls"]):
            frames, states, _ = D.moment(meta, c)
            r = ex.modality(pol.conditions(frames, states))
            rows.append(dict(call=c, progress=c / max(1, meta["n_calls"] - 1), vision=r["vision"], state=r["state"]))
        _dump(path, rows)
        if verbose:
            print(f"F {os.path.basename(d.rstrip('/'))}: vision {np.mean([r['vision'] for r in rows]):.2f}", flush=True)


# ── summary ───────────────────────────────────────────────────────────────────
def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, c - h), min(1.0, c + h)


def _load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern)) if not f.endswith("sanity.json")]


def summary(out, task):
    L = [f"== Diffusion Policy, {task} =="]
    A = _load(os.path.join(out, "A", f"{task}_ep*.json"))
    if A:
        k = sum(r["success"] for r in A)
        p, lo, hi = _wilson(k, len(A))
        L.append(f"A  success {k}/{len(A)} = {p:.0%} (95% CI {lo:.0%}-{hi:.0%})")
    B = [r for r in _load(os.path.join(out, "B", f"{task}_ep*.json"))]
    if B:
        ms = list(B[0]["methods"])
        L.append(f"B  faithfulness over {len(B)} moments (median; rank 1 = best, mean over moments):")
        ranks = {m: [] for m in ms}
        for r in B:
            order = sorted(ms, key=lambda m: -r["methods"][m]["deletion_morf"])
            for i, m in enumerate(order):
                ranks[m].append(i + 1)
        for m in sorted(ms, key=lambda m: np.mean(ranks[m])):
            dm = np.median([r["methods"][m]["deletion_morf"] for r in B])
            ins = np.median([r["methods"][m]["insertion"] for r in B])
            L.append(f"     {m:10s} deletion {dm:.2f}  insertion {ins:.2f}  deletion rank {np.mean(ranks[m]):.1f}")
        G = [r["grounding"] for r in B if "grounding" in r]
        if G:
            gs = G[0]["groups"]
            eff = np.nanmean([g["effect"] for g in G], 0)
            ctl = np.nanmean([g["control"] for g in G], 0)
            L.append("G  removing each object (plan change / D_full; control = as many background patches): "
                     + ", ".join(f"{n} {e:.2f} vs {c:.2f}" for n, e, c in zip(gs[:-1], eff[:-1], ctl[:-1])))
            for m in ms:
                rho = np.nanmean([g["spearman"][m] for g in G])
                top = np.nanmean([g["top1"][m] for g in G])
                share = np.mean([g["claim"][m] for g in G], 0)
                L.append(f"     {m:10s} claim vs effect spearman {rho:+.2f}, top-1 {top:.0%}, claimed share "
                         + " ".join(f"{n} {s:.2f}" for n, s in zip(gs, share)))
    C = _load(os.path.join(out, "C", f"{task}_ep*.json"))
    if C:
        f = lambda k: np.round(np.mean([r[k] for r in C], 0), 2).tolist()
        L.append(f"C  over {len(C)} moments, 5 bins of 20 DDPM steps (noisy -> clean): EIG share {f('share')}, "
                 f"plan change with vision removed in one bin {f('drop')}, recovered with vision in one bin only "
                 f"{f('only')}")
    Dr = _load(os.path.join(out, "D", f"{task}_*.json"))
    if Dr:
        L.append("D  closed loop:")
        for cnd in sorted({(r["method"], r["mode"], r["frac"]) for r in Dr}):
            rs = [r for r in Dr if (r["method"], r["mode"], r["frac"]) == cnd]
            k = sum(r["success"] for r in rs)
            p, lo, hi = _wilson(k, len(rs))
            L.append(f"     {cnd[0]}:{cnd[1]} ({cnd[2]:.0%} hidden) {k}/{len(rs)} = {p:.0%} ({lo:.0%}-{hi:.0%})")
    sp = os.path.join(out, "E", "sanity.json")
    if os.path.exists(sp):
        E = json.load(open(sp))
        ms = sorted({r["method"] for r in E})
        L.append("E  map rank correlation with the trained model's, randomizing from the output (|map|, lower = "
                 "passes):")
        for lv in dict.fromkeys(r["upto"] for r in E):
            L.append(f"     +{lv.split('+')[-1]:10s} " + " ".join(
                f"{m} {np.nanmean([r['spearman_abs'] for r in E if r['upto'] == lv and r['method'] == m]):.2f}"
                for m in ms))
    Fr = [x for r in _load(os.path.join(out, "F", f"{task}_ep*.json")) for x in r]
    if Fr:
        L.append(f"F  over {len(Fr)} calls: vision {np.mean([r['vision'] for r in Fr]):.2f}, "
                 f"proprioception {np.mean([r['state'] for r in Fr]):.2f}")
    return "\n".join(L)
