"""Resumable experiment stages shared by the Colab notebook and the CLI scripts.

Every stage writes one small file per unit of work into `out` and skips units
that already exist, so an interrupted session resumes where it stopped.

  A  rollouts         closed-loop episodes, official protocol, decision points saved
  B  benchmark        all attribution methods at matched compute + pixel faithfulness
                      (+ seed stability on a subset)
     rescore          faithfulness of the saved B maps under another removal fill
  C  denoise          when during denoising does vision matter: eig split by step + knockout
  D  closed_loop      success rate when top / bottom / random patches are hidden online
     modality_timeline vision / language / state Shapley at every decision of whole episodes
  E  sanity           cascading model-parameter randomization (Adebayo et al. 2018)
  F  language         word x joint attribution (word removal + IG) and vision/language/state Shapley
  G  grounding        attribution mass per scene object vs the plan change when that object is removed
"""
import glob
import json
import os
import time

import numpy as np
import torch

from .attribution import Explainer
from .closedloop import run_condition
from .grounding import Grounding, group_index, groups, replay_segmentation, to_groups
from .language import LanguageExplainer
from .metrics import Faithfulness, rank_agreement, topk_iou
from .rollout import _render, load_episode, make_env, rollout


def _dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ── A ─────────────────────────────────────────────────────────────────────────
def rollouts(pol, tasks, episodes, out, render_backend=None, verbose=True):
    os.makedirs(out, exist_ok=True)
    rows = []
    for task in tasks:
        env = None
        for ep in episodes:
            d = os.path.join(out, f"{task}_ep{ep:03d}")
            if os.path.exists(os.path.join(d, "episode.json")):
                with open(os.path.join(d, "episode.json")) as f:
                    m = json.load(f)
            else:
                env = env or make_env(task, render_backend)
                m = rollout(pol, task, ep, env=env, save_dir=out)
            row = {k: m[k] for k in ("task", "episode", "seed", "success", "steps", "n_calls", "wall")}
            rows.append(row)
            if verbose:
                print(f"{task:20s} ep{ep:03d} success={row['success']!s:5} steps={row['steps']:3d} "
                      f"calls={row['n_calls']:2d} {row['wall']:6.1f}s", flush=True)
        if env is not None:
            env.close()
    return rows


def select_moments(rollout_dir, tasks=None, every=3, max_per_episode=4, only_success=True):
    """(episode_dir, call) pairs: every `every`-th decision, evenly thinned to
    at most `max_per_episode` per episode."""
    out = []
    for d in sorted(glob.glob(os.path.join(rollout_dir, "*_ep*"))):
        f = os.path.join(d, "episode.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            m = json.load(fh)
        if (tasks and m["task"] not in tasks) or (only_success and not m["success"]):
            continue
        calls = list(range(0, m["n_calls"], every))
        if len(calls) > max_per_episode:
            calls = [calls[int(round(i))] for i in np.linspace(0, len(calls) - 1, max_per_episode)]
        out += [(d, c) for c in calls]
    return out


def balanced_moments(moments, n=None):
    """Interleave tasks (round robin) so any prefix is task-balanced; keep the first n."""
    by_task = {}
    for d, c in moments:
        by_task.setdefault(os.path.basename(d).split("_ep")[0], []).append((d, c))
    groups = list(by_task.values())
    out = [groups[g][i] for i in range(max(map(len, groups), default=0)) for g in range(len(groups)) if i < len(groups[g])]
    return out if n is None else out[:n]


def _cond(pol, meta, c):
    dp = meta["decisions"][c]
    embs = pol.encode([dp["prev"], dp["curr"]])
    return dp, pol.conditions(meta["task"], embs[0:1], embs[1:2], dp["qpos8"])


# ── B ─────────────────────────────────────────────────────────────────────────
def benchmark(pol, moments, methods, out, n_evals=16, n_seeds=4, fast=False, stability=False,
              batch=8, joints=None, fill="gray", verbose=True):
    os.makedirs(out, exist_ok=True)
    ex = Explainer(pol, joints=joints, batch=batch)
    fm = Faithfulness.fast(pol, fill=fill) if fast else Faithfulness(pol, fill=fill)
    occ = 4 if fast else 3
    cache = {}
    for d, c in moments:
        tag = f"{os.path.basename(d)}_call{c:03d}"
        path = os.path.join(out, tag + ".json")
        if os.path.exists(path):
            continue
        if d not in cache:
            cache = {d: load_episode(d)}
        meta = cache[d]
        dp, cond = _cond(pol, meta, c)
        seed = dp["seed"] % 100000
        maps, rec = {}, dict(moment=tag, task=meta["task"], episode=meta["episode"], call=c, step=dp["step"],
                             n_calls=meta["n_calls"], n_evals=n_evals, fast=fast, methods={})
        t0 = time.time()
        for m in methods:
            r = ex.explain(cond, m, n_evals=n_evals, seed=seed, n_seeds=n_seeds, occ_block=occ)
            maps[m] = r["map"]
            rec["methods"][m] = {k: v for k, v in r.items() if k not in ("map", "prev", "curr", "steps")}
            if stability and m != "random":
                r2 = ex.explain(cond, m, n_evals=n_evals, seed=seed + 50000, n_seeds=n_seeds, occ_block=occ)
                rec["methods"][m]["seed_spearman"] = rank_agreement(r["map"], r2["map"])
                rec["methods"][m]["seed_top10_iou"] = topk_iou(r["map"], r2["map"])
        t1 = time.time()
        ff = fm.evaluate(cond, dp["prev"], dp["curr"], maps)
        for m in methods:
            rec["methods"][m].update(ff[m])
        rec["seconds_explain"], rec["seconds_faith"], rec["fill"] = t1 - t0, time.time() - t1, fill
        np.savez_compressed(os.path.join(out, tag + ".npz"), **maps)
        _dump(path, rec)
        if verbose:
            best = sorted(methods, key=lambda m: -ff[m]["deletion_morf"])[:3]
            print(f"{tag}  explain {t1-t0:5.1f}s  faith {time.time()-t1:5.1f}s  top deletion: "
                  + ", ".join(f"{m}={ff[m]['deletion_morf']:.2f}" for m in best), flush=True)


_FAITH_KEYS = ("d_full", "deletion_morf", "deletion_lerf", "insertion", "morf_minus_lerf",
               "curve_morf", "curve_lerf", "curve_ins")


def rescore(pol, src, out, rollout_dir, fill="inpaint", fast=False, verbose=True):
    """Faithfulness of the maps `benchmark` saved in `src` under another removal fill, without
    recomputing the explanations: each record is copied with its faithfulness metrics replaced."""
    os.makedirs(out, exist_ok=True)
    fm = Faithfulness.fast(pol, fill=fill) if fast else Faithfulness(pol, fill=fill)
    cache = {}
    for f in sorted(glob.glob(os.path.join(src, "*.json"))):
        tag = os.path.basename(f)[:-5]
        path, npz = os.path.join(out, tag + ".json"), os.path.join(src, tag + ".npz")
        if os.path.exists(path) or not os.path.exists(npz):
            continue
        with open(f) as fh:
            rec = json.load(fh)
        ep, c = tag.rsplit("_call", 1)
        d = os.path.join(rollout_dir, ep)
        if d not in cache:
            cache = {d: load_episode(d)}
        dp, cond = _cond(pol, cache[d], int(c))
        z = np.load(npz)
        maps = {m: z[m] for m in rec["methods"] if m in z.files}
        t0 = time.time()
        ff = fm.evaluate(cond, dp["prev"], dp["curr"], maps)
        for m, v in rec["methods"].items():
            for k in _FAITH_KEYS:
                v.pop(k, None)
            v.update(ff.get(m, {}))
        rec["seconds_faith"], rec["fill"], rec["faith_fast"] = time.time() - t0, fill, fast
        _dump(path, rec)
        if verbose:
            best = sorted(maps, key=lambda m: -ff[m]["deletion_morf"])[:3]
            print(f"{tag}  {fill} faith {time.time()-t0:5.1f}s  top deletion: "
                  + ", ".join(f"{m}={ff[m]['deletion_morf']:.2f}" for m in best), flush=True)


# ── C ─────────────────────────────────────────────────────────────────────────
def denoise(pol, moments, out, n_evals=16, n_seeds=4, batch=8, verbose=True):
    os.makedirs(out, exist_ok=True)
    ex = Explainer(pol, batch=batch)
    cache = {}
    for d, c in moments:
        tag = f"{os.path.basename(d)}_call{c:03d}"
        path = os.path.join(out, tag + ".json")
        if os.path.exists(path):
            continue
        if d not in cache:
            cache = {d: load_episode(d)}
        meta = cache[d]
        dp, cond = _cond(pol, meta, c)
        seed = dp["seed"] % 100000
        r = ex.explain(cond, "eig_steps", n_evals=n_evals, n_seeds=n_seeds, seed=seed)
        ko = ex.knockout(cond, seed=seed)
        steps = r["steps"]
        mass = np.abs(steps).reshape(len(steps), -1).sum(1)
        np.savez_compressed(os.path.join(out, tag + ".npz"), steps=steps)
        _dump(path, dict(moment=tag, task=meta["task"], call=c, step=dp["step"], n_calls=meta["n_calls"],
                         step_mass=(mass / mass.sum()).tolist(), knock_drop=ko["drop"].tolist(),
                         knock_only=ko["only"].tolist(), d_none=ko["d_none"],
                         completeness_rel_err=r["completeness_rel_err"]))
        if verbose:
            print(f"{tag}  attribution mass by step {np.round(mass / mass.sum(), 2)}  "
                  f"drop-one {np.round(ko['drop'], 2)}", flush=True)


# ── D ─────────────────────────────────────────────────────────────────────────
def closed_loop(pol, tasks, episodes, conditions, out, frac=0.1, n_evals=8, batch=8, render_backend=None,
                fill="inpaint", verbose=True):
    """conditions: list of (method, mode), e.g. [('none','none'), ('eig','top'), ('random','random')]."""
    os.makedirs(out, exist_ok=True)
    ex = Explainer(pol, batch=batch)
    for task in tasks:
        env = make_env(task, render_backend)
        for method, mode in conditions:
            for ep in episodes:
                path = os.path.join(out, f"{task}_{method}-{mode}-{int(frac*100)}_ep{ep:03d}.json")
                if os.path.exists(path):
                    continue
                row = run_condition(pol, ex, task, [ep], method, mode, frac, env=env, n_evals=n_evals, fill=fill)[0]
                _dump(path, row)
                if verbose:
                    print(f"{task:16s} {method:7s} {mode:10s} ep{ep:03d} success={row['success']!s:5} "
                          f"steps={row['steps']:3d} {row['wall']:6.1f}s", flush=True)
        env.close()


# ── E ─────────────────────────────────────────────────────────────────────────
def sanity(pol, moments, methods, out, n_evals=8, levels=(7, 14, 28), batch=8, verbose=True):
    """Re-initialize the last k RDT blocks (cascading from the output) and
    compare each method's map to the trained model's: a faithful method must
    change (low rank correlation) when the model it explains is destroyed."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "sanity.json")
    if os.path.exists(path):
        return
    ex = Explainer(pol, batch=batch)
    blocks = pol.rdt.model.blocks
    levels = sorted({min(k, len(blocks)) for k in levels})             # RDT-1B has 28 blocks
    backup ={i: {k: v.detach().cpu().clone() for k, v in blocks[i].state_dict().items()} for i in range(len(blocks))}
    rows = []
    prepared = [(d, c, *_cond(pol, load_episode(d), c)) for d, c in moments]
    base = {(d, c): {m: ex.explain(cond, m, n_evals=n_evals, seed=1)["map"] for m in methods}
            for d, c, _, cond in prepared}
    g = torch.Generator(device="cpu").manual_seed(0)
    try:
        for k in levels:
            for i in range(len(blocks) - k, len(blocks)):
                for name, prm in blocks[i].named_parameters():
                    if prm.dim() >= 2:                       # weights: redraw at the trained scale
                        std = backup[i][name].float().std().item() or 0.02
                        new = torch.randn(prm.shape, generator=g) * std
                    elif "norm" in name:                     # RMSNorm scales: reset to init
                        new = torch.ones(prm.shape)
                    else:                                    # biases
                        new = torch.zeros(prm.shape)
                    prm.data.copy_(new.to(prm.device, prm.dtype))
            for d, c, _, cond in prepared:
                for m in methods:
                    rnd = ex.explain(cond, m, n_evals=n_evals, seed=1)["map"]
                    rows.append(dict(moment=f"{os.path.basename(d)}_call{c:03d}", method=m, randomized_blocks=k,
                                     spearman=rank_agreement(base[(d, c)][m], rnd),
                                     abs_spearman=rank_agreement(np.abs(base[(d, c)][m]), np.abs(rnd))))
            if verbose:
                cur = [r for r in rows if r["randomized_blocks"] == k]
                print(f"randomized last {k:2d} blocks: " + "  ".join(
                    f"{m}={np.mean([r['spearman'] for r in cur if r['method']==m]):.2f}" for m in methods), flush=True)
    finally:
        for i in range(len(blocks)):
            blocks[i].load_state_dict({kk: vv.to(pol.device) for kk, vv in backup[i].items()})
    _dump(path, rows)


# ── F ─────────────────────────────────────────────────────────────────────────
def language(pol, moments, out, n_evals=16, batch=8, verbose=True):
    os.makedirs(out, exist_ok=True)
    lx = LanguageExplainer(pol, batch=batch)
    cache = {}
    for d, c in moments:
        tag = f"{os.path.basename(d)}_call{c:03d}"
        path = os.path.join(out, tag + ".json")
        if os.path.exists(path):
            continue
        if d not in cache:
            cache = {d: load_episode(d)}
        meta = cache[d]
        dp, cond = _cond(pol, meta, c)
        seed = dp["seed"] % 100000
        groups, labels = lx.words(meta["task"])
        t0 = time.time()
        drop = lx.word_drop(cond, groups, seeds=(seed, seed + 1))
        ig, rel = lx.word_ig(cond, groups, n_evals=n_evals, seed=seed)
        mod = lx.modality(cond, seeds=(seed, seed + 1))
        agree = [rank_agreement(np.abs(ig[j]), drop[j]) for j in range(8)]
        _dump(path, dict(moment=tag, task=meta["task"], call=c, step=dp["step"], n_calls=meta["n_calls"],
                         words=labels, word_drop=drop.tolist(), word_ig=ig.tolist(), ig_completeness_rel_err=rel.tolist(),
                         ig_drop_spearman=agree, modality=mod, seconds=time.time() - t0))
        if verbose:
            tot = sum(abs(v) for v in mod.values()) + 1e-12
            top = labels[int(np.argmax(drop.sum(0)))]
            print(f"{tag}  vision/lang/state share {mod['vision']/tot:.2f}/{mod['language']/tot:.2f}/"
                  f"{mod['state']/tot:.2f}  most important word: '{top}'  IG~drop rho={np.nanmean(agree):.2f}  "
                  f"{time.time()-t0:.1f}s", flush=True)


def modality_timeline(pol, rollout_dir, tasks, n_episodes, out, verbose=True):
    """Vision / language / state Shapley shares at every decision of the first `n_episodes`
    successful episodes per task (8 sampler calls per seed), for reliance over the episode."""
    os.makedirs(out, exist_ok=True)
    lx = LanguageExplainer(pol)
    for task in tasks:
        eps = sorted({d for d, _ in select_moments(rollout_dir, tasks=[task], every=1, max_per_episode=10 ** 6)})
        for d in eps[:n_episodes]:
            path = os.path.join(out, os.path.basename(d) + ".json")
            if os.path.exists(path):
                continue
            meta, rows, t0 = load_episode(d), [], time.time()
            for c in range(meta["n_calls"]):
                dp, cond = _cond(pol, meta, c)
                seed = dp["seed"] % 100000
                rows.append(dict(call=c, step=dp["step"], **lx.modality(cond, seeds=(seed, seed + 1))))
            _dump(path, dict(task=meta["task"], episode=meta["episode"], n_calls=meta["n_calls"], calls=rows))
            if verbose:
                v = np.array([[abs(r[k]) for k in ("vision", "language", "state")] for r in rows])
                v = v / (v.sum(1, keepdims=True) + 1e-12)
                print(f"{os.path.basename(d)}  {len(rows)} decisions  vision share first/last "
                      f"{v[0, 0]:.2f}/{v[-1, 0]:.2f}  {time.time()-t0:.1f}s", flush=True)


# ── G ─────────────────────────────────────────────────────────────────────────
def episode_segmentation(d, seg_dir, env_for):
    """Object ids of every frame the policy saw in episode `d` (replayed once, then cached)."""
    os.makedirs(seg_dir, exist_ok=True)
    npz, js = os.path.join(seg_dir, os.path.basename(d) + ".npz"), os.path.join(seg_dir, os.path.basename(d) + ".json")
    if not os.path.exists(js):
        with open(os.path.join(d, "episode.json")) as f:
            meta = json.load(f)
        segs, names, state, diffs = replay_segmentation(meta, d, env_for(meta["task"]), _render)
        np.savez_compressed(npz, **{k: v.astype(np.int16) for k, v in segs.items()})
        _dump(js, dict(names=names, state=state, rgb_diff_max=max(diffs, default=0.0),
                       rgb_diff_mean=float(np.mean(diffs)) if diffs else 0.0))
    with open(js) as f:
        info = json.load(f)
    z = np.load(npz)
    return {k: z[k] for k in z.files}, {int(k): v for k, v in info["names"].items()}, info


def grounding(pol, bench_dir, rollout_dir, out, fill="inpaint", render_backend=None, verbose=True):
    """Stage G on the maps saved by `benchmark`: per moment, each method's attribution share on each
    scene object vs the plan change when that object is removed (see grounding.py)."""
    os.makedirs(out, exist_ok=True)
    gr = Grounding(pol, fill=fill)
    envs = {}
    env_for = lambda task: envs.setdefault(task, make_env(task, render_backend))
    cache = {}
    try:
        for f in sorted(glob.glob(os.path.join(bench_dir, "*.npz"))):
            tag = os.path.basename(f)[:-4]
            path = os.path.join(out, tag + ".json")
            if os.path.exists(path):
                continue
            ep, c = tag.rsplit("_call", 1)
            c, d = int(c), os.path.join(rollout_dir, ep)
            segs, names, info = episode_segmentation(d, os.path.join(out, "seg"), env_for)
            if info["rgb_diff_max"] > 1.0:
                print(f"{ep}: replay differs from the saved frames (max mean |diff| {info['rgb_diff_max']:.2f}); "
                      "skipped", flush=True)
                continue
            if d not in cache:
                cache = {d: load_episode(d)}
            meta = cache[d]
            dp, cond = _cond(pol, meta, c)
            lut, G = group_index(meta["task"], names), len(groups(meta["task"]))
            gm = {k: None if f"call{c:03d}_{k}" not in segs else to_groups(segs[f"call{c:03d}_{k}"], lut, G)
                  for k in ("prev", "curr")}
            z = np.load(f)
            t0 = time.time()
            r = gr.evaluate(meta["task"], cond, dp["prev"], dp["curr"], gm["prev"], gm["curr"],
                            {m: z[m] for m in z.files})
            r.update(moment=tag, task=meta["task"], call=c, step=dp["step"], n_calls=meta["n_calls"],
                     state=info["state"].get(str(c), {}), fill=fill, seconds=time.time() - t0)
            _dump(path, r)
            if verbose:
                eff = ", ".join(f"{g}={e:.2f}" for g, e in zip(r["groups"], r["effect"]) if np.isfinite(e))
                print(f"{tag}  effect of removing: {eff}  | eig claims "
                      + ", ".join(f"{g}={v:.2f}" for g, v in zip(r["groups"], r["claim"]["eig"]))
                      + f"  rho={r['spearman']['eig']:.2f}  {time.time()-t0:.1f}s", flush=True)
    finally:
        for e in envs.values():
            e.close()
