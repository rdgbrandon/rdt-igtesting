#!/usr/bin/env python3
"""Run one study stage from a JSON config, in whichever Python environment
executes this script (the Colab kernel, the RDT-era venv, or an Expanse node).

    python scripts/run_stage.py --config cfg.json --stage A
    stages: verify, A, B, C, D, E, F, G, video

Config keys (all optional except out): see DEFAULTS. Every stage is resumable.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULTS = dict(
    tasks=["PickCube-v1", "StackCube-v1", "PushCube-v1"], n_episodes=16,
    methods=["random", "daam", "gae", "grad", "gxi", "occlusion", "iss", "ig_zero", "ig_bg", "blurig", "eig"],
    n_evals=16, fast=False, b_moments=60, b_stability_moments=24, moments_every=3, moments_per_episode=3,
    b_fill="inpaint", b_fill_fast=False,
    c_moments=30, d_tasks=["PickCube-v1", "StackCube-v1"], d_episodes=16, d_frac=0.05, d_fill="inpaint",
    d_conditions=[["none", "none"], ["random", "random"], ["eig", "top_abs"], ["eig", "bottom_abs"], ["eig", "top"],
                  ["daam", "top"], ["blurig", "top"]],
    e_moments=8, e_methods=["eig", "ig_bg", "blurig", "grad", "gae", "daam"], f_moments=30, f_timeline_episodes=8,
    dtype="fp32", obs_mode="rgb", batch=8,
)

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--stage", required=True, choices=["verify", "A", "B", "C", "D", "E", "F", "G", "video"])
ap.add_argument("--episode", default="", help="video: episode dir name under <out>/A (default: first successful)")
args = ap.parse_args()
cfg = dict(DEFAULTS, **json.load(open(args.config)))
out = cfg["out"]
os.environ["RDT_OBS_MODE"] = cfg["obs_mode"]
if cfg["dtype"] == "bf16":
    os.environ["RDT_DTYPE"] = "bf16"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rdt_xai import experiments as X  # noqa: E402
from rdt_xai.policy import RDTPolicy  # noqa: E402


def versions():
    import importlib.metadata as md
    v = {"python": sys.version.split()[0]}
    for p in ("torch", "transformers", "diffusers", "timm", "mani_skill", "sapien", "numpy"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    return v


t_start = time.time()
if os.environ.get("RDT_TINY"):            # test hook: tiny random model with the real architecture
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from tiny import tiny_policy
    pol = tiny_policy()
else:
    pol = RDTPolicy()
print(f"[{args.stage}] policy on {pol.device} {pol.dtype}  {versions()}", flush=True)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, f"env_{args.stage}.json"), "w") as f:
    json.dump(dict(versions=versions(), config=cfg), f)


def moments():
    m = X.select_moments(os.path.join(out, "A"), tasks=cfg["tasks"], every=cfg["moments_every"],
                         max_per_episode=cfg["moments_per_episode"])
    return X.balanced_moments(m)


if args.stage == "verify":
    from PIL import Image
    from rdt_xai.attribution import Explainer
    from rdt_xai.metrics import Faithfulness
    from rdt_xai.rollout import _render, make_env
    sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (lambda: None)
    rng = np.random.default_rng(0)
    ip, ic = (Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)) for _ in range(2))
    q = np.array([0.0, 0.4, 0.0, -2.0, 0.0, 2.4, 0.8, 0.04], dtype=np.float32)
    with torch.no_grad():
        lang, noise = pol.lang("PickCube-v1"), pol.noise(123)
        e6 = pol.encode([ip, None, None, ic, None, None]).reshape(1, -1, pol.img_dim)
        _r = torch.randn
        torch.randn = lambda *a, **k: noise.clone()
        ref = pol.rdt.predict_action(lang_tokens=lang, lang_attn_mask=torch.ones(lang.shape[:2], dtype=torch.bool,
                                     device=pol.device), img_tokens=e6, state_tokens=pol.state_token(q),
                                     action_mask=pol.elem_mask, ctrl_freqs=pol.ctrl_freqs)
        torch.randn = _r
        e = pol.encode([ip, ic])
        cond = pol.conditions("PickCube-v1", e[0:1], e[1:2], q)
        ours = pol.sample(cond, noise=noise)
    err = (pol.unnormalize(ref) - pol.unnormalize(ours)).abs().max().item()
    print(f"cached sampler vs predict_action: {err:.2e} rad ->", "OK" if err < 5e-3 else "MISMATCH", flush=True)
    env = make_env("PickCube-v1")
    env.reset(seed=20241201)
    frame = _render(env)
    env.close()
    frame.save(os.path.join(out, "verify_render.png"))
    print("render", frame.size, "mean pixel", float(np.asarray(frame).mean()), flush=True)
    ex = Explainer(pol, batch=cfg["batch"])
    t = {}
    sync(); t0 = time.time(); pol.act("PickCube-v1", ip, ic, q); sync(); t["policy call"] = time.time() - t0
    for m in ("eig", "occlusion", "gae"):
        sync(); t0 = time.time(); ex.explain(cond, m, n_evals=cfg["n_evals"], seed=1); sync(); t[m] = time.time() - t0
    fm = Faithfulness.fast(pol, fill=cfg["b_fill"]) if cfg["fast"] else Faithfulness(pol, fill=cfg["b_fill"])
    maps = {m: ex.explain(cond, m, n_evals=4, seed=1)["map"] for m in cfg["methods"]}
    sync(); t0 = time.time(); fm.evaluate(cond, ip, ic, maps); sync(); t["faithfulness"] = time.time() - t0
    per_moment = 4 * t["eig"] + t["occlusion"] + 3 * t["gae"] + t["faithfulness"]
    calls = 12
    proj = {"A": len(cfg["tasks"]) * cfg["n_episodes"] * (calls * t["policy call"] + 8),
            "B": cfg["b_moments"] * per_moment + cfg["b_stability_moments"] * 4 * t["eig"],
            "C": cfg["c_moments"] * (2.2 * t["eig"] + 12 * t["policy call"]),
            "D": len(cfg["d_tasks"]) * len(cfg["d_conditions"]) * cfg["d_episodes"]
            * (calls * (t["policy call"] + t["eig"] / 2) + 8),
            "E": cfg["e_moments"] * 4 * (2 * t["eig"] + t["gae"]),
            "F": cfg["f_moments"] * (8 * t["eig"] + 60 * t["policy call"])}
    print("timings (s):", {k: round(v, 2) for k, v in t.items()})
    print("projected minutes:", {k: round(v / 60, 1) for k, v in proj.items()},
          f"total {sum(proj.values())/3600:.1f} h", flush=True)
    json.dump(dict(sampler_err=err, timings=t, projection_min={k: v / 60 for k, v in proj.items()}),
              open(os.path.join(out, "verify.json"), "w"))
elif args.stage == "A":
    rows = X.rollouts(pol, cfg["tasks"], range(cfg["n_episodes"]), os.path.join(out, "A"))
    for task in cfg["tasks"]:
        r = [x for x in rows if x["task"] == task]
        print(f"SUCCESS {task}: {sum(x['success'] for x in r)}/{len(r)} = {np.mean([x['success'] for x in r]):.0%}")
elif args.stage == "B":
    B = moments()[:cfg["b_moments"]]
    print(f"B: {len(B)} moments", flush=True)
    S = cfg["b_stability_moments"]
    X.benchmark(pol, B[:S], cfg["methods"], os.path.join(out, "B"), n_evals=cfg["n_evals"], fast=cfg["fast"],
                stability=True, batch=cfg["batch"])
    X.benchmark(pol, B[S:], cfg["methods"], os.path.join(out, "B"), n_evals=cfg["n_evals"], fast=cfg["fast"],
                stability=False, batch=cfg["batch"])
    if cfg["b_fill"] != "gray":           # B/ keeps the pilot's grey-fill scores; re-score the same maps
        X.rescore(pol, os.path.join(out, "B"), os.path.join(out, "B_" + cfg["b_fill"]), os.path.join(out, "A"),
                  fill=cfg["b_fill"], fast=cfg["b_fill_fast"])
elif args.stage == "C":
    X.denoise(pol, moments()[:cfg["c_moments"]], os.path.join(out, "C"), n_evals=cfg["n_evals"], batch=cfg["batch"])
elif args.stage == "D":
    d_out = os.path.join(out, "D" if cfg["d_fill"] == "gray" else "D_" + cfg["d_fill"])
    X.closed_loop(pol, cfg["d_tasks"], range(cfg["d_episodes"]), [tuple(c) for c in cfg["d_conditions"]],
                  d_out, frac=cfg["d_frac"], n_evals=8, batch=cfg["batch"], fill=cfg["d_fill"])
elif args.stage == "E":
    X.sanity(pol, moments()[:cfg["e_moments"]], cfg["e_methods"], os.path.join(out, "E"), n_evals=8, batch=cfg["batch"])
elif args.stage == "F":
    X.language(pol, moments()[:cfg["f_moments"]], os.path.join(out, "F"), n_evals=cfg["n_evals"], batch=cfg["batch"])
    X.modality_timeline(pol, os.path.join(out, "A"), cfg["tasks"], cfg["f_timeline_episodes"],
                        os.path.join(out, "F_modality"))
elif args.stage == "G":
    X.grounding(pol, os.path.join(out, "B"), os.path.join(out, "A"), os.path.join(out, "G"), fill=cfg["b_fill"])
elif args.stage == "video":
    import imageio
    import matplotlib
    from rdt_xai.attribution import Explainer
    from rdt_xai.rollout import load_episode
    ex = Explainer(pol, batch=cfg["batch"])
    ep_dir = os.path.join(out, "A", args.episode) if args.episode else sorted({d for d, _ in moments()})[0]
    meta = load_episode(ep_dir)
    cmap = matplotlib.colormaps["inferno"]
    frames = []
    for k, dp in enumerate(meta["decisions"]):
        e = pol.encode([dp["prev"], dp["curr"]])
        c = pol.conditions(meta["task"], e[0:1], e[1:2], dp["qpos8"])
        a = np.clip(ex.explain(c, "eig", n_evals=8, seed=k)["map"], 0, None)
        a = a / (a.max() + 1e-12)
        up = np.pad(np.kron(a, np.ones((14, 14))), ((0, 6), (0, 6)))
        img = np.asarray(dp["curr"].resize((384, 384))) / 255.0
        al = (np.sqrt(up) * 0.8)[..., None]
        frames.append((255 * (img * (1 - al) + cmap(up)[..., :3] * al)).astype(np.uint8))
    os.makedirs(os.path.join(out, "report"), exist_ok=True)
    path = os.path.join(out, "report", f"video_{os.path.basename(ep_dir)}_eig.mp4")
    try:
        imageio.mimsave(path, frames, fps=2, macro_block_size=1)
    except Exception as err:                   # no ffmpeg plugin: fall back to a gif
        print("mp4 failed (", str(err).splitlines()[0], ") - writing gif")
        path = path[:-4] + ".gif"
        imageio.mimsave(path, frames, duration=500, loop=0)
    print("VIDEO", path)
print(f"[{args.stage}] done in {(time.time()-t_start)/60:.1f} min", flush=True)
