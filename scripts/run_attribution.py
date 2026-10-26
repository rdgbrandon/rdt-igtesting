#!/usr/bin/env python3
"""Explain recorded decision points with every method at matched compute and
score each explanation with pixel-space deletion/insertion.

    python scripts/run_attribution.py --episodes runs/rollouts/PickCube-v1_ep000 \
        --calls 0,3,6 --methods random,attn,grad,ig_bg,blurig,eig --n-evals 16 --out runs/attr
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", nargs="+", required=True, help="episode dirs or globs")
ap.add_argument("--calls", default="all", help="comma list of decision indices, 'all', or 'every:K'")
ap.add_argument("--methods", default="random,attn,grad,gxi,ig_zero,ig_bg,blurig,eig")
ap.add_argument("--n-evals", type=int, default=16)
ap.add_argument("--n-seeds", type=int, default=4, help="noise seeds for eig")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--joints", default="all", help="'all' or comma list of joint ids, e.g. 7 for gripper")
ap.add_argument("--no-faithfulness", action="store_true")
ap.add_argument("--stability", action="store_true", help="also recompute each map with a second seed")
ap.add_argument("--only-success", action="store_true")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--out", default="runs/attr")
args = ap.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

from rdt_xai.attribution import Explainer  # noqa: E402
from rdt_xai.metrics import Faithfulness, rank_agreement, topk_iou  # noqa: E402
from rdt_xai.policy import RDTPolicy  # noqa: E402
from rdt_xai.rollout import load_episode  # noqa: E402

pol = RDTPolicy()
joints = None if args.joints == "all" else [int(j) for j in args.joints.split(",")]
ex = Explainer(pol, joints=joints, batch=args.batch)
fm = None if args.no_faithfulness else Faithfulness(pol)
methods = args.methods.split(",")
os.makedirs(args.out, exist_ok=True)

ep_dirs = sorted({d for pat in args.episodes for d in glob.glob(pat)})
for ep_dir in ep_dirs:
    meta = load_episode(ep_dir)
    if args.only_success and not meta["success"]:
        continue
    n = len(meta["decisions"])
    if args.calls == "all":
        calls = list(range(n))
    elif args.calls.startswith("every:"):
        calls = list(range(0, n, int(args.calls.split(":")[1])))
    else:
        calls = [int(c) for c in args.calls.split(",") if int(c) < n]
    tag = os.path.basename(ep_dir.rstrip("/\\"))
    for c in calls:
        out_json = os.path.join(args.out, f"{tag}_call{c:03d}_j{args.joints.replace(',', '-')}.json")
        if os.path.exists(out_json):
            print("skip", out_json)
            continue
        dp = meta["decisions"][c]
        embs = pol.encode([dp["prev"], dp["curr"]])
        cond = pol.conditions(meta["task"], embs[0:1], embs[1:2], dp["qpos8"])
        maps, rec = {}, dict(episode=tag, task=meta["task"], success=meta["success"], call=c, step=dp["step"],
                             joints=args.joints, n_evals=args.n_evals, methods={})
        for m in methods:
            t0 = time.time()
            r = ex.explain(cond, m, n_evals=args.n_evals, seed=dp["seed"] % 100000, n_seeds=args.n_seeds)
            maps[m] = r["map"]
            rec["methods"][m] = {k: v for k, v in r.items() if k not in ("map", "prev", "curr")}
            if args.stability and m not in ("random",):
                r2 = ex.explain(cond, m, n_evals=args.n_evals, seed=dp["seed"] % 100000 + 50000, n_seeds=args.n_seeds)
                rec["methods"][m]["seed_spearman"] = rank_agreement(r["map"], r2["map"])
                rec["methods"][m]["seed_top10_iou"] = topk_iou(r["map"], r2["map"])
                maps[m + "@seed2"] = r2["map"]
            print(f"{tag} call {c:3d} {m:10s} {time.time()-t0:7.1f}s"
                  + (f"  compl_err={r['completeness_rel_err']:.3f}" if "completeness_rel_err" in r else "")
                  + (f"  seed_rho={rec['methods'][m].get('seed_spearman', float('nan')):.3f}" if args.stability else ""),
                  flush=True)
        if fm is not None:
            t0 = time.time()
            ff = fm.evaluate(cond, dp["prev"], dp["curr"], {m: maps[m] for m in methods})
            for m in methods:
                rec["methods"][m].update(ff[m])
            print(f"{tag} call {c:3d} faithfulness {time.time()-t0:.1f}s  "
                  + "  ".join(f"{m}:{ff[m]['deletion_morf']:.2f}/{ff[m]['insertion']:.2f}" for m in methods), flush=True)
        np.savez_compressed(out_json.replace(".json", ".npz"), **maps)
        with open(out_json, "w") as f:
            json.dump(rec, f)
