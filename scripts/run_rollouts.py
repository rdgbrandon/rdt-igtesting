#!/usr/bin/env python3
"""Closed-loop RDT rollouts on ManiSkill3 (official protocol), recording every
decision point for later attribution.

    python scripts/run_rollouts.py --task PickCube-v1 --episodes 0:25 --out runs/rollouts
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="PickCube-v1")
ap.add_argument("--episodes", default="0:1", help="python-style range a:b")
ap.add_argument("--out", default="runs/rollouts")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--render-backend", default=None)
ap.add_argument("--max-steps", type=int, default=400)
args = ap.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

from rdt_xai.policy import RDTPolicy  # noqa: E402
from rdt_xai.rollout import make_env, rollout  # noqa: E402

a, b = (int(x) for x in args.episodes.split(":"))
os.makedirs(args.out, exist_ok=True)
pol = RDTPolicy()
env = make_env(args.task, args.render_backend)
summary = os.path.join(args.out, f"summary_{args.task}.jsonl")
for ep in range(a, b):
    d = os.path.join(args.out, f"{args.task}_ep{ep:03d}", "episode.json")
    if os.path.exists(d):
        print("skip (done)", d)
        continue
    r = rollout(pol, args.task, ep, env=env, save_dir=args.out, max_steps=args.max_steps, verbose=True)
    row = {k: r[k] for k in ("task", "episode", "seed", "success", "steps", "n_calls", "wall")}
    print(json.dumps(row), flush=True)
    with open(summary, "a") as f:
        f.write(json.dumps(row) + "\n")
env.close()
