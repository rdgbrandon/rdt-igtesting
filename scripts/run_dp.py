"""Run the attribution study's stages on a trained Diffusion Policy checkpoint.

  python scripts/run_dp.py --task PickCube-v1 --ckpt <dp/PickCube-v1/final.pt> --out <dir> --stages A,B,C,D,E,F
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rdt_xai import dp as D  # noqa: E402
from rdt_xai import dp_stages as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stages", default="A,B,C,D,E,F")
ap.add_argument("--a_episodes", type=int, default=20)
ap.add_argument("--b_moments", type=int, default=40)
ap.add_argument("--b_fast", action="store_true")
ap.add_argument("--c_moments", type=int, default=30)
ap.add_argument("--d_episodes", type=int, default=16)
ap.add_argument("--d_frac", type=float, default=0.05)
ap.add_argument("--d_conditions", default="none:none,random:random,eig:top_abs,eig:bottom_abs,eig:top,"
                                          "gradcam:top,blurig:top")
ap.add_argument("--e_moments", type=int, default=10)
ap.add_argument("--f_episodes", type=int, default=8)
ap.add_argument("--n_evals", type=int, default=16)
ap.add_argument("--methods", default=",".join(D.METHODS))
ap.add_argument("--fill", default="inpaint")
a = ap.parse_args()

out = os.path.join(a.out, a.task)
stages = a.stages.split(",")
pol = D.DPPolicy(D.load_agent(a.ckpt))
A_dir = os.path.join(out, "A")
if "A" in stages:
    S.rollouts(pol, a.task, range(a.a_episodes), A_dir)
pol.set_background(*S.background(A_dir, a.task))
methods = a.methods.split(",")
moms = S.moments(A_dir, a.task)
spread = lambda n: [moms[i] for i in sorted({int(round(x)) for x in
                                              (list(range(len(moms))) if n >= len(moms) else
                                               [k * (len(moms) - 1) / max(1, n - 1) for k in range(n)])})]
if "B" in stages:
    S.benchmark(pol, a.task, spread(a.b_moments), methods, os.path.join(out, "B"), n_evals=a.n_evals,
                fast=a.b_fast, fill=a.fill)
if "C" in stages:
    S.denoise(pol, spread(a.c_moments), os.path.join(out, "C"), n_evals=a.n_evals)
if "D" in stages:
    S.closed_loop(pol, a.task, range(a.d_episodes), a.d_conditions.split(","), os.path.join(out, "D"),
                  frac=a.d_frac, fill=a.fill)
if "E" in stages:
    S.sanity(pol, spread(a.e_moments), [m for m in methods if m != "random"] + ["random"], os.path.join(out, "E"))
if "F" in stages:
    S.modality(pol, A_dir, a.task, a.f_episodes, os.path.join(out, "F"))
txt = S.summary(out, a.task)
open(os.path.join(out, "summary.txt"), "w").write(txt)
print(txt, flush=True)
