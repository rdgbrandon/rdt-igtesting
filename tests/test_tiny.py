"""Fast CPU test of the whole pipeline on a tiny random RDT + SigLIP.

Checks numerics that must hold for any weights (cached sampler == RDT's own
predict_action; IG completeness; per-step split sums to the whole) and that
every method / metric / rollout path runs.

    python tests/test_tiny.py [--rollout]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rdt_xai.policy import RDTPolicy, RDTRunner, default_paths  # noqa: E402
from rdt_xai.attribution import Explainer, METHODS  # noqa: E402
from rdt_xai.metrics import Faithfulness  # noqa: E402
from tiny import tiny_policy  # noqa: E402


def check(name, ok, msg=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {msg}")
    if not ok:
        check.failed = True


check.failed = False
ap = argparse.ArgumentParser()
ap.add_argument("--rollout", action="store_true", help="also run a 32-step ManiSkill rollout")
args = ap.parse_args()

t0 = time.time()
pol = tiny_policy()
rng = np.random.default_rng(0)
img_p = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
img_c = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
q = np.array([0.0, 0.4, 0.0, -2.0, 0.0, 2.4, 0.8, 0.04], dtype=np.float32)
task = "StackCube-v1"
print(f"tiny policy ready {time.time()-t0:.1f}s")

print("sampler")
with torch.no_grad():
    embs6 = pol.encode([img_p, None, None, img_c, None, None]).reshape(1, -1, pol.img_dim)
    lang, noise = pol.lang(task), pol.noise(7)
    _r = torch.randn
    torch.randn = lambda *a, **k: noise.clone()
    ref = pol.rdt.predict_action(lang_tokens=lang, lang_attn_mask=torch.ones(lang.shape[:2], dtype=torch.bool),
                                 img_tokens=embs6, state_tokens=pol.state_token(q), action_mask=pol.elem_mask,
                                 ctrl_freqs=pol.ctrl_freqs)
    torch.randn = _r
    e = pol.encode([img_p, img_c])
    cond = pol.conditions(task, e[0:1], e[1:2], q)
    ours = pol.sample(cond, noise=noise)
    check("cached sample == predict_action", (ours - ref).abs().max() < 1e-4, f"{(ours-ref).abs().max():.1e}")
    allv = pol.sample(cond, noise=noise, vision_steps=range(5))
    check("vision_steps=all == default", (allv - ours).abs().max() < 1e-6)
    split = pol.sample(cond, noise=noise, step_embs=[(e[0:1], e[1:2])] * 5)
    check("step_embs identical == default", (split - ours).abs().max() < 1e-6)
    none = pol.sample(cond, noise=noise, vision_steps=[])
    check("vision off changes plan", (none - ours).abs().max() > 1e-4)
    b = pol.sample(cond, emb_curr=e[1:2].expand(3, -1, -1), emb_prev=e[0:1].expand(3, -1, -1),
                   noise=torch.cat([noise] * 3))
    check("batched == single", (b - ours).abs().max() < 1e-5)

print("methods")
ex = Explainer(pol, batch=8)
maps = {}
for m in METHODS:
    r = ex.explain(cond, m, n_evals=8, seed=3)
    maps[m] = r["map"]
    ok = r["map"].shape == (27, 27) and np.isfinite(r["map"]).all() and np.abs(r["map"]).sum() > 0
    extra = f"compl_err={r['completeness_rel_err']:.3f}" if "completeness_rel_err" in r else ""
    check(f"{m:10s} {r['seconds']:5.2f}s", ok, extra)

r = ex.explain(cond, "ig_bg", n_evals=64, seed=3)
check("IG completeness (64 steps)", r["completeness_rel_err"] < 0.05, f"rel_err={r['completeness_rel_err']:.4f}")
r = ex.explain(cond, "eig", n_evals=64, seed=3, n_seeds=4)
check("EIG completeness (64 evals, 4 seeds)", r["completeness_rel_err"] < 0.05, f"rel_err={r['completeness_rel_err']:.4f}")
rs = ex.explain(cond, "eig_steps", n_evals=8, seed=3, n_seeds=4)
re = ex.explain(cond, "eig", n_evals=8, seed=3, n_seeds=4)
err = np.abs(rs["steps"].sum(0) - re["map"]).max() / (np.abs(re["map"]).max() + 1e-12)
check("sum over steps of eig_steps == eig", err < 1e-3, f"rel={err:.1e}  steps shape={rs['steps'].shape}")
ko = ex.knockout(cond, seed=3)
check("knockout", ko["drop"].shape == (5,) and np.isfinite(ko["drop"]).all(), f"drop={np.round(ko['drop'],2)}")

print("language")
from rdt_xai.language import LanguageExplainer  # noqa: E402
lx = LanguageExplainer(pol, batch=8)
groups, labels = lx.words(task)
check("words cover all tokens", sum(len(g) for g in groups) == pol.lang(task).shape[1] and labels[-1] == "</s>",
      " ".join(labels))
with torch.no_grad():
    L = pol.lang(task).shape[1]
    keep = pol.sample(cond, noise=noise, lang_mask=torch.ones(1, L, dtype=torch.bool))
    m = torch.ones(1, L, dtype=torch.bool); m[0, groups[3]] = False
    drop1 = pol.sample(cond, noise=noise, lang_mask=m)
check("lang_mask all-true == default", (keep - ours).abs().max() < 1e-6)
check("masking a word changes plan", (drop1 - ours).abs().max() > 1e-6)
wd = lx.word_drop(cond, groups, seeds=(0,))
check("word_drop", wd.shape == (8, len(groups)) and np.isfinite(wd).all())
wi, rel = lx.word_ig(cond, groups, n_evals=32, seed=0)
check("word IG completeness per joint (32 steps)", wi.shape == (8, len(groups)) and rel.max() < 0.05,
      f"max rel_err={rel.max():.4f}")
mod = lx.modality(cond, seeds=(0,))
with torch.no_grad():
    cns = pol.conditions(task, e[0:1], e[1:2], q, null_state=True)
    a_all = pol.sample(cond, noise=pol.noise(0))[:, 0:64:4][:, :, pol.idx]
    a_none = pol.sample(cns, noise=pol.noise(0), emb_prev=pol.bg_emb, emb_curr=pol.bg_emb,
                        lang_kv=pol.lang_kv(pol.empty_lang()))[:, 0:64:4][:, :, pol.idx]
eff = float((a_all - a_none).norm())
check("modality Shapley efficiency", abs(sum(mod.values()) - eff) < 1e-3 * max(1, eff),
      f"sum={sum(mod.values()):.4f} vs {eff:.4f}  {({k: round(v, 3) for k, v in mod.items()})}")

print("faithfulness")
for fm, tag in ((Faithfulness.fast(pol), "fast"), (Faithfulness(pol), "full"), (Faithfulness.fast(pol, fill="gray"), "fast gray")):
    t1 = time.time()
    res = fm.evaluate(cond, img_p, img_c, {k: maps[k] for k in ("random", "eig", "grad")})
    ok = all(0 <= v["deletion_morf"] <= 3 and np.isfinite(v["insertion"]) for v in res.values())
    check(f"{tag} {time.time()-t1:.1f}s", ok, "  ".join(f"{k}:{v['deletion_morf']:.2f}/{v['insertion']:.2f}" for k, v in res.items()))

print("experiment stages (synthetic episode)")
import tempfile  # noqa: E402
from rdt_xai import experiments as X  # noqa: E402
from rdt_xai.closedloop import MaskHook  # noqa: E402
from rdt_xai.rollout import save_episode  # noqa: E402

tmp = tempfile.mkdtemp()
frames = [Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)) for _ in range(4)]
decs = [dict(step=16 * k, qpos8=q.tolist(), seed=1000 + k, actions=[], prev=frames[k], curr=frames[k + 1])
        for k in range(3)]
save_episode(dict(task=task, episode=0, seed=0, success=True, steps=48, n_calls=3, wall=0.0), decs, frames,
             os.path.join(tmp, "A"))
mom = X.select_moments(os.path.join(tmp, "A"), every=1, max_per_episode=2)
check("select_moments", len(mom) == 2, str([c for _, c in mom]))
t1 = time.time()
X.benchmark(pol, mom, ["random", "daam", "gae", "iss", "eig"], os.path.join(tmp, "B"), n_evals=8, fast=True,
            stability=True, verbose=False)
recs = [json.load(open(f)) for f in glob.glob(os.path.join(tmp, "B", "*.json"))]
check("benchmark", len(recs) == 2 and "seed_spearman" in recs[0]["methods"]["eig"], f"{time.time()-t1:.1f}s")
X.rescore(pol, os.path.join(tmp, "B"), os.path.join(tmp, "B_inpaint"), os.path.join(tmp, "A"), fast=True, verbose=False)
rs = [json.load(open(f)) for f in glob.glob(os.path.join(tmp, "B_inpaint", "*.json"))]
check("rescore", len(rs) == 2 and all(r["fill"] == "inpaint" and "seed_spearman" in r["methods"]["eig"] for r in rs)
      and recs[0]["fill"] == "gray", " ".join(f"{m}:{v['deletion_morf']:.2f}" for m, v in rs[0]["methods"].items()))
X.denoise(pol, mom[:1], os.path.join(tmp, "C"), n_evals=8, verbose=False)
check("denoise", len(glob.glob(os.path.join(tmp, "C", "*.json"))) == 1)
X.sanity(pol, mom[:1], ["eig", "daam"], os.path.join(tmp, "E"), n_evals=4, levels=(2, 4), verbose=False)
s = json.load(open(os.path.join(tmp, "E", "sanity.json")))
check("sanity", len(s) == 4, " ".join(f"{r['method']}@{r['randomized_blocks']}={r['spearman']:.2f}" for r in s))
after = pol.sample(cond, noise=noise)
check("weights restored after sanity", (after - ours).abs().max() < 1e-6)
for mode, fill in (("top", "inpaint"), ("bottom", "inpaint"), ("top_abs", "inpaint"), ("bottom_abs", "inpaint"),
                   ("random", "inpaint"), ("top", "gray")):
    hk = MaskHook(pol, ex, method="eig", mode=mode, frac=0.1, n_evals=4, fill=fill)
    mp, mc = hk(img_p, img_c, dict(call=0, step=0, qpos8=q, task=task))
    changed = (np.asarray(mc) != np.asarray(img_c)).any(-1)
    ok = 0.05 < changed.mean() < 0.15 and mp is not None
    if fill == "gray":
        ok = ok and (np.asarray(mc)[changed] == np.array(pol.bg_color)).all()
    check(f"mask hook {mode:10s} {fill:7s}", ok, f"masked pixel fraction={changed.mean():.3f}")

X.language(pol, mom[:1], os.path.join(tmp, "F"), n_evals=8, verbose=False)
check("language stage", len(glob.glob(os.path.join(tmp, "F", "*.json"))) == 1)
X.modality_timeline(pol, os.path.join(tmp, "A"), [task], 1, os.path.join(tmp, "F_modality"), verbose=False)
tl = [json.load(open(f)) for f in glob.glob(os.path.join(tmp, "F_modality", "*.json"))]
check("modality timeline", len(tl) == 1 and len(tl[0]["calls"]) == 3)
# closed-loop rows with 0/n and n/n successes: Wilson bounds must not cross the point estimate
os.makedirs(os.path.join(tmp, "D_inpaint"))
for cond, succ in ((("none", "none"), True), (("random", "random"), False), (("eig", "top_abs"), False),
                   (("eig", "bottom_abs"), True)):
    for ep in range(6):
        json.dump(dict(task=task, episode=ep, method=cond[0], mode=cond[1], frac=0.05, fill="inpaint",
                       success=succ, steps=100, wall=1.0),
                  open(os.path.join(tmp, "D_inpaint", f"{task}_{cond[0]}-{cond[1]}-5_ep{ep:03d}.json"), "w"))
from rdt_xai.report import build_report  # noqa: E402
txt = build_report(tmp)
figs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(tmp, "report", "*.png")))
check("report", "Faithfulness" in txt and "Modality" in txt and "fig_word_joint.png" in figs
      and "fig_closed_loop.png" in figs and "fig_faithfulness_curves_gray.png" in figs and "peak_morf" in txt
      and "inpaint fill" in txt and "every decision" in txt and "Low-signal moments" in txt
      and "deletion_morf_rank" in txt, " ".join(figs))

if args.rollout:
    print("rollout")
    from rdt_xai.rollout import rollout
    r = rollout(pol, "PickCube-v1", 0, max_steps=32, verbose=False)
    check("32-step rollout", r["steps"] == 32 and len(r["decisions"]) == 2, f"{r['wall']:.1f}s")

print("ALL PASS" if not check.failed else "SOME CHECKS FAILED")
sys.exit(1 if check.failed else 0)
