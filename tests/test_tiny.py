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
from rdt_xai.policy import RDTPolicy, RDTRunner, default_paths  # noqa: E402
from rdt_xai.attribution import Explainer, METHODS  # noqa: E402
from rdt_xai.metrics import Faithfulness  # noqa: E402
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel  # noqa: E402


def tiny_policy():
    torch.manual_seed(0)
    vcfg = SiglipVisionConfig(image_size=384, patch_size=14, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=2)
    vision = SiglipVisionModel(vcfg)
    proc = SiglipImageProcessor.from_pretrained(default_paths()["siglip"])
    cfg = dict(lang_adaptor="mlp2x_gelu", img_adaptor="mlp2x_gelu", state_adaptor="mlp3x_gelu",
               rdt=dict(hidden_size=64, depth=4, num_heads=4, cond_pos_embed_type="multimodal"),
               noise_scheduler=dict(type="ddpm", num_train_timesteps=1000, num_inference_timesteps=5,
                                    beta_schedule="squaredcos_cap_v2", prediction_type="sample", clip_sample=False))
    rdt = RDTRunner(action_dim=128, pred_horizon=64, config=cfg, lang_token_dim=4096, img_token_dim=32,
                    state_token_dim=128, max_lang_cond_len=1024, img_cond_len=6 * 729,
                    img_pos_embed_config=[("image", (2, 3, -729))], lang_pos_embed_config=[("lang", -1024)],
                    dtype=torch.float32)
    torch.nn.init.normal_(rdt.model.final_layer.ffn_final.fc2.weight, std=0.5)   # RDT zero-inits it
    return RDTPolicy(model=rdt, vision=vision, processor=proc, dtype=torch.float32)


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

print("faithfulness")
for fm, tag in ((Faithfulness.fast(pol), "fast"), (Faithfulness(pol), "full")):
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
X.denoise(pol, mom[:1], os.path.join(tmp, "C"), n_evals=8, verbose=False)
check("denoise", len(glob.glob(os.path.join(tmp, "C", "*.json"))) == 1)
X.sanity(pol, mom[:1], ["eig", "daam"], os.path.join(tmp, "E"), n_evals=4, levels=(2, 4), verbose=False)
s = json.load(open(os.path.join(tmp, "E", "sanity.json")))
check("sanity", len(s) == 4, " ".join(f"{r['method']}@{r['randomized_blocks']}={r['spearman']:.2f}" for r in s))
after = pol.sample(cond, noise=noise)
check("weights restored after sanity", (after - ours).abs().max() < 1e-6)
for mode in ("top", "bottom", "random"):
    hk = MaskHook(pol, ex, method="eig", mode=mode, frac=0.1, n_evals=4)
    mp, mc = hk(img_p, img_c, dict(call=0, step=0, qpos8=q, task=task))
    frac = (np.asarray(mc) != np.asarray(img_c)).any(-1).mean()
    check(f"mask hook {mode:6s}", 0.05 < frac < 0.15 and mp is not None, f"masked pixel fraction={frac:.3f}")

from rdt_xai.report import build_report  # noqa: E402
txt = build_report(tmp)
figs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(tmp, "report", "*.png")))
check("report", "Faithfulness" in txt and len(figs) >= 4, " ".join(figs))

if args.rollout:
    print("rollout")
    from rdt_xai.rollout import rollout
    r = rollout(pol, "PickCube-v1", 0, max_steps=32, verbose=False)
    check("32-step rollout", r["steps"] == 32 and len(r["decisions"]) == 2, f"{r['wall']:.1f}s")

print("ALL PASS" if not check.failed else "SOME CHECKS FAILED")
sys.exit(1 if check.failed else 0)
