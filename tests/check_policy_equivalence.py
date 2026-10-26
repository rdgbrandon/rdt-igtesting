"""Check that RDTPolicy.sample (KV-cached, slot-cached) reproduces RDT's own
predict_action on the official 6-slot input with identical initial noise.

    python tests/check_policy_equivalence.py [--task PickCube-v1]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rdt_xai.policy import RDTPolicy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="PickCube-v1")
ap.add_argument("--threads", type=int, default=0)
args = ap.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

t0 = time.time()
pol = RDTPolicy()
print(f"loaded in {time.time()-t0:.1f}s  device={pol.device} dtype={pol.dtype} threads={torch.get_num_threads()}")

rng = np.random.default_rng(0)
from PIL import Image  # noqa: E402
img_prev = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
img_curr = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
qpos = np.array([0.0, 0.4, 0.0, -2.0, 0.0, 2.4, 0.8, 0.04], dtype=np.float32)

# reference: the official path (6 slots encoded, predict_action, global randn)
with torch.no_grad():
    embs6 = pol.encode([img_prev, None, None, img_curr, None, None]).reshape(1, -1, pol.img_dim)
    lang = pol.lang(args.task)
    noise = pol.noise(123)
    _randn = torch.randn
    torch.randn = lambda *a, **k: noise.clone()
    try:
        t0 = time.time()
        ref = pol.rdt.predict_action(
            lang_tokens=lang, lang_attn_mask=torch.ones(lang.shape[:2], dtype=torch.bool, device=pol.device),
            img_tokens=embs6, state_tokens=pol.state_token(qpos), action_mask=pol.elem_mask,
            ctrl_freqs=pol.ctrl_freqs)
        t_ref = time.time() - t0
    finally:
        torch.randn = _randn

    t0 = time.time()
    embs = pol.encode([img_prev, img_curr])
    cond = pol.conditions(args.task, embs[0:1], embs[1:2], qpos)
    ours = pol.sample(cond, noise=noise)
    t_ours = time.time() - t0

a_ref, a_ours = pol.unnormalize(ref), pol.unnormalize(ours)
err = (a_ref - a_ours).abs().max().item()
print(f"reference predict_action: {t_ref:.1f}s (excl. SigLIP)   cached sample: {t_ours:.1f}s (incl. 2x SigLIP)")
print(f"max |diff| in joint units: {err:.2e}   action range: {a_ref.abs().max().item():.3f}")
assert err < 1e-3, "cached sampler diverges from RDT reference"
print("OK")
