"""Real-weights check: RDT's official RoboticDiffusionTransformerModel.step()
(scripts/maniskill_model.py) vs this repo's pipeline on identical inputs and
identical initial noise. Shares one copy of the weights, so it fits in ~10 GB RAM.

    python tests/check_official_step.py [--device cpu] [--task StackCube-v1]

Covers everything between raw inputs and joint targets: image padding/preprocessing,
SigLIP, the 6-slot layout, state normalization, language mask, sampling, action
un-normalization. Prints the max joint-target difference (radians).
"""
import argparse
import os
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rdt_xai.policy import RDT_REPO, RDTPolicy, default_paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--task", default="StackCube-v1")
ap.add_argument("--threads", type=int, default=0)
args = ap.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

paths = default_paths()
cwd = os.getcwd()
os.chdir(RDT_REPO)
import scripts.maniskill_model as mm  # noqa: E402

mm.RoboticDiffusionTransformerModel.get_text_encoder = lambda self, p: (None, torch.nn.Identity())
if args.device == "cpu":   # the checkpoint was saved from GPU; the official loader has no map_location
    _load = torch.load
    mm.torch.load = lambda f, *a, **k: _load(f, *a, **dict(k, map_location="cpu"))
with open("configs/base.yaml") as fp:
    config = yaml.safe_load(fp)
t0 = time.time()
official = mm.create_model(args=config, dtype=torch.float32, pretrained=paths["ckpt"], device=args.device,
                           pretrained_text_encoder_name_or_path=None,
                           pretrained_vision_encoder_name_or_path=paths["siglip"])
os.chdir(cwd)
print(f"official model loaded {time.time()-t0:.0f}s", flush=True)

# this repo's pipeline on the very same modules (no second copy of the weights)
pol = RDTPolicy(model=official.policy, vision=official.vision_model.vision_tower,
                processor=official.image_processor, device=args.device, dtype=torch.float32)

# realistic-size inputs: two different 512x512 frames (the render size), first finger + 7 joints
rng = np.random.default_rng(0)
base = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
prev = Image.fromarray(base)
curr = Image.fromarray(np.clip(base.astype(int) + rng.integers(-20, 20, base.shape), 0, 255).astype(np.uint8))
qpos = torch.tensor([[0.02, 0.35, -0.03, -2.1, 0.01, 2.45, 0.78, 0.039]])          # (1, 8) like obs qpos[:, :-1]
text_embed = torch.load(os.path.join(paths["lang_dir"], f"text_embed_{args.task}.pt"))
noise = pol.noise(1234)

_randn = torch.randn
torch.randn = lambda *a, **k: noise.clone()
try:
    t0 = time.time()
    a_off = official.step(qpos, [prev, None, None, curr, None, None], text_embed)      # (1, 64, 8) joint units
    t_off = time.time() - t0
finally:
    torch.randn = _randn

t0 = time.time()
with torch.no_grad():
    embs = pol.encode([prev, curr])
    cond = pol.conditions(args.task, embs[0:1], embs[1:2], qpos.numpy().reshape(-1))
    a_ours = pol.unnormalize(pol.sample(cond, noise=noise)).float()
t_ours = time.time() - t0

diff = (a_off.float().cpu() - a_ours.cpu()).abs()
print(f"official step {t_off:.0f}s   ours {t_ours:.0f}s")
print(f"max |official - ours| = {diff.max().item():.2e} rad   (executed rows: {diff[:, ::4].max().item():.2e})")
print("per joint max:", np.round(diff.amax(dim=(0, 1)).numpy(), 6))
print("OK" if diff.max().item() < 1e-3 else "MISMATCH")
