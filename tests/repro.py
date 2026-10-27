"""Reproduction check: RDT success rate on a ManiSkill task with either RDT's
official evaluation code or this repo's pipeline, in whatever environment runs it.

    python tests/repro.py --impl official --task PickCube-v1 -n 20
    python tests/repro.py --impl ours --dtype fp32 --obs state_dict -n 20

Prints one JSON line: {"impl", "task", "n", "success_rate", "versions", ...}.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--impl", choices=["official", "ours"], required=True)
ap.add_argument("--task", default="PickCube-v1")
ap.add_argument("-n", type=int, default=20)
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--obs", default="rgb", choices=["rgb", "state_dict"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--progress", default="", help="jsonl of finished episodes: appended live, skipped on restart")
args = ap.parse_args()
if args.dtype == "bf16":
    os.environ["RDT_DTYPE"] = "bf16"
# the official code calls torch.load() on a DeepSpeed checkpoint; torch>=2.6 defaults to weights_only=True
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch  # noqa: E402
from rdt_xai.policy import RDT_REPO, TASK2LANG, default_paths  # noqa: E402  (sets up sys.path / hub fallback)


def versions():
    import importlib.metadata as md
    out = {"python": sys.version.split()[0]}
    for p in ("torch", "transformers", "diffusers", "timm", "mani_skill", "sapien", "numpy"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


t0 = time.time()
paths = default_paths()
done_eps = {}
if args.progress and os.path.exists(args.progress):
    for line in open(args.progress):
        if line.strip():
            r = json.loads(line)
            done_eps[r["ep"]] = r


def record(ep, success, n_steps, resumed=False):
    done_eps[ep] = dict(ep=ep, success=bool(success), steps=int(n_steps))
    if args.progress and not resumed:
        with open(args.progress, "a") as f:
            f.write(json.dumps(done_eps[ep]) + "\n")
    print(f"  ep {ep:3d} success={bool(success)} steps={n_steps}" + ("  (saved earlier)" if resumed else ""),
          file=sys.stderr, flush=True)


for ep in sorted(done_eps):
    if ep < args.n:
        record(ep, done_eps[ep]["success"], done_eps[ep]["steps"], resumed=True)
todo = [ep for ep in range(args.n) if ep not in done_eps]

if args.impl == "official":
    # eval_sim/eval_rdt_maniskill.py, verbatim except: T5-XXL is not loaded (the released
    # instruction embeddings are used, exactly as that script does when text_embed_*.pt exists),
    # SigLIP and the checkpoint come from local paths, and only `-n` episodes are run.
    import yaml
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from collections import deque
    from PIL import Image
    os.chdir(RDT_REPO)
    import scripts.maniskill_model as mm

    mm.RoboticDiffusionTransformerModel.get_text_encoder = lambda self, p: (None, torch.nn.Identity())
    seed = args.seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    env = gym.make(args.task, obs_mode="rgb", control_mode="pd_joint_pos", render_mode="rgb_array",
                   reward_mode="dense", sensor_configs=dict(shader_pack="default"),
                   human_render_camera_configs=dict(shader_pack="default"),
                   viewer_camera_configs=dict(shader_pack="default"), sim_backend="auto")
    with open("configs/base.yaml") as fp:
        config = yaml.safe_load(fp)
    policy = mm.create_model(args=config, dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
                             pretrained=paths["ckpt"], pretrained_text_encoder_name_or_path=None,
                             pretrained_vision_encoder_name_or_path=paths["siglip"])
    text_embed = torch.load(os.path.join(paths["lang_dir"], f"text_embed_{args.task}.pt"))
    MAX_EPISODE_STEPS = 400
    base_seed = 20241201
    # note: after a resume the official script's global RNG stream (diffusion noise) differs from an
    # uninterrupted run; each episode is still a valid independent evaluation on the same seed.
    for episode in todo:
        obs_window = deque(maxlen=2)
        obs, _ = env.reset(seed=episode + base_seed)
        policy.reset()
        img = env.render().squeeze(0).detach().cpu().numpy()
        obs_window.append(None)
        obs_window.append(np.array(img))
        proprio = obs["agent"]["qpos"][:, :-1]
        global_steps, done, info = 0, False, {}
        while global_steps < MAX_EPISODE_STEPS and not done:
            image_arrs = []
            for window_img in obs_window:
                image_arrs.append(window_img)
                image_arrs.append(None)
                image_arrs.append(None)
            images = [Image.fromarray(arr) if arr is not None else None for arr in image_arrs]
            actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
            actions = actions[::4, :]
            for idx in range(actions.shape[0]):
                obs, reward, terminated, truncated, info = env.step(actions[idx])
                img = env.render().squeeze(0).detach().cpu().numpy()
                obs_window.append(img)
                proprio = obs["agent"]["qpos"][:, :-1]
                global_steps += 1
                if terminated or truncated:
                    if info["success"]:
                        done = True
                        break
        s = info.get("success", False)
        record(episode, s.item() if hasattr(s, "item") else s, global_steps)
else:
    from rdt_xai.policy import RDTPolicy
    from rdt_xai.rollout import make_env, rollout
    pol = RDTPolicy()
    env = make_env(args.task, obs_mode=args.obs)
    for ep in todo:
        r = rollout(pol, args.task, ep, env=env)
        record(ep, r["success"], r["steps"])

res = [done_eps[e] for e in range(args.n)]
succ, steps = [r["success"] for r in res], [r["steps"] for r in res]
print("RESULT_JSON " + json.dumps(dict(impl=args.impl, task=args.task, dtype=args.dtype, obs=args.obs if args.impl == "ours" else "rgb",
                      n=len(succ), success_rate=float(np.mean(succ)), mean_steps=float(np.mean(steps)),
                      minutes=(time.time() - t0) / 60, versions=versions())))
