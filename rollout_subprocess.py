#!/usr/bin/env python3
"""
Standalone rollout worker.
Uses RoboticDiffusionTransformerModel directly (official inference pipeline).
Run in a fresh subprocess to avoid Issue #116 state accumulation.
Usage: python rollout_subprocess.py [--task TASK] [--n N] [--base-seed SEED] [--random-seed SEED]
"""
import os, sys, argparse, random

os.environ.setdefault('DISPLAY', '')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

sys.path.insert(0, '/content/RoboticsDiffusionTransformer')
os.chdir('/content/rdt-igtesting')

parser = argparse.ArgumentParser()
parser.add_argument('--task',        default='PickCube-v1')
parser.add_argument('--n',           type=int, default=25)
parser.add_argument('--base-seed',   type=int, default=20241201)
parser.add_argument('--random-seed', type=int, default=None)
args = parser.parse_args()

# ── Set random seeds exactly as official eval does ────────────────────────────
import time
seed = args.random_seed if args.random_seed is not None else int(time.time()) % 100000
args.random_seed = seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

import numpy as np
np.random.seed(seed)

import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f'WORKER: task={args.task}  n={args.n}  base_seed={args.base_seed}  random_seed={seed}', flush=True)

# ── Stubs: prevent wandb/deepspeed from loading ───────────────────────────────
import types, importlib.machinery

def _make_stub(name):
    class _S(types.ModuleType):
        def __getattr__(self, n):
            if n.startswith('__') and n.endswith('__'): raise AttributeError(n)
            c = _make_stub(f'{self.__name__}.{n}'); object.__setattr__(self, n, c)
            sys.modules[c.__name__] = c; return c
        def __call__(self, *a, **kw): return None
    m = _S(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    m.__path__ = []; m.__file__ = f'/tmp/_stub_{name.replace(".", "_")}.py'
    m.__package__ = name.split('.')[0]
    return m

for _pkg in ('wandb', 'deepspeed', 'flash_attn'):
    for _k in [k for k in sys.modules if k == _pkg or k.startswith(_pkg + '.')]:
        del sys.modules[_k]
    sys.modules[_pkg] = _make_stub(_pkg)

# ── Monkey-patch T5 loading so we can skip it (we use pre-computed embeddings) ─
import torch.nn as nn
import scripts.maniskill_model as _msm

class _NoopTextModel(nn.Module):
    """Drop-in for the T5 text encoder when using pre-computed embeddings."""
    def forward(self, *a, **kw): return None

def _skip_text_encoder(self, path):
    print('WORKER: Skipping T5 load (using pre-computed embeddings)', flush=True)
    return _NoopTextModel(), None
_msm.RoboticDiffusionTransformerModel.get_text_encoder = _skip_text_encoder

# Also patch reset() — when pretrained_text_encoder_name_or_path=None, __init__
# sets self.text_model=None directly without calling get_text_encoder, so reset()
# crashes on self.text_model.eval(). Swap in the noop before delegating.
_orig_reset = _msm.RoboticDiffusionTransformerModel.reset
def _patched_reset(self):
    if self.text_model is None:
        self.text_model = _NoopTextModel()
    _orig_reset(self)
_msm.RoboticDiffusionTransformerModel.reset = _patched_reset

# ── Load config and create model ──────────────────────────────────────────────
import yaml
from scripts.maniskill_model import RoboticDiffusionTransformerModel
from huggingface_hub import hf_hub_download

config_path = '/content/RoboticsDiffusionTransformer/configs/base.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

print('WORKER: Creating RoboticDiffusionTransformerModel ...', flush=True)
policy = RoboticDiffusionTransformerModel(
    args=config,
    dtype=torch.bfloat16,
    device='cuda',
    pretrained_vision_encoder_name_or_path='google/siglip-so400m-patch14-384',
    pretrained_text_encoder_name_or_path=None,
    control_frequency=25,
)

print('WORKER: Loading ManiSkill fine-tuned weights ...', flush=True)
_ckpt_path = hf_hub_download('robotics-diffusion-transformer/maniskill-model',
                              'rdt/mp_rank_00_model_states.pt')
_ckpt = torch.load(_ckpt_path, map_location='cpu', weights_only=False)
_sd   = _ckpt.get('module', _ckpt)
missing, unexpected = policy.policy.load_state_dict(_sd, strict=False)
print(f'WORKER: Weights loaded (missing={len(missing)}, unexpected={len(unexpected)})', flush=True)
policy.reset()

# ── Load pre-computed language embedding ──────────────────────────────────────
lang_embed_path = f'lang_embed_{args.task}.pt'
if not os.path.exists(lang_embed_path):
    lang_embed_path = 'lang_embed.pt'
text_embed = torch.load(lang_embed_path, map_location='cpu', weights_only=False)
if isinstance(text_embed, dict):
    text_embed = text_embed.get('embeddings', text_embed)
if text_embed.dim() == 2:
    text_embed = text_embed.unsqueeze(0)
print(f'WORKER: text_embed shape: {text_embed.shape}', flush=True)

# ── Env setup ─────────────────────────────────────────────────────────────────
import gymnasium as gym
import mani_skill.envs
from PIL import Image as PILImage

def _make_env(max_ep=400):
    _e = gym.make(args.task, obs_mode='rgb', render_mode='rgb_array',
                  control_mode='pd_joint_pos')
    _w = _e
    while _w is not None:
        if hasattr(_w, '_max_episode_steps'):
            _w._max_episode_steps = max_ep; break
        _w = getattr(_w, 'env', None)
    return _e

def _render_pil(env):
    r = env.render()
    if hasattr(r, 'cpu'): r = r.cpu().numpy()
    return PILImage.fromarray(np.array(r).squeeze().astype(np.uint8))

# ── Rollout (exact official eval loop) ───────────────────────────────────────
from collections import deque

print(f"\n{'ep':>4}  {'seed':>10}  {'result':>10}  {'steps':>6}", flush=True)
results = []

for ep in range(args.n):
    _env = _make_env()
    obs, _ = _env.reset(seed=ep + args.base_seed)
    policy.reset()  # clear internal action buffer between episodes

    obs_window = deque(maxlen=2)
    img = _render_pil(_env)
    obs_window.append(None)
    obs_window.append(img)
    proprio = obs['agent']['qpos'][:, :-1]  # (1, 8) tensor

    global_steps = 0
    done = False
    info = {}

    while global_steps < 400 and not done:
        images = []
        for window_img in obs_window:
            images.append(window_img)  # already PIL or None
            images.append(None)
            images.append(None)

        # Official: policy.step(proprio, images, text_embed)
        with torch.no_grad():
            actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
        actions = actions[::4, :]  # stride-4: 64 → 16

        for idx in range(actions.shape[0]):
            obs, _, terminated, truncated, info = _env.step(actions[idx])
            img = _render_pil(_env)
            obs_window.append(img)
            proprio = obs['agent']['qpos'][:, :-1]
            global_steps += 1
            if terminated or truncated:
                done = True
                break

    _env.close()
    s = bool(info.get('success', False))
    results.append(s)
    print(f'  {ep:2d}  {ep+args.base_seed:10d}    {"SUCCESS" if s else "fail   "} ({global_steps:4d})',
          flush=True)

n = sum(results)
print(f'\nSuccess rate: {n}/{args.n}  ({100*n/args.n:.0f}%)', flush=True)
