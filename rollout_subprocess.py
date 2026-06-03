#!/usr/bin/env python3
"""
Standalone rollout worker for PickCube-v1.
Run in a fresh subprocess to avoid Issue #116 state accumulation.
Usage: python rollout_subprocess.py [--task TASK] [--n N] [--base-seed SEED]
"""
import os, sys, argparse

os.environ.setdefault('DISPLAY', '')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ['RDT_REPO']   = '/content/RoboticsDiffusionTransformer'
os.environ['SKIP_IG']    = '1'

sys.path.insert(0, '/content/RoboticsDiffusionTransformer')
os.chdir('/content/rdt-igtesting')

parser = argparse.ArgumentParser()
parser.add_argument('--task',      default='PickCube-v1')
parser.add_argument('--n',         type=int, default=25)
parser.add_argument('--base-seed', type=int, default=20241201)
args = parser.parse_args()

os.environ['MANISKILL_TASK'] = args.task
os.environ['LANG_EMBED']     = f'lang_embed_{args.task}.pt'

# Check lang embed exists (cell-2 downloads it as lang_embed.pt for PickCube)
if not os.path.exists(os.environ['LANG_EMBED']):
    os.environ['LANG_EMBED'] = 'lang_embed.pt'

print(f'WORKER: task={args.task}  n={args.n}  base_seed={args.base_seed}', flush=True)
print('WORKER: Loading models (from HF cache) ...', flush=True)

import runpy
_g = runpy.run_path('rdt_blurig.py', run_name='__main__')

print('WORKER: Models loaded.', flush=True)

import torch, numpy as np
from PIL import Image as PILImage
from collections import deque
import gymnasium as gym
import mani_skill.envs

rdt               = _g['rdt']
siglip            = _g['siglip']
siglip_proc       = _g['siglip_proc']
lang_tokens       = _g['lang_tokens']
lang_attn_mask    = _g['lang_attn_mask']
MANISKILL_INDICES = _g['MANISKILL_INDICES']
STATE_MIN         = _g['STATE_MIN']
STATE_MAX         = _g['STATE_MAX']
ACTION_MIN        = _g['ACTION_MIN']
ACTION_MAX        = _g['ACTION_MAX']
DEVICE            = _g['DEVICE']
DTYPE             = _g['DTYPE']
N_DDPM_STEPS      = _g['N_DDPM_STEPS']
embed_dim         = _g['embed_dim']

_bg_color = tuple(int(x * 255) for x in siglip_proc.image_mean)
_bg_img   = PILImage.new('RGB', (384, 384), _bg_color)
_use_pa   = hasattr(rdt, 'predict_action')
print(f'WORKER: bg={_bg_color}  predict_action={_use_pa}  DDPM_steps={N_DDPM_STEPS}', flush=True)

STATE_MIN_t  = STATE_MIN.to(DEVICE, dtype=DTYPE)
STATE_MAX_t  = STATE_MAX.to(DEVICE, dtype=DTYPE)
ACTION_MIN_t = ACTION_MIN.to(DEVICE, dtype=DTYPE)
ACTION_MAX_t = ACTION_MAX.to(DEVICE, dtype=DTYPE)


def _encode_6(pil_list):
    imgs = [(_bg_img if img is None else img) for img in pil_list]
    pvs  = siglip_proc(images=imgs, return_tensors='pt')['pixel_values'].to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        embs = siglip(pixel_values=pvs).last_hidden_state
    return embs.reshape(1, -1, embed_dim)


def _render_pil(env):
    r = env.render()
    if hasattr(r, 'cpu'): r = r.cpu().numpy()
    return PILImage.fromarray(np.array(r).squeeze().astype(np.uint8))


def _make_env(max_ep=1000):
    _e = gym.make(args.task, obs_mode='rgb', render_mode='rgb_array',
                  control_mode='pd_joint_pos')
    _w = _e
    while _w is not None:
        if hasattr(_w, '_max_episode_steps'):
            _w._max_episode_steps = max_ep; break
        _w = getattr(_w, 'env', None)
    return _e


def rollout(ep_idx):
    _env = _make_env()
    _obs, _ = _env.reset(seed=ep_idx + args.base_seed)
    _hist = deque([None, _render_pil(_env)], maxlen=2)
    chunk, cp, done, step, info = None, 16, False, 0, {}

    while not done and step < 1000:
        if cp >= 16:
            _raw = _encode_6([_hist[0], None, None, _hist[1], None, None])
            _qp  = _obs['agent']['qpos']
            if hasattr(_qp, 'cpu'): _qp = _qp.cpu()
            _j8  = torch.tensor(np.array(_qp).flatten()[:8], dtype=DTYPE, device=DEVICE).unsqueeze(0)
            _jn  = (_j8 - STATE_MIN_t) / (STATE_MAX_t - STATE_MIN_t).clamp(min=1e-6) * 2 - 1
            _st  = torch.zeros(1, 1, 128, dtype=DTYPE, device=DEVICE)
            _st[0, 0, MANISKILL_INDICES] = _jn[0]
            _mk  = torch.zeros(1, 128, dtype=DTYPE, device=DEVICE)
            _mk[0, MANISKILL_INDICES] = 1.0
            _cf  = torch.tensor([25.0], device=DEVICE, dtype=DTYPE)
            with torch.no_grad():
                if _use_pa:
                    _tr = rdt.predict_action(
                        lang_tokens=lang_tokens, lang_attn_mask=lang_attn_mask,
                        img_tokens=_raw, state_tokens=_st,
                        action_mask=_mk.unsqueeze(1), ctrl_freqs=_cf)
                else:
                    _ic = rdt.img_adaptor(_raw)
                    _sc = rdt.state_adaptor(torch.cat([_st, _mk.unsqueeze(1)], dim=2))
                    _tr = rdt.conditional_sample(
                        _g['lang_cond'], lang_attn_mask, _ic, _sc, _mk.unsqueeze(1), _cf)
            _acts = (_tr[0, :, MANISKILL_INDICES] + 1) / 2 * (ACTION_MAX_t - ACTION_MIN_t) + ACTION_MIN_t
            chunk = _acts[::4].cpu().float().numpy()
            cp = 0

        _obs, _, term, trunc, info = _env.step(chunk[cp].reshape(1, 8))
        _hist.append(_render_pil(_env))
        done = bool(term) or bool(trunc)
        cp += 1; step += 1

    _env.close()
    return bool(info.get('success', False)), step


print(f"\n{'ep':>4}  {'seed':>10}  {'result':>10}  {'steps':>6}", flush=True)
results = []
for ep in range(args.n):
    s, t = rollout(ep)
    results.append(s)
    print(f"  {ep:2d}  {ep+args.base_seed:10d}    {'SUCCESS' if s else 'fail   '} ({t:4d})", flush=True)

n = sum(results)
print(f"\nSuccess rate: {n}/{args.n}  ({100*n/args.n:.0f}%)", flush=True)
