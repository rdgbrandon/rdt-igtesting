#!/usr/bin/env python3
"""
Standalone rollout worker for PickCube-v1.
Run in a fresh subprocess to avoid Issue #116 state accumulation.
Usage: python rollout_subprocess.py [--task TASK] [--n N] [--base-seed SEED]
"""
import os, sys, subprocess, argparse

# Self-healing: mani_skill can silently lose its utils subpackage between Colab sessions.
# Root cause (confirmed): mani_skill.utils.common does `import sapien.physx`, and sapien
# (mani_skill's physics/rendering engine dependency) can go missing independently of
# mani_skill itself. Reinstalling ONLY mani-skill with --no-deps never fixes that —
# sapien must be reinstalled too. Detect here (fresh subprocess), reinstall both, and
# retry the import so the rest of the script sees a clean installation.
try:
    import mani_skill.utils
except (ImportError, ModuleNotFoundError):
    print('WORKER: mani_skill.utils missing — diagnosing...', flush=True)
    for pkg in ['sapien', 'mani-skill']:
        print(f'WORKER: reinstalling {pkg}...', flush=True)
        # --no-cache-dir: force a fresh download instead of reusing a possibly-corrupted
        # local pip cache. --no-deps on each avoids dragging in the full dependency tree
        # (torch etc.), which is what made a plain --force-reinstall mani-skill slow.
        # timeout=180: avoid hanging forever if PyPI is unreachable from this VM.
        try:
            r = subprocess.run([sys.executable, '-m', 'pip', 'install',
                                '--force-reinstall', '--no-deps', '--no-cache-dir', pkg],
                               capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f'WORKER: {pkg} install timed out after 180s — likely a network issue '
                'reaching PyPI from this Colab VM. Try a fresh runtime.'
            )
        if r.returncode != 0:
            print(f'WORKER: {pkg} reinstall FAILED:', flush=True)
            print(r.stdout[-3000:], flush=True)
            print(r.stderr[-3000:], flush=True)
            raise RuntimeError(f'{pkg} reinstall failed — see pip output above')
        print(f'WORKER: {pkg} reinstalled OK.', flush=True)

    # Refresh import state in this process instead of re-exec'ing — simpler and
    # avoids any chance of the process restart being mishandled by the parent.
    for _m in list(sys.modules):
        if _m == 'mani_skill' or _m.startswith('mani_skill.') \
           or _m == 'sapien' or _m.startswith('sapien.'):
            del sys.modules[_m]
    import importlib
    importlib.invalidate_caches()
    import mani_skill.utils
    print('WORKER: mani_skill fixed in-place — continuing...', flush=True)

os.environ.setdefault('DISPLAY', '')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ['RDT_REPO']   = '/content/RoboticsDiffusionTransformer'
os.environ['SKIP_IG']    = '1'

sys.path.insert(0, '/content/RoboticsDiffusionTransformer')
os.chdir('/content/rdt-igtesting')

parser = argparse.ArgumentParser()
parser.add_argument('--task',         default='PickCube-v1')
parser.add_argument('--n',            type=int, default=1,   help='number of successes to collect')
parser.add_argument('--max-attempts', type=int, default=500, help='episode cap to avoid infinite loop')
parser.add_argument('--base-seed',    type=int, default=20241201)
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


def _expand2square(img, bg):
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = PILImage.new(img.mode, (side, side), bg)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out

def _encode_6(pil_list):
    bg = tuple(int(x * 255) for x in siglip_proc.image_mean)
    imgs = [_expand2square(_bg_img if img is None else img, bg) for img in pil_list]
    pvs  = siglip_proc(images=imgs, return_tensors='pt')['pixel_values'].to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        embs = siglip(pixel_values=pvs).last_hidden_state
    return embs.reshape(1, -1, embed_dim)


def _render_pil(env):
    r = env.render()
    if hasattr(r, 'cpu'): r = r.cpu().numpy()
    img = PILImage.fromarray(np.array(r).squeeze().astype(np.uint8))
    if img.width != img.height:
        print(f'WORKER: non-square render {img.width}x{img.height}', flush=True)
    return img


def _make_env(max_ep=400):
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
    first_frame = _render_pil(_env)
    _hist  = deque([None, first_frame], maxlen=2)
    frames = [first_frame]  # collect all frames
    chunk, cp, done, step, info = None, 16, False, 0, {}

    while not done and step < 400:
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
        f = _render_pil(_env)
        _hist.append(f)
        frames.append(f)
        done = bool(term) or bool(trunc)
        cp += 1; step += 1

    _env.close()
    success = bool(info.get('success', False))
    if success:
        # Sample 10 evenly spaced frames (including first and last)
        n_total = len(frames)
        indices = [int(round(i)) for i in np.linspace(0, n_total - 1, 10)]
        frame_paths = []
        for k, idx in enumerate(indices):
            path = f'success_ep{ep_idx:02d}_seed{ep_idx+args.base_seed}_f{k:02d}.png'
            frames[idx].save(path)
            frame_paths.append({'step': idx, 'path': path})
        return True, step, frame_paths
    return False, step, []


import json
print(f"Target: {args.n} success(es)  |  max attempts: {args.max_attempts}", flush=True)
print(f"\n{'ep':>4}  {'seed':>10}  {'result':>10}  {'steps':>6}  {'found':>6}", flush=True)
results, success_meta = [], []
ep = 0
while len(success_meta) < args.n and ep < args.max_attempts:
    s, t, frame_paths = rollout(ep)
    results.append(s)
    if s:
        success_meta.append({'ep': ep, 'seed': ep + args.base_seed,
                             'steps': t, 'frames': frame_paths})
    print(f"  {ep:2d}  {ep+args.base_seed:10d}    {'SUCCESS' if s else 'fail   '} ({t:4d})"
          f"  {len(success_meta)}/{args.n}", flush=True)
    ep += 1

n_found = len(success_meta)
print(f"\nSuccess rate: {n_found}/{ep}  ({100*n_found/max(ep,1):.0f}%)", flush=True)
with open('success_frames.json', 'w') as f:
    json.dump({'task': args.task, 'base_seed': args.base_seed, 'successes': success_meta}, f)
print(f"Saved {n_found} success episodes (10 frames each) + success_frames.json", flush=True)
