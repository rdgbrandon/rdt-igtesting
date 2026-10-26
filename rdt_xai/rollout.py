"""ManiSkill3 closed-loop rollout matching eval_sim/eval_rdt_maniskill.py.

Differences from the official loop are bookkeeping only:
  * every policy call ("decision point") is recorded with the exact inputs the
    policy saw (prev frame, current frame, proprio) so attribution is computed
    on real inputs rather than re-synthesized ones;
  * an optional `obs_hook(prev_img, curr_img, ctx) -> (prev_img, curr_img)`
    lets closed-loop experiments intervene on what the policy sees.
"""
import json
import os
import time
from collections import deque

import numpy as np
from PIL import Image

BASE_SEED = 20241201
MAX_EPISODE_STEPS = 400
STRIDE = 4                      # official: actions[::4] -> 16 env steps per call


def make_env(task, render_backend=None, obs_mode=None):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers envs)
    # The official eval uses obs_mode="rgb" but only consumes env.render() and qpos;
    # "state_dict" (RDT_OBS_MODE) skips rendering the unused sensor cameras.
    obs_mode = obs_mode or os.environ.get("RDT_OBS_MODE", "rgb")
    kw = dict(obs_mode=obs_mode, control_mode="pd_joint_pos", render_mode="rgb_array",
              reward_mode="dense", sim_backend="cpu")
    if render_backend:
        kw["render_backend"] = render_backend
    try:
        return gym.make(task, **kw)
    except TypeError:
        kw.pop("render_backend", None)
        return gym.make(task, **kw)


def _render(env):
    r = env.render()
    if hasattr(r, "cpu"):
        r = r.cpu().numpy()
    return Image.fromarray(np.asarray(r).squeeze().astype(np.uint8))


def _qpos8(obs):
    q = obs["agent"]["qpos"]
    if hasattr(q, "cpu"):
        q = q.cpu().numpy()
    return np.asarray(q, dtype=np.float32).reshape(-1)[:-1]        # official: qpos[:, :-1]


def rollout(policy, task, episode, env=None, base_seed=BASE_SEED, noise_seed=None, obs_hook=None,
            save_dir=None, max_steps=MAX_EPISODE_STEPS, render_backend=None, verbose=False):
    """Run one episode. Returns a dict with success, steps, decision points.
    noise_seed: seed of the sampler noise for call k is noise_seed*1000+k
    (defaults to the episode seed) so runs are reproducible."""
    own_env = env is None
    env = env or make_env(task, render_backend)
    seed = base_seed + episode
    noise_seed = seed if noise_seed is None else noise_seed
    obs, _ = env.reset(seed=seed)
    window = deque([None, _render(env)], maxlen=2)
    q = _qpos8(obs)
    decisions, frames = [], [window[-1]]
    step, success, t0 = 0, False, time.time()
    while step < max_steps and not success:
        prev_img, curr_img = window[0], window[1]
        ctx = dict(step=step, call=len(decisions), qpos8=q, task=task)
        if obs_hook is not None:
            prev_img, curr_img = obs_hook(prev_img, curr_img, ctx)
        call_seed = noise_seed * 1000 + len(decisions)
        acts, _, _ = policy.act(task, prev_img, curr_img, q, seed=call_seed, stride=STRIDE)
        decisions.append(dict(step=step, qpos8=q.tolist(), seed=call_seed, actions=acts.tolist(),
                              prev=window[0], curr=window[1]))
        for a in acts:
            obs, _, term, trunc, info = env.step(a)
            window.append(_render(env))
            frames.append(window[-1])
            q = _qpos8(obs)
            step += 1
            s = info.get("success", False)
            success = bool(s.item() if hasattr(s, "item") else s)
            if success or step >= max_steps:
                break
        if verbose:
            print(f"  call {len(decisions):3d} step {step:4d}  success={success}  {time.time()-t0:6.1f}s", flush=True)
    if own_env:
        env.close()
    res = dict(task=task, episode=episode, seed=seed, success=success, steps=step,
               n_calls=len(decisions), wall=time.time() - t0)
    if save_dir:
        save_episode(res, decisions, frames, save_dir)
    res["decisions"], res["frames"] = decisions, frames
    return res


def save_episode(res, decisions, frames, save_dir):
    d = os.path.join(save_dir, f"{res['task']}_ep{res['episode']:03d}")
    os.makedirs(d, exist_ok=True)
    meta = dict(res, decisions=[])
    for k, dp in enumerate(decisions):
        rec = {kk: vv for kk, vv in dp.items() if kk not in ("prev", "curr")}
        for name in ("prev", "curr"):
            if dp[name] is None:
                rec[name] = None
            else:
                fn = f"call{k:03d}_{name}.png"
                dp[name].save(os.path.join(d, fn))
                rec[name] = fn
        meta["decisions"].append(rec)
    with open(os.path.join(d, "episode.json"), "w") as f:
        json.dump(meta, f)
    try:
        import imageio
        imageio.mimsave(os.path.join(d, "video.mp4"), [np.asarray(f) for f in frames], fps=25, macro_block_size=1)
    except Exception as e:                                   # video is a convenience only
        print("video not written:", e)
    return d


def load_episode(ep_dir):
    with open(os.path.join(ep_dir, "episode.json")) as f:
        meta = json.load(f)
    for dp in meta["decisions"]:
        for name in ("prev", "curr"):
            dp[name] = None if dp[name] is None else Image.open(os.path.join(ep_dir, dp[name])).convert("RGB")
    return meta
