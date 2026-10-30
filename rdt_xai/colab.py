"""Colab environment setup, kept in the repo so fixes arrive with `git pull`
instead of requiring a fresh copy of the notebook."""
import os
import subprocess
import sys

REPO = "/content/rdt-igtesting"
RDT_REPO = "/content/RoboticsDiffusionTransformer"
DATA = "/content/rdt-data"
OUT = "/content/out"
MARKER = "/content/.rdt_xai_installed_v2"

_ICD = {
    "/usr/share/vulkan/icd.d/nvidia_icd.json":
        '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.2.155"}}',
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json":
        '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}',
    "/etc/vulkan/implicit_layer.d/nvidia_layers.json":
        '{"file_format_version":"1.0.0","layer":{"name":"VK_LAYER_NV_optimus","type":"INSTANCE",'
        '"library_path":"libGLX_nvidia.so.0","api_version":"1.2.155","implementation_version":"1",'
        '"description":"NVIDIA Optimus layer","functions":{"vkGetInstanceProcAddr":"vk_optimusGetInstanceProcAddr",'
        '"vkGetDeviceProcAddr":"vk_optimusGetDeviceProcAddr"},"enable_environment":{"__NV_PRIME_RENDER_OFFLOAD":"1"},'
        '"disable_environment":{"DISABLE_LAYER_NV_OPTIMUS_1":""}}}',
}


def sh(cmd):
    """Run a shell command; show its output only if it fails (os.system output is invisible in Colab)."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise RuntimeError(f"failed: {cmd}")
    return r.stdout


def install():
    if os.path.exists(MARKER):
        return
    sh("apt-get -qq update && apt-get -qq install -y --no-install-recommends libvulkan1")
    # Colab's preinstalled torch / transformers / timm provide everything RDT needs (SigLIP, timm Attention).
    # ManiSkill 3.0.1 pins mplib==0.1.1 (motion-planning demos only, never imported by the tasks), which has
    # no Python 3.13 build: install ManiSkill without deps, then its dependency list minus mplib.
    sh('pip install -q --no-deps "mani-skill==3.0.1"')
    sh('pip install -q "sapien==3.0.3" "pytorch_kinematics==0.7.6" "tyro>=0.8.5" "gymnasium>=0.29.1" dacite h5py '
       'pyyaml tqdm GitPython tabulate transforms3d trimesh "imageio[ffmpeg]" IPython defusedxml nvidia-ml-py '
       'huggingface_hub diffusers einops sentencepiece')
    open(MARKER, "w").close()


def vulkan():
    """Vulkan/EGL ICD files for SAPIEN rendering (ManiSkill's Colab recipe; only written if missing)."""
    for p, s in _ICD.items():
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(s)


import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_SYNC_LOCK = threading.Lock()


def _copy(ab):
    try:
        shutil.copyfile(*ab)
        return 1
    except FileNotFoundError:                      # file replaced while walking; next round picks it up
        return 0


def sync(src, dst, workers=16):
    """Copy new/changed files only. Drive is slow per file, not per byte, so copies run in parallel
    threads in-process (one `cp` process per file made restoring a pilot take ~10 min). Stages write
    each unit's .json last (atomically), so a unit is only skipped on resume once it is complete."""
    jobs = []
    with _SYNC_LOCK:
        for root, _, files in os.walk(src):
            rel = os.path.relpath(root, src)
            os.makedirs(os.path.join(dst, rel), exist_ok=True)
            for f in files:
                if f.endswith(".tmp"):
                    continue
                a, b = os.path.join(root, f), os.path.join(dst, rel, f)
                try:
                    if not os.path.exists(b) or os.path.getsize(a) != os.path.getsize(b):
                        jobs.append((a, b))
                except FileNotFoundError:
                    pass
        with ThreadPoolExecutor(workers) as ex:
            return sum(ex.map(_copy, jobs))


def autosync(src, dst, every=300):
    """Copy results to Drive every `every` seconds in the background, so a disconnect
    loses at most a few minutes of work instead of the stage in progress."""
    global _AUTOSYNC
    if globals().get("_AUTOSYNC") and _AUTOSYNC.is_alive():
        return _AUTOSYNC

    def loop():
        import time
        while True:
            time.sleep(every)
            try:
                sync(src, dst)
            except Exception as e:                 # never kill the run because Drive hiccuped
                print("autosync:", e)
    _AUTOSYNC = threading.Thread(target=loop, daemon=True)
    _AUTOSYNC.start()
    print(f"autosync to Drive every {every // 60} min")
    return _AUTOSYNC


def prepare():
    """GPU check, installs, Vulkan, RDT source, env vars; no Google Drive needed."""
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print("GPU:", gpu or "NONE", "| python", sys.version.split()[0])
    assert gpu, "No GPU. Runtime > Change runtime type > A100 GPU"
    if "A100" not in gpu and "H100" not in gpu:
        print("WARNING: not an A100/H100 - everything works, just slower.")
    install()
    vulkan()
    if not os.path.exists(RDT_REPO):
        sh(f"git clone -q https://github.com/thu-ml/RoboticsDiffusionTransformer {RDT_REPO}")
    os.environ["RDT_REPO"], os.environ["RDT_DATA"] = RDT_REPO, DATA
    os.chdir(REPO)
    import importlib.metadata as md
    vers = {}
    for p in ("torch", "transformers", "diffusers", "timm", "mani_skill", "sapien", "numpy"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = "MISSING"
    print(vers)
    print("code:", sh("git log --oneline -1").strip())
    os.makedirs(OUT, exist_ok=True)


def setup(out_drive):
    prepare()
    os.makedirs(out_drive, exist_ok=True)
    print("restored from Drive:", sync(out_drive, OUT), "files")
    autosync(OUT, out_drive)
    return OUT


def download_models():
    """RDT-1B ManiSkill fine-tune, SigLIP-SO400M and the released instruction embeddings (~6 GB)."""
    import shutil
    from huggingface_hub import hf_hub_download
    for sub in ("rdt", "siglip", "lang"):
        os.makedirs(f"{DATA}/{sub}", exist_ok=True)
    for f in ["config.json", "preprocessor_config.json", "special_tokens_map.json", "spiece.model",
              "tokenizer.json", "tokenizer_config.json", "model.safetensors"]:
        hf_hub_download("google/siglip-so400m-patch14-384", f, local_dir=f"{DATA}/siglip")
    hf_hub_download("robotics-diffusion-transformer/maniskill-model", "rdt/mp_rank_00_model_states.pt", local_dir=DATA)
    for t in ["PickCube", "StackCube", "PushCube", "PegInsertionSide", "PlugCharger"]:
        p = hf_hub_download("robotics-diffusion-transformer/maniskill-model",
                            f"lang_embeds/text_embed_{t}-v1.pt", local_dir=DATA)
        if not os.path.exists(f"{DATA}/lang/text_embed_{t}-v1.pt"):
            shutil.copy(p, f"{DATA}/lang/text_embed_{t}-v1.pt")
    print("models ready")


def run_stage(py, cfg, stage, out_drive, *extra):
    """Run one study stage (scripts/run_stage.py) in the chosen environment, streaming its
    progress, then copy results to Drive. Raises if the stage fails."""
    env = dict(os.environ, RDT_REPO=RDT_REPO, RDT_DATA=DATA)
    p = subprocess.Popen([py, "scripts/run_stage.py", "--config", cfg, "--stage", stage, *extra], cwd=REPO, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if "warn" not in line.lower():
            print(line.rstrip(), flush=True)
    code = p.wait()
    print("synced", sync(OUT, out_drive), "files to Drive")
    if code != 0:
        raise RuntimeError(f"stage {stage} failed (exit {code}); last output:\n" + "".join(tail))


FIGURES = ("fig_faithfulness_curves", "fig_faithfulness_curves_gray", "fig_qualitative", "fig_denoising_steps",
           "fig_denoising_step_maps", "fig_closed_loop", "fig_sanity", "fig_word_joint", "fig_modality",
           "fig_grounding")


def show_report(out, out_drive=None):
    """Build the tables and figures, show them in the notebook, and copy them to Drive."""
    from IPython.display import Image, Markdown, display
    from rdt_xai.report import build_report
    display(Markdown(build_report(out)))
    for f in FIGURES:
        path = f"{out}/report/{f}.png"
        if os.path.exists(path):
            print(f)
            display(Image(path, width=900))
    if out_drive and os.path.abspath(out_drive) != os.path.abspath(out):
        print("synced", sync(out, out_drive), "files ->", out_drive)


# ── second policy: ManiSkill's official Diffusion Policy baseline (RGB) ──────────────────
MANISKILL_SRC = "/content/ManiSkill"
DP_SETTINGS = {"PickCube-v1": (100, 30000), "PushCube-v1": (100, 30000), "StackCube-v1": (200, 100000)}
DEMOS = os.path.expanduser("~/.maniskill/demos")


def _stream(cmd, cwd=None):
    """Run a command, printing its output as it comes; raise with the tail if it fails."""
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1)
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if "warn" not in line.lower():
            print(line.rstrip(), flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"failed (exit {p.returncode}): {cmd}\n" + "".join(tail))


_SAME_STEP = ('**({"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP} '
              'if hasattr(gym.vector, "AutoresetMode") else {})')
_SPACE_ORDER = '''from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper


def _space_order(obs, space):
    if not hasattr(space, "keys"):
        return obs
    return {k: _space_order(obs[k], space[k]) for k in space.keys() if k in obs}


class SpaceOrderFlatten(FlattenRGBDObservationWrapper):
    """FlattenRGBDObservationWrapper that concatenates the state in the key order of the env's (batched)
    observation space, the order train_rgbd.py puts the training data in. gymnasium >= 1.0's batch_space rebuilds
    Dict spaces from plain dicts, which sorts their keys, so that order is no longer the env's own dict order."""

    def __init__(self, env, *args, **kwargs):
        self._raw_space = env.unwrapped.observation_space
        super().__init__(env, *args, **kwargs)

    def observation(self, observation):
        return super().observation(_space_order(observation, self._raw_space))
'''
DP_PATCHES = (
    # CPU eval envs start with "forkserver"; ManiSkill's reset then touches CUDA in the child and fails ("Cannot
    # re-initialize CUDA in forked subprocess"). "spawn" workers start clean. Gymnasium >= 1.0 also resets a
    # finished env one step late by default (the policy would act on the last frame of the previous episode);
    # same-step autoreset restores the gymnasium 0.29 behaviour the baseline was written for.
    ("diffusion_policy/make_env.py",
     ('gym.vector.AsyncVectorEnv(x, context="forkserver")', 'gym.vector.AsyncVectorEnv(x, context="spawn")'),
     f'gym.vector.AsyncVectorEnv(x, context="spawn", {_SAME_STEP})'),
    # gymnasium >= 1.0 reports the ending step's infos as a dict of arrays (final_info, or the info itself)
    ("diffusion_policy/evaluate.py",
     ('''                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:''',),
     '''                final = info.get("final_info", info)
                if isinstance(final, dict):
                    for k, v in final["episode"].items():
                        eval_metrics[k].extend(np.asarray(v.float().cpu().numpy() if hasattr(v, "cpu") else v).reshape(-1))
                else:
                    for final_info in final:'''),
    # gymnasium >= 1.0's batch_space sorts Dict keys, so the env's batched observation_space, whose order the
    # dataset puts the h5 keys in, is alphabetical (extra: goal_pos, is_grasped, tcp_pose); FlattenRGBDObservationWrapper
    # concatenates live obs in the env's dict order (is_grasped, tcp_pose, goal_pos): the evaluated policy got the
    # goal where it learned the gripper pose and failed every episode. Flatten live obs in the space order.
    ("diffusion_policy/make_env.py",
     ("from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv\n",),
     "from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv\n" + _SPACE_ORDER),
    ("train_rgbd.py", ("from diffusion_policy.make_env import make_eval_envs",),
     "from diffusion_policy.make_env import SpaceOrderFlatten, make_eval_envs"),
    ("train_rgbd.py", ("        wrappers=[FlattenRGBDObservationWrapper],",), "        wrappers=[SpaceOrderFlatten],"),
)


def _patch_dp_baseline(code):
    for rel, olds, new in DP_PATCHES:
        path = f"{code}/{rel}"
        src = open(path).read()
        if new in src:
            continue
        old = next((o for o in olds if o in src), None)
        if old is None:
            raise RuntimeError(f"{rel}: code to patch not found (ManiSkill baseline changed?)")
        open(path, "w").write(src.replace(old, new))


def train_dp(task, out_drive, demos=100, total_iters=None, seed=1):
    """Train ManiSkill's Diffusion Policy baseline (train_rgbd.py, RGB, pd_ee_delta_pos, motion-planning demos)
    with the settings of its baselines.sh, in the current ManiSkill. Checkpoints go to
    <out_drive>/dp/<task>/. Skipped if a final checkpoint is already there."""
    import glob
    import json
    dst = f"{out_drive}/dp/{task}"
    if os.path.exists(f"{dst}/final.pt"):
        print(f"{task}: trained checkpoint already in {dst}")
        return dst
    import mani_skill
    code = _dp_code()
    steps, iters = DP_SETTINGS.get(task, (200, 100000))
    iters = total_iters or iters
    name = f"dp-{task}-rgb-{demos}demos-s{seed}"
    run = f"{code}/runs/{name}/checkpoints"
    last = (iters - 1) // 5000 * 5000                   # the last numbered checkpoint a finished run saves
    for d in (run, dst):                                # a run that got that far before a disconnect
        if os.path.exists(f"{d}/{last}.pt"):
            print(f"{task}: {last}.pt of an earlier run is in {d}; using it instead of retraining")
            dp_adopt(task, out_drive, f"{d}/{last}.pt", demos, seed)
            return dst
    raw = f"{DEMOS}/{task}/motionplanning/trajectory.h5"
    rgb = f"{DEMOS}/{task}/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
    if not os.path.exists(raw):
        _stream(f"{sys.executable} -m mani_skill.utils.download_demo {task}")
    if not os.path.exists(rgb):                         # only 2x the demos trained on: a few fail to convert
        _stream(f"{sys.executable} -m mani_skill.trajectory.replay_trajectory --traj-path {raw} --use-first-env-state "
                f"-c pd_ee_delta_pos -o rgb --save-traj --num-envs 10 -b physx_cpu --count {2 * demos}")
    os.makedirs(dst, exist_ok=True)
    stop = threading.Event()

    def backup():                                       # the baseline cannot resume, so keep what it saved
        while not stop.wait(60):
            for f in glob.glob(f"{run}/*.pt"):
                b = f"{dst}/{os.path.basename(f)}"
                try:                                    # skip files torch.save may still be writing
                    if time.time() - os.path.getmtime(f) > 30 and (
                            not os.path.exists(b) or os.path.getmtime(f) > os.path.getmtime(b)):
                        shutil.copyfile(f, b)
                except OSError as e:
                    print("checkpoint backup:", e)
    threading.Thread(target=backup, daemon=True).start()
    try:
        _stream(f"{sys.executable} train_rgbd.py --env-id {task} --demo-path {rgb} --control-mode pd_ee_delta_pos "
            f"--sim-backend physx_cpu --num-demos {demos} --max_episode_steps {steps} --total_iters {iters} "
                f"--obs-mode rgb --exp-name {name} --demo_type=motionplanning --seed {seed} --save_freq 5000",
                cwd=code)
    finally:
        stop.set()
    ck = sorted(glob.glob(f"{run}/*.pt"))
    for f in ck:
        shutil.copy(f, f"{dst}/{os.path.basename(f)}")
    # numbered checkpoints stop at the last multiple of save_freq below total_iters; the baseline's own
    # pick is the best evaluation success at the end of an eval round
    numbered = sorted((f for f in ck if os.path.basename(f)[:-3].isdigit()), key=lambda f: int(os.path.basename(f)[:-3]))
    best = [f for f in ck if os.path.basename(f) == "best_eval_success_at_end.pt"]
    final = (best or numbered[-1:] or [None])[0]
    if final:
        shutil.copy(final, f"{dst}/final.pt")
    json.dump(dict(task=task, demos=demos, total_iters=iters, max_episode_steps=steps, seed=seed, run=name,
                   maniskill=mani_skill.__version__, checkpoints=[os.path.basename(f) for f in ck],
                   final=os.path.basename(final) if final else None),
              open(f"{dst}/train.json", "w"), indent=1)
    print(f"{task}: {len(ck)} checkpoints -> {dst}")
    if final:
        dp_evaluate(task, out_drive)
    return dst


def _dp_code():
    """ManiSkill's Diffusion Policy baseline code of the installed release: cloned, patched, installed."""
    import mani_skill
    code = f"{MANISKILL_SRC}/examples/baselines/diffusion_policy"
    if not os.path.exists(MANISKILL_SRC):
        tag = "v" + mani_skill.__version__
        if subprocess.run(f"git clone -q --depth 1 --branch {tag} https://github.com/haosulab/ManiSkill {MANISKILL_SRC}",
                          shell=True).returncode != 0:
            print(f"no tag {tag}; using ManiSkill main")
            sh(f"git clone -q --depth 1 https://github.com/haosulab/ManiSkill {MANISKILL_SRC}")
    _patch_dp_baseline(code)
    try:
        import diffusion_policy  # noqa: F401
    except ImportError:
        sh(f"pip install -q --no-deps -e {code}")
        sh('pip install -q tyro tensorboard h5py')
        import importlib
        importlib.invalidate_caches()
    return code


def _dp_baseline():
    """The patched baseline's modules (make_env, evaluate, train_rgbd), reloaded so a patch made after a first
    import takes effect."""
    import importlib
    code = _dp_code()
    if code not in sys.path:
        sys.path.insert(0, code)
    import diffusion_policy.evaluate
    import diffusion_policy.make_env
    import train_rgbd
    return tuple(importlib.reload(m) for m in (diffusion_policy.make_env, diffusion_policy.evaluate, train_rgbd))


def _dp_env_kwargs(task):
    return dict(control_mode="pd_ee_delta_pos", reward_mode="sparse", obs_mode="rgb", render_mode="rgb_array",
                human_render_camera_configs=dict(shader_pack="default"),
                max_episode_steps=DP_SETTINGS.get(task, (200, 100000))[0])


def _dp_agent(T, make_env, task, path):
    """The baseline's Agent with a checkpoint's EMA weights, and the single-env wrapper chain it was built for."""
    import types
    import gymnasium as gym
    import torch
    from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack
    kw = _dp_env_kwargs(task)
    chain = CPUGymWrapper(FrameStack(make_env.SpaceOrderFlatten(gym.make(task, reconfiguration_freq=1, **kw)),
                                     num_stack=2), ignore_terminations=True, record_metrics=True)
    spaces = types.SimpleNamespace(single_observation_space=chain.observation_space,
                                   single_action_space=chain.action_space)
    args = T.Args(env_id=task, obs_mode="rgb", control_mode="pd_ee_delta_pos", max_episode_steps=kw["max_episode_steps"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = T.Agent(spaces, args).to(device)
    agent.load_state_dict(torch.load(path, map_location=device)["ema_agent"])
    agent.eval()
    return agent, chain, device


def _dp_ckpt(task, out_drive, ckpt):
    return ckpt if os.path.isabs(ckpt) else f"{out_drive}/dp/{task}/{ckpt}"


def dp_evaluate(task, out_drive, ckpt="final.pt", episodes=100):
    """Success rate of a checkpoint through the baseline's (patched) evaluation: 10 CPU envs, the task's training
    max_episode_steps, fresh random episodes. Saved as <out_drive>/dp/<task>/eval_<ckpt name>.json."""
    import json
    import numpy as np
    make_env, ev, T = _dp_baseline()
    path = _dp_ckpt(task, out_drive, ckpt)
    agent, chain, device = _dp_agent(T, make_env, task, path)
    chain.close()
    envs = make_env.make_eval_envs(task, 10, "physx_cpu", _dp_env_kwargs(task), dict(obs_horizon=2), video_dir=None,
                                   wrappers=[make_env.SpaceOrderFlatten])
    m = ev.evaluate(episodes, agent, envs, device, "physx_cpu")
    envs.close()
    r = {k: float(np.mean(v)) for k, v in m.items() if not k.startswith("_")}
    r.update(task=task, ckpt=path, episodes=int(len(m["success_once"])))
    print(f"{task} {os.path.basename(path)}: success_once {r['success_once']:.2f}, "
          f"success_at_end {r['success_at_end']:.2f} over {r['episodes']} episodes")
    os.makedirs(f"{out_drive}/dp/{task}", exist_ok=True)
    json.dump(r, open(f"{out_drive}/dp/{task}/eval_{os.path.basename(path)[:-3]}.json", "w"), indent=1)
    return r


def dp_adopt(task, out_drive, ckpt, demos=100, seed=1, episodes=100):
    """Make a checkpoint of an earlier training run (on this runtime's disk) the task's final.pt, without
    retraining: copy that run's checkpoints to Drive, then evaluate it with the fixed evaluation."""
    import glob
    import json
    import mani_skill
    name = f"dp-{task}-rgb-{demos}demos-s{seed}"
    run = f"{MANISKILL_SRC}/examples/baselines/diffusion_policy/runs/{name}/checkpoints"
    dst = f"{out_drive}/dp/{task}"
    os.makedirs(dst, exist_ok=True)
    ck = sorted(glob.glob(f"{run}/*.pt"))
    for f in ck:
        shutil.copy(f, f"{dst}/{os.path.basename(f)}")
    src = ckpt if os.path.isabs(ckpt) else f"{run}/{ckpt}"
    shutil.copy(src, f"{dst}/final.pt")
    steps, iters = DP_SETTINGS.get(task, (200, 100000))
    json.dump(dict(task=task, demos=demos, total_iters=iters, max_episode_steps=steps, seed=seed, run=name,
                   maniskill=mani_skill.__version__, checkpoints=[os.path.basename(f) for f in ck],
                   final=os.path.basename(src), adopted=True), open(f"{dst}/train.json", "w"), indent=1)
    print(f"{task}: {os.path.basename(src)} -> {dst}/final.pt ({len(ck)} checkpoints copied)")
    return dp_evaluate(task, out_drive, "final.pt", episodes)


def dp_xai(task, out_drive, stages="A,B,C,D,E,F", ckpt="final.pt", **cfg):
    """The attribution study's stages on a task's trained Diffusion Policy (scripts/run_dp.py; options
    as its flags, e.g. a_episodes=20, b_moments=40, b_fast=True). Results go to <out_drive>/dp_xai/<task>/,
    synced every 5 min and at the end; a rerun resumes where the last one stopped."""
    code = _dp_code()
    os.environ["DP_CODE"] = code
    local, remote = f"{OUT}/dp_xai", f"{out_drive}/dp_xai"
    if os.path.exists(remote):
        sync(remote, local)
    flags = " ".join(f"--{k}" if v is True else f"--{k} {v}" for k, v in cfg.items() if v is not False)
    stop = threading.Event()

    def loop():
        while not stop.wait(300):
            try:
                sync(local, remote)
            except Exception as e:                     # never kill the run because Drive hiccuped
                print("sync:", e)
    threading.Thread(target=loop, daemon=True).start()
    try:
        _stream(f"{sys.executable} scripts/run_dp.py --task {task} --ckpt {_dp_ckpt(task, out_drive, ckpt)} "
                f"--out {local} --stages {stages} {flags}", cwd=REPO)
    finally:
        stop.set()
        sync(local, remote)
    return open(f"{local}/{task}/summary.txt").read()


def dp_diagnose(task, out_drive, ckpt="final.pt", episodes=10, demo=0):
    """Why does a trained Diffusion Policy fail the baseline's evaluation? Each check isolates one stage:
      1. a live env set to a demo's first state gives the training data's observation (rgb and state);
      2. the demo's actions, replayed open loop from that state, still solve the task (controller);
      3. the policy reproduces the demo's actions from training observations (network, preprocessing);
      4. the policy solves fresh episodes in one plain env, stepping it by hand;
      5. the same, through the baseline's own vector env and evaluate() (evaluation loop)."""
    import json
    import gymnasium as gym
    import h5py
    import numpy as np
    import torch
    import diffusers
    import mani_skill
    import mani_skill.envs  # noqa: F401
    from mani_skill.trajectory import utils as trajectory_utils
    make_env, ev, T = _dp_baseline()
    from diffusion_policy.utils import build_state_obs_extractor, load_content_from_h5_file
    describe = subprocess.run(f"git -C {MANISKILL_SRC} describe --tags --always", shell=True, capture_output=True,
                              text=True).stdout.strip()
    print(f"mani_skill {mani_skill.__version__} (baseline code {describe}), gymnasium {gym.__version__} "
          f"(AutoresetMode: {hasattr(gym.vector, 'AutoresetMode')}), diffusers {diffusers.__version__}, "
          f"torch {torch.__version__}, numpy {np.__version__}")
    rgb_path = f"{DEMOS}/{task}/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
    meta = json.load(open(rgb_path[:-2] + "json"))
    env_kwargs = _dp_env_kwargs(task)
    ep = meta["episodes"][demo]
    with h5py.File(rgb_path, "r") as f:
        traj = load_content_from_h5_file(f[f"traj_{ep['episode_id']}"])
    extract = build_state_obs_extractor(task)
    raw_env = gym.make(task, reconfiguration_freq=1, **env_kwargs)
    space = raw_env.observation_space                 # the dataset reorders the h5 keys to this
    traj["obs"] = T.reorder_keys(traj["obs"], space)
    live_keys = {k: list(v.keys()) for k, v in raw_env.unwrapped.get_obs().items() if k in ("agent", "extra")}
    space_keys = {k: list(space[k].keys()) for k in live_keys}
    print(f"state keys: env {live_keys}" + ("" if live_keys == space_keys else
                                            f"; space/dataset {space_keys} (the eval flattens in this order)"))

    def demo_state(o):
        parts = [np.asarray(x, dtype=np.float32) for x in extract(o)]
        return np.column_stack([x.reshape(len(x), -1) for x in parts])

    cam = next(iter(traj["obs"]["sensor_data"]))
    d_rgb, d_state, d_act = traj["obs"]["sensor_data"][cam]["rgb"], demo_state(traj["obs"]), traj["actions"]
    print(f"demo {ep['episode_id']}: {len(d_act)} actions, camera {cam} {d_rgb.shape[1:]}, state dim "
          f"{d_state.shape[1]}, |action| mean {np.abs(d_act).mean(0).round(3).tolist()}")
    if d_rgb.max() == 0:
        print("  !! the training images are all black")

    # 1. same state, same observation?
    env = make_env.SpaceOrderFlatten(raw_env)
    env.reset(**ep["reset_kwargs"])
    states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
    env.unwrapped.set_state_dict(states[0])
    live = env.observation(env.unwrapped.get_obs())
    l_rgb, l_state = live["rgb"][0].cpu().numpy(), live["state"][0].cpu().numpy()
    px = np.abs(l_rgb.astype(np.float32) - d_rgb[0].astype(np.float32))
    print(f"1. observation: live state dim {l_state.shape[0]} vs demo {d_state.shape[1]}; ", end="")
    if l_state.shape[0] == d_state.shape[1]:
        diff = np.abs(l_state - d_state[0])
        print(f"max |state diff| {diff.max():.4f} (at dims {np.argsort(diff)[::-1][:3].tolist()}); ", end="")
    print(f"rgb mean |diff| {px.mean():.2f}/255, max {px.max():.0f}; live rgb mean {l_rgb.mean():.1f}, "
          f"demo rgb mean {d_rgb[0].mean():.1f}")
    try:
        from PIL import Image
        os.makedirs(f"{out_drive}/dp/{task}", exist_ok=True)
        Image.fromarray(np.concatenate([d_rgb[0], l_rgb], 1)).resize((512, 256), Image.NEAREST).save(
            f"{out_drive}/dp/{task}/diag_demo_vs_live.png")
    except Exception as e:
        print("   (image not saved:", e, ")")

    # 2. open-loop replay of the demo actions
    succ = False
    for a in d_act:
        _, _, _, _, info = env.step(a)
        succ = succ or bool(np.asarray(info["success"]).any())
    print(f"2. demo actions replayed open loop: {'success' if succ else 'FAIL'}")
    env.close()

    path = _dp_ckpt(task, out_drive, ckpt)
    agent, chain, device = _dp_agent(T, make_env, task, path)
    print(f"policy {path}")

    def act(rgb, state):                         # rgb (2, H, W, C) uint8, state (2, D)
        obs = dict(rgb=torch.as_tensor(np.asarray(rgb))[None].to(device),
                   state=torch.as_tensor(np.asarray(state, dtype=np.float32))[None].to(device))
        return agent.get_action(obs)[0].cpu().numpy()

    # 3. the demo's actions from training observations (frames t-1, t; padded at the start like the dataset)
    errs, base = [], []
    for t in range(0, len(d_act) - 8, max(1, (len(d_act) - 8) // 6)):
        i = [max(t - 1, 0), t]
        pred = act(d_rgb[i], d_state[i])
        errs.append(np.abs(pred - d_act[t:t + 8]).mean())
        base.append(np.abs(d_act.mean(0) - d_act[t:t + 8]).mean())
    print(f"3. policy vs demo actions on training observations: mean |error| {np.mean(errs):.3f} "
          f"(predicting the mean action: {np.mean(base):.3f})")

    # 4. closed loop, one env stepped by hand
    wins, lens = 0, []
    for k in range(episodes):
        obs, _ = chain.reset(seed=1000 + k)
        done, n, ok = False, 0, False
        while not done:
            for a in act(obs["rgb"], obs["state"]):
                obs, _, term, trunc, info = chain.step(a)
                n += 1
                ok = ok or bool(info["success"])
                done = bool(term or trunc)
                if done:
                    break
        wins += ok
        lens.append(n)
    print(f"4. closed loop, plain env: {wins}/{episodes} solved (episode length {np.mean(lens):.0f})")
    chain.close()

    # 5. the baseline's own evaluation path
    envs = make_env.make_eval_envs(task, min(episodes, 10), "physx_cpu", env_kwargs, dict(obs_horizon=2),
                                   video_dir=None, wrappers=[make_env.SpaceOrderFlatten])
    m = ev.evaluate(episodes, agent, envs, device, "physx_cpu", progress_bar=False)
    envs.close()
    print(f"5. baseline evaluate(): success_once {np.mean(m['success_once']):.2f} over {len(m['success_once'])} "
          f"values, episode_len {np.mean(m['episode_len']):.0f}")


PAPER = {"PickCube-v1": 0.772, "StackCube-v1": 0.74, "PushCube-v1": 1.0, "PegInsertionSide-v1": 0.132,
         "PlugCharger-v1": 0.012}


def _repro_run(tag, python, task, n, extra):
    import json
    path = f"{OUT}/repro_{tag}_{task}_{n}.json"
    if os.path.exists(path):
        r = json.load(open(path))
        print(f"{tag}: {r['success_rate']:.0%} (cached)")
        return r
    print(f">> {tag}", flush=True)
    env = dict(os.environ, RDT_REPO=RDT_REPO, RDT_DATA=DATA)
    # one merged stream: reading stderr and stdout one after the other deadlocks once the
    # child fills the unread pipe (e.g. per-step simulator warnings)
    progress = f"{OUT}/repro_{tag}_{task}.progress.jsonl"          # finished episodes survive interruptions
    p = subprocess.Popen([python, "tests/repro.py", "--task", task, "-n", str(n), "--progress", progress, *extra],
                         cwd=REPO, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail, result = [], None
    for line in p.stdout:
        if line.startswith("  ep"):
            print(line.rstrip(), flush=True)
        elif line.startswith("RESULT_JSON "):
            result = line[len("RESULT_JSON "):]
        else:
            tail = (tail + [line])[-25:]
    if p.wait() != 0 or result is None:
        print(f"{tag} FAILED (exit {p.returncode}):\n" + "".join(tail))
        return None
    r = json.loads(result)
    json.dump(r, open(path, "w"))
    print(f"{tag}: {r['success_rate']:.0%}  ({r['minutes']:.1f} min)\n", flush=True)
    return r


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


def repro_check(task="PickCube-v1", n=50, sims=("current", "2024")):
    """Does RDT reproduce its paper success rate, and with which simulator? 2 x 2 design:
    {official eval code, this repo} x {current ManiSkill 3.0.1, Nov-2024 ManiSkill 3.0.0b15}, same seeds.
    Prints per-run success with 95% Wilson CIs and pooled exact tests for the simulator and code effects.
    sims: which simulators to run; finished episodes are kept across interruptions and reused."""
    from scipy.stats import fisher_exact
    prepare()
    download_models()
    runs = [("official", "current", "official_current", ["--impl", "official", "--dtype", "bf16"]),
            ("this repo", "current", "ours_current", ["--impl", "ours", "--dtype", "fp32", "--obs", "rgb"]),
            ("official", "2024", "official_rdt_era", ["--impl", "official", "--dtype", "bf16"]),
            ("this repo", "2024", "ours_rdt_era", ["--impl", "ours", "--dtype", "fp32", "--obs", "rgb"])]
    py = {"current": sys.executable}
    res = {}
    for impl, sim, tag, extra in runs:
        if sim not in sims:
            continue
        if sim not in py:
            py[sim] = rdt_era_env()
        res[(impl, sim)] = _repro_run(tag, py[sim], task, n, extra)
    paper = PAPER.get(task)
    label = {"current": "current sim (ManiSkill 3.0.1)", "2024": "2024 sim (ManiSkill 3.0.0b15)"}
    print(f"\n===== {task}: RDT paper {paper:.1%} (250 trials) =====")
    counts = {}
    for (impl, sim), r in res.items():
        if r is None:
            print(f"{impl + ' / ' + label[sim]:42s} FAILED")
            continue
        k = round(r["success_rate"] * r["n"])
        counts[(impl, sim)] = (k, r["n"])
        lo, hi = _wilson(k, r["n"])
        _, pv = fisher_exact([[k, r["n"] - k], [round(paper * 250), 250 - round(paper * 250)]], alternative="less")
        print(f"{impl + ' / ' + label[sim]:42s} {k:3d}/{r['n']:<3d} = {k / r['n']:4.0%}  95% CI [{lo:4.0%}, {hi:4.0%}]  "
              f"p(below paper) = {pv:.3f}")

    def pooled(keys):
        ks = [counts[x] for x in keys if x in counts]
        return sum(k for k, _ in ks), sum(m for _, m in ks)

    for name, a_keys, b_keys in (
            ("simulator: 2024 vs current", [("official", "2024"), ("this repo", "2024")],
             [("official", "current"), ("this repo", "current")]),
            ("code: this repo vs official", [("this repo", "current"), ("this repo", "2024")],
             [("official", "current"), ("official", "2024")])):
        (ka, na), (kb, nb) = pooled(a_keys), pooled(b_keys)
        if na and nb:
            _, pv = fisher_exact([[ka, na - ka], [kb, nb - kb]])
            print(f"{name:30s} {ka}/{na} = {ka / na:.0%}  vs  {kb}/{nb} = {kb / nb:.0%}   two-sided exact p = {pv:.3f}")
    return res


def rdt_era_env(path="/content/py311"):
    """Python 3.11 venv with the stack RDT was evaluated on (Nov 2024): ManiSkill 3.0.0b15,
    SAPIEN 3.0.0b1, transformers 4.41, diffusers 0.27.2. Completed installs are marked, so an
    interrupted build is resumed instead of being mistaken for a finished one."""
    py, marker = f"{path}/bin/python", f"{path}/.rdt_era_ready"
    check = (f'{py} -c "import torch, mani_skill, sapien; '
             f'print(torch.__version__, mani_skill.__version__, sapien.__version__, torch.cuda.is_available())"')
    if not os.path.exists(marker):
        sh("pip install -q uv")
        if not os.path.exists(py):
            sh(f"uv venv -q --python 3.11 {path}")
        sh(f'uv pip install -q --python {py} "torch==2.5.1" "torchvision==0.20.1" '
           "--index-url https://download.pytorch.org/whl/cu121")
        sh(f'uv pip install -q --python {py} "mani-skill==3.0.0b15" "sapien==3.0.0b1" "transformers==4.41.0" '
           '"diffusers==0.27.2" "timm==1.0.3" "huggingface_hub==0.23.4" "numpy<2" einops sentencepiece pyyaml '
           "imageio imageio-ffmpeg scipy nvidia-ml-py \"setuptools<70\"")   # sapien 3.0.0b1 imports pkg_resources
        sh(check)
        open(marker, "w").close()
    print("RDT-era env:", sh(check).strip())
    return py
