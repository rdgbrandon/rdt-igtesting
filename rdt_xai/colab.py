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
