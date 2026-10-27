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


def sync(src, dst):
    """Copy new/changed files only (Drive is slow with many small files)."""
    n = 0
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            a, b = os.path.join(root, f), os.path.join(dst, rel, f)
            if not os.path.exists(b) or os.path.getsize(a) != os.path.getsize(b):
                subprocess.run(["cp", a, b])
                n += 1
    return n


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
    p = subprocess.Popen([python, "tests/repro.py", "--task", task, "-n", str(n), *extra], cwd=REPO, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tail = []
    for line in p.stderr:
        if line.startswith("  ep"):
            print(line.rstrip(), flush=True)
        else:
            tail = (tail + [line])[-25:]
    out = p.stdout.read().strip().splitlines()
    if p.wait() != 0 or not out:
        print(f"{tag} FAILED (exit {p.returncode}):\n" + "".join(tail))
        return None
    r = json.loads(out[-1])
    json.dump(r, open(path, "w"))
    print(f"{tag}: {r['success_rate']:.0%}  ({r['minutes']:.1f} min)\n", flush=True)
    return r


def repro_check(task="PickCube-v1", n=10):
    """Does RDT reproduce its paper success rate, and with which simulator? Three runs:
    official code on the current stack, official code on the Nov-2024 stack, this repo on the Nov-2024 stack."""
    from scipy.stats import fisher_exact
    prepare()
    download_models()
    res = {"official / current sim (ManiSkill 3.0.1)": _repro_run("official_current", sys.executable, task, n,
                                                                   ["--impl", "official", "--dtype", "bf16"])}
    py = rdt_era_env()
    res["official / 2024 sim (ManiSkill 3.0.0b15)"] = _repro_run("official_rdt_era", py, task, n,
                                                                 ["--impl", "official", "--dtype", "bf16"])
    res["this repo / 2024 sim (ManiSkill 3.0.0b15)"] = _repro_run("ours_rdt_era", py, task, n,
                                                                  ["--impl", "ours", "--dtype", "fp32", "--obs", "rgb"])
    paper = PAPER.get(task)
    print(f"\n===== {task}: RDT paper {paper:.1%} =====")
    for k, r in res.items():
        if r is None:
            print(f"{k:45s} FAILED")
            continue
        k_succ = round(r["success_rate"] * r["n"])
        # one-sided exact test against the paper rate as if it came from 250 trials
        _, pv = fisher_exact([[k_succ, r["n"] - k_succ], [round(paper * 250), 250 - round(paper * 250)]],
                             alternative="less")
        print(f"{k:45s} {k_succ:2d}/{r['n']} = {r['success_rate']:4.0%}   "
              f"p(below paper) = {pv:.3f}   sapien {r['versions']['sapien']}")
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
           "imageio imageio-ffmpeg scipy")
        sh(check)
        open(marker, "w").close()
    print("RDT-era env:", sh(check).strip())
    return py
