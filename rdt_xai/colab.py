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


def setup(out_drive):
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
    os.makedirs(out_drive, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    print("restored from Drive:", sync(out_drive, OUT), "files")
    return OUT
