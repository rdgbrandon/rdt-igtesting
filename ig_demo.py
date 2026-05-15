# CLIP + BlurIG across video frames (Xu et al., CVPR 2020)
# See CITATIONS.txt for references.

import os, urllib.request
import cv2
import torch
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from transformers import CLIPModel, CLIPProcessor

VIDEO_URL  = "https://upload.wikimedia.org/wikipedia/commons/3/38/Dog_wagging_tail.webm"
VIDEO_PATH = "dog_video.webm"
if not os.path.exists(VIDEO_PATH):
    print("Downloading video ...")
    req = urllib.request.Request(VIDEO_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(VIDEO_PATH, "wb") as f:
        f.write(resp.read())

N_FRAMES    = 8
MAX_SECONDS = 12

cap   = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total/fps:.1f}s  {fps:.1f}fps")

end_frame = min(total - 1, int(MAX_SECONDS * fps))
frames, timestamps = [], []
for idx in np.linspace(0, end_frame, N_FRAMES, dtype=int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    if ret:
        frames.append(PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        timestamps.append(idx / fps)
cap.release()

print("Loading CLIP ViT-B/32 ...")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

QUESTION = "a dog wagging its tail"

def encode_text(text):
    inp = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        out  = model.text_model(**inp)
        feat = model.text_projection(out.pooler_output)
    return feat / feat.norm(dim=-1, keepdim=True)

def image_feat(pv):
    out  = model.vision_model(pixel_values=pv)
    feat = model.visual_projection(out.pooler_output)
    return feat / feat.norm(dim=-1, keepdim=True)

text_feat = encode_text(QUESTION)


def blur_ig(pv, target, sigma_max=8.0, n_steps=30):
    # Path: x(t) = GaussianBlur(x, sigma_max*(1-t)), t in [0,1]
    # Keeps every interpolation step in natural-image space unlike a black baseline.
    # attr = sum_k grad(x_t) * (x_{t+1} - x_t)  — trapezoid rule
    sigmas = torch.linspace(sigma_max, 0.0, n_steps + 1)
    total  = torch.zeros_like(pv)

    def blur(sig):
        s = sig.item()
        if s < 0.05:
            return pv.detach().clone()
        ks = 2 * int(3 * s + 0.5) + 1
        ks = ks if ks % 2 == 1 else ks + 1
        return TF.gaussian_blur(pv.detach(), kernel_size=[ks, ks], sigma=s)

    for k in range(n_steps):
        x_t    = blur(sigmas[k]).requires_grad_(True)
        x_next = blur(sigmas[k + 1])
        sim    = (image_feat(x_t) * target).sum(dim=-1)
        grad   = torch.autograd.grad(sim.sum(), x_t)[0]
        total  = total + grad.detach() * (x_next - x_t.detach())

    return total


def to_map(attr):
    # ViT-B/32 has a 7x7 patch grid — average within each 32px patch,
    # then upsample so the map aligns with the displayed image.
    a    = np.abs(attr.squeeze().permute(1, 2, 0).detach().numpy()).sum(-1)
    grid = a.reshape(7, 32, 7, 32).mean(axis=(1, 3))
    return np.array(
        PILImage.fromarray((grid / (grid.max() + 1e-8) * 255).astype(np.uint8))
                 .resize((224, 224), PILImage.BILINEAR)
    ) / 255.0


print(f'\nBlurIG on {len(frames)} frames — "{QUESTION}"')
results = []
for i, (frame_pil, ts) in enumerate(zip(frames, timestamps)):
    pv = processor(images=frame_pil, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        sim = (image_feat(pv) * text_feat).sum().item()
    amap     = to_map(blur_ig(pv, text_feat))
    img_disp = (pv.squeeze().permute(1, 2, 0).numpy() * CLIP_STD.squeeze().numpy()
                + CLIP_MEAN.squeeze().numpy()).clip(0, 1)
    results.append({"ts": ts, "img": img_disp, "amap": amap, "sim": sim})
    print(f"  [{i+1}/{len(frames)}]  t={ts:.2f}s  sim={sim:.4f}")

n   = len(results)
fig = plt.figure(figsize=(n * 2.4, 10.5))
gs  = plt.GridSpec(4, n, height_ratios=[2.2, 2.2, 2.2, 1.0],
                   hspace=0.10, wspace=0.04, top=0.91, bottom=0.08)

for row, (label, kind) in enumerate([("Frame", "img"), ("BlurIG\noverlay", "overlay"), ("BlurIG\nonly", "only")]):
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[row, i])
        if kind == "img":
            ax.imshow(r["img"])
            ax.set_title(f't={r["ts"]:.1f}s', fontsize=8)
        elif kind == "overlay":
            ax.imshow(r["img"])
            ax.imshow(r["amap"], cmap="inferno", alpha=0.65, vmin=0, vmax=1)
        else:
            ax.imshow(r["amap"], cmap="inferno", vmin=0, vmax=1)
        if i == 0:
            ax.set_ylabel(label, fontsize=9)
        ax.axis("off")

ax_sim = fig.add_subplot(gs[3, :])
ts_v, sim_v = [r["ts"] for r in results], [r["sim"] for r in results]
ax_sim.plot(ts_v, sim_v, "o-", color="steelblue", lw=2, ms=6, zorder=3)
ax_sim.fill_between(ts_v, min(sim_v) * 0.995, sim_v, alpha=0.25, color="steelblue")
peak = int(np.argmax(sim_v))
ax_sim.scatter([ts_v[peak]], [sim_v[peak]], color="crimson", zorder=5, s=80,
               label=f"peak t={ts_v[peak]:.1f}s")
ax_sim.set_xlabel("Time (s)", fontsize=9)
ax_sim.set_ylabel("CLIP\nsimilarity", fontsize=8)
ax_sim.legend(fontsize=8, loc="lower right")
ax_sim.tick_params(labelsize=8)
ax_sim.set_xlim(ts_v[0] - 0.1, ts_v[-1] + 0.1)

fig.suptitle(f'CLIP + BlurIG (Xu et al., CVPR 2020)\nFixed question: "{QUESTION}"', fontsize=11)
plt.savefig("ig_demo_output.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: ig_demo_output.png")


# Word-level decomposition
# sim(I, T) = v_I · v_T is linear in v_T, so attributions decompose:
#   BlurIG(I, v_T) ≈ sum_k  alpha_k * BlurIG(I, u_k)
# where u_k is the CLIP embedding of word k and alpha_k = v_T · u_k
# (projection of the sentence direction onto each word's direction).

peak_idx   = int(np.argmax(sim_v))
peak_r     = results[peak_idx]
peak_pv    = processor(images=frames[peak_idx], return_tensors="pt")["pixel_values"]
words      = QUESTION.split()
word_feats = {w: encode_text(w) for w in words}
alphas     = {w: (text_feat * word_feats[w]).sum().item() for w in words}

print(f'\nWord decomposition — peak frame t={peak_r["ts"]:.1f}s')
for w, a in alphas.items():
    print(f"  {w:12s}  alpha={a:+.4f}")

print("  Running per-word BlurIG ...")
word_maps = {w: to_map(blur_ig(peak_pv, word_feats[w])) for w in words}
full_map  = to_map(blur_ig(peak_pv, text_feat))

recon_raw = sum(max(alphas[w], 0) * word_maps[w] for w in words)
recon_map = recon_raw / (recon_raw.max() + 1e-8)

n_w  = len(words)
fig2 = plt.figure(figsize=(n_w * 2.8, 11))
gs2  = plt.GridSpec(3, n_w, height_ratios=[2.2, 2.2, 1.0],
                    hspace=0.18, wspace=0.06, top=0.88, bottom=0.07)

for col, (title, amap, is_img) in enumerate([
    ("Original\n(peak frame)", None, True),
    ("Full sentence\nBlurIG",  full_map, False),
    ("Reconstructed\n(sum a*map)", recon_map, False),
]):
    ax = fig2.add_subplot(gs2[0, col])
    ax.imshow(peak_r["img"] if is_img else amap,
              **({} if is_img else {"cmap": "inferno", "vmin": 0, "vmax": 1}))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
for col in range(3, n_w):
    fig2.add_subplot(gs2[0, col]).axis("off")

for i, w in enumerate(words):
    ax = fig2.add_subplot(gs2[1, i])
    m  = word_maps[w]
    ax.imshow(m / (m.max() + 1e-8), cmap="inferno", vmin=0, vmax=1)
    ax.set_title(f'"{w}"\na={alphas[w]:+.3f}', fontsize=9)
    ax.axis("off")

ax_bar = fig2.add_subplot(gs2[2, :])
colors = ["steelblue" if alphas[w] >= 0 else "tomato" for w in words]
bars   = ax_bar.bar(words, list(alphas.values()), color=colors, edgecolor="k", lw=0.8)
ax_bar.axhline(0, color="k", lw=0.6)
ax_bar.set_ylabel("alpha_k  (v_T · u_k)", fontsize=9)
ax_bar.tick_params(labelsize=9)
for bar, w in zip(bars, words):
    h = alphas[w]
    ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.004 * (1 if h >= 0 else -1),
                f"{h:.3f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=8)

fig2.suptitle(
    f'Word-level attribution decomposition — "{QUESTION}"\n'
    r'BlurIG(I, v$_T$) $\approx$ $\sum_k$ $\alpha_k$ $\cdot$ BlurIG(I, u$_k$)',
    fontsize=10,
)
plt.savefig("ig_word_decomp.png", dpi=150, bbox_inches="tight")
print("Word decomp saved: ig_word_decomp.png")
