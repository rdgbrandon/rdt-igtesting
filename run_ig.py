import json, os
import torch, numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from IPython.display import display, Image as IPyImage
from transformers import AutoTokenizer


with open('success_frames.json') as f:
    meta = json.load(f)
task           = meta['task']
success_frames = meta['successes']
print('Task:', task, '|', len(success_frames), 'successful episodes')
if not success_frames:
    print('No successes to analyse.')
    raise SystemExit

task2lang = {
    'PickCube-v1':         'Grasp a red cube and move it to a target goal position.',
    'StackCube-v1':        'Pick up a red cube and stack it on top of a green cube.',
    'PushCube-v1':         'Push and move a cube to a goal region in front of it.',
    'PegInsertionSide-v1': 'Pick up a orange-white peg and insert the orange end into the box.',
    'PlugCharger-v1':      'Pick up one of the misplaced shapes and insert it into the correct slot.',
}

_embed_path = 'lang_embed_' + task + '.pt'
_ld = torch.load(_embed_path, map_location='cpu', weights_only=False)
_lt = _ld if not isinstance(_ld, dict) else _ld['embeddings']
if _lt.dim() == 2: _lt = _lt.unsqueeze(0)
lang_tokens    = _lt.to(DEVICE, dtype=DTYPE)
lang_attn_mask = torch.ones(lang_tokens.shape[:2], dtype=torch.bool, device=DEVICE)

_tok  = AutoTokenizer.from_pretrained('t5-small')
L_emb = lang_tokens.shape[1]
_desc = task2lang.get(task, task)
_ids  = _tok(_desc, return_tensors='pt').input_ids[0]
n_use = min(len(_ids), L_emb)
raw_words = _tok.convert_ids_to_tokens(_ids[:n_use])
groups, cur = [], [0]
for i in range(1, n_use):
    if raw_words[i].startswith(chr(9601)): groups.append(cur); cur = [i]
    else: cur.append(i)
groups.append(cur)
for i in range(n_use, L_emb): groups.append([i])
plot_labels = [
    (_tok.decode([_ids[i].item() for i in g if i < len(_ids)], skip_special_tokens=True).strip()
     or '[' + str(g[0]) + ']') if g[0] < n_use else '[pad]'
    for g in groups
]
W = len(plot_labels)
JOINT_NAMES = ['base rot', 'shoulder', 'upper arm', 'elbow',
               'forearm rot', 'wrist pitch', 'wrist rot', 'gripper']
_midx = torch.tensor(MANISKILL_INDICES, device=DEVICE)
_bl   = torch.zeros_like(lang_tokens)
_dl   = lang_tokens - _bl
N_IG  = 15


def _blur_emb(e, sigma):
    """Gaussian blur in SigLIP patch-embedding space.
    Uses TF.gaussian_blur (reflect-padded) to avoid zero-pad boundary artifacts."""
    B, T, C = e.shape; G = int(T ** 0.5)
    img = e.reshape(B, G, G, C).permute(0, 3, 1, 2).float()
    if sigma >= 0.05:
        ks = 2 * int(3 * sigma + 0.5) + 1
        if ks % 2 == 0: ks += 1
        img = TF.gaussian_blur(img, kernel_size=[ks, ks], sigma=sigma)
    return img.to(e.dtype).permute(0, 2, 3, 1).reshape(B, T, C)


def blurig_image(scene_emb, scene_state, am0):
    """BlurIG: overall image attribution map (summed over all actions/joints).
    sigma range matches rdt_blurig.py (2.0 → 0.0) to avoid extreme blurring."""
    sigmas = [2.0, 1.5, 1.0, 0.5, 0.0]
    wi_tot = torch.zeros_like(scene_emb)
    for k in range(len(sigmas) - 1):
        E_t    = _blur_emb(scene_emb.detach(), sigmas[k]).requires_grad_(True)
        E_next = _blur_emb(scene_emb.detach(), sigmas[k + 1])
        with torch.enable_grad():
            _ac = rdt.conditional_sample(
                rdt.lang_adaptor(lang_tokens), lang_attn_mask,
                rdt.img_adaptor(E_t.repeat(1, 6, 1)),
                scene_state, am0, ctrl_freqs)
        score = _ac[:, :8, :][:, :, _midx].norm()
        g = torch.autograd.grad(score, E_t)[0]
        wi_tot = wi_tot + g.detach() * (E_next - E_t.detach())
    G      = int(scene_emb.shape[1] ** 0.5)
    wi_map = wi_tot.squeeze(0).float().cpu().abs().sum(-1).reshape(G, G).numpy()
    wi_map = (wi_map - wi_map.min()) / (wi_map.max() - wi_map.min() + 1e-8)
    return wi_map


def word_joint_ig(scene_emb, scene_state, am0):
    """Token IG: word x joint attribution."""
    wj = [torch.zeros_like(lang_tokens) for _ in range(8)]
    for k in range(N_IG):
        alpha  = (k + 0.5) / N_IG
        interp = (_bl + alpha * _dl).requires_grad_(True)
        with torch.enable_grad():
            _ac = rdt.conditional_sample(
                rdt.lang_adaptor(interp), lang_attn_mask,
                rdt.img_adaptor(scene_emb.detach().repeat(1, 6, 1)),
                scene_state, am0, ctrl_freqs)
        js = _ac[:, :8, :][:, :, _midx].norm(dim=(0, 1))
        for j in range(8):
            g = torch.autograd.grad(js[j], interp, retain_graph=(j < 7))[0]
            wj[j] = wj[j] + g.detach() * _dl.detach()
    attr_jt = np.array([wj[j].squeeze(0).float().cpu().abs().sum(-1).numpy() for j in range(8)])
    attr_jw = np.array([[attr_jt[j, g].sum() for g in groups] for j in range(8)])
    return attr_jw


_bil = getattr(PILImage, 'Resampling', PILImage).BILINEAR

def _upsample(wi_map):
    return np.array(PILImage.fromarray((wi_map * 255).astype(np.uint8)).resize((384, 384), _bil)) / 255.0

def _overlay_img(ax, img_np, wi_map):
    """Overlay heatmap using sqrt-alpha so mid-range patches are visible
    while near-zero regions stay transparent."""
    ax.imshow(img_np)
    up   = _upsample(wi_map)
    rgba = mcm.get_cmap('inferno')(up)        # (H, W, 4) RGBA
    rgba[..., 3] = np.sqrt(up) * 0.85        # sqrt: boosts mid-values; *0.85 keeps original visible
    ax.imshow(rgba)

def _make_state(frame):
    with torch.no_grad():
        emb = encode_image(frame)
        _s0 = torch.zeros(1, 1, 128, dtype=DTYPE, device=DEVICE)
        _a0 = torch.zeros(1, 1, 128, dtype=DTYPE, device=DEVICE)
        _a0[0, 0, MANISKILL_INDICES] = 1.0
        state = rdt.state_adaptor(torch.cat([_s0, _a0], dim=2))
    return emb, state, _a0


for info in success_frames:
    ep, seed, steps, flist = info['ep'], info['seed'], info['steps'], info['frames']
    print('Episode', ep, 'seed=', seed, 'steps=', steps)

    frames_pil  = [PILImage.open(fd['path']).convert('RGB') for fd in flist]
    step_labels = (['start']
                   + ['step ' + str(flist[k]['step']) for k in range(1, 9)]
                   + ['SUCCESS'])

    # ── Row 1: raw frames ─────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 10, figsize=(22, 3))
    for k, img_k in enumerate(frames_pil):
        axes1[k].imshow(np.array(img_k))
        axes1[k].set_title(step_labels[k], fontsize=7,
                           fontweight='bold' if k in (0, 9) else 'normal',
                           color='green' if k == 9 else 'black')
        axes1[k].axis('off')
    fig1.suptitle(task + '  ep' + str(ep) + '  seed' + str(seed) +
                  '  (' + str(steps) + ' steps)', fontsize=10, y=1.02)
    plt.tight_layout()
    strip_path = 'strip_ep' + str(ep).zfill(2) + '.png'
    plt.savefig(strip_path, dpi=140, bbox_inches='tight')
    plt.close()
    display(IPyImage(strip_path))

    # ── Row 2: BlurIG overlay per frame ───────────────────────────────────────
    print('  BlurIG on 10 frames...')
    wi_means = []
    fig2, axes2 = plt.subplots(1, 10, figsize=(22, 3))
    fig2.subplots_adjust(right=0.91, wspace=0.04)
    for k, img_k in enumerate(frames_pil):
        emb, state, am0 = _make_state(img_k)
        wi_map = blurig_image(emb, state, am0)
        wi_means.append(float(wi_map.mean()))
        _overlay_img(axes2[k], np.array(img_k) / 255.0, wi_map)
        axes2[k].set_title(step_labels[k], fontsize=7,
                           fontweight='bold' if k in (0, 9) else 'normal',
                           color='green' if k == 9 else 'black')
        axes2[k].axis('off')
        print('   ', k + 1, '/ 10', end='\r', flush=True)
    print()
    # Colorbar in its own axes — avoid stealing space from frame panels
    cax2 = fig2.add_axes([0.92, 0.12, 0.012, 0.72])
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(0, 1))
    fig2.colorbar(sm, cax=cax2, label='Attribution strength')
    fig2.suptitle('BlurIG image attribution per frame  |  ' + task +
                  '  ep' + str(ep), fontsize=10, y=1.02)
    ig_strip_path = 'ig_strip_ep' + str(ep).zfill(2) + '.png'
    plt.savefig(ig_strip_path, dpi=140, bbox_inches='tight')
    plt.close()
    display(IPyImage(ig_strip_path))

    # ── Temporal attribution evolution ────────────────────────────────────────
    fig_t, ax_t = plt.subplots(figsize=(8, 2.5))
    ax_t.plot(range(10), wi_means, 'o-', color='darkorange', linewidth=2, markersize=6)
    ax_t.set_xticks(range(10))
    ax_t.set_xticklabels(step_labels, rotation=30, ha='right', fontsize=8)
    ax_t.set_ylabel('Mean attribution', fontsize=9)
    ax_t.set_title('Attribution magnitude over episode  (ep' + str(ep) + ')', fontsize=9)
    ax_t.grid(True, alpha=0.3)
    plt.tight_layout()
    t_path = 'attr_temporal_ep' + str(ep).zfill(2) + '.png'
    plt.savefig(t_path, dpi=120, bbox_inches='tight')
    plt.close()
    display(IPyImage(t_path))

    # ── Word x Joint (from first frame) ──────────────────────────────────────
    print('  Word x Joint IG...')
    emb0, state0, am0 = _make_state(frames_pil[0])
    attr_jw  = word_joint_ig(emb0, state0, am0)
    _jw_row  = attr_jw / (attr_jw.max(axis=1, keepdims=True) + 1e-8)   # per-joint norm
    _jw_glob = attr_jw / (attr_jw.max() + 1e-8)                         # global norm

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(max(14, W * 1.4), 7),
                                       gridspec_kw={'hspace': 0.6})
    for ax, data, title in [
        (ax3a, _jw_row,  'Row-normalised  — which words each joint cares about'),
        (ax3b, _jw_glob, 'Global-normalised  — cross-joint magnitude comparison'),
    ]:
        im = ax.imshow(data, cmap='inferno', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(W))
        ax.set_xticklabels(plot_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(8))
        ax.set_yticklabels(JOINT_NAMES, fontsize=9)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        for idx in np.argsort(data.ravel())[-5:][::-1]:
            r, c = divmod(int(idx), W)
            ax.text(c, r, f'{data[r, c]:.2f}', ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold')

    fig3.suptitle('Word × Joint attribution  |  ' + task +
                  '  ep' + str(ep) + '  (start frame)', fontsize=10)
    wj_path = 'wj_ep' + str(ep).zfill(2) + '.png'
    plt.savefig(wj_path, dpi=130, bbox_inches='tight')
    plt.close()
    display(IPyImage(wj_path))
    print()

print('Done.')
