import json, os
import torch, numpy as np
import matplotlib.pyplot as plt
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
    import torch.nn.functional as F
    B, T, C = e.shape; G = int(T ** 0.5)
    m = e.reshape(B, G, G, C).permute(0, 3, 1, 2)
    if sigma > 0:
        ks = max(3, int(6 * sigma) // 2 * 2 + 1)
        m = F.avg_pool2d(m, kernel_size=ks, stride=1, padding=ks // 2)
    return m.permute(0, 2, 3, 1).reshape(B, T, C)

def blurig_image(scene_emb, scene_state, am0):
    """BlurIG: overall image attribution map (summed over all actions/joints)."""
    sigmas = [10.0, 7.5, 5.0, 2.5, 0.0]
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

    frames_pil = [PILImage.open(fd['path']).convert('RGB') for fd in flist]

    # ── Row 1: raw frames ─────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 10, figsize=(20, 2))
    for k, (fd, img_k) in enumerate(zip(flist, frames_pil)):
        axes1[k].imshow(np.array(img_k))
        axes1[k].set_title('start' if k == 0 else ('SUCCESS' if k == 9 else 'step ' + str(fd['step'])), fontsize=6)
        axes1[k].axis('off')
    fig1.suptitle(task + '  ep' + str(ep) + '  seed' + str(seed) + '  (' + str(steps) + ' steps)', fontsize=9)
    plt.tight_layout()
    strip_path = 'strip_ep' + str(ep).zfill(2) + '.png'
    plt.savefig(strip_path, dpi=120, bbox_inches='tight')
    plt.close()
    display(IPyImage(strip_path))

    # ── Row 2: BlurIG overlay per frame ───────────────────────────────────────
    print('  BlurIG on 10 frames...')
    fig2, axes2 = plt.subplots(1, 10, figsize=(20, 2))
    for k, (fd, img_k) in enumerate(zip(flist, frames_pil)):
        emb, state, am0 = _make_state(img_k)
        wi_map = blurig_image(emb, state, am0)
        img_np = np.array(img_k) / 255.0
        axes2[k].imshow(img_np)
        axes2[k].imshow(_upsample(wi_map), cmap='inferno', alpha=0.6, vmin=0, vmax=1)
        axes2[k].set_title('start' if k == 0 else ('SUCCESS' if k == 9 else 'step ' + str(fd['step'])), fontsize=6)
        axes2[k].axis('off')
        print('   ', k + 1, '/ 10', end='\r', flush=True)
    print()
    fig2.suptitle('Word x Image (BlurIG) per frame', fontsize=9)
    plt.tight_layout()
    ig_strip_path = 'ig_strip_ep' + str(ep).zfill(2) + '.png'
    plt.savefig(ig_strip_path, dpi=120, bbox_inches='tight')
    plt.close()
    display(IPyImage(ig_strip_path))

    # ── Word x Joint (from first frame) ──────────────────────────────────────
    print('  Word x Joint IG...')
    emb0, state0, am0 = _make_state(frames_pil[0])
    attr_jw = word_joint_ig(emb0, state0, am0)
    _jw_n = attr_jw / (attr_jw.max(axis=1, keepdims=True) + 1e-8)
    fig3, ax3 = plt.subplots(figsize=(max(10, W * 1.2), 3))
    im = ax3.imshow(_jw_n, cmap='inferno', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(W)); ax3.set_xticklabels(plot_labels, rotation=40, ha='right', fontsize=8)
    ax3.set_yticks(range(8)); ax3.set_yticklabels(JOINT_NAMES, fontsize=8)
    ax3.set_title('Word x Joint (token IG, start frame)')
    plt.colorbar(im, ax=ax3, fraction=0.03)
    plt.tight_layout()
    wj_path = 'wj_ep' + str(ep).zfill(2) + '.png'
    plt.savefig(wj_path, dpi=120, bbox_inches='tight')
    plt.close()
    display(IPyImage(wj_path))
    print()

print('Done.')
