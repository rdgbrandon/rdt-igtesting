# Colab cell: play each successful episode as a video — raw frames, and (if the
# IG cell has been run) the same frames with the BlurIG patch-attribution overlay.
#
# Run order: rollout cell (saves frames + success_frames.json)  →  IG cell
# (run_ig.py, saves ig_maps_epXX.npy)  →  this cell. Only files on disk are
# used, so it needs no notebook globals and the raw video works even before
# the IG cell has run.
import json, os, base64
import numpy as np
import matplotlib.cm as mcm
from PIL import Image as PILImage
from IPython.display import display, HTML, Image as IPyImage

FPS = 5   # 30 sampled frames -> ~6 s clip

with open('success_frames.json') as f:
    meta = json.load(f)
print('Task:', meta['task'], '|', len(meta['successes']), 'successful episodes')

_bil = getattr(PILImage, 'Resampling', PILImage).BILINEAR
try:
    _inferno = mcm.get_cmap('inferno')          # matplotlib < 3.9
except AttributeError:
    import matplotlib
    _inferno = matplotlib.colormaps['inferno']  # matplotlib >= 3.9


def _overlay_frame(img, wi_map):
    """Composite the BlurIG heatmap onto one frame — same inferno colormap and
    sqrt-alpha blend as run_ig.py's _overlay_img, but returning a plain uint8
    image instead of drawing on a matplotlib axis."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    up = np.asarray(PILImage.fromarray((wi_map * 255).astype(np.uint8))
                    .resize((w, h), _bil)) / 255.0
    rgb   = _inferno(up)[..., :3]
    alpha = (np.sqrt(up) * 0.85)[..., None]
    return (255 * (arr * (1 - alpha) + rgb * alpha)).astype(np.uint8)


def _write_video(frames_np, stem):
    """Write an .mp4 (imageio + ffmpeg); fall back to a looping .gif if the
    ffmpeg plugin isn't available. Returns the written path."""
    try:
        import imageio
        path = stem + '.mp4'
        imageio.mimsave(path, frames_np, fps=FPS, macro_block_size=1)
        return path
    except Exception as e:
        print('  mp4 unavailable (' + str(e).split(chr(10))[0] + ') — writing gif')
        path = stem + '.gif'
        pil = [PILImage.fromarray(f) for f in frames_np]
        pil[0].save(path, save_all=True, append_images=pil[1:],
                    duration=int(1000 / FPS), loop=0)
        return path


def _show_video(path, title):
    print(' ', title, '->', path)
    if path.endswith('.mp4'):
        b64 = base64.b64encode(open(path, 'rb').read()).decode()
        display(HTML('<div style="display:inline-block;margin:4px;text-align:center">'
                     '<div style="font:12px sans-serif">' + title + '</div>'
                     '<video controls autoplay loop muted width="420" '
                     'src="data:video/mp4;base64,' + b64 + '"></video></div>'))
    else:
        display(IPyImage(path))


for info in meta['successes']:
    ep, seed, steps, flist = info['ep'], info['seed'], info['steps'], info['frames']
    print('Episode', ep, 'seed=', seed, 'steps=', steps, '|', len(flist), 'frames')
    frames_pil = [PILImage.open(fd['path']).convert('RGB') for fd in flist]

    raw_path = _write_video([np.asarray(f) for f in frames_pil],
                            'video_raw_ep' + str(ep).zfill(2))
    _show_video(raw_path, meta['task'] + ' ep' + str(ep) + ' — raw rollout')

    maps_path = 'ig_maps_ep' + str(ep).zfill(2) + '.npy'
    if os.path.exists(maps_path):
        wi_maps = np.load(maps_path)
        if len(wi_maps) != len(frames_pil):
            print('  WARNING:', maps_path, 'has', len(wi_maps), 'maps but',
                  len(frames_pil), 'frames — rerun the IG cell on this rollout.')
        ig_frames = [_overlay_frame(img, m) for img, m in zip(frames_pil, wi_maps)]
        ig_path = _write_video(ig_frames, 'video_ig_ep' + str(ep).zfill(2))
        _show_video(ig_path, meta['task'] + ' ep' + str(ep) + ' — BlurIG overlay')
    else:
        print(' ', maps_path, 'not found — run the IG cell (run_ig.py) first '
              'to get the attribution-overlay video.')

print('Done.')
