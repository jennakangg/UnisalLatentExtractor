# Export DIEM Latents

For each video under example folder:

1. Extracts frames.
2. Runs UNISAL inference.
3. Saves one latent tensor per frame as `.pt`.

Latents are saved to:

`<DIEM_ROOT>/saliency_unisal_latents/<video_name>/*.pt`

## Prerequisites

1. Install dependencies from `environment.yml`.
2. Ensure pretrained weights are available under your training directory.

Default model id used by this repo is `pretrained_unisal` (inside `TRAIN_DIR`).

## Expected DIEM layout

`export_diem_latents.py` expects:

```text
<DIEM_ROOT>/data/<video_name>/video/<video_file>.mp4
```

Supported video extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.

## Run command

From the repository root:

### Minimal

```powershell
python export_diem_latents.py export --data_root "examples"
```
## Outputs

- Latents:
  - `<DIEM_ROOT>/saliency_unisal_latents/<video_name>/*.pt`
- Temporary work directory (removed unless `--keep_workdir=True`):
  - `<DIEM_ROOT>/_saliency_unisal_workdir/<video_name>/`
