# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "deepfilternet==0.5.6",
#     "torch>=2.0,<2.2",
#     "torchaudio>=2.0,<2.2",
#     "onnx>=1.15",
#     "onnxruntime>=1.15",
#     "numpy>=1.22,<2",
#     "soundfile>=0.12",
#     "MonkeyType",
# ]
# ///
"""Convert DeepFilterNet3's published weights into the model RoomMute runs.

    uv run scripts/export_dfn3.py [OUTPUT_DIR] [MODEL_DIR]

That is the whole thing. uv installs the dependencies into a throwaway
environment, the script downloads the official checkpoint (or uses MODEL_DIR,
a training run directory holding `config.ini` and `checkpoints/`), and it writes a
`DeepFilterNet3_onnx.tar.gz` that `roommute --model <that file>` loads
directly. Nothing needs to be installed first except uv, and nothing is left
behind on your system.

Two pins are load-bearing, so leave them alone unless you are testing an
upgrade:

* **Python < 3.12** — DeepFilterLib, the packaged Rust front-end that does the
  STFT and ERB banding, publishes wheels only up to cp311. uv fetches a
  suitable interpreter itself if you do not have one.
* **torch/torchaudio < 2.2** — `deepfilternet` 0.5.6 imports
  `torchaudio.backend.common.AudioMetaData`, which later torchaudio moved and
  then deleted. Newer versions fail at import with `No module named
  'torchaudio.backend'`.

## What comes out

Three graphs plus a config, tarred together:

    enc.onnx       features      -> embedding + skip connections
    erb_dec.onnx   embedding     -> ERB gain mask
    df_dec.onnx    embedding     -> deep-filter coefficients

Upstream splits it this way on purpose: the STFT front-end belongs in the host
application, not the graph. RoomMute runs these through DeepFilterNet's own
tract runner, which streams them correctly — see docs/model-pipeline.md for
why running them frame-at-a-time under ONNX Runtime silently destroys ~6 dB of
the near voice.

Two of the three graphs come out byte-identical to upstream's own release,
which is the evidence that this conversion is faithful rather than merely
plausible.

## Provenance

Weights: DeepFilterNet3, © 2021 Hendrik Schröter, MIT/Apache-2.0.
<https://github.com/Rikorose/DeepFilterNet>. See models/NOTICE.md.
"""

import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

# MUST come before anything imports torch. `df` pulls torch in, torch loads the
# system libstdc++, and on some Linux setups a later `import sqlite3` then dies
# with "CXXABI_1.3.15 not found" from the interpreter's own libicui18n.
# Upstream's export guards itself with `import monkeytype`, which imports
# sqlite3 transitively — so the failure surfaces as "Failed to import
# monkeytype" and installing MonkeyType does not fix it. Getting in first does.
import sqlite3  # noqa: E402,F401

CHECKPOINT_URL = (
    "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip"
)


def fetch_checkpoint(into: Path) -> Path:
    """Download and unpack the published checkpoint. Cached after the first run."""
    model_dir = into / "DeepFilterNet3"
    if (model_dir / "config.ini").exists():
        print(f"checkpoint already present: {model_dir}")
        return model_dir

    print(f"fetching {CHECKPOINT_URL}")
    into.mkdir(parents=True, exist_ok=True)
    with urlopen(CHECKPOINT_URL) as r:  # noqa: S310  fixed https URL
        blob = r.read()
    print(f"  {len(blob) / 1e6:.1f} MB")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(into)

    if not (model_dir / "config.ini").exists():
        raise SystemExit(f"unpacked archive has no config.ini under {model_dir}")
    return model_dir


def patch_windows_url_bug() -> None:
    """Work around `get_test_sample` building its URL with `os.path.join`.

    `df.io.get_test_sample` fetches a speech clip that the exporter runs
    through both torch and ONNX to check they agree. It composes the URL with
    `os.path.join`, which on Windows yields
    `.../main/assets\\clean_freesound_33711.wav` and 404s. Nothing is wrong
    with the export itself — it just never gets there.

    Rather than skip the check, fetch the same file with a correct URL. If the
    machine is offline, fall back to noise: any signal exercises the
    comparison, speech merely makes a failure easier to interpret.
    """
    import df.io

    url = "https://github.com/Rikorose/DeepFilterNet/raw/main/assets/clean_freesound_33711.wav"

    def get_test_sample(sr: int = 48000):
        import torch

        try:
            with urlopen(url) as r:  # noqa: S310  fixed https URL
                blob = r.read()
            clip = Path(sys.argv[0]).parent / ".test_sample.wav"
            clip.write_bytes(blob)
            sample, _ = df.io.load_audio(str(clip), sr=sr)
            clip.unlink(missing_ok=True)
            return sample
        except Exception as e:  # noqa: BLE001  any failure -> synthetic
            print(f"  (test clip unavailable: {e}; checking against noise instead)")
            return torch.randn(1, sr * 2) * 0.05

    df.io.get_test_sample = get_test_sample


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "models/dfn3-export").resolve()
    if len(sys.argv) > 2:
        model_dir = Path(sys.argv[2]).resolve()
    else:
        model_dir = fetch_checkpoint(out / "checkpoint")

    out.mkdir(parents=True, exist_ok=True)
    patch_windows_url_bug()

    import torch

    import df

    print(
        f"python {sys.version.split()[0]}  torch {torch.__version__}  "
        f"deepfilternet {getattr(df, '__version__', '?')}"
    )
    print(f"exporting to {out}")

    import argparse

    import df.scripts.export as exporter

    # Upstream scripts two of the three graphs through `torch.jit.script`
    # before exporting. On the published 0.5.6 wheel that fails with
    # "Unsupported value kind: Tensor" — the model definition there is not
    # scriptable under torch 2.1, though the same code in their git tree is.
    #
    # jit is only an aid to the export; upstream already passes jit=False for
    # the full-model path. Forcing it off everywhere is what makes this work
    # from PyPI alone, with no clone and no Rust toolchain.
    original = exporter.export_impl

    def without_jit(*args, **kwargs):
        kwargs["jit"] = False
        return original(*args, **kwargs)

    exporter.export_impl = without_jit

    exporter.main(
        argparse.Namespace(
            model_base_dir=str(model_dir),
            export_dir=str(out),
            pf=False,
            log_level="INFO",
            epoch="best",
            check=True,
            simplify=False,
            opset=17,
        )
    )

    produced = sorted(p.name for p in out.glob("*.onnx"))
    tar = next(out.glob("*_onnx.tar.gz"), None)
    print(f"\ngraphs: {', '.join(produced) or 'NONE'}")
    if tar is None:
        raise SystemExit("no *_onnx.tar.gz was produced")
    print(f"model:  {tar}  ({tar.stat().st_size / 1e6:.1f} MB)")
    print(f"\n  roommute --denoise in.wav out.wav --model {tar}")


if __name__ == "__main__":
    main()
