"""Build the "someone talking in the next room" noise class.

Takes VCTK speakers that are *never* used as targets (speaker-disjoint, so the
model cannot shortcut on voice identity), convolves each utterance with a FAR
RIR - high RT60, several metres away, low-passed to imitate a wall - and writes
the result as ordinary noise clips.

DeepFilterNet then mixes these through its normal noise path, so the target
stays dry and near while the interferer arrives reverberant and quiet. That
contrast is the entire point: it is what should teach the model to discriminate
on proximity rather than on "is this speech".

Note we do NOT use libDF's built-in `p_interfer_sp`. That code path has a bug
(libDF/src/dataset.rs:1327): inside the interferer loop it calls
`self.reverb.transform_single(&mut speech, rir)`, reverberating the *target*
signal 1-3 times over, rather than `sample`, the interferer. It applies the
reverberation to exactly the wrong signal.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

SR = 48000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--speakers", required=True, help="file with interferer speaker ids"
    )
    ap.add_argument("--vctk-root", required=True)
    ap.add_argument("--rir-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--per-speaker", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)

    rirs = sorted(
        os.path.join(args.rir_dir, f)
        for f in os.listdir(args.rir_dir)
        if f.endswith(".wav")
    )
    speakers = Path(args.speakers).read_text().split()
    print(f"{len(speakers)} interferer speakers, {len(rirs)} far RIRs", flush=True)

    made, written = 0, []
    for spk in speakers:
        d = os.path.join(args.vctk_root, spk)
        if not os.path.isdir(d):
            print(f"  skip missing {spk}", flush=True)
            continue
        utts = sorted(f for f in os.listdir(d) if f.endswith("_mic1.flac"))
        rng.shuffle(utts)
        for u in utts[: args.per_speaker]:
            x, sr = sf.read(os.path.join(d, u), dtype="float32")
            if sr != SR or x.size < SR // 2:
                continue
            if x.ndim > 1:
                x = x.mean(1)
            h, _ = sf.read(rng.choice(rirs), dtype="float32")
            y = fftconvolve(x, h)[: len(x)]
            peak = float(np.abs(y).max())
            if not np.isfinite(peak) or peak < 1e-6:
                continue
            # Normalise per clip; the mixing SNR sets the actual level later.
            y = (y / peak * rng.uniform(0.3, 0.9)).astype(np.float32)
            p = os.path.join(args.outdir, f"int_{spk}_{u[:-5]}.wav")
            sf.write(p, y, SR, subtype="PCM_16")
            written.append(p)
            made += 1
        print(f"  {spk}: {made} total", flush=True)

    listfile = os.path.join(args.outdir, "..", os.path.basename(args.outdir) + ".txt")
    with open(listfile, "w") as f:
        f.write("\n".join(written) + "\n")
    dur = sum(sf.info(p).duration for p in written[:200]) / min(200, len(written))
    print(
        f"wrote {made} interferer clips (~{made * dur / 3600:.2f} h) -> {listfile}",
        flush=True,
    )


if __name__ == "__main__":
    main()
