"""Synthesise 48 kHz RIRs with pyroomacoustics (BSD) - no licence entanglement.

Two families, used for different purposes:

  NEAR  - small rooms, low RT60, source close to the mic. These go into the RIR
          HDF5 that DeepFilterNet samples for its own `p_reverb` augmentation,
          which makes the *input* more reverberant than the *target* and so
          teaches mild dereverberation of the near speaker.

  FAR   - bigger rooms, high RT60, source metres away, plus a low-pass to
          emulate transmission through a wall. These are convolved offline into
          the interfering-speech noise set: the "child in the next room".

Keeping the two pools separate is deliberate. If the far RIRs were in the pool
DeepFilterNet samples from, the near target itself would sometimes be heavily
reverberant, which is the opposite of the proximity cue we want it to learn.
"""

import argparse
import os

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48000


def one_rir(rng, kind):
    if kind == "near":
        dims = rng.uniform([3.0, 2.8, 2.3], [6.0, 5.0, 3.0])
        rt60 = rng.uniform(0.15, 0.35)
        dist = rng.uniform(0.3, 1.2)
    else:
        dims = rng.uniform([4.0, 4.0, 2.4], [10.0, 9.0, 3.4])
        rt60 = rng.uniform(0.40, 1.00)
        dist = rng.uniform(3.0, 8.0)

    absorption, max_order = pra.inverse_sabine(rt60, dims)
    max_order = int(min(max_order, 17))  # keep image-source count sane at 48 kHz

    room = pra.ShoeBox(
        dims, fs=SR, materials=pra.Material(absorption), max_order=max_order
    )
    mic = rng.uniform([0.6, 0.6, 1.0], dims - np.array([0.6, 0.6, 1.2]))

    # Place the source `dist` away from the mic, at a random bearing, clamped
    # inside the room.
    for _ in range(40):
        ang = rng.uniform(0, 2 * np.pi)
        elev = rng.uniform(-0.25, 0.25)
        off = np.array([dist * np.cos(ang), dist * np.sin(ang), dist * np.sin(elev)])
        src = mic + off
        if np.all(src > 0.35) and np.all(src < dims - 0.35):
            break
    else:
        return None

    room.add_source(src).add_microphone(mic)
    room.compute_rir()
    h = np.asarray(room.rir[0][0], dtype=np.float32)

    if kind == "far":
        # Through-a-wall: a wall is a low-pass. Cut-off between 900 Hz and 3 kHz
        # gives the muffled quality of speech from an adjacent room.
        fc = rng.uniform(900.0, 3000.0)
        sos = butter(4, fc / (SR / 2), btype="low", output="sos")
        h = sosfilt(sos, h).astype(np.float32)

    h = h[: int(1.5 * SR)]
    peak = np.abs(h).max()
    if not np.isfinite(peak) or peak < 1e-8:
        return None
    return (h / (peak * 1.05)).astype(np.float32), rt60, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["near", "far"])
    ap.add_argument("outdir")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    made, rt60s = 0, []
    while made < args.n:
        r = one_rir(rng, args.kind)
        if r is None:
            continue
        h, rt60, _ = r
        sf.write(
            os.path.join(args.outdir, f"rir_{args.kind}_{made:04d}.wav"),
            h,
            SR,
            subtype="PCM_16",
        )
        rt60s.append(rt60)
        made += 1
        if made % 50 == 0:
            print(f"  {made}/{args.n}", flush=True)
    print(
        f"{args.kind}: {made} RIRs, RT60 {np.min(rt60s):.2f}-{np.max(rt60s):.2f}s "
        f"-> {args.outdir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
