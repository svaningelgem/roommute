"""Split the noise corpora into train/valid/test and write prepare_data file lists.

DEMAND is split by *environment* and the interferers by *speaker*, so the valid
and test sets contain acoustic conditions and voices the model never trained on.
A random file-level split would leak both.
"""

import argparse
import os
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="data root, e.g. /root/dfn-data")
    args = ap.parse_args()
    d = args.data
    lists = f"{d}/lists"

    def write(name, files):
        with open(f"{lists}/{name}.txt", "w") as f:
            f.write("\n".join(files) + "\n")
        print(f"  {name:28s} {len(files):6d} files")

    # DEMAND: 4 of the 16 channels per environment; the array is highly correlated.
    demand = sorted(
        os.path.join(dp, f)
        for dp, _, fs in os.walk(f"{d}/demand")
        for f in fs
        if f in ("ch01.wav", "ch05.wav", "ch09.wav", "ch13.wav")
    )
    envs = sorted({os.path.basename(os.path.dirname(p)) for p in demand})
    random.Random(7).shuffle(envs)
    splits = {"TRAIN": envs[:14], "VALID": envs[14:16], "TEST": envs[16:]}
    print(
        f"DEMAND environments: {len(envs)}, valid={splits['VALID']} test={splits['TEST']}"
    )
    for name, es in splits.items():
        write(
            f"{name}_NOISE_DEMAND",
            [p for p in demand if os.path.basename(os.path.dirname(p)) in es],
        )

    fsd_root = f"{d}/fsd50k/FSD50K.dev_audio"
    fsd = sorted(
        os.path.join(fsd_root, f) for f in os.listdir(fsd_root) if f.endswith(".wav")
    )
    random.Random(11).shuffle(fsd)
    write("TRAIN_NOISE_FSD", fsd[:8000])
    write("VALID_NOISE_FSD", fsd[8000:8400])
    write("TEST_NOISE_FSD", fsd[8400:8800])

    # Near RIRs only: the far family is baked into the interferers, never sampled
    # by the loader (see gen_rirs.py).
    near = sorted(
        f"{d}/rir_near/{f}" for f in os.listdir(f"{d}/rir_near") if f.endswith(".wav")
    )
    random.Random(13).shuffle(near)
    write("TRAIN_RIR", near[:340])
    write("VALID_RIR", near[340:370])
    write("TEST_RIR", near[370:])

    for split in ("train", "valid", "test"):
        src = f"{d}/interferers_{split}.txt"
        write(f"{split.upper()}_NOISE_INTERFER", Path(src).read_text().split())


if __name__ == "__main__":
    main()
