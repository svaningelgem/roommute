"""Split VCTK speakers into disjoint target and interferer pools.

Interferers are never targets, so the model cannot learn "this voice is the one
to keep". Valid and test speakers are held out of both training pools.

Writes, under <data>/lists:
  spk_{tgt,int}_{train,valid,test}.txt   speaker ids
  {TRAIN,VALID,TEST}_SPEECH.txt          target utterances, mic1 only
"""

import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="data root, e.g. /root/dfn-data")
    args = ap.parse_args()

    root = f"{args.data}/vctk/wav48_silence_trimmed"
    lists = f"{args.data}/lists"
    os.makedirs(lists, exist_ok=True)

    spk = sorted(s for s in os.listdir(root) if os.path.isdir(os.path.join(root, s)))
    # p315 has no text and p280 has a known mic2 fault; we only use mic1, but
    # drop any speaker with too few usable utterances anyway.
    spk = [
        s
        for s in spk
        if len(
            [f for f in os.listdir(os.path.join(root, s)) if f.endswith("_mic1.flac")]
        )
        >= 50
    ]
    random.Random(1234).shuffle(spk)
    print("usable speakers:", len(spk))

    pools = {
        "tgt_train": spk[:76],
        "tgt_valid": spk[76:80],
        "tgt_test": spk[80:84],
        "int_train": spk[84:104],
        "int_valid": spk[104:107],
        "int_test": spk[107:110],
    }
    for name, ids in pools.items():
        with open(f"{lists}/spk_{name}.txt", "w") as f:
            f.write("\n".join(ids) + "\n")
        print(f"  {name:9s} {len(ids):3d} speakers")

    for split in ("train", "valid", "test"):
        files = []
        for s in pools[f"tgt_{split}"]:
            d = os.path.join(root, s)
            files += [
                os.path.join(d, f)
                for f in sorted(os.listdir(d))
                if f.endswith("_mic1.flac")
            ]
        with open(f"{lists}/{split.upper()}_SPEECH.txt", "w") as f:
            f.write("\n".join(files) + "\n")
        print(f"  {split.upper()}_SPEECH: {len(files)} utterances")


if __name__ == "__main__":
    main()
