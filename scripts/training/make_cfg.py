"""Write dataset.cfg and config.ini for a training run.

config.ini is upstream DeepFilterNet3's own, changed only where this machine
needs it. Runs 1 and 2 departed from it on schedule (no batch-size scheduling,
60 epochs, patience 8-12) and both over-suppressed; run 3 tests whether the
recipe, not the data, was the problem.

Sampling factors matter more than they look. libDF combines `n_noises =
uniform(2,6)` noise clips per mixture and draws them from a pool whose size per
dataset is `sampling_factor * n_keys`. So if the interferer set is a fraction f
of the pool, the chance a mixture contains at least one interfering speaker is
1 - (1-f)^3.5 for the average of 2..5 draws. Solving that for one third gives
f = 0.109, which is what we target.
"""

import argparse
import configparser
import json
import os

import h5py

TARGET_SHARE = 0.109  # -> ~1/3 of mixtures contain interfering speech
UPSTREAM = "/root/.cache/DeepFilterNet/DeepFilterNet3/config.ini"

# The only departures from upstream. None of them changes the arithmetic.
OVERRIDES = {
    ("train", "device"): "cuda:0",
    ("train", "log_timings"): "False",
}


def nkeys(path):
    with h5py.File(path, "r") as f:
        return sum(len(f[g]) for g in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5", help="directory holding the *.hdf5 datasets")
    ap.add_argument("out", help="run directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = {}
    for split in ("TRAIN", "VALID", "TEST"):
        n_d = nkeys(f"{args.hdf5}/{split}_NOISE_DEMAND.hdf5")
        n_f = nkeys(f"{args.hdf5}/{split}_NOISE_FSD.hdf5")
        n_i = nkeys(f"{args.hdf5}/{split}_NOISE_INTERFER.hdf5")

        # FSD50K carries the diversity; DEMAND is only 17 environments so it
        # gets a boost without dominating; the interferer factor is then solved
        # for the share we want.
        a_f, a_d = 1.0, 20.0
        other = a_f * n_f + a_d * n_d
        a_i = TARGET_SHARE / (1 - TARGET_SHARE) * other / n_i
        share = a_i * n_i / (other + a_i * n_i)
        print(
            f"{split}: demand={n_d} fsd={n_f} interfer={n_i} -> a_i={a_i:.3f} "
            f"P(mixture has interferer)={1 - (1 - share) ** 3.5:.3f}"
        )
        cfg[split.lower()] = [
            [
                f"{split}_SET_SPEECH.hdf5",
                {"TRAIN": 1.0, "VALID": 0.25, "TEST": 0.35}[split],
            ],
            [f"{split}_RIR.hdf5", 1.0],
            [f"{split}_NOISE_DEMAND.hdf5", a_d],
            [f"{split}_NOISE_FSD.hdf5", a_f],
            [f"{split}_NOISE_INTERFER.hdf5", round(a_i, 4)],
        ]

    with open(f"{args.out}/dataset.cfg", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {args.out}/dataset.cfg")

    c = configparser.ConfigParser()
    if not c.read(UPSTREAM):
        raise SystemExit(
            f"missing {UPSTREAM}; see docs/training.md, 'Reference implementation'"
        )
    # libDF's own interferer path reverberates the target (dataset.rs:1327);
    # interference comes from the noise set instead. Upstream ships it off.
    assert c["distortion"]["p_interfer_sp"] == "0.0", "upstream enabled p_interfer_sp"
    for (section, key), value in OVERRIDES.items():
        c[section][key] = value
    with open(f"{args.out}/config.ini", "w") as f:
        c.write(f)
    print(
        f"wrote {args.out}/config.ini (upstream + {', '.join(k for _, k in OVERRIDES)})"
    )


if __name__ == "__main__":
    main()
