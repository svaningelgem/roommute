# Training a model in WSL2

Notes for rebuilding the DeepFilterNet training and reference environment
without rediscovering the workarounds. Everything here was learned the hard
way; the non-obvious bits are called out.

## Status: RoomMute does not use a model trained here

It ships DeepFilterNet3's published weights, which its MIT/Apache-2.0 licence
permits us to redistribute, exported by `scripts/export_dfn3.py` from the
published checkpoint. That removed the reason this document originally
existed — the fear was that we had no right to ship anything, and we do. See
[`model-pipeline.md`](model-pipeline.md).

Two training runs were attempted anyway, to try to beat DFN3 at competing
speech specifically. Both produced models that suppress rather than separate;
neither is used. Run 3 below repeats the attempt on upstream's own recipe; a
model that holds up would also stop the shipped weights depending on how
upstream's licence grant is read. What follows is the environment, the dataset choices and the
failure analysis, for anyone who wants to try again.

## Environment

Source [`scripts/wsl_env.sh`](../scripts/wsl_env.sh) before anything else.

| Trap | Symptom | Fix |
|---|---|---|
| WSL inherits the **Windows PATH** | `bash: syntax error near unexpected token '('` | rebuild `PATH` rather than appending to it |
| miniforge's `sqlite3` needs miniforge's `libstdc++` | imports fail as a misleading *"Failed to import monkeytype"* | `LD_LIBRARY_PATH=/opt/miniforge3/lib` |
| `$VAR` and `$(...)` are eaten by `wsl.exe -d Ubuntu -- bash -c '...'` | silent missing-argument errors | put anything with shell variables in a `.sh` file and run that |
| stdout is lossy over the same bridge | truncated output on long/fast runs | redirect to a file and read it back |
| HDF5 locking fails on `drvfs` | dataset creation or training dies | keep datasets on the WSL filesystem (`/root/...`), never `/mnt/...` |
| `requests`/`urllib3` in that Python cannot TLS to **GitHub** | `ConnectionError: RemoteDisconnected` | `curl` works fine, and so does `pip` against PyPI — this is GitHub-specific, **not** a WSL networking problem |

## Reference implementation (for validating a port)

```bash
pip install deepfilternet 'torch==2.1.2' 'torchaudio==2.1.2' \
  --index-url https://download.pytorch.org/whl/cpu
```

`torchaudio` must be pinned: 2.2 moved `torchaudio.backend.common.AudioMetaData`
and 2.9 removed it. Place the checkpoint at
`/root/.cache/DeepFilterNet/DeepFilterNet3/` by hand — the automatic download
hits the GitHub TLS problem above.

`df.enhance.df_features()` returns exactly the tensors a host must compute
(`spec`, `feat_erb`, `feat_spec`), which is how
[`crates/dsp/src/dfn_frontend.rs`](../crates/dsp/src/dfn_frontend.rs) was
verified: on real audio all three match to floating-point noise.

## The data loader

```bash
python3 -m maturin build --release -m pyDF-data/Cargo.toml \
  --features hdf5-static --compatibility linux -o /tmp/wheels
```

`--compatibility linux` (or installing `patchelf`) is **required**: without it
the manylinux repair step fails and maturin writes a 22-byte empty wheel that
installs without complaint and fails at import. `--features hdf5-static` does
not actually static-link; `libhdf5_serial.so.103` still has to be present.

`df/train.py` also imports `icecream`, which is not pulled in as a dependency.

## AMD GPU under WSL2

ROCm **does** work in WSL2 — an RX 7900 XTX reaches 14.3 TFLOP/s fp32,
including GRU forward and backward, which is what matters for a GRU-heavy
model.

Install `torch` for ROCm into its own venv, then apply the one fix that makes
it see the card:

```bash
cp -f /opt/rocm/lib/libhsa-runtime64.so.1.14.0 \
      "$VENV/lib/python3.11/site-packages/torch/lib/libhsa-runtime64.so"
```

PyTorch ships a generic HSA runtime that talks to `/dev/kfd`; WSL2 has only
`/dev/dxg`, so ROCm's own runtime has to replace it. `rocm-smi` failing with
*"amdgpu not found in modules"* is expected on WSL and harmless — that is the
monitoring tool, not compute.

Newer `torchaudio` in the ROCm venv needs `torchaudio.save` bound to
soundfile at `df.io` import time, because `df/train.py` calls
`torchaudio.save` directly as well. Pinned memory must also be disabled —
it fails with `HIP error: out of memory` on WSL regardless of free RAM.

Measured, batch 64, 5 s segments, warm cache:

| | epoch | throughput | 100 epochs on VCTK |
|---|---|---|---|
| RX 7900 XTX | 9 s | ~430 audio-s/s | ~10 h |
| 7950X3D, 32 threads | 33 s | ~116 audio-s/s | ~38 h |

The GPU is only ~3.7× the CPU here: the model is small and the run is partly
dataloader-bound. CPU-only training is viable.

## Datasets

Clean licences and 48 kHz throughout, 33 GB downloaded, 23.7 GB of HDF5.

| Corpus | Size | Licence | Rate |
|---|---|---|---|
| VCTK 0.92 | 10.94 GB | CC BY 4.0 | 48 kHz |
| DEMAND (17 environments) | 4.91 GB | CC BY-SA 3.0 | 48 kHz |
| FSD50K dev | 17.15 GB | per-clip CC-BY / CC0 | 44.1 kHz → resampled |
| RIRs | — | synthesised, pyroomacoustics (BSD) | 48 kHz |

Deliberately **not** used:

- **MUSAN** and **OpenSLR28** RIRs — 16 kHz only, useless for a 48 kHz model.
- **DNS-Challenge noise** — Audioset/YouTube-derived, which would reintroduce
  exactly the provenance problem we are escaping.
- `assets/clean_freesound_33711.wav` — **CC BY-NC**, fine for plumbing tests,
  must never reach a shipped model.

The Edinburgh DataShare URL for VCTK returns a *wrapper* zip containing the
real `VCTK-Corpus-0.92.zip`; extracting it looks like a no-op because the inner
archive has the same name.

## Training for competing speech

The point of training our own model is not to reproduce DeepFilterNet3 but to
be *deliberately* good at what it is accidentally good at.

On a competing voice DFN3 gains 3.6 dB of SNR where RNNoise gains none at all,
because speech-enhancement models are trained to *preserve* speech and a
neighbour is speech. DFN3 appears to discriminate on proximity and
reverberation instead. Training on speech-plus-noise alone would very likely
lose that property.

So interfering speech is a first-class noise class:

- **Speaker-disjoint pools** — interferers are never targets, so the model
  cannot shortcut on voice identity.
- **Two separate RIR families.** Near (RT60 0.15–0.35 s, 0.3–1.2 m) for the
  loader's own augmentation; far (RT60 0.40–1.00 s, 3–8 m, plus a 900 Hz–3 kHz
  low-pass for the wall) used *only* to build interferers offline. Keeping them
  apart matters — far RIRs in the loader's pool would make the near target
  itself heavily reverberant.
- **Mixture rate solved, not guessed.** libDF combines `uniform(2,6)` noise
  clips per mixture, so the interferer share `f` of the noise pool follows
  `1-(1-f)^3.5 = 1/3` → `f = 0.109`. Verified at 0.332 across all splits.
- `p_reverb`: DFN3 ships 0.1. Run 1 used **0.2**, on the theory that libDF
  makes the *input* more reverberant than the *target*, and that this
  dereverberation pressure is the mechanism behind proximity discrimination —
  so it was leaned on deliberately. **That was too much; see below.**

Use the **official DFN3 recipe**, not the generated defaults — the defaults
leave the multi-resolution spectral loss at zero and shrink `conv_ch` from 64
to 16, training a different and much weaker network.

### A bug in libDF's own interferer path

`libDF/src/dataset.rs:1327`, inside the interferer loop:

```rust
self.reverb.transform_single(&mut speech, rir)?;   // `speech` is the TARGET
```

It reverberates the training **target**, once per interferer and therefore
compounded, instead of `sample`, the interferer. The target becomes reverberant
while the interferer stays dry — exactly backwards. Set `p_interfer_sp = 0.0`
and build interference yourself.

## Judging the result

Build the reproducible sample and measure SNR improvement *while the speaker is
talking*:

```bash
uv run scripts/make_demo_sample.py
roommute --denoise samples/demo_raw.wav samples/demo_cleaned.wav --model <candidate>
uv run scripts/analyse_demo.py
```

**Do not judge on how quiet the gaps get.** Level percentiles over a whole file
reward a model that simply outputs less: a model that mutes between sentences
scores beautifully and is useless. That is precisely how the two runs below
were read as successes before anyone listened to them. Measuring against the
clean reference while the voice is present cannot be gamed that way, because
the near voice has to survive for the error to fall.

Pair it with listening regardless.

## Run 1: what went wrong

Stopped early at epoch 40 (patience 12), best at **epoch 29**, valid 1.00606.
The percentile numbers looked like a landslide, and were not:

| near-voice band vs raw, loudest 5% of windows | 80–300 | 300–800 | 800–2k | 2–4k | 4–8k | 8–16k |
|---|---|---|---|---|---|---|
| DeepFilterNet3 | −1.3 | −1.0 | −0.7 | +0.5 | **+1.7** | +1.0 |
| Run 1 (epoch 29) | −9.3 | −3.0 | −2.6 | −5.1 | **−13.7** | −8.7 |

Every band down, worst in the presence band where DFN3 adds a little. On the
percentile measure it removed 5.3 dB from the near voice at p95 on `kid` where
DFN3 removes 0.1, and **33.9 dB** on `chat`. Meanwhile the noise floor dropped
65.8 dB at p50 against DFN3's 27.0. That combination is the signature of a
model that learned to **suppress rather than separate** — and it is exactly
what the warning above describes, so the warning earned its place.

The suspect was `p_reverb = 0.2`. Push dereverberation hard enough and the
model treats the near voice's *own* room reflections as noise, taking the
brightness with them.

**Do not bundle speed work into an experiment run.** Raising `batch_size` from
64 to 96 is the obvious win — the bottleneck is sequential GRU steps, so bigger
batches buy throughput almost free — but it moves gradient noise and the LR
schedule, and a better result would no longer be attributable to `p_reverb`.
`early_stopping_patience` is safe to change (12 → 8; run 1 spent 11 epochs
after its best proving it was finished), as are `jit` and `torch.compile`,
which do not change the arithmetic.

## Run 2: same failure, so the suspect was wrong

Run 2 set `p_reverb` back to DFN3's 0.1 and changed nothing else affecting the
arithmetic, making it single-variable against run 1's epoch 29. It
over-suppressed in the same way. Whatever costs these runs the near voice's
brightness, it is not `p_reverb` alone.

Three differences from the official DFN3 recipe are still unexplored, and all
three are in the schedule rather than the data:

| | DFN3 | both runs |
|---|---|---|
| `batch_size_scheduling` | `0/16,2/24,5/32,10/64,20/128,40/256` | empty |
| epochs | 120 | 60 |
| `early_stopping_patience` | 25 | 8–12 |

A run without batch-size scheduling sees a very different effective learning
rate over its life, and 60 epochs with patience 8 stops a long way short of
where DFN3 was still improving. Start there before touching the data.

(`40/9999` in upstream's shipped config is harmless: `df.train` caps each
scheduled size at `batch_size`, 64, so the schedule is 16/24/32/64.)

## Run 3: upstream's recipe, our data

Everything above as one command, from Windows:

```bash
wsl -d Ubuntu -u root -- bash /mnt/e/roommute/scripts/training/train.sh
```

[`scripts/training/`](../scripts/training/) downloads the corpora, builds the
RIRs, interferers, speaker splits and HDF5 files into `/root/dfn-data` (each
step stamped in `.done/`, so an interrupted run resumes), writes
`/root/dfn-run3/config.ini` once, and trains. The config is upstream DFN3's own
with only `device` and `log_timings` changed, so if run 3 fails the same way,
the data is to blame and not the recipe.

Budget: ~1.5 h of downloads, ~1 h of prep, then ~7 min per epoch at batch 64
(the batch-16 epochs at the start are slower). Plan on 120 epochs taking two
nights. Rerunning the same command resumes from the latest checkpoint. Each new
batch size makes MIOpen recompile its kernels, which takes minutes, but happens
only four times.

Keep the terminal open, and stop Windows from sleeping. A sleeping host
suspends the WSL VM and the run with it.

To judge it, export the best checkpoint and score it as in
[Judging the result](#judging-the-result):

```bash
uv run scripts/export_dfn3.py models/run3-export //wsl.localhost/Ubuntu/root/dfn-run3
```

