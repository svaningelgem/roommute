# Where the model comes from, and how to rebuild it

Everything here is reproducible from a clean machine. No step depends on a file
that exists on only one PC.

## What RoomMute runs

`models/dfn3/` — upstream DeepFilterNet3's own three-graph export (`enc`,
`erb_dec`, `df_dec`) plus its config, as five ordinary files, loaded by
[`crates/dsp/src/tract.rs`](../crates/dsp/src/tract.rs) through libDF's tract
runner. The installer copies them to `model\` beside the executable and the app
finds them there.

libDF accepts only a `.tar.gz` — `DfParams` exposes `new(path)` and
`from_bytes(buf)` and nothing else — so RoomMute packs the directory in memory
on load. That is libDF's interface, not something this program needs on disk,
and it is not a reason for an install directory to contain a tarball. Passing
a `.tar.gz` still works, since that is the form upstream distributes.

One command rebuilds it, from the published checkpoint, on any machine:

```bash
uv run scripts/export_dfn3.py
```

No clone, no GPU, no manual pip, no Rust toolchain — [uv](https://docs.astral.sh/uv/)
resolves the pinned environment from the script's own header. It takes about a
minute of CPU work. Two of the three graphs come out **byte-identical to
upstream's release** (`enc.onnx`, `erb_dec.onnx`); `df_dec.onnx` differs by
1.4 KB. That is the evidence the export is faithful.

`models/model.onnx` is a second, older artefact kept for compatibility: it is a
single-file streaming graph that installs from before the switch have beside
them, and `model_path` can still point at one. It is **not** what ships, and it
is not reproducible — see [the caveat on `model.onnx`](#the-caveat-on-modelonnx).

## Licence: we may redistribute

DeepFilterNet is dual-licensed **MIT or Apache-2.0**, and the statement in its
README covers the whole repository, including `models/` where the weights live.
There is no carve-out for the checkpoints. Both licences permit redistribution
provided the licence text and copyright notice travel with it (`LICENSE-MIT`,
`LICENSE-APACHE`, © 2021 Hendrik Schröter).

So shipping the weights inside RoomMute is allowed, which is why the app works
on first run with nothing to download.

One caveat worth stating rather than hiding: weights are shaped by their
training data, and whether a dataset's licence reaches through to the trained
weights is unsettled in general. Upstream distributes these weights under
MIT/Apache-2.0 without qualification, and that is the licence being relied on.

### The trap in the export

The export dies with:

    Failed to import monkeytype. Please install it via
    $ pip install MonkeyType

and installing MonkeyType does not help, because that is not the problem. `df`
imports torch, torch loads the system `libstdc++`, and miniforge's `libicui18n`
then needs a newer `CXXABI` than that provides — so `sqlite3` fails to load.
Upstream guards its export with `import monkeytype`, which imports `sqlite3`
transitively, so an unrelated ABI mismatch surfaces as a missing package.

Fix: `import sqlite3` **before** anything imports torch. That is most of what
`scripts/export_dfn3.py` does beyond calling upstream's own script.

## The contract

| graph | in | out |
|---|---|---|
| `enc` | `feat_erb[1,1,S,32]`, `feat_spec[1,2,S,96]` | `e0..e3`, `emb`, `c0`, `lsnr` |
| `erb_dec` | `emb`, `e0..e3` | `m` — ERB gains, 32 bands |
| `df_dec` | `emb`, `c0` | `coefs[S,96,10]` — 5 complex taps per bin |

The STFT front-end and both output stages live outside the graphs, in the host.
Per frame:

1. `analysis(480 samples)` -> spectrum, 481 bins
2. `feat_erb` = `erb_norm(erb(spec))`, `feat_spec` = `unit_norm(spec[..96])`
3. run `enc`, then `erb_dec` and `df_dec`
4. **ERB mask**: expand the 32 gains back over the 481 bins by band width and
   multiply (`apply_interp_band_gain`)
5. **Deep filter**: for the lowest 96 bins, replace the value with a complex FIR
   across 5 frames of the *noisy* history, taps from `coefs`
6. `synthesis(spec)` -> 480 samples

`df_lookahead = 2`, so output for frame *t* uses noisy frames *t+2 … t-2* and
the result lags the input by two hops (960 samples).

libDF implements all of it, and `tract.rs` calls into libDF rather than
reimplementing it. [`crates/dsp/src/dfn_frontend.rs`](../crates/dsp/src/dfn_frontend.rs)
is a standalone Rust port of steps 1–2, verified against
`df.enhance.df_features()` on real audio; it is not on the path the app runs.

## Stage switching is off, deliberately

libDF's runner picks a different treatment for every 10 ms frame from the
model's own SNR estimate:

| lsnr | treatment |
|---|---|
| < `min_db_thresh` | output zeroed |
| > `max_db_erb_thresh` | passed through untouched |
| > `max_db_df_thresh` | ERB mask only |
| otherwise | ERB mask + deep filter |

On speech with background chatter the estimate sits near a boundary constantly,
so neighbouring frames get wholly different processing and the seams are
audible — the voice cracks. On a 60-second sample it also produces **17 seconds
of exact digital silence**.

`tract.rs` pushes the thresholds out of reach (`NEVER` / `ALWAYS`) so one
treatment stays in force. This is not a tuning compromise: the single-file
DeepFilterNet3 export everyone compares against has no equivalent switching —
it is one graph that always runs both stages — so disabling it is what makes
the output *match* the reference rather than approximate it.

| variant | correlation vs reference | silent seconds |
|---|---|---|
| stage switching on | 0.9295 | 17 |
| **switching off (shipped)** | **1.0000** | 1 |
| switching off + 25 dB limit | 0.9997 | 0 |

End to end, the shipped build differs from the reference by at most **one
sample step out of 32768** — float rounding through a differently shaped graph,
inaudible.

The 25 dB variant is worth remembering: it never emits digital silence at all,
at the cost of a little background. If the reference's silences ever feel like
a dropped call, that is the knob — `attenuation_db` in `config.toml`.

## The caveat on `model.onnx`

Upstream publishes DeepFilterNet3 as three ONNX files, because the STFT
front-end is meant to live in the host application. `models/model.onnx` is a
single streaming graph:

    input_frame[480], states[45304], atten_lim_db
      -> enhanced_audio_frame[480], new_states[45304], lsnr

Somebody repackaged upstream's graphs into that form, and who is unknown. The
weights are provably genuine — 23 of its large tensors match the published
`model_120.ckpt.best` byte for byte, including `enc.df_fc_emb.0.weight` and
`enc.emb_gru.linear_in.0.weight`, and the rest are ONNX Runtime's fused
conv/bias forms expected from a pytorch 1.13.1 graph after ORT optimisation.

So it is not a trust problem; it is a reproducibility one. It cannot be
regenerated from source, which is the reason `models/dfn3/` exists and is
what ships.

## Re-checking a change

`scripts/onnx3_runner.py` runs the three graphs over a whole file in Python and
applies both stages — useful for isolating which stage a discrepancy is in.
`scripts/diag_align.py <noisy.wav> <reference.wav>` scans lags before
correlating, so a pure delay cannot be mistaken for a broken stage; that is how
`df_lookahead = 2` was pinned down.

For end-to-end judgement, prefer the reproducible sample: `scripts/make_demo_sample.py`
builds it from openly licensed audio and `scripts/analyse_demo.py` measures SNR
improvement while the speaker is talking. See
[testing.md](testing.md) and the README.
