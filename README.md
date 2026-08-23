# RoomMute

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#licence)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows%2010%2F11-0078d4.svg)](#install)
[![Rust](https://img.shields.io/badge/built%20with-Rust-orange.svg)](https://www.rust-lang.org/)
[![Model: DeepFilterNet3](https://img.shields.io/badge/model-DeepFilterNet3-purple.svg)](#the-model)
[![No network code](https://img.shields.io/badge/network%20code-none-brightgreen.svg)](#privacy)

### Your voice goes to the call. The rest of the room doesn't.

Everything removes fans and keyboard clatter. RoomMute removes **the people talking around you** — the child in the next room, a partner on the phone, the desk behind you — and leaves your own voice untouched.

It runs in the tray, uses about 4% of one core, and needs nothing from the internet.

> **Status: pre-alpha.** Windows 10/11, MSVC toolchain. It works and it is in daily use, but it has not been through many hands yet.

---

## The problem nobody else solves

Fans, hum and typing are *stationary* noise. Every denoiser handles those; Windows does some of it for free.

Your neighbour's voice is a different problem — because every speech-enhancement model is trained to **preserve speech**, and your neighbour is speech.

### Hear it

Twenty seconds, everything at once: ventilation, cafeteria babble, street traffic and a neighbour through the wall, all at the same time. You hear the raw microphone first, then the same twenty seconds cleaned. **Orange is whichever one you are hearing**, so you can watch the difference at the moment it happens.

https://github.com/user-attachments/assets/40c809fc-33a0-4098-a334-eb2d9c3184c0

> **Unmute it.** GitHub always starts videos muted, and a silent noise-cancellation demo proves very little.

<details>
<summary><b>▶ The full two-minute tour</b> — all six environments, one at a time</summary>

<video src="https://github.com/user-attachments/assets/4c229f14-d712-4d34-ba94-69c7313be1ff" controls></video>

Clean speech first as a reference, then ventilation, cafeteria babble, street traffic, a neighbour through the wall, and finally everything together. Each one runs twice: input, then output.

</details>

Audio only: **[before](samples/demo_raw.mp3)** · **[after](samples/demo_cleaned.mp3)** · **[RNNoise, for comparison](samples/demo_rnnoise.mp3)**. Both videos are committed in [`samples/`](samples/), and [Measuring it](#measuring-it-and-rebuilding-the-proof) has the numbers and the scripts that rebuild all of it.

---

## Install

Two things to install, and **one of them needs administrator rights**:

1. **RoomMute.** [Download the installer](https://github.com/svaningelgem/roommute/releases) and run it. It installs into your own user profile, so Windows raises no prompt, and it adds no driver and no service. The model ships inside it — nothing is fetched on first run.
2. **A virtual audio cable**, which is what lets other apps hear the cleaned microphone. [VB-Cable](https://vb-audio.com/Cable/) is free and takes about two minutes. It installs a sound device system-wide, so **Windows will ask for administrator permission** for this one. RoomMute detects that it is missing on first run and offers to open the download page.

Then, in Zoom, Teams, Discord, OBS or your browser: **pick `CABLE Output` as your microphone.** That's it.

If you would rather not install a driver at all, RoomMute still works offline on files — see [Try it without touching your audio setup](#try-it-without-touching-your-audio-setup).

<details>
<summary><b>Build from source instead</b></summary>

```powershell
git clone https://github.com/svaningelgem/roommute
cd roommute
cargo build --release
```

Every backend compiles in — there are no cargo features to choose, because the only thing a switch achieved was shipping a build that quietly could not load the model. The model is in `models/dfn3/` as ordinary files — three ONNX graphs and a readable `config.ini` — so a fresh clone gives you a working binary.

Requires the MSVC toolchain (`rustup default stable-x86_64-pc-windows-msvc`).
</details>

---

## Using it

Left-click the tray icon to toggle denoising. The icon shows the state at a glance:

| | |
|---|---|
| **teal** | on, cleaning your microphone |
| **orange** | bypassed, passing audio straight through |
| **⚠ badge** | audio is not flowing — the device went away, and it is trying to recover |

Right-click for the menu: pick a microphone, switch backend, toggle start-with-Windows, open the log folder, or **Help** — which repeats the first-run message naming the microphone in use and the device to select in Zoom, Discord and the rest.

**Microphones are remembered in preference order.** Click one and it becomes first choice; the rest shift down. Unplug it mid-call and the next one down takes over on its own, rather than the app stopping to ask. Plug it back in and it is picked up again.

A cable's own output is greyed out in the picker, because selecting it would have RoomMute record from the very cable it writes into.

---

## Try it without touching your audio setup

You do not have to install a cable, or route anything, to hear what it does:

```powershell
# Record 10 seconds from your microphone
.\roommute.exe --record 10 test.wav

# Clean it up
.\roommute.exe --denoise test.wav clean.wav
```

It prints what it did:

```
noise_floor="-67.1 -> -100.2 dB (-33.2)"  speech="-25.4 -> -26.0 dB (-0.7)"  rtf="0.038"
```

`noise_floor` is the quietest tenth of the file and `speech` the loudest twentieth. So: between words it went to digital silence, your own voice lost 0.7 dB, and it ran at 26× realtime.

Useful as a smoke test. Do not read too much into the first number, though: it is the level between words, and a plain noise gate scores well on it too. [Measuring it](#measuring-it-and-rebuilding-the-proof) does the honest version.

---

## Privacy

**There is no network code in this program.** No telemetry, no update check, no model download, no crash reporting. The binary does not link a HTTP client.

Your microphone audio goes to the DSP thread and into the virtual cable. Nowhere else. This started as a security audit of an audio app, and that property is deliberate — the model is bundled precisely so that first run does not need to fetch anything.

The only things RoomMute writes are `%APPDATA%\RoomMute\config.toml` and a log file that rotates at 5 MB. The log records device names, so it is worth a glance before pasting into a bug report.

---

## The model

DeepFilterNet3, by [Hendrik Schröter](https://github.com/Rikorose/DeepFilterNet) — dual MIT/Apache-2.0, which is why we can ship it. Attribution travels with it in [`models/NOTICE.md`](models/NOTICE.md).

We do not ship someone's prebuilt file. The model is **exported from the published checkpoint by a script in this repo**, so every artefact in the chain can be rebuilt:

```bash
uv run scripts/export_dfn3.py
```

That fetches the checkpoint, converts it, and writes the model the app loads. No clone, no Rust toolchain, no manual pip — [uv](https://docs.astral.sh/uv/) handles the environment. [`docs/model-pipeline.md`](docs/model-pipeline.md) covers how it works and the several traps involved.

Two backends, switchable from the tray while running:

| backend | a neighbour talking | a busy room | CPU | needs |
|---|---|---|---|---|
| **DeepFilterNet3** (default) | **+3.6 dB** | **+9.6 dB** | ~4% of one core | nothing — bundled |
| **RNNoise** | −0.0 dB | +6.8 dB | ~0.3% of one core | nothing — embedded |

RNNoise earns its place: ~50× less compute, no model file at all, and genuinely good at fans and hiss. It simply cannot touch a voice in the next room, because it was trained not to. RoomMute falls back to it automatically if no model is found, so it always does something useful.

## Measuring it, and rebuilding the proof

Every number here comes from one 60-second sample, and nothing about it is private: it is assembled from openly licensed audio by scripts in this repo, and these four commands rebuild it byte for byte on any machine.

```bash
uv run scripts/make_demo_sample.py     # LibriSpeech + DEMAND, ~1.3 GB once
roommute --denoise samples/demo_raw.wav samples/demo_cleaned.wav
roommute --denoise samples/demo_raw.wav samples/demo_rnnoise.wav --rnnoise
uv run scripts/analyse_demo.py         # the table and the spectrogram
uv run scripts/make_demo_video.py      # the two videos (needs ffmpeg)
```

One voice throughout, six ten-second segments, a different problem in each:

![spectrogram of the demo, before and after](docs/demo-spectrogram.png)

| segment | DeepFilterNet3 | RNNoise |
|---|---|---|
| office ventilation | +2.6 dB | **+7.4 dB** |
| cafeteria babble | **+8.0 dB** | +4.5 dB |
| street traffic | **+10.2 dB** | +7.5 dB |
| **neighbour through wall** | **+3.6 dB** | **−0.0 dB** |
| everything at once | **+9.6 dB** | +6.8 dB |

Signal-to-noise improvement measured **while the person is speaking** — the only moment that is hard, because you cannot solve it by muting.

Read the last row first: against a competing voice **RNNoise achieves nothing at all**, and it is not a broken build — it is a 2017 model trained on noise, doing exactly what it was designed to do. Read the first row second: for plain ventilation hum RNNoise is *nearly three times better* than the model we ship, at 1/50th the CPU. Both facts are in the box.

<details>
<summary><b>Why not just measure the quiet bits between sentences?</b></summary>

Because silence is free. Scoring the gaps rewards whichever model mutes hardest, and by that metric RNNoise wins every row here — including against a competing voice, where it removes nothing whatsoever. A denoiser that outputs digital silence between your sentences would score perfectly and be useless.

A neighbour is only a problem while *you* are talking, because that is when you cannot simply mute. So that is when it is measured. The mixture is synthetic for exactly this reason: `demo_clean_reference.wav` holds the near voice on its own, which makes the difference between it and the output precisely the part that should not be there.

[`docs/training.md`](docs/training.md) has the expensive version of the same trap.
</details>

Caveat on the sample: LibriSpeech is 16 kHz, so nothing above 8 kHz is real. DEMAND has no airshow, so "traffic" and "cafeteria" stand in for the outdoor and crowd cases; inventing a synthetic aeroplane would have proved nothing.

---

## Configuration

`%APPDATA%\RoomMute\config.toml` — created on first run, edited live by the tray menu.

```toml
microphones = ["Microphone (Yeti)", "Microphone (Webcam)"]  # preference order
output_device = ""        # empty = find the virtual cable automatically
enabled = true            # master on/off
use_onnx = true           # true = DeepFilterNet3, false = RNNoise. The key
                          # is named for ONNX; the model runs through tract
attenuation_db = 100.0    # how hard to suppress; 100 = no limit, 25 = gentler
model_path = ""           # empty = use the model beside the executable
welcomed = true           # the one-time welcome has been shown; Help reopens it
```

Devices are named, not numbered. Endpoint IDs are opaque GUIDs that change every time you replug a device — names survive.

**`attenuation_db` is the one worth knowing about.** At 100 the model may suppress as much as it likes, which occasionally means a passage of exact digital silence between sentences. If that reads as a dropped call to the person listening, try 25: a little background stays, and the output never goes fully dead.

---

## How it works

```
physical mic ─► WASAPI capture ─► ring A ─► DSP thread ─► ring B ─► WASAPI render ─► CABLE Input
                                                                                          │
                                                     other apps select "CABLE Output" ◄────┘
```

Three dedicated MMCSS "Pro Audio" threads, lock-free SPSC ring buffers with 80 ms of headroom, 480-sample (10 ms) frames end to end — the native frame size for both models, so nothing is reblocked inside the DSP path.

A watchdog watches both halves. WASAPI reports a dead stream by simply never signalling again — no error, no callback — so the honest signal is whether frames are still moving. If either the capture or the output side stops, the icon badges and the pipeline is rebuilt quietly, without a dialog interrupting your call.

---

## Contributing

Every pull request runs format, lint, build, test and a coverage ratchet on Windows.

```powershell
cargo fmt --all -- --check
cargo clippy --all-targets --all-features
cargo test --all-features
```

[`docs/signing.md`](docs/signing.md) covers release signing, and why it does not make the SmartScreen warning go away.

[`docs/testing.md`](docs/testing.md) explains the ratchet — a floor that may be raised but never lowered — and is honest about which parts cannot be covered without hardware, and about two fixes that have no test at all and why.

Tests are written **test-first**: write it, watch it fail for the right reason, then fix. A regression test that has never been seen to fail is not a regression test.

---

## Not included

- macOS / Linux (the `audio-io` crate is Windows-only; the rest is portable)
- Far-end denoising — only your microphone is cleaned, not what you hear
- Acoustic echo cancellation
- Auto-update

---

## Credits

**Forked from [Yashsomalkar/noisegate](https://github.com/Yashsomalkar/noisegate)**, which is where the WASAPI capture and render core came from. This repository is where the work continued; the original author is credited in `Cargo.toml` alongside me.

- **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)** — Hendrik Schröter et al. The model, and the tract runner that streams it correctly.
- **[RNNoise](https://gitlab.xiph.org/xiph/rnnoise)** — Jean-Marc Valin / Xiph.Org, via **[`nnnoiseless`](https://github.com/jneem/nnnoiseless)** by jneem.
- **[VB-Cable](https://vb-audio.com/Cable/)** — VB-Audio. The free virtual driver every Windows routing app depends on.
- **[`windows`](https://github.com/microsoft/windows-rs)**, **[`tray-icon`](https://github.com/tauri-apps/tray-icon)**, **[`winit`](https://github.com/rust-windowing/winit)**, **[`ringbuf`](https://github.com/agerasev/ringbuf)**, **[`ort`](https://github.com/pykeio/ort)**.
- Inspired by **[NoiseTorch](https://github.com/noisetorch/NoiseTorch)**, the Linux equivalent.

## Licence

Code: **MIT or Apache-2.0**, your choice.

The bundled DeepFilterNet3 weights are MIT/Apache-2.0 — redistributable, commercial use included. RNNoise is BSD. There is nothing here you cannot ship.

The demo audio in `samples/demo_*.mp3` is **not** MIT. It is built from [LibriSpeech](https://www.openslr.org/12/) (CC BY 4.0) and [DEMAND](https://zenodo.org/records/1227121), and is redistributed here under **CC BY-SA 3.0** — full attribution in [`samples/demo_ATTRIBUTION.txt`](samples/demo_ATTRIBUTION.txt). DEMAND's licence is stated inconsistently by its own sources (Zenodo says CC BY 4.0, the paper says CC BY-SA 3.0), so the share-alike reading is the one applied.

## Issues & discussion

- [Report a bug or request a feature](https://github.com/svaningelgem/roommute/issues)
- [Discussions](https://github.com/svaningelgem/roommute/discussions) for questions and ideas
