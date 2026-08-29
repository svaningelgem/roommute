//! Pipeline glue: capture → ring A → DSP thread → ring B → render.
//!
//! Two SPSC ring buffers connect the three audio threads. We size the rings
//! at 8 frames (~80 ms) — enough headroom to absorb a scheduler hiccup,
//! small enough that we don't hide actual problems.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::HeapRb;
use tracing::{info, warn};

use audio_io::devices::DeviceList;
use audio_io::wasapi_capture::{Frame, FrameSink};
use audio_io::wasapi_render::FrameSource;
use dsp::{DenoiserHost, Stats};

use crate::config::Config;
use crate::parking_lot_compat::RwLock;

const RING_FRAMES: usize = 8;

/// Where the pipeline gets and puts its audio.
///
/// Production is WASAPI. Tests substitute a fake so the whole ring-buffer,
/// DSP-thread and shutdown path can run with no sound card present — which is
/// the only way it gets exercised on CI, and the only way to make a device
/// "disappear" on demand to test the stall handling.
///
/// The handles come back boxed because the pipeline only holds them to keep
/// the audio threads alive; dropping them stops the audio. Boxing keeps the
/// generics from leaking into every caller of `Pipeline`.
pub trait AudioIo {
    fn start_capture(&self, device_id: &str, sink: Box<dyn FrameSink>) -> Result<Box<dyn Send>>;
    fn start_render(&self, device_id: &str, source: Box<dyn FrameSource>) -> Result<Box<dyn Send>>;
}

/// The real thing.
pub struct Wasapi;

impl AudioIo for Wasapi {
    fn start_capture(&self, device_id: &str, sink: Box<dyn FrameSink>) -> Result<Box<dyn Send>> {
        let c = audio_io::WasapiCapture::start(device_id, sink).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Box::new(c))
    }

    fn start_render(&self, device_id: &str, source: Box<dyn FrameSource>) -> Result<Box<dyn Send>> {
        let r = audio_io::WasapiRender::start(device_id, source).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Box::new(r))
    }
}

pub struct Pipeline {
    /// Held to keep the audio threads alive. Dropping these stops them.
    #[allow(dead_code)]
    capture: Box<dyn Send>,
    #[allow(dead_code)]
    render: Box<dyn Send>,
    #[allow(dead_code)]
    dsp_thread: Option<std::thread::JoinHandle<()>>,

    bypass: Arc<AtomicBool>,
    stats: Arc<Stats>,
    /// See [`Pipeline::render_polls`].
    render_polls: Arc<AtomicU64>,
    denoiser_name: &'static str,
    /// Used to ask the DSP thread to exit cleanly when we drop.
    shutdown: Arc<AtomicBool>,
}

/// DSP threads currently alive.
///
/// An instrument, not a statistic. The thread owns the denoiser — a whole
/// DeepFilterNet3 graph — and polls every 2 ms, so one that outlives its
/// pipeline costs both memory and a core forever. Counting them makes
/// "starting audio and failing must leave nothing behind" a thing a test can
/// assert instead of a thing to be careful about.
pub(crate) static DSP_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Decrements on the way out however the thread ends, panic included.
struct ThreadCount;

impl Drop for ThreadCount {
    fn drop(&mut self) {
        DSP_THREADS.fetch_sub(1, Ordering::Release);
    }
}

impl Pipeline {
    /// Start against the real sound card.
    pub fn start(cfg: Arc<RwLock<Config>>) -> Result<Self> {
        let (input_ids, output_id) = Self::resolve_devices(&cfg.read().unwrap())?;
        Self::build_ranked(&Wasapi, cfg, &input_ids, &output_id)
    }

    /// Pick the devices to use. Split out from [`Pipeline::build`] because it
    /// is the only part that needs a real sound card, which keeps the wiring
    /// testable.
    fn resolve_devices(snapshot: &Config) -> Result<(Vec<String>, String)> {
        let devices = DeviceList::enumerate().context("enumerating audio devices")?;
        Self::pick_devices(&devices, snapshot)
    }

    /// Choose input and output from a device inventory. Separate from the
    /// enumeration so the rules — ranking, the feedback-loop guard, failing
    /// closed when there is no cable — can be tested without a sound card.
    fn pick_devices(devices: &DeviceList, snapshot: &Config) -> Result<(Vec<String>, String)> {
        // Config names devices; ids are an internal detail resolved here,
        // fresh on every start and every restart.
        //
        // Every candidate, not just the winner: a device can enumerate and
        // still refuse to open, and the caller tries them in turn.
        let candidates = devices.capture_candidates(&snapshot.microphones);
        let input = *candidates
            .first()
            .ok_or_else(|| anyhow::anyhow!("no microphone is available"))?;
        // Say so when the preferred one wasn't there, so a silent downgrade to
        // the laptop's built-in mic isn't a mystery later.
        if let Some(preferred) = snapshot.microphones.first() {
            if !audio_io::devices::same_device_name(preferred, &input.friendly_name) {
                warn!(
                    using = %input.friendly_name,
                    preferred = %preferred,
                    "preferred microphone unavailable; using the next one down"
                );
            }
        }

        // Installing a virtual cable usually makes it the default *capture*
        // device too. Left alone, "follow the Windows default" then means
        // recording from the same cable we render into — the cable feeding
        // itself, with no real microphone anywhere in the loop.
        if let Some(product) = input.virtual_cable_output() {
            anyhow::bail!(
                "the selected microphone is {product}'s own output, which is where RoomMute \
                 sends audio — routing it back in would loop. Pick a real microphone from the \
                 tray menu"
            );
        }

        // Auto-detection failing must not fall back to the default render
        // device: that would play the microphone out of whatever speakers,
        // Bluetooth headset or meeting-room HDMI display happens to be
        // default. Fail closed and let the user fix the routing.
        let output = devices
            .resolve_render(&snapshot.output_device)
            .with_context(|| {
                format!(
                    "no virtual audio cable is installed (looked for {}). Cleaned audio needs \
                     one so other apps can hear it as a microphone. Install VB-Cable, or name \
                     one in output_device",
                    audio_io::devices::known_cable_products().join(", ")
                )
            })?
            .clone();
        // The feedback-loop guard above tested the first choice; the rest of
        // the list still has to be checked, or a fallback could route the
        // cable into itself.
        let input_ids: Vec<String> = candidates
            .iter()
            .filter(|d| d.virtual_cable_output().is_none())
            .map(|d| d.id.clone())
            .collect();
        info!(
            mic = %input.friendly_name,
            fallbacks = input_ids.len().saturating_sub(1),
            to = %output.friendly_name,
            "using devices"
        );
        Ok((input_ids, output.id.clone()))
    }

    /// Wire capture -> ring -> DSP -> ring -> render, against whatever audio
    /// backend is handed in.
    /// Try each microphone in turn, keeping the first that actually opens.
    ///
    /// A device being listed does not mean it can be used: an endpoint whose
    /// effects chain is broken enumerates perfectly and then fails to open,
    /// which is how one reporter ended up staring at a dialog with three
    /// working microphones plugged in. Picking the next candidate is exactly
    /// what the preference list is for, and it already happens when a device
    /// disappears mid-call — it just never happened at startup.
    ///
    /// Only when every candidate has failed does the error reach the user, and
    /// it is the last real failure rather than a summary.
    fn build_ranked(
        io: &dyn AudioIo,
        cfg: Arc<RwLock<Config>>,
        input_ids: &[String],
        output_id: &str,
    ) -> Result<Self> {
        let mut last: Option<anyhow::Error> = None;
        for id in input_ids {
            match Self::build(io, cfg.clone(), id, output_id) {
                Ok(p) => {
                    if last.is_some() {
                        info!(device = %id, "started on a fallback microphone");
                    }
                    return Ok(p);
                }
                Err(e) => {
                    warn!(device = %id, error = %format!("{e:#}"), "microphone would not open; trying the next");
                    last = Some(e);
                }
            }
        }
        Err(last.unwrap_or_else(|| anyhow::anyhow!("no microphone is available")))
    }

    fn build(
        io: &dyn AudioIo,
        cfg: Arc<RwLock<Config>>,
        input_id: &str,
        output_id: &str,
    ) -> Result<Self> {
        let snapshot = cfg.read().unwrap().clone();

        // Build the rings.
        let (prod_a, mut cons_a) = HeapRb::<Frame>::new(RING_FRAMES).split();
        let (mut prod_b, cons_b) = HeapRb::<Frame>::new(RING_FRAMES).split();

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_dsp = shutdown.clone();

        // DSP thread: pulls from ring A, processes, pushes to ring B.
        // Capture sink: pushes into ring A.
        struct Sink<P: Producer<Item = Frame> + Send> {
            prod: P,
        }
        impl<P: Producer<Item = Frame> + Send> audio_io::wasapi_capture::FrameSink for Sink<P> {
            fn on_frame(&mut self, frame: &Frame) {
                if self.prod.try_push(*frame).is_err() {
                    // DSP behind — overwrite oldest by popping (we don't
                    // have direct access here; the simplest cheap option is
                    // to just drop. Audible as a tiny click; far better
                    // than blocking the audio engine.)
                    tracing::warn!("ring A full; dropping captured frame");
                }
            }
            fn on_glitch(&mut self, flags: u32) {
                tracing::warn!(flags, "capture glitch reported by audio engine");
            }
        }

        let capture = io.start_capture(input_id, Box::new(Sink { prod: prod_a }))?;

        // Render source: pulls from ring B.
        struct Source<C: Consumer<Item = Frame> + Send> {
            cons: C,
            /// `None` = never logged, so the first underrun reports at once.
            /// Not `Instant::now() - 60s`: subtracting from a fresh Instant
            /// panics when the machine has been up for less than the offset,
            /// and starting with Windows means booting is exactly when that
            /// happens.
            last_underrun_log: Option<std::time::Instant>,
            /// Bumped on every poll, which is the render thread's pulse: the
            /// engine only asks for audio while that thread is alive and its
            /// device is signalling. See `Pipeline::render_polls`.
            polls: Arc<AtomicU64>,
        }
        impl<C: Consumer<Item = Frame> + Send> audio_io::wasapi_render::FrameSource for Source<C> {
            fn next_frame(&mut self) -> Option<Frame> {
                self.polls.fetch_add(1, Ordering::Relaxed);
                self.cons.try_pop()
            }
            fn on_underrun(&mut self) {
                let due = self
                    .last_underrun_log
                    .is_none_or(|t| t.elapsed() > std::time::Duration::from_secs(5));
                if due {
                    tracing::warn!("render underrun (cleaned audio not arriving from DSP)");
                    self.last_underrun_log = Some(std::time::Instant::now());
                }
            }
        }

        let render_polls = Arc::new(AtomicU64::new(0));
        let render = io.start_render(
            output_id,
            Box::new(Source {
                cons: cons_b,
                last_underrun_log: None,
                polls: render_polls.clone(),
            }),
        )?;

        // Built after the devices, not before. Loading DeepFilterNet3 takes
        // a few hundred milliseconds; doing it first meant every failed start
        // paid for a model it then threw away. On the install that reported
        // this, the watchdog retried three thousand times against a microphone
        // that could not open, and each retry loaded the model twice.
        // DSP setup.
        let model_path = snapshot.active_model();
        let denoiser = dsp::build_denoiser(model_path.as_deref(), snapshot.attenuation_db)
            .context("loading denoiser")?;
        let denoiser_name = denoiser.name();
        let (mut host, bypass, stats) = DenoiserHost::new(denoiser);
        bypass.store(!snapshot.enabled, Ordering::Relaxed);

        // The DSP thread is spawned only once both devices are open.
        //
        // It used to start before them, and it owns the denoiser — a whole
        // DeepFilterNet3 graph. When opening a device failed, `?` returned
        // without ever setting the shutdown flag, so the thread stayed alive
        // for the life of the process, polling every 2 ms and holding the
        // model. The watchdog retried every few seconds and leaked another
        // one each time: a real install reached 7 GB and a permanent 11% of
        // the CPU, with 3,085 of them running.
        //
        // Nothing above this point owns a thread, so every `?` in between is
        // now just a drop.
        DSP_THREADS.fetch_add(1, Ordering::Release);
        let dsp_thread = std::thread::Builder::new()
            .name("roommute-dsp".into())
            .spawn(move || {
                let _count = ThreadCount;
                #[cfg(windows)]
                let _mmcss = audio_io::mmcss_pro_audio_for_current_thread();
                // An empty ring is not news: this thread polls every 2 ms
                // while frames arrive every 10 ms, so it finds nothing most
                // of the time even when everything is healthy. Only a long
                // gap means anything, and that is what gets reported.
                let mut last_frame_at = std::time::Instant::now();
                let mut reported_gap = false;
                let mut last_drop_log: Option<std::time::Instant> = None;
                while !shutdown_dsp.load(Ordering::Acquire) {
                    let mut frame = match cons_a.try_pop() {
                        Some(f) => f,
                        None => {
                            if !reported_gap
                                && last_frame_at.elapsed() > std::time::Duration::from_secs(2)
                            {
                                warn!(
                                    gap_ms = last_frame_at.elapsed().as_millis() as u64,
                                    "no audio from the microphone — it may have been unplugged, \
                                     muted, or blocked in Windows privacy settings"
                                );
                                reported_gap = true;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(2));
                            continue;
                        }
                    };
                    if reported_gap {
                        info!("microphone audio resumed");
                        reported_gap = false;
                    }
                    last_frame_at = std::time::Instant::now();
                    if let Err(e) = host.process(&mut frame) {
                        warn!(error = %e, "denoiser error; passing frame through");
                    }
                    if prod_b.try_push(frame).is_err() {
                        // Render is behind — drop. Audible as a click; better
                        // than blocking the DSP thread.
                        //
                        // Throttled: when the render side dies the ring stays
                        // full forever, and one line per frame is 100 a second
                        // for as long as the app runs.
                        let due = last_drop_log.is_none_or(|t: std::time::Instant| {
                            t.elapsed() > std::time::Duration::from_secs(5)
                        });
                        if due {
                            warn!(
                                "ring B full; dropping frames (is the output device still there?)"
                            );
                            last_drop_log = Some(std::time::Instant::now());
                        }
                    }
                }
            })
            .context("spawn dsp thread")?;

        Ok(Self {
            capture,
            render,
            dsp_thread: Some(dsp_thread),
            bypass,
            stats,
            render_polls,
            denoiser_name,
            shutdown,
        })
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.bypass.store(!enabled, Ordering::Relaxed);
    }

    pub fn is_enabled(&self) -> bool {
        !self.bypass.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    pub fn denoiser_name(&self) -> &'static str {
        self.denoiser_name
    }

    /// Frames processed so far. The tray watches this: a counter that stops
    /// advancing means the capture stream died, which WASAPI reports by
    /// simply never signalling its event again. Silence still produces
    /// frames, so this distinguishes "quiet room" from "dead device".
    pub fn frames_processed(&self) -> u64 {
        self.stats.frames.load(Ordering::Relaxed)
    }

    /// The render thread's pulse, for the same watchdog.
    ///
    /// `frames_processed` only proves the *capture* half is alive: it is
    /// bumped by the DSP thread as it drains ring A. If the output endpoint
    /// dies — the cable reconfigured in Windows sound settings, say — capture
    /// and DSP carry on happily and that counter keeps climbing while nothing
    /// reaches the far end of the call at all. This one stops when the render
    /// thread stops asking for audio, which is the only honest signal that
    /// half of the pipeline is still running.
    pub fn render_polls(&self) -> u64 {
        self.render_polls.load(Ordering::Relaxed)
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(t) = self.dsp_thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A sound card that isn't one.
    ///
    /// Capture replays a fixed signal at full speed and then stops delivering,
    /// which is also how a real device behaves when it's unplugged — so the
    /// same fake exercises both the happy path and the stall.
    struct FakeIo {
        input: Vec<Frame>,
        rendered: Arc<Mutex<Vec<Frame>>>,
        stop: Arc<AtomicBool>,
    }

    impl FakeIo {
        fn new(input: Vec<Frame>) -> Self {
            Self {
                input,
                rendered: Arc::new(Mutex::new(Vec::new())),
                stop: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    struct FakeHandle(Arc<AtomicBool>, Option<std::thread::JoinHandle<()>>);

    impl Drop for FakeHandle {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
            if let Some(t) = self.1.take() {
                let _ = t.join();
            }
        }
    }

    impl AudioIo for FakeIo {
        fn start_capture(&self, _id: &str, mut sink: Box<dyn FrameSink>) -> Result<Box<dyn Send>> {
            let frames = self.input.clone();
            let stop = self.stop.clone();
            let t = std::thread::spawn(move || {
                for f in frames {
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    sink.on_frame(&f);
                    // Roughly real time so the ring doesn't overflow; the DSP
                    // thread consumes at its own pace.
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                // Then nothing more, like a device that vanished.
                while !stop.load(Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            });
            Ok(Box::new(FakeHandle(self.stop.clone(), Some(t))))
        }

        fn start_render(
            &self,
            _id: &str,
            mut source: Box<dyn FrameSource>,
        ) -> Result<Box<dyn Send>> {
            let out = self.rendered.clone();
            let stop = self.stop.clone();
            let t = std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    match source.next_frame() {
                        Some(f) => out.lock().unwrap().push(f),
                        None => std::thread::sleep(std::time::Duration::from_millis(1)),
                    }
                }
            });
            Ok(Box::new(FakeHandle(self.stop.clone(), Some(t))))
        }
    }

    /// Deterministic tone plus hiss. Generated rather than checked in: no
    /// binary in git, no licence to honour, and identical on every machine.
    pub(super) fn test_signal(frames: usize) -> Vec<Frame> {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        (0..frames)
            .map(|i| {
                let mut f = [0.0f32; dsp::FRAME_SAMPLES];
                for (j, s) in f.iter_mut().enumerate() {
                    let t = (i * dsp::FRAME_SAMPLES + j) as f32 / 48_000.0;
                    // xorshift, so the "noise" is reproducible.
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let hiss = (seed >> 40) as f32 / 8_388_608.0 - 1.0;
                    *s = (t * 220.0 * std::f32::consts::TAU).sin() * 0.25 + hiss * 0.02;
                }
                f
            })
            .collect()
    }

    pub(super) fn config(enabled: bool) -> Arc<RwLock<Config>> {
        Arc::new(RwLock::new(Config {
            enabled,
            ..Config::default()
        }))
    }

    fn run(io: &FakeIo, cfg: Arc<RwLock<Config>>, settle: Duration) -> Vec<Frame> {
        let p = Pipeline::build(io, cfg, "in", "out").expect("pipeline should build");
        std::thread::sleep(settle);
        let out = io.rendered.lock().unwrap().clone();
        drop(p);
        out
    }

    use std::time::Duration;

    #[test]
    fn audio_flows_from_capture_through_to_render() {
        let io = FakeIo::new(test_signal(40));
        let out = run(&io, config(true), Duration::from_millis(400));
        assert!(!out.is_empty(), "nothing reached the render side");
    }

    /// Bypass must be a true passthrough, not "denoising turned down". This is
    /// the assertion a real recording could not make any more strongly.
    #[test]
    fn bypass_passes_audio_through_bit_exactly() {
        let input = test_signal(30);
        let io = FakeIo::new(input.clone());
        let out = run(&io, config(false), Duration::from_millis(400));

        assert!(!out.is_empty());
        for (i, frame) in out.iter().enumerate() {
            assert_eq!(
                frame.as_slice(),
                input[i].as_slice(),
                "frame {i} was altered while bypassed"
            );
        }
    }

    #[test]
    fn enabling_the_denoiser_changes_the_audio() {
        let input = test_signal(30);
        let processed = run(
            &FakeIo::new(input.clone()),
            config(true),
            Duration::from_millis(400),
        );
        assert!(!processed.is_empty());
        let any_different = processed
            .iter()
            .enumerate()
            .any(|(i, f)| f.as_slice() != input[i].as_slice());
        assert!(any_different, "denoiser left the signal untouched");
    }

    /// The render side needs its own pulse: `frames_processed` is bumped by
    /// the DSP thread and keeps climbing happily while the output endpoint is
    /// dead, which is precisely the failure the tray watchdog exists to catch.
    #[test]
    fn render_polls_advance_alongside_the_frame_counter() {
        let io = FakeIo::new(test_signal(25));
        let p = Pipeline::build(&io, config(true), "in", "out").unwrap();
        std::thread::sleep(Duration::from_millis(400));
        assert!(
            p.render_polls() > 0,
            "the render thread's liveness signal never moved"
        );
    }

    #[test]
    fn frames_processed_counts_what_went_through() {
        let io = FakeIo::new(test_signal(25));
        let p = Pipeline::build(&io, config(true), "in", "out").unwrap();
        std::thread::sleep(Duration::from_millis(400));
        let n = p.frames_processed();
        assert!(n > 0, "frame counter never advanced");
        assert!(n <= 25, "counted more frames than were fed: {n}");
    }

    /// The watchdog's signal: a device that stops delivering must stop
    /// advancing the counter, while one that is merely quiet keeps going.
    #[test]
    fn the_frame_counter_stops_when_capture_stops_delivering() {
        let io = FakeIo::new(test_signal(10)); // then silence forever
        let p = Pipeline::build(&io, config(true), "in", "out").unwrap();
        std::thread::sleep(Duration::from_millis(300));
        let first = p.frames_processed();
        std::thread::sleep(Duration::from_millis(300));
        let second = p.frames_processed();
        assert!(first > 0, "should have processed the frames it was given");
        assert_eq!(first, second, "counter advanced after capture went silent");
    }

    fn device(
        name: &str,
        direction: audio_io::devices::DeviceDirection,
    ) -> audio_io::devices::Device {
        audio_io::devices::Device {
            id: format!("{{0.0.0.00000000}}.{name}"),
            friendly_name: name.into(),
            direction,
            is_default: false,
        }
    }

    /// The first entry in each list is the Windows default, as it would be on
    /// a real machine — without one, "follow the default" resolves to nothing.
    fn inventory(mics: &[&str], outputs: &[&str]) -> DeviceList {
        use audio_io::devices::DeviceDirection::{Capture, Render};
        let mark_first = |mut devices: Vec<audio_io::devices::Device>| {
            if let Some(d) = devices.first_mut() {
                d.is_default = true;
            }
            devices
        };
        DeviceList {
            capture: mark_first(mics.iter().map(|n| device(n, Capture)).collect()),
            render: mark_first(outputs.iter().map(|n| device(n, Render)).collect()),
        }
    }

    fn with_mics(mics: &[&str]) -> Config {
        Config {
            microphones: mics.iter().map(|s| s.to_string()).collect(),
            ..Config::default()
        }
    }

    #[test]
    fn the_highest_ranked_microphone_that_is_present_wins() {
        let devices = inventory(
            &["Webcam Mic", "Yeti"],
            &["CABLE Input (VB-Audio Virtual Cable)"],
        );
        let (inputs, output) =
            Pipeline::pick_devices(&devices, &with_mics(&["Yeti", "Webcam Mic"])).unwrap();
        assert!(inputs[0].ends_with("Yeti"), "got {inputs:?}");
        assert!(output.contains("CABLE Input"));
        // The rest stay behind it as fallbacks, in preference order, so a mic
        // that enumerates but will not open is not the end of the road.
        assert!(
            inputs.len() > 1 && inputs[1].ends_with("Webcam Mic"),
            "the runner-up has to survive as a fallback: {inputs:?}"
        );

        // Unplug the Yeti: the next one down takes over rather than failing.
        let devices = inventory(&["Webcam Mic"], &["CABLE Input (VB-Audio Virtual Cable)"]);
        let (inputs, _) =
            Pipeline::pick_devices(&devices, &with_mics(&["Yeti", "Webcam Mic"])).unwrap();
        assert!(inputs[0].ends_with("Webcam Mic"), "got {inputs:?}");
    }

    /// Installing a cable usually makes it the default capture device too, so
    /// "follow the Windows default" silently becomes the cable feeding itself.
    /// Nothing real is in that loop and the user hears nothing.
    #[test]
    fn capturing_from_the_cables_own_output_is_refused() {
        let devices = inventory(
            &["CABLE Output (VB-Audio Virtual Cable)"],
            &["CABLE Input (VB-Audio Virtual Cable)"],
        );
        let err = Pipeline::pick_devices(&devices, &Config::default())
            .expect_err("must not wire the cable to itself");
        let msg = format!("{err:#}");
        assert!(msg.contains("loop"), "should explain the loop: {msg}");
        assert!(msg.contains("tray menu"), "should say how to fix it: {msg}");
    }

    /// Failing closed matters here: falling back to the default render device
    /// would play the microphone out of whatever speakers or meeting-room
    /// display happens to be default.
    #[test]
    fn no_cable_is_an_error_rather_than_a_fallback_to_the_speakers() {
        let devices = inventory(&["Yeti"], &["Speakers (Realtek)", "Headphones"]);
        let err = Pipeline::pick_devices(&devices, &with_mics(&["Yeti"]))
            .expect_err("must not fall back to the speakers");
        let msg = format!("{err:#}");
        assert!(msg.contains("VB-Cable"), "{msg}");
        assert!(
            !msg.contains("Speakers"),
            "picked the speakers anyway: {msg}"
        );
    }

    #[test]
    fn no_microphone_at_all_is_reported_as_such() {
        let devices = inventory(&[], &["CABLE Input (VB-Audio Virtual Cable)"]);
        let err = Pipeline::pick_devices(&devices, &Config::default()).unwrap_err();
        assert!(format!("{err:#}").contains("no microphone"));
    }

    /// A named output wins over auto-detection — the escape hatch for anyone
    /// running a cable we do not recognise.
    #[test]
    fn a_named_output_device_overrides_cable_detection() {
        let devices = inventory(&["Yeti"], &["Line 1 (Some Other Cable)"]);
        let cfg = Config {
            output_device: "Line 1 (Some Other Cable)".into(),
            ..with_mics(&["Yeti"])
        };
        let (_, output) = Pipeline::pick_devices(&devices, &cfg).unwrap();
        assert!(output.contains("Line 1"), "got {output}");
    }

    #[test]
    fn dropping_the_pipeline_stops_its_threads() {
        let io = FakeIo::new(test_signal(200));
        let p = Pipeline::build(&io, config(true), "in", "out").unwrap();
        std::thread::sleep(Duration::from_millis(100));
        drop(p); // joins the DSP thread; the fake handles join theirs
        let settled = io.rendered.lock().unwrap().len();
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(
            settled,
            io.rendered.lock().unwrap().len(),
            "frames kept arriving after the pipeline was dropped"
        );
    }
}

#[cfg(test)]
mod fallback_tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use super::tests::{config, test_signal};
    use super::{AudioIo, FrameSink, Pipeline};
    use anyhow::Result;

    /// A sound card where one named device exists but refuses to open.
    ///
    /// This is what a reporter hit: `GetMixFormat` returned 0x8007007E for the
    /// microphone we picked, and RoomMute stopped at a dialog even though
    /// other working microphones were plugged in. Selecting one by hand fixed
    /// it, which is the app's job, not the user's.
    struct OneBadMic {
        bad: &'static str,
        attempts: Arc<Mutex<Vec<String>>>,
        stop: Arc<AtomicBool>,
    }

    struct Handle(Arc<AtomicBool>);
    impl Drop for Handle {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    impl AudioIo for OneBadMic {
        fn start_capture(&self, id: &str, mut sink: Box<dyn FrameSink>) -> Result<Box<dyn Send>> {
            self.attempts.lock().unwrap().push(id.to_string());
            if id == self.bad {
                anyhow::bail!(
                    "GetMixFormat failed: the device's audio enhancements could not be \
                     loaded (0x8007007E)"
                );
            }
            let frames = test_signal(4);
            let stop = self.stop.clone();
            std::thread::spawn(move || {
                for f in frames {
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    sink.on_frame(&f);
                }
            });
            Ok(Box::new(Handle(self.stop.clone())))
        }

        fn start_render(
            &self,
            _id: &str,
            _source: Box<dyn super::FrameSource>,
        ) -> Result<Box<dyn Send>> {
            Ok(Box::new(Handle(self.stop.clone())))
        }
    }

    #[test]
    fn a_microphone_that_will_not_open_falls_through_to_the_next() {
        let attempts = Arc::new(Mutex::new(Vec::new()));
        let io = OneBadMic {
            bad: "broken-mic",
            attempts: attempts.clone(),
            stop: Arc::new(AtomicBool::new(false)),
        };

        let p = Pipeline::build_ranked(
            &io,
            config(true),
            &["broken-mic".to_string(), "good-mic".to_string()],
            "out",
        );

        assert!(
            p.is_ok(),
            "a working microphone was available; refusing to start is the bug"
        );
        assert_eq!(
            *attempts.lock().unwrap(),
            vec!["broken-mic", "good-mic"],
            "it has to try the next candidate, in order"
        );
    }

    #[test]
    fn the_error_survives_when_every_microphone_fails() {
        let io = OneBadMic {
            bad: "broken-mic",
            attempts: Arc::new(Mutex::new(Vec::new())),
            stop: Arc::new(AtomicBool::new(false)),
        };
        let err = Pipeline::build_ranked(&io, config(true), &["broken-mic".to_string()], "out");
        // Matched rather than `expect_err`: Pipeline owns thread handles and
        // is not Debug, so unwrapping the error needs no formatting of the Ok.
        let Err(err) = err else {
            panic!("nothing could open, so this must fail");
        };
        let text = format!("{err:#}");
        assert!(
            text.contains("0x8007007E"),
            "the last real reason has to reach the user: {text}"
        );
    }
}

#[cfg(test)]
mod leak_tests {
    use std::sync::atomic::Ordering;

    use super::tests::config;
    use super::{AudioIo, FrameSink, FrameSource, Pipeline, DSP_THREADS};
    use anyhow::Result;

    /// A sound card where opening the microphone always fails.
    struct DeadCard;

    impl AudioIo for DeadCard {
        fn start_capture(&self, _id: &str, _sink: Box<dyn FrameSink>) -> Result<Box<dyn Send>> {
            anyhow::bail!("GetMixFormat failed (0x8007007E)")
        }
        fn start_render(&self, _id: &str, _s: Box<dyn FrameSource>) -> Result<Box<dyn Send>> {
            Ok(Box::new(()))
        }
    }

    /// A card that takes the microphone and then refuses to render.
    struct NoRender;

    impl AudioIo for NoRender {
        fn start_capture(&self, _id: &str, _sink: Box<dyn FrameSink>) -> Result<Box<dyn Send>> {
            Ok(Box::new(()))
        }
        fn start_render(&self, _id: &str, _s: Box<dyn FrameSource>) -> Result<Box<dyn Send>> {
            anyhow::bail!("no such render endpoint")
        }
    }

    /// Reported from a real install: 6.8 GB and a permanent 11% of the CPU.
    ///
    /// The DSP thread was spawned before the devices were opened, and it owns
    /// the denoiser — an entire DeepFilterNet3 graph. When opening a device
    /// failed, `?` returned without ever setting the shutdown flag, so the
    /// thread stayed alive forever, polling every 2 ms and holding the model.
    /// The watchdog then retried, and each retry leaked another one.
    #[test]
    fn a_failed_start_leaves_no_thread_behind() {
        let before = DSP_THREADS.load(Ordering::Acquire);

        for _ in 0..5 {
            assert!(Pipeline::build(&DeadCard, config(true), "in", "out").is_err());
        }
        for _ in 0..5 {
            let io = NoRender;
            assert!(Pipeline::build(&io, config(true), "in", "out").is_err());
        }

        // Threads that were asked to stop may take a moment to notice.
        for _ in 0..100 {
            if DSP_THREADS.load(Ordering::Acquire) <= before {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(
            DSP_THREADS.load(Ordering::Acquire),
            before,
            "ten failed starts left DSP threads running, each holding a model"
        );
    }
}
