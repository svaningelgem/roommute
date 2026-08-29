//! System tray UI: enable/disable toggle, microphone picker, start-at-login,
//! CPU meter, log folder, quit.
//!
//! `tray-icon` needs a window-message pump on the main thread, so we drive
//! it with a `winit` event loop (no actual window — just the tray icon).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use tracing::{info, warn};
use tray_icon::menu::{
    CheckMenuItem, Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem, Submenu,
};
use tray_icon::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};
use winit::application::ApplicationHandler;
use winit::event_loop::{ControlFlow, EventLoop};

use audio_io::devices::DeviceList;

use crate::config::Config;
use crate::parking_lot_compat::RwLock;
use crate::pipeline::Pipeline;

/// `startup_error`, if any, is reported *after* the tray icon exists — a modal
/// dialog with nothing behind it reads as an error from nowhere.
pub fn run(
    cfg: Arc<RwLock<Config>>,
    pipeline: Option<Pipeline>,
    startup_error: Option<crate::StartupProblem>,
) -> Result<()> {
    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();

    // Forward tray events to winit so we run on a single event loop.
    let menu_proxy = proxy.clone();
    MenuEvent::set_event_handler(Some(move |e| {
        let _ = menu_proxy.send_event(UserEvent::Menu(e));
    }));
    TrayIconEvent::set_event_handler(Some(move |e| {
        let _ = proxy.send_event(UserEvent::Tray(e));
    }));

    let mut app = App {
        cfg,
        pipeline,
        startup_error,
        tray: None,
        items: None,
        last_tooltip_update: Instant::now(),
        health: Health::new(),
        failures: FailureLog::default(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Debug)]
enum UserEvent {
    Menu(MenuEvent),
    Tray(TrayIconEvent),
}

struct App {
    cfg: Arc<RwLock<Config>>,
    /// `None` only while a device switch is restarting it, or if a restart
    /// failed — the tray stays alive either way so the user can pick again.
    pipeline: Option<Pipeline>,
    /// Shown once, after the icon is up.
    startup_error: Option<crate::StartupProblem>,
    tray: Option<TrayIcon>,
    items: Option<Items>,
    last_tooltip_update: Instant,
    health: Health,
    /// Keeps the retry loop from writing the same sentence every 5 seconds.
    failures: FailureLog,
}

/// Watches for the pipeline going quiet without saying so.
///
/// WASAPI reports a dead stream by simply never signalling its event again —
/// no error, no callback. The threads stay alive and the app looks perfectly
/// healthy while nothing has reached the far end of the call for a quarter of
/// an hour. Counters are the only honest signal, because silence still
/// produces frames: a quiet room and a dead device look identical otherwise.
///
/// Both halves need watching, and this originally only watched one. The
/// capture counter is bumped by the DSP thread, so it keeps climbing quite
/// happily when the *output* endpoint dies — reconfigure the virtual cable in
/// Windows sound settings and that is exactly what happens. Capture and DSP
/// carry on, the icon stays teal, and nobody on the call hears anything again.
struct Health {
    /// The DSP thread draining ring A — proves capture is alive.
    capture: Progress,
    /// The render thread asking for audio — proves the *output* is alive.
    /// Watching only the first would miss the cable being reconfigured
    /// underneath us, which stops the call hearing anything while capture and
    /// DSP carry on and the counter keeps climbing.
    render: Progress,
    last_recovery: Instant,
    stalled: bool,
}

/// A counter that is supposed to keep moving, and when it last did.
#[derive(Debug)]
struct Progress {
    last: u64,
    last_change: Instant,
}

impl Progress {
    fn new(now: Instant) -> Self {
        Self {
            last: 0,
            last_change: now,
        }
    }

    /// Feed the current value; `true` while it is still moving.
    fn moving(&mut self, value: u64, now: Instant) -> bool {
        if value != self.last {
            self.last = value;
            self.last_change = now;
        }
        now.duration_since(self.last_change) < STALL_AFTER
    }

    fn reset(&mut self, value: u64, now: Instant) {
        self.last = value;
        self.last_change = now;
    }
}

/// How often the event loop wakes to check health and refresh the tooltip.
const TICK: Duration = Duration::from_millis(500);
/// Frames stop for this long => the stream is dead, not the room quiet.
const STALL_AFTER: Duration = Duration::from_secs(2);
/// Don't hammer the device if it stays gone.
const RECOVERY_INTERVAL: Duration = Duration::from_secs(5);

/// How often a failure that has not changed is worth repeating in the log.
const REPEAT_FAILURE_AFTER: Duration = Duration::from_secs(300);

/// Decides whether a recovery failure is news.
///
/// The watchdog retries every 5 seconds forever, which is right — plug a
/// microphone in and it recovers with no restart. Logging every attempt is
/// not: a machine with no working microphone wrote two lines every five
/// seconds, and the log that should have explained a runaway was 6,000
/// repetitions of one sentence with the useful lines buried in it.
///
/// So: say it the first time, say it again whenever the reason changes, and
/// otherwise once every five minutes so a long outage is still visible.
#[derive(Default)]
struct FailureLog {
    last: Option<(String, Instant)>,
}

impl FailureLog {
    fn should_report(&mut self, reason: &str, now: Instant) -> bool {
        let news = match &self.last {
            Some((seen, at)) => seen != reason || now.duration_since(*at) >= REPEAT_FAILURE_AFTER,
            None => true,
        };
        if news {
            self.last = Some((reason.to_string(), now));
        }
        news
    }

    /// Audio came back, so the next failure is news again.
    fn clear(&mut self) {
        self.last = None;
    }
}

/// What the watchdog wants done about what it just saw. Two independent
/// flags rather than one verdict because the badge has to go up *before* a
/// recovery attempt: if the restart fails we must already have stopped
/// looking healthy.
#[derive(Debug, Default, PartialEq, Eq)]
struct HealthAction {
    /// The stall state changed; the icon no longer matches it.
    redraw: bool,
    /// Rebuild the pipeline.
    restart: bool,
}

impl Health {
    fn new() -> Self {
        Self::new_at(Instant::now())
    }

    fn new_at(now: Instant) -> Self {
        Self {
            capture: Progress::new(now),
            render: Progress::new(now),
            last_recovery: now,
            stalled: false,
        }
    }

    /// Feed the watchdog one observation of both counters.
    ///
    /// Takes `now` rather than reading the clock so the stall and recovery
    /// timings can be tested without waiting seconds per case.
    fn observe(&mut self, frames: u64, render_polls: u64, now: Instant) -> HealthAction {
        // Both must be evaluated: each records its own last-changed time.
        let capture_moving = self.capture.moving(frames, now);
        let render_moving = self.render.moving(render_polls, now);

        // Silence still produces frames and the engine still asks for them,
        // so movement on both sides means the stream is alive — a quiet room
        // is not a stall.
        if capture_moving && render_moving {
            if self.stalled {
                info!("audio recovered");
                self.stalled = false;
                return HealthAction {
                    redraw: true,
                    ..Default::default()
                };
            }
            return HealthAction::default();
        }

        let mut action = HealthAction::default();
        if !self.stalled {
            let side = match (capture_moving, render_moving) {
                (false, false) => "capture and output",
                (false, true) => "the capture device",
                _ => "the output device",
            };
            warn!(side, "audio stopped flowing — {side} probably went away");
            self.stalled = true;
            action.redraw = true;
        }
        if now.duration_since(self.last_recovery) >= RECOVERY_INTERVAL {
            self.last_recovery = now;
            info!("attempting to restart audio");
            action.restart = true;
        }
        action
    }

    /// Nothing is running, so there are no counters to watch. Recovery still
    /// has to keep trying: a restart that failed because the microphone was
    /// unplugged must succeed once it is plugged back in.
    fn observe_stopped(&mut self, now: Instant) -> HealthAction {
        self.stalled = true;
        if now.duration_since(self.last_recovery) < RECOVERY_INTERVAL {
            return HealthAction::default();
        }
        self.last_recovery = now;
        // No log line here. This fires every five seconds for as long as
        // audio is down, and the attempt itself is not news — the reason it
        // failed is, and restart_pipeline_quietly reports that.
        HealthAction {
            restart: true,
            ..Default::default()
        }
    }
}

/// One entry in the microphone submenu.
struct MicEntry {
    item: CheckMenuItem,
    /// Empty string = follow the Windows default device.
    device_id: String,
    friendly_name: String,
}

/// One entry in the denoiser submenu.
struct DenoiserEntry {
    item: CheckMenuItem,
    /// True for the ONNX model, false for built-in RNNoise.
    onnx: bool,
}

struct Items {
    enable: CheckMenuItem,
    mics: Vec<MicEntry>,
    denoisers: Vec<DenoiserEntry>,
    auto_start: CheckMenuItem,
    help: MenuItem,
    open_logs: MenuItem,
    quit: MenuItem,
}

impl Items {
    fn mic_by_id(&self, id: &MenuId) -> Option<&MicEntry> {
        self.mics.iter().find(|m| m.item.id() == id)
    }

    fn denoiser_by_id(&self, id: &MenuId) -> Option<&DenoiserEntry> {
        self.denoisers.iter().find(|d| d.item.id() == id)
    }
}

/// How a capture device should appear in the picker.
///
/// A virtual cable's output side is offered by Windows like any other
/// microphone, but choosing it would have RoomMute record from the very
/// cable it writes to. The pipeline refuses to start in that case; the menu
/// shouldn't let it get that far. Greyed out with the reason attached beats an
/// error dialog after the fact.
fn mic_entry(
    name: &str,
    is_system_default: bool,
    cable: Option<&str>,
    rank: Option<usize>,
) -> (String, bool) {
    if let Some(product) = cable {
        return (
            format!("{name}  — {product}'s own output, would loop"),
            false,
        );
    }
    // Rank first so the preference order reads down the menu.
    let prefix = match rank {
        Some(i) => format!("{}.  ", i + 1),
        None => "     ".to_string(),
    };
    let suffix = if is_system_default {
        "  (system default)"
    } else {
        ""
    };
    (format!("{prefix}{name}{suffix}"), true)
}

/// The "Windows default" entry, which is the subtle one: following the system
/// default is fine until the system default *is* a cable, which is precisely
/// what installing one does. Then the safe-looking option is the trap.
fn default_entry(current_default: Option<&str>) -> (String, bool) {
    match current_default {
        Some(product) => (
            format!("Windows default  — currently {product}, would loop"),
            false,
        ),
        None => ("Windows default".to_string(), true),
    }
}

/// Record a microphone choice. An empty name is the "Windows default" entry,
/// which means *stop* expressing a preference rather than remembering one.
fn choose_microphone(cfg: &mut Config, name: &str) {
    if name.is_empty() {
        cfg.microphones.clear();
    } else {
        cfg.prefer_microphone(name);
    }
}

/// Record a denoiser choice, remembering where the model was found so that
/// switching back to it later doesn't ask the user to configure it again.
fn choose_denoiser(cfg: &mut Config, onnx: bool) {
    if onnx {
        if let Some(p) = cfg.available_model() {
            cfg.model_path = p.to_string_lossy().into_owned();
        }
    }
    cfg.use_onnx = onnx;
}

/// Share of real time the denoiser is using, as a percentage.
///
/// Each frame is 10 ms of audio, so the budget is 10 ms of processing per
/// frame; 100% means the DSP is exactly keeping up and any hiccup drops audio.
fn cpu_percent(frames: u64, total_dsp_ns: u64) -> f64 {
    if frames == 0 {
        return 0.0;
    }
    let avg_dsp_ms = (total_dsp_ns as f64 / frames as f64) / 1_000_000.0;
    avg_dsp_ms / 10.0 * 100.0
}

fn tooltip_text(denoiser: &str, enabled: bool, cpu_pct: f64, peak_ms: f64) -> String {
    format!(
        "RoomMute ({})\n{}  |  CPU: {:.1}%  peak: {:.1}ms",
        denoiser,
        if enabled { "ON" } else { "BYPASS" },
        cpu_pct,
        peak_ms,
    )
}

fn build_menu(cfg: &Config) -> (Menu, Items) {
    let menu = Menu::new();

    let enable = CheckMenuItem::new("Enabled", true, cfg.enabled, None);
    menu.append(&enable).ok();
    menu.append(&PredefinedMenuItem::separator()).ok();

    // Microphone picker.
    let mic_menu = Submenu::new("Microphone", true);
    let mut mics = Vec::new();

    let follow_default = cfg.microphones.is_empty();
    let devices = DeviceList::enumerate();

    // Is the system default itself a cable? Then "Windows default" is a trap.
    let default_is_cable = devices
        .as_ref()
        .ok()
        .and_then(|l| l.default_capture())
        .and_then(|d| d.virtual_cable_output());
    let (label, enabled) = default_entry(default_is_cable);
    let default_item = CheckMenuItem::new(label, enabled, follow_default, None);
    mic_menu.append(&default_item).ok();
    mics.push(MicEntry {
        item: default_item,
        device_id: String::new(),
        friendly_name: String::new(),
    });

    match devices {
        Ok(list) => {
            if !list.capture.is_empty() {
                mic_menu.append(&PredefinedMenuItem::separator()).ok();
            }
            for d in &list.capture {
                let rank = cfg
                    .microphones
                    .iter()
                    .position(|m| audio_io::devices::same_device_name(m, &d.friendly_name));
                let checked = rank == Some(0);
                let (label, enabled) = mic_entry(
                    &d.friendly_name,
                    d.is_default,
                    d.virtual_cable_output(),
                    rank,
                );
                let item = CheckMenuItem::new(label, enabled, checked, None);
                mic_menu.append(&item).ok();
                mics.push(MicEntry {
                    item,
                    device_id: d.id.clone(),
                    friendly_name: d.friendly_name.clone(),
                });
            }
        }
        Err(e) => {
            warn!(error = %e, "could not enumerate capture devices for the tray menu");
            let item = MenuItem::new("(no microphones found)", false, None);
            mic_menu.append(&item).ok();
        }
    }
    menu.append(&mic_menu).ok();

    // Denoiser picker.
    let dsp_menu = Submenu::new("Denoiser", true);
    let mut denoisers = Vec::new();
    let model = cfg.available_model();
    let onnx_active = cfg.active_model().is_some();

    let rnnoise = CheckMenuItem::new("RNNoise (built-in)", true, !onnx_active, None);
    dsp_menu.append(&rnnoise).ok();
    denoisers.push(DenoiserEntry {
        item: rnnoise,
        onnx: false,
    });

    match &model {
        Some(path) => {
            // Named after the backend that will actually load it, decided the
            // same way `dsp::build_denoiser` decides. A directory or a .tar.gz
            // is DeepFilterNet3 through tract; anything else goes to the ONNX
            // loader. Calling both "(ONNX)" was wrong for the one we ship.
            let label = if path.is_dir() || path.extension().is_some_and(|e| e == "gz") {
                "DeepFilterNet3".to_string()
            } else {
                format!(
                    "{} (ONNX)",
                    path.file_name().unwrap_or_default().to_string_lossy()
                )
            };
            let item = CheckMenuItem::new(label, true, onnx_active, None);
            dsp_menu.append(&item).ok();
            denoisers.push(DenoiserEntry { item, onnx: true });
        }
        // Explain the absence rather than silently offering one option. There
        // is no "built without" case any more: every backend is always
        // compiled in, so a missing model is the only way to get here.
        None => {
            dsp_menu
                .append(&MenuItem::new(
                    "(no model found next to the app)",
                    false,
                    None,
                ))
                .ok();
        }
    }
    menu.append(&dsp_menu).ok();
    menu.append(&PredefinedMenuItem::separator()).ok();

    // Ask the registry rather than trusting the config file: the user may have
    // removed the Run entry by hand since we last wrote it.
    let auto_start = CheckMenuItem::new(
        "Start with Windows",
        true,
        crate::autostart::is_enabled(),
        None,
    );
    menu.append(&auto_start).ok();

    let help = MenuItem::new("Help — how to use this", true, None);
    menu.append(&help).ok();

    let open_logs = MenuItem::new("Open log folder", true, None);
    menu.append(&open_logs).ok();
    menu.append(&PredefinedMenuItem::separator()).ok();

    let quit = MenuItem::new("Quit RoomMute", true, None);
    menu.append(&quit).ok();

    (
        menu,
        Items {
            enable,
            mics,
            denoisers,
            auto_start,
            help,
            open_logs,
            quit,
        },
    )
}

impl App {
    /// Rebuild the pipeline without telling the user. Used by the watchdog,
    /// where a dialog every five seconds would be its own kind of failure.
    fn restart_pipeline_quietly(&mut self) {
        drop(self.pipeline.take());
        match Pipeline::start(self.cfg.clone()) {
            Ok(p) => {
                info!(denoiser = p.denoiser_name(), "audio restarted");
                self.failures.clear();
                let now = Instant::now();
                self.health.capture.reset(p.frames_processed(), now);
                self.health.render.reset(p.render_polls(), now);
                self.health.stalled = false;
                self.pipeline = Some(p);
            }
            Err(e) => {
                let reason = format!("{e:#}");
                if self.failures.should_report(&reason, Instant::now()) {
                    warn!(error = %reason, "audio will not start; retrying every 5s");
                }
            }
        }
        self.refresh_icon();
    }

    /// Restart the audio pipeline against the current config. The old one is
    /// dropped first so it releases the capture device before we reopen it.
    fn restart_pipeline(&mut self) {
        drop(self.pipeline.take());
        match Pipeline::start(self.cfg.clone()) {
            Ok(p) => {
                info!(denoiser = p.denoiser_name(), "pipeline restarted");
                self.pipeline = Some(p);
            }
            Err(e) => {
                // Leave the tray running: the user picked a bad device and the
                // fix is to pick a different one from this very menu.
                warn!(error = %e, "restarting the pipeline failed");
                // Badge first, so it's already showing behind the dialog.
                self.refresh_icon();
                crate::message_box(&format!("Could not start audio with that device:\n\n{e:#}"));
                return;
            }
        }
        self.refresh_icon();
    }

    /// Watch the frame counter and quietly rebuild the pipeline if it stops.
    ///
    /// Silently delivering nothing is the worst failure this app has: the
    /// icon stays teal, the log says "running", and the call on the other end
    /// hears nothing. Recovery is deliberately quiet — no dialog, because the
    /// usual cause is a device that vanished and came back, and interrupting
    /// someone mid-call to announce that would be worse than fixing it.
    fn check_health(&mut self) {
        let now = Instant::now();
        let action = match self.pipeline.as_ref() {
            Some(p) => self
                .health
                .observe(p.frames_processed(), p.render_polls(), now),
            // A restart that failed leaves no pipeline. Keep trying rather
            // than giving up for the rest of the session: the usual reason is
            // a device that was still missing at the moment we retried.
            None => self.health.observe_stopped(now),
        };
        // Badge first: even if recovery fails, stop looking healthy.
        if action.redraw {
            self.refresh_icon();
        }
        if action.restart {
            self.restart_pipeline_quietly();
        }
    }

    /// Keep the icon in step with both bits of state it shows: teal vs orange
    /// for on/off, and the badge for "audio isn't running at all".
    fn refresh_icon(&self) {
        if let Some(tray) = &self.tray {
            let enabled = self.cfg.read().unwrap().enabled;
            let _ = tray.set_icon(Some(build_icon(enabled, self.audio_is_broken())));
        }
    }

    /// No pipeline at all, or one that has stopped delivering. Both mean the
    /// microphone isn't reaching anybody, which is what the badge is for.
    fn audio_is_broken(&self) -> bool {
        self.pipeline.is_none() || self.health.stalled
    }

    /// The single path for turning denoising on and off, whichever surface
    /// asked — the menu checkbox or a left click on the icon. Both have to
    /// leave the checkbox, the icon and the config agreeing with each other.
    fn set_enabled(&mut self, enabled: bool) {
        if let Some(p) = &self.pipeline {
            p.set_enabled(enabled);
        }
        {
            let mut c = self.cfg.write().unwrap();
            c.enabled = enabled;
            if let Err(e) = c.save() {
                warn!(error = %e, "saving config failed");
            }
        }
        if let Some(items) = &self.items {
            // A left click didn't touch the menu, so sync it by hand.
            if items.enable.is_checked() != enabled {
                items.enable.set_checked(enabled);
            }
        }
        self.refresh_icon();
        info!(enabled, "denoising toggled");
    }

    /// Show the welcome the first time, and never again.
    ///
    /// Only when audio is actually running: the message says which microphone
    /// is in use and where to point other apps, and neither is true yet if
    /// startup failed. Someone in that state has already been told what went
    /// wrong, and stacking a cheerful welcome on top of it helps nobody — they
    /// get it on the first run that works.
    ///
    /// The flag is written before the dialog, not after. A modal blocks this
    /// thread, and a crash or a kill while it is open would otherwise mean the
    /// same message again on every start.
    fn welcome_once(&mut self) {
        if self.pipeline.is_none() || self.cfg.read().unwrap().welcomed {
            return;
        }
        {
            let mut c = self.cfg.write().unwrap();
            c.welcomed = true;
            if let Err(e) = c.save() {
                warn!(error = %e, "could not record that the welcome was shown");
            }
        }
        self.show_welcome();
    }

    /// The welcome, with whatever is actually true right now.
    ///
    /// Both names are read live rather than remembered: the pipeline may have
    /// fallen through to a different microphone than the configured first
    /// choice, and telling someone the wrong one is worse than telling them
    /// nothing. The cable is looked up as the *capture* endpoint, because that
    /// is the half other applications select.
    fn show_welcome(&self) {
        let devices = audio_io::devices::DeviceList::enumerate().ok();
        let mic = devices
            .as_ref()
            .and_then(|d| {
                let prefs = &self.cfg.read().unwrap().microphones;
                d.capture_candidates(prefs)
                    .first()
                    .map(|d| d.friendly_name.clone())
            })
            .unwrap_or_else(|| "your default microphone".to_string());
        let cable = devices
            .as_ref()
            .and_then(|d| d.virtual_cable_output_device())
            .map(|d| d.friendly_name.clone());
        crate::firstrun::show_welcome(&mic, cable.as_deref());
    }

    /// Clicking a microphone makes it first choice; everything else shifts
    /// down. One rule, and it builds the fallback order out of ordinary use.
    fn select_mic(&mut self, device_id: String, name: String) {
        {
            let mut c = self.cfg.write().unwrap();
            choose_microphone(&mut c, &name);
            if let Err(e) = c.save() {
                warn!(error = %e, "saving config failed");
            }
        }
        // Exactly one entry stays checked — these are radio buttons wearing
        // checkbox clothing, and muda won't enforce that for us.
        if let Some(items) = &self.items {
            for m in &items.mics {
                m.item.set_checked(m.device_id == device_id);
            }
        }
        info!(device = %if device_id.is_empty() { "Windows default" } else { &device_id }, "microphone selected");
        self.restart_pipeline();
    }

    fn select_denoiser(&mut self, onnx: bool) {
        // Choosing the high-quality model means accepting someone else's
        // licence, so ask every time it's deliberately selected. Declining
        // leaves the built-in backend running rather than nothing.
        if onnx {
            if !crate::firstrun::model_licence() {
                info!("model licence declined; staying on RNNoise");
                if let Some(items) = &self.items {
                    for d in &items.denoisers {
                        d.item.set_checked(!d.onnx);
                    }
                }
                return;
            }
            let available = self.cfg.read().unwrap().available_model();
            if available.is_none() {
                // Nothing to load yet — say where to put it and stay put.
                // The directory, because that is the layout the installer
                // lays down and the one to recreate by hand.
                let expected = std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|d| d.join("model")))
                    .unwrap_or_else(|| std::path::PathBuf::from("model"));
                crate::firstrun::model_missing(&expected);
                if let Some(items) = &self.items {
                    for d in &items.denoisers {
                        d.item.set_checked(!d.onnx);
                    }
                }
                return;
            }
        }
        {
            let mut c = self.cfg.write().unwrap();
            choose_denoiser(&mut c, onnx);
            if let Err(e) = c.save() {
                warn!(error = %e, "saving config failed");
            }
        }
        if let Some(items) = &self.items {
            for d in &items.denoisers {
                d.item.set_checked(d.onnx == onnx);
            }
        }
        info!(onnx, "denoiser selected");
        self.restart_pipeline();
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.tray.is_some() {
            return;
        }
        self.welcome_once();
        let (menu, items) = build_menu(&self.cfg.read().unwrap());

        let tooltip = self
            .pipeline
            .as_ref()
            .map(initial_tooltip)
            .unwrap_or_else(|| "RoomMute — stopped".to_string());

        let enabled = self.cfg.read().unwrap().enabled;
        let tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip(tooltip)
            .with_icon(build_icon(enabled, self.pipeline.is_none()))
            // Left click is the on/off toggle, so it must not also open the
            // menu. Right click still does.
            .with_menu_on_left_click(false)
            .build()
            .expect("build tray icon");

        self.tray = Some(tray);
        self.items = Some(items);

        // Tick periodically so we can refresh the tooltip CPU meter.
        // Re-armed at the end of every `about_to_wait`; see there for why.
        event_loop.set_control_flow(ControlFlow::wait_duration(TICK));

        // Now that there's an icon in the tray, it's safe to interrupt with a
        // dialog: the user can see what it belongs to, and the badge is still
        // there once they dismiss it.
        // A missing cable never arrives here: `real_main` handles that case
        // itself, before the tray exists, because both answers end the run.
        if let Some(problem) = self.startup_error.take() {
            crate::message_box(&problem.message);
        }
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, ev: UserEvent) {
        let id = match ev {
            UserEvent::Menu(MenuEvent { id, .. }) => id,
            // Left click toggles; releases only, so press-and-drag-away is not
            // a toggle. Right click is handled by the menu itself.
            UserEvent::Tray(TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            }) => {
                let now = !self.cfg.read().unwrap().enabled;
                self.set_enabled(now);
                return;
            }
            UserEvent::Tray(_) => return,
        };
        let Some(items) = &self.items else { return };

        if id == *items.enable.id() {
            let now_enabled = items.enable.is_checked();
            self.set_enabled(now_enabled);
        } else if id == *items.auto_start.id() {
            let wanted = items.auto_start.is_checked();
            match crate::autostart::set(wanted) {
                // Nothing to persist: the Run key is the state.
                Ok(()) => info!(auto_start = wanted, "start-with-Windows toggled"),
                Err(e) => {
                    warn!(error = %e, "could not update the Run key");
                    // Put the checkbox back where it was — it must reflect the
                    // registry, not what the user wished for.
                    items.auto_start.set_checked(!wanted);
                    crate::message_box(&format!("Could not change start-with-Windows:\n\n{e:#}"));
                }
            }
        } else if id == *items.help.id() {
            self.show_welcome();
        } else if id == *items.open_logs.id() {
            let _ = std::process::Command::new(explorer_path())
                .arg(crate::config::log_dir())
                .spawn();
        } else if id == *items.quit.id() {
            info!("quit requested");
            event_loop.exit();
        } else if let Some((device_id, name)) = items
            .mic_by_id(&id)
            .map(|m| (m.device_id.clone(), m.friendly_name.clone()))
        {
            self.select_mic(device_id, name);
        } else if let Some(onnx) = items.denoiser_by_id(&id).map(|d| d.onnx) {
            self.select_denoiser(onnx);
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        _event: winit::event::WindowEvent,
    ) {
        // No window — nothing to do.
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // `wait_duration` is `WaitUntil(now + d)` — a deadline, not a period.
        // Set once, it expires after the first tick and every later pass sees
        // a deadline already in the past, so the loop never sleeps again: a
        // background tray app quietly pinning a core. Re-arm on every pass.
        event_loop.set_control_flow(ControlFlow::wait_duration(TICK));

        self.check_health();
        if self.last_tooltip_update.elapsed() >= Duration::from_millis(1000) {
            self.last_tooltip_update = Instant::now();
            if let Some(tray) = &self.tray {
                let text = match status_text(self.pipeline.is_some(), self.health.stalled) {
                    Some(fixed) => fixed.to_string(),
                    None => tooltip(self.pipeline.as_ref().expect("running")),
                };
                let _ = tray.set_tooltip(Some(text));
            }
        }
    }
}

/// Absolute path to Explorer. Spawning bare `"explorer"` would resolve it
/// through PATH, so any writable directory sitting earlier on PATH gets to
/// supply the binary we launch.
fn explorer_path() -> std::path::PathBuf {
    std::env::var_os("SystemRoot")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(r"C:\Windows"))
        .join("explorer.exe")
}

fn initial_tooltip(p: &Pipeline) -> String {
    format!("RoomMute ({}) — starting", p.denoiser_name())
}

fn tooltip(p: &Pipeline) -> String {
    let s = p.stats();
    let frames = s.frames.load(Ordering::Relaxed);
    let total_ns = s.dsp_ns.load(Ordering::Relaxed);
    let peak_ns = s.peak_frame_ns.load(Ordering::Relaxed);
    tooltip_text(
        p.denoiser_name(),
        p.is_enabled(),
        cpu_percent(frames, total_ns),
        peak_ns as f64 / 1_000_000.0,
    )
}

/// What the tray should say about the audio, given whether a pipeline exists
/// and whether the watchdog thinks it has gone quiet.
fn status_text(running: bool, stalled: bool) -> Option<&'static str> {
    match (running, stalled) {
        (true, true) => Some("RoomMute — no audio from the microphone"),
        (false, _) => Some("RoomMute — stopped (pick a microphone)"),
        // Running and healthy: the caller has real statistics to show instead.
        (true, false) => None,
    }
}

/// Icons are generated procedurally so v1 doesn't need to ship a .ico.
/// 32x32 rather than 16x16: the warning badge needs the room, and Windows
/// scales down more gracefully than up.
const ICON_SIZE: usize = 32;

/// Is a pixel inside a triangle that has its apex at the top and widens
/// linearly to `max_half` at `bottom`?
fn in_triangle(x: usize, y: usize, top: usize, bottom: usize, cx: usize, max_half: f32) -> bool {
    if y < top || y > bottom {
        return false;
    }
    let t = (y - top) as f32 / (bottom - top) as f32;
    (x as f32 - cx as f32).abs() <= t * max_half
}

/// Denoising is on — the same blue-green the app has always used.
const ACTIVE: [u8; 4] = [0x2a, 0xa1, 0x98, 0xff];
/// Denoising is bypassed. Orange rather than a dimmed teal: at tray size a
/// brightness difference is invisible, a hue change isn't.
const BYPASSED: [u8; 4] = [0xd0, 0x6b, 0x18, 0xff];

/// The tray icon: a disc coloured by whether denoising is on, plus a warning
/// badge when audio isn't running at all.
///
/// Both states matter because the icon is the only surface this app has. Without
/// the colour there's no way to tell a bypassed RoomMute from a working one;
/// without the badge, a *stopped* one looks identical too, and the error dialog
/// is long gone by the time anyone notices their microphone is dead.
fn icon_rgba(enabled: bool, warning: bool) -> Vec<u8> {
    let mut rgba = vec![0u8; ICON_SIZE * ICON_SIZE * 4];
    let mut put = |x: usize, y: usize, c: [u8; 4]| {
        let i = (y * ICON_SIZE + x) * 4;
        rgba[i..i + 4].copy_from_slice(&c);
    };

    let disc = if enabled { ACTIVE } else { BYPASSED };
    let centre = ICON_SIZE as i32 / 2;
    for y in 0..ICON_SIZE {
        for x in 0..ICON_SIZE {
            let d2 = (x as i32 - centre).pow(2) + (y as i32 - centre).pow(2);
            if d2 <= 14 * 14 {
                put(x, y, disc);
            }
        }
    }
    if !warning {
        return rgba;
    }

    // Warning badge, bottom-right: dark triangle outline, amber fill, and an
    // exclamation mark punched back out in the dark colour.
    const DARK: [u8; 4] = [0x1c, 0x1c, 0x1c, 0xff];
    const AMBER: [u8; 4] = [0xf5, 0xa6, 0x23, 0xff];
    for y in 0..ICON_SIZE {
        for x in 0..ICON_SIZE {
            if in_triangle(x, y, 14, 31, 23, 9.0) {
                put(x, y, DARK);
            }
            if in_triangle(x, y, 17, 29, 23, 6.0) {
                put(x, y, AMBER);
            }
        }
    }
    for y in 20..=25 {
        put(22, y, DARK);
        put(23, y, DARK);
    }
    for y in 27..=28 {
        put(22, y, DARK);
        put(23, y, DARK);
    }
    rgba
}

fn build_icon(enabled: bool, warning: bool) -> tray_icon::Icon {
    tray_icon::Icon::from_rgba(
        icon_rgba(enabled, warning),
        ICON_SIZE as u32,
        ICON_SIZE as u32,
    )
    .expect("valid icon")
}

#[cfg(test)]
mod tests {
    use super::*;

    // The cable download URL lives in firstrun.rs, which is the only place
    // that still offers it, and is pinned by a test there.

    #[test]
    fn explorer_is_resolved_absolutely_and_exists() {
        let p = explorer_path();
        assert!(p.is_absolute(), "must not be resolved through PATH");
        assert_eq!(p.file_name().unwrap(), "explorer.exe");
        assert!(p.exists(), "expected the real Explorer at {}", p.display());
    }

    #[test]
    fn system_default_mic_is_marked_in_the_label() {
        // Unranked entries are padded so their names line up under ranked ones.
        let (label, enabled) = mic_entry("Yeti", false, None, None);
        assert_eq!(label.trim(), "Yeti");
        assert!(
            !label.trim_start().starts_with(char::is_numeric),
            "no rank: {label}"
        );
        assert!(enabled);

        let (label, enabled) = mic_entry("Yeti", true, None, Some(0));
        assert!(label.contains("Yeti") && label.contains("system default"));
        assert!(
            label.trim_start().starts_with("1."),
            "rank should lead: {label}"
        );
        assert!(enabled);
    }

    /// Selecting a cable's output would have RoomMute record from the cable
    /// it writes to. Not selectable, and the menu says why.
    #[test]
    fn a_cables_own_output_cannot_be_chosen_as_the_microphone() {
        let (label, enabled) = mic_entry(
            "CABLE Output (VB-Audio Virtual Cable)",
            false,
            Some("VB-Cable"),
            None,
        );
        assert!(!enabled, "must be greyed out");
        assert!(label.contains("loop"), "should say why: {label}");
        assert!(
            label.contains("VB-Cable"),
            "should name the product: {label}"
        );
    }

    /// The trap that actually bit: installing a cable makes it the system
    /// default, so "Windows default" silently becomes the looping choice.
    #[test]
    fn windows_default_is_disabled_when_the_default_is_a_cable() {
        let (label, enabled) = default_entry(None);
        assert_eq!(label, "Windows default");
        assert!(enabled);

        let (label, enabled) = default_entry(Some("VB-Cable"));
        assert!(!enabled, "following the default would loop here");
        assert!(
            label.contains("VB-Cable") && label.contains("loop"),
            "{label}"
        );
    }

    /// A cable being present must not disable ordinary microphones.
    #[test]
    fn real_microphones_stay_selectable_alongside_a_cable() {
        let (_, enabled) = mic_entry("Microphone (fifine Microphone)", true, None, Some(1));
        assert!(enabled);
    }

    fn count(px: &[u8], colour: [u8; 4]) -> usize {
        px.chunks(4).filter(|p| *p == colour).count()
    }

    #[test]
    fn every_icon_is_the_size_tray_icon_expects() {
        for enabled in [false, true] {
            for warning in [false, true] {
                assert_eq!(icon_rgba(enabled, warning).len(), ICON_SIZE * ICON_SIZE * 4);
            }
        }
    }

    #[test]
    fn on_and_off_are_different_colours_not_just_different_brightness() {
        let on = icon_rgba(true, false);
        let off = icon_rgba(false, false);
        assert_ne!(on, off);
        assert!(count(&on, ACTIVE) > 400 && count(&on, BYPASSED) == 0);
        assert!(count(&off, BYPASSED) > 400 && count(&off, ACTIVE) == 0);

        // Hue must actually differ: at tray size a brightness-only change is
        // invisible. Teal is green-dominant, the bypass colour red-dominant.
        assert!(ACTIVE[1] > ACTIVE[0], "active should be green-dominant");
        assert!(BYPASSED[0] > BYPASSED[1], "bypassed should be red-dominant");
    }

    #[test]
    fn the_warning_badge_is_visible_in_both_toggle_states() {
        const AMBER: [u8; 4] = [0xf5, 0xa6, 0x23, 0xff];
        for enabled in [false, true] {
            let ok = icon_rgba(enabled, false);
            let bad = icon_rgba(enabled, true);
            assert_ne!(ok, bad, "error state must look different");
            assert!(
                count(&bad, AMBER) > 20,
                "badge too small to see: {} px",
                count(&bad, AMBER)
            );
            assert_eq!(count(&ok, AMBER), 0);
        }
    }

    #[test]
    fn the_badge_sits_in_the_corner_not_over_the_whole_icon() {
        let bad = icon_rgba(true, true);
        // Top-left must stay the plain disc colour, so the icon is still
        // recognisable and the on/off hue still readable.
        let i = (8 * ICON_SIZE + 12) * 4;
        assert_eq!(&bad[i..i + 4], &ACTIVE);
    }

    // ---- the watchdog -------------------------------------------------
    //
    // This is the app's most important safety feature. WASAPI reports a dead
    // capture stream by simply never signalling again: no error, no callback,
    // the icon stays teal and the log says "running" while the microphone has
    // been off the air for a quarter of an hour. Driving `observe` with an
    // explicit clock is what makes those timings testable at all.

    fn at(base: Instant, secs: f32) -> Instant {
        base + Duration::from_secs_f32(secs)
    }

    #[test]
    fn a_counter_that_keeps_moving_is_never_disturbed() {
        let base = Instant::now();
        let mut h = Health::new_at(base);
        for i in 1..=20u64 {
            let action = h.observe(i, i, at(base, i as f32));
            assert_eq!(action, HealthAction::default(), "tick {i} was acted on");
        }
        assert!(!h.stalled);
    }

    /// A quiet room still produces frames — silence is data. Only a frozen
    /// counter means the stream itself has died.
    #[test]
    fn a_brief_gap_is_not_a_stall() {
        let base = Instant::now();
        let mut h = Health::new_at(base);
        h.observe(1, 1, base);
        assert_eq!(h.observe(1, 1, at(base, 1.9)), HealthAction::default());
        assert!(!h.stalled, "1.9s is under the 2s threshold");
    }

    #[test]
    fn a_frozen_counter_badges_the_icon_immediately() {
        let base = Instant::now();
        let mut h = Health::new_at(base);
        h.observe(1, 1, base);

        let action = h.observe(1, 1, at(base, 2.1));
        assert!(action.redraw, "the icon must stop looking healthy");
        assert!(h.stalled);
    }

    /// The badge appears as soon as the stall is detected, but the first
    /// recovery attempt waits for the recovery interval measured from
    /// start-up. That grace matters: a device still settling in the first
    /// seconds after launch should not be torn down and reopened.
    ///
    /// A device that dies *later* is not delayed — its last recovery is by
    /// then long past, so the attempt goes out on the same tick as the badge.
    #[test]
    fn recovery_waits_out_the_start_up_grace_but_not_a_later_failure() {
        let base = Instant::now();
        let mut h = Health::new_at(base);
        h.observe(1, 1, base);
        assert!(
            !h.observe(1, 1, at(base, 2.1)).restart,
            "still within grace"
        );
        assert!(h.observe(1, 1, at(base, 5.1)).restart, "grace is over");

        // Now the same failure arriving after ten minutes of healthy audio.
        let base = Instant::now();
        let mut h = Health::new_at(base);
        h.observe(1, 1, at(base, 600.0));
        let action = h.observe(1, 1, at(base, 602.1));
        assert!(action.redraw && action.restart, "must not wait: {action:?}");
    }

    /// A device that stays gone must not be reopened every tick.
    #[test]
    fn recovery_is_not_attempted_more_than_once_per_interval() {
        let base = Instant::now();
        let mut h = Health::new_at(base);
        h.observe(1, 1, base);

        assert!(h.observe(1, 1, at(base, 5.1)).restart);

        // Ticks arrive every 500 ms; none of these may retry.
        for t in [5.6, 6.1, 7.0, 9.0, 10.0] {
            let action = h.observe(1, 1, at(base, t));
            assert!(!action.restart, "retried after only {}s", t - 5.1);
            assert!(!action.redraw, "already badged at 2.1s");
        }

        // Past the interval, try again.
        assert!(h.observe(1, 1, at(base, 10.2)).restart);
    }

    /// The counter moving again is the only evidence of recovery there is.
    #[test]
    fn audio_coming_back_clears_the_badge() {
        let base = Instant::now();
        let mut h = Health::new_at(base);
        h.observe(1, 1, base);
        assert!(h.observe(1, 1, at(base, 2.1)).redraw);
        assert!(h.stalled);

        let action = h.observe(2, 2, at(base, 2.5));
        assert!(action.redraw, "the badge has to come off");
        assert!(!action.restart);
        assert!(!h.stalled);

        // And a later freeze is treated as a fresh stall, not a continuation.
        assert_eq!(h.observe(2, 2, at(base, 3.0)), HealthAction::default());
        assert!(h.observe(2, 2, at(base, 4.6)).redraw, "must badge again");
    }

    /// The first observation happens before any audio has flowed. Starting
    /// from zero must not read as an immediate stall.
    #[test]
    fn a_pipeline_that_has_not_produced_anything_yet_is_given_time() {
        let base = Instant::now();
        let mut h = Health::new_at(base);
        assert_eq!(h.observe(0, 0, base), HealthAction::default());
        assert_eq!(h.observe(0, 0, at(base, 1.0)), HealthAction::default());
        // But a pipeline that never produces a single frame is still broken.
        assert!(h.observe(0, 0, at(base, 2.5)).redraw);
    }

    /// The output half can die on its own: reconfigure the virtual cable in
    /// Windows sound settings and its endpoint is invalidated while the
    /// microphone is untouched. Capture and DSP carry on, so the frame counter
    /// keeps climbing — and everyone on the call stops hearing anything.
    /// Watching only that counter reports perfect health throughout.
    #[test]
    fn a_dead_output_is_caught_even_while_capture_keeps_running() {
        let base = Instant::now();
        let mut h = Health::new_at(base);

        // Healthy: both sides moving.
        for i in 1..=4u64 {
            let t = at(base, i as f32 * 0.5);
            assert_eq!(h.observe(i, i, t), HealthAction::default());
        }

        // Render stops asking for audio. Capture carries on regardless.
        let action = h.observe(20, 4, at(base, 4.6));
        assert!(
            action.redraw,
            "a dead output must badge the icon even though frames still flow"
        );
        assert!(h.stalled);
    }

    /// And the mirror image, so the two counters are not silently swapped.
    #[test]
    fn a_dead_capture_is_caught_even_while_the_output_keeps_polling() {
        let base = Instant::now();
        let mut h = Health::new_at(base);
        h.observe(1, 1, base);

        // Ring B empties, so render keeps polling and underrunning.
        let action = h.observe(1, 50, at(base, 2.1));
        assert!(action.redraw, "a dead microphone must still be caught");
    }

    /// The bug this branch exists to fix, one level up: when a restart fails
    /// there is no pipeline left, so there are no counters to watch. Giving up
    /// there means a mic unplugged for a few seconds is never picked up again
    /// and the only way out is the menu — for the rest of the session.
    #[test]
    fn recovery_keeps_trying_when_there_is_no_pipeline_at_all() {
        let base = Instant::now();
        let mut h = Health::new_at(base);

        // A restart has just failed; nothing is running.
        assert!(!h.observe_stopped(at(base, 1.0)).restart, "within grace");
        assert!(h.stalled, "and it must not look healthy meanwhile");

        let first = h.observe_stopped(at(base, 5.1));
        assert!(first.restart, "the retry loop must keep running");

        // Still paced, not hammering the device.
        for t in [5.6, 7.0, 9.9] {
            assert!(!h.observe_stopped(at(base, t)).restart, "retried too soon");
        }
        assert!(h.observe_stopped(at(base, 10.2)).restart, "and again after");
    }

    /// Once a retry succeeds, the counters restart from whatever the new
    /// pipeline reports — not from the dead one's values, which would read as
    /// an immediate stall.
    #[test]
    fn a_successful_restart_rebases_both_counters() {
        let base = Instant::now();
        let mut h = Health::new_at(base);
        h.observe(900, 900, base);
        h.observe(900, 900, at(base, 2.1));
        assert!(h.stalled);

        // A fresh pipeline starts its counters at zero.
        let now = at(base, 6.0);
        h.capture.reset(0, now);
        h.render.reset(0, now);
        h.stalled = false;

        assert_eq!(
            h.observe(1, 1, at(base, 6.2)),
            HealthAction::default(),
            "the new pipeline's low counter must not read as a stall"
        );
    }

    // ---- menu choices -------------------------------------------------

    #[test]
    fn choosing_windows_default_stops_expressing_a_preference() {
        let mut cfg = Config {
            microphones: vec!["Yeti".into(), "Webcam".into()],
            ..Config::default()
        };
        choose_microphone(&mut cfg, "");
        assert!(
            cfg.microphones.is_empty(),
            "the default entry means follow Windows, not remember a name"
        );
    }

    #[test]
    fn choosing_a_microphone_promotes_it_above_the_rest() {
        let mut cfg = Config {
            microphones: vec!["Yeti".into(), "Webcam".into()],
            ..Config::default()
        };
        choose_microphone(&mut cfg, "Webcam");
        assert_eq!(cfg.microphones, vec!["Webcam", "Yeti"]);
    }

    /// Switching to RNNoise and back must not lose where the model was, or
    /// the user has to configure it again every time they experiment.
    #[test]
    fn switching_away_from_the_model_remembers_where_it_was() {
        let dir = std::env::temp_dir().join(format!("roommute-tray-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("dfn3.onnx");
        std::fs::write(&model, b"stand-in").unwrap();

        let mut cfg = Config {
            model_path: model.to_string_lossy().into_owned(),
            use_onnx: false,
            ..Config::default()
        };

        choose_denoiser(&mut cfg, true);
        assert!(cfg.use_onnx);
        assert_eq!(cfg.model_path, model.to_string_lossy());

        choose_denoiser(&mut cfg, false);
        assert!(!cfg.use_onnx);
        assert_eq!(
            cfg.model_path,
            model.to_string_lossy(),
            "path was forgotten"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- what the tooltip says ----------------------------------------

    #[test]
    fn the_cpu_meter_reads_a_tenth_of_realtime_as_ten_percent() {
        // A frame is 10 ms of audio, so 1 ms of DSP per frame is 10%.
        assert!((cpu_percent(100, 100 * 1_000_000) - 10.0).abs() < 1e-9);
        // Exactly keeping up is 100%.
        assert!((cpu_percent(50, 50 * 10_000_000) - 100.0).abs() < 1e-9);
        // No frames yet must not divide by zero.
        assert_eq!(cpu_percent(0, 0), 0.0);
        assert_eq!(cpu_percent(0, 12_345), 0.0);
    }

    #[test]
    fn the_tooltip_says_whether_denoising_is_on() {
        let on = tooltip_text("DeepFilterNet3", true, 4.25, 1.5);
        assert!(on.contains("DeepFilterNet3") && on.contains("ON"));
        assert!(on.contains("4.2") || on.contains("4.3"), "{on}");

        let off = tooltip_text("RNNoise", false, 0.0, 0.0);
        assert!(off.contains("BYPASS"));
        assert!(!off.contains("ON  "), "BYPASS must not also read as ON");
    }

    /// The two states where there are no statistics worth showing have to say
    /// something useful instead — "stopped" with no hint is a support ticket.
    #[test]
    fn a_broken_pipeline_explains_itself_in_the_tooltip() {
        assert!(status_text(true, true).unwrap().contains("no audio"));
        assert!(status_text(false, false)
            .unwrap()
            .contains("pick a microphone"));
        assert!(
            status_text(true, false).is_none(),
            "a healthy pipeline should show its real statistics"
        );
    }
}

#[cfg(test)]
mod failure_log_tests {
    use std::time::{Duration, Instant};

    use super::{FailureLog, REPEAT_FAILURE_AFTER};

    fn at(base: Instant, secs: f64) -> Instant {
        base + Duration::from_secs_f64(secs)
    }

    /// A machine with no working microphone retries every five seconds for as
    /// long as it is switched on. One real report arrived with 6,000 copies of
    /// a single sentence, which is how the lines that mattered got lost.
    #[test]
    fn the_same_failure_is_not_repeated_every_five_seconds() {
        let base = Instant::now();
        let mut log = FailureLog::default();

        assert!(log.should_report("no microphone is available", base));
        for i in 1..60 {
            assert!(
                !log.should_report("no microphone is available", at(base, i as f64 * 5.0)),
                "attempt {i} said the same thing; saying it again helps nobody"
            );
        }
    }

    /// A different reason means something changed, and that is worth knowing
    /// straight away — it is the difference between "no microphone" and "this
    /// microphone will not open".
    #[test]
    fn a_changed_reason_is_reported_at_once() {
        let base = Instant::now();
        let mut log = FailureLog::default();

        assert!(log.should_report("no microphone is available", base));
        assert!(
            log.should_report("GetMixFormat failed (0x8007007E)", at(base, 5.0)),
            "the reason changed, so it is news"
        );
    }

    /// Silence for hours is its own failure: an outage still has to be visible
    /// to anyone reading the log later.
    #[test]
    fn a_long_outage_is_repeated_occasionally() {
        let base = Instant::now();
        let mut log = FailureLog::default();
        let reason = "no microphone is available";

        assert!(log.should_report(reason, base));
        let just_before = REPEAT_FAILURE_AFTER.as_secs_f64() - 1.0;
        assert!(!log.should_report(reason, at(base, just_before)));
        assert!(
            log.should_report(reason, at(base, REPEAT_FAILURE_AFTER.as_secs_f64() + 0.1)),
            "a five-minute reminder keeps a long outage visible"
        );
    }

    /// Recovering resets it, so the next failure is reported immediately
    /// rather than swallowed as a repeat of something hours old.
    #[test]
    fn recovery_makes_the_next_failure_news_again() {
        let base = Instant::now();
        let mut log = FailureLog::default();
        let reason = "no microphone is available";

        assert!(log.should_report(reason, base));
        log.clear();
        assert!(log.should_report(reason, at(base, 5.0)));
    }
}
