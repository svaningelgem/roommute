//! RoomMute — real-time mic noise cancellation for Windows.
//!
//! Architecture:
//!   physical mic ─► WASAPI capture ─► ring buffer A ─► DSP (DeepFilterNet3)
//!                                                               │
//!                                                               ▼
//!                                                       ring buffer B
//!                                                               │
//!                                              WASAPI render ──┘──► VB-Cable Input
//!
//! The tray UI lives on the main thread. Audio runs on three dedicated
//! MMCSS "Pro Audio" threads spawned by the audio-io and pipeline modules.

// GUI subsystem: double-clicking goes straight to the tray with no console
// window behind it. The CLI modes still work — `console::attach_to_parent`
// borrows the terminal's console when we were launched from one, so
// `--list-devices` and friends print where you'd expect. See console.rs.
#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(windows)]
mod autostart;
mod banner;
mod config;
#[cfg(windows)]
mod console;
#[cfg(windows)]
mod firstrun;
mod log_format;
mod offline;
mod pipeline;
#[cfg(windows)]
mod tray;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::info;

/// Roll the log over once it gets big. It's an append-only file that records
/// every device name we see on every run, so left alone it grows forever and
/// accumulates more of the user's hardware inventory than anyone expects to
/// hand over when they paste a log into a bug report.
const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

fn rotate_log_if_large(log_file: &std::path::Path, max_bytes: u64) {
    let too_big = std::fs::metadata(log_file)
        .map(|m| m.len() > max_bytes)
        .unwrap_or(false);
    if too_big {
        let _ = std::fs::rename(log_file, log_file.with_extension("log.old"));
    }
}

fn init_tracing() {
    let log_dir = config::log_dir();
    let _ = std::fs::create_dir_all(&log_dir);
    let log_file = log_dir.join("roommute.log");
    rotate_log_if_large(&log_file, MAX_LOG_BYTES);

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)
        .ok();

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,roommute=debug"));

    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    // Stdout: green-themed, custom format; ANSI disabled in the layer
    // because GreenFormat writes its own escape codes.
    let stdout_layer = fmt::layer()
        .with_ansi(false)
        .event_format(log_format::GreenFormat::new());

    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(stdout_layer);

    // File: plain, no colors, default format. Useful for sharing logs.
    if let Some(file) = file {
        registry
            .with(
                fmt::layer()
                    .with_writer(file)
                    .with_ansi(false)
                    .with_target(false),
            )
            .init();
    } else {
        registry.init();
    }
}

/// Report a fatal problem. With no console attached (the normal tray launch)
/// an error message printed to stderr goes nowhere, and the app looks like it
/// simply failed to start.
/// What went wrong at startup, and whether the tray can offer a way out.
#[cfg(windows)]
pub struct StartupProblem {
    pub message: String,
}

/// Does this error chain bottom out in "no virtual cable installed"?
#[cfg(windows)]
fn is_missing_cable(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        matches!(
            c.downcast_ref::<audio_io::AudioError>(),
            Some(audio_io::AudioError::VirtualCableMissing)
        )
    })
}

/// A yes/no dialog. Returns true if the user said yes.
#[cfg(windows)]
pub fn message_box_yes_no(text: &str) -> bool {
    use windows::core::HSTRING;
    use windows::Win32::UI::WindowsAndMessaging::{MessageBoxW, IDYES, MB_ICONWARNING, MB_YESNO};
    unsafe {
        MessageBoxW(
            None,
            &HSTRING::from(text),
            &HSTRING::from("RoomMute"),
            MB_YESNO | MB_ICONWARNING,
        ) == IDYES
    }
}

/// Hand a URL to Explorer, which opens it in the default browser. Absolute
/// path, never resolved through PATH.
#[cfg(windows)]
pub fn open_url(url: &str) {
    let explorer = std::env::var_os("SystemRoot")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(r"C:\Windows"))
        .join("explorer.exe");
    let _ = std::process::Command::new(explorer).arg(url).spawn();
}

/// Informational, not an error. Same box, different icon — a welcome with a
/// red cross on it reads as a failure.
#[cfg(windows)]
pub fn message_box_info(text: &str) {
    use windows::core::HSTRING;
    use windows::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONINFORMATION, MB_OK};
    unsafe {
        MessageBoxW(
            None,
            &HSTRING::from(text),
            &HSTRING::from("RoomMute"),
            MB_OK | MB_ICONINFORMATION,
        );
    }
}

#[cfg(windows)]
pub fn message_box(text: &str) {
    use windows::core::HSTRING;
    use windows::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONERROR, MB_OK};
    unsafe {
        MessageBoxW(
            None,
            &HSTRING::from(text),
            &HSTRING::from("RoomMute"),
            MB_OK | MB_ICONERROR,
        );
    }
}

#[cfg(windows)]
fn main() -> Result<()> {
    // Anything that writes to stdout/stderr needs this first: Rust caches the
    // standard handles on first use, so redirecting them later is too late.
    let has_console = console::attach_to_parent();

    match real_main(has_console) {
        Ok(()) => Ok(()),
        Err(e) => {
            // A console gets the full anyhow chain; without one, a dialog is
            // the only way the user learns anything at all.
            if has_console {
                Err(e)
            } else {
                message_box(&format!("{e:#}"));
                std::process::exit(1);
            }
        }
    }
}

/// Which model offline `--denoise` should load.
///
/// The flag wins; otherwise whatever the tray would run. Returning `None` when
/// no flag was given sent offline runs to RNNoise while the app itself used
/// DeepFilterNet3 — so `--denoise`, the mode people use to check what the app
/// does, reported a different backend's numbers.
fn offline_model(
    flag: Option<&str>,
    configured: Option<PathBuf>,
    force_rnnoise: bool,
) -> Option<PathBuf> {
    if force_rnnoise {
        return None;
    }
    flag.map(PathBuf::from).or(configured)
}

#[cfg(windows)]
fn real_main(has_console: bool) -> Result<()> {
    let args = parse_args();
    // The banner is terminal decoration; skip it when there's no terminal.
    if has_console {
        banner::print();
    }
    if args.help {
        print_help();
        return Ok(());
    }

    if args.list_devices {
        return list_devices();
    }

    init_tracing();

    if let Some((input, output)) = &args.denoise {
        let model = offline_model(
            args.model.as_deref(),
            config::Config::load_or_default().active_model(),
            args.rnnoise,
        );
        match (&model, args.rnnoise) {
            (Some(m), _) => info!(model = %m.display(), "denoising with"),
            (None, true) => info!("using the built-in RNNoise, as asked"),
            (None, false) => info!("no model found; using the built-in RNNoise"),
        }
        return offline::denoise_file(
            std::path::Path::new(input),
            std::path::Path::new(output),
            model.as_deref(),
            args.atten.unwrap_or(0.0),
        );
    }
    if let Some((seconds, path)) = &args.record {
        let device = match args.mic.as_deref() {
            Some(filter) => resolve_mic_by_substring(filter)?,
            None => String::new(),
        };
        return offline::record(&device, *seconds, std::path::Path::new(path));
    }

    // Single-instance lock via a named mutex. Prevents two trays from
    // fighting over the same audio devices.
    //
    // Launching an already-running tray app is not an error — it's what
    // happens when someone double-clicks the shortcut, or when autostart
    // races a manual launch. The existing instance is already doing the job,
    // so this one bows out without a word. Only a *failure* to determine
    // whether we're alone is worth reporting.
    let _lock = match single_instance::acquire() {
        Ok(lock) => lock,
        Err(single_instance::AlreadyRunning) => {
            info!("another instance is already running; leaving it to it");
            return Ok(());
        }
    };

    info!("RoomMute starting");

    // Always print the device inventory at startup so users can identify
    // which mic to pick — especially important to spot Bluetooth-HFP
    // endpoints (which sound terrible) vs USB/wired mics.
    log_input_devices();

    let mut cfg_value = config::Config::load_or_default();
    // Apply CLI overrides. Not persisted — that's what the tray menu is for.
    if let Some(mic_filter) = args.mic.as_deref() {
        match resolve_mic_by_substring(mic_filter) {
            Ok(name) => {
                info!(filter = mic_filter, resolved = %name, "--mic override applied");
                // Front of the preference list, not the legacy single field:
                // the pipeline reads `microphones`, so writing anywhere else
                // means the flag is silently ignored. Keeping the rest of the
                // list intact leaves the usual fallbacks in place if the
                // requested device disappears mid-session.
                cfg_value.prefer_microphone(&name);
            }
            Err(e) => {
                anyhow::bail!("--mic '{}' did not match any input device: {e}", mic_filter);
            }
        }
    }

    let cfg = Arc::new(parking_lot_compat::RwLock::new(cfg_value));
    // A failure here is usually a missing VB-Cable or an unplugged mic —
    // both fixable without restarting. Show the tray anyway so the user has
    // the microphone picker to hand, rather than exiting on them.
    let (pipeline, startup_error) = match pipeline::Pipeline::start(cfg.clone()) {
        Ok(p) => {
            info!(denoiser = p.denoiser_name(), "audio pipeline running");
            (Some(p), None)
        }
        // The one failure with a scripted way out, handled before the tray
        // exists because both answers end the run.
        Err(e) if is_missing_cable(&e) => {
            tracing::error!("no virtual audio cable installed");
            match firstrun::cable_missing() {
                firstrun::CableChoice::OpenSite => open_url(firstrun::CABLE_URL),
                firstrun::CableChoice::Close => {
                    // Otherwise this dialog greets them on every single boot,
                    // a loop they could only escape via the registry.
                    if autostart::is_enabled() {
                        info!("turning off start-with-Windows so this doesn't repeat");
                        if let Err(e) = autostart::set(false) {
                            tracing::warn!(error = %e, "could not turn off start-with-Windows");
                        }
                    }
                }
            }
            return Ok(());
        }
        Err(e) => {
            tracing::error!(error = ?e, "audio pipeline did not start");
            let problem = StartupProblem {
                message: format!("RoomMute is running, but audio isn't:\n\n{e:#}"),
            };
            (None, Some(problem))
        }
    };

    tray::run(cfg, pipeline, startup_error)?;
    info!("RoomMute exiting");
    Ok(())
}

#[derive(Default)]
struct CliArgs {
    help: bool,
    list_devices: bool,
    mic: Option<String>,
    /// `--record <SECONDS> <FILE.wav>`
    record: Option<(f32, String)>,
    /// `--denoise <IN.wav> <OUT.wav>`
    denoise: Option<(String, String)>,
    /// `--model <FILE>` — overrides config for the offline tools.
    model: Option<String>,
    /// `--rnnoise` — ignore any model and use the built-in backend, which is
    /// how the comparison in the README is produced.
    rnnoise: bool,
    /// `--atten <DB>` — attenuation limit for models that support one.
    atten: Option<f32>,
}

fn parse_args() -> CliArgs {
    parse_args_from(std::env::args().skip(1))
}

fn parse_args_from<I, S>(args: I) -> CliArgs
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut out = CliArgs::default();
    let mut args = args.into_iter().map(Into::into);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => out.help = true,
            "--list-devices" => out.list_devices = true,
            "--mic" => out.mic = args.next(),
            other if other.starts_with("--mic=") => {
                out.mic = Some(other["--mic=".len()..].to_string());
            }
            "--model" => out.model = args.next(),
            "--rnnoise" => out.rnnoise = true,
            "--atten" => out.atten = args.next().and_then(|v| v.parse().ok()),
            "--record" => {
                if let (Some(secs), Some(path)) = (args.next(), args.next()) {
                    out.record = secs.parse().ok().map(|s| (s, path));
                }
            }
            "--denoise" => {
                if let (Some(input), Some(output)) = (args.next(), args.next()) {
                    out.denoise = Some((input, output));
                }
            }
            _ => {} // ignore unknowns silently
        }
    }
    out
}

fn print_help() {
    println!("{HELP}");
}

const HELP: &str = "RoomMute — real-time mic noise cancellation\n\
         \n\
         USAGE:\n\
             roommute.exe [OPTIONS]\n\
         \n\
         With no options it starts in the tray and cleans your microphone.\n\
         \n\
         OPTIONS:\n\
             --list-devices         Print all input/output devices and exit.\n\
             --mic <SUBSTRING>      Pick the input device whose friendly name contains\n\
                                    SUBSTRING (case-insensitive). Useful when your\n\
                                    default mic is a Bluetooth headset and you want\n\
                                    a USB mic instead.\n\
             --record <SECS> <WAV>  Record from the chosen mic straight to a WAV file.\n\
             --denoise <IN> <OUT>   Clean up an existing mono 48 kHz WAV file.\n\
             --model <FILE>         Use this model for --denoise.\n\
             --atten <DB>           Attenuation limit for models that support one.\n\
             -h, --help             Show this help.\n\
         \n\
         CONFIG FILE:\n\
             %APPDATA%\\RoomMute\\config.toml\n\
             \n\
             microphones     list of mic names, best first; falls down the list\n\
                             as devices come and go, Windows' default is the\n\
                             final fallback\n\
             output_device   where cleaned audio goes; empty auto-detects the\n\
                             virtual cable\n\
             enabled         master on/off\n\
             use_onnx        run the high-quality model rather than RNNoise\n\
             model_path      where that model lives\n\
             attenuation_db  how hard to suppress, 6 = subtle, 100 = maximum\n\
             \n\
             Devices are named, not numbered: ids are opaque GUIDs that change\n\
             whenever you replug the device.\n";

#[cfg(windows)]
fn list_devices() -> Result<()> {
    use audio_io::devices::DeviceList;
    let list = DeviceList::enumerate().context("enumerating devices")?;
    println!("Capture (input) devices:");
    for d in &list.capture {
        let tag = if d.is_default { " [default]" } else { "" };
        println!("  - {}{}", d.friendly_name, tag);
        println!("    id: {}", d.id);
    }
    println!("\nRender (output) devices:");
    for d in &list.render {
        let tag = if d.is_default { " [default]" } else { "" };
        let vb = match d.virtual_cable_input() {
            Some(product) => format!("  [{product}]"),
            None => String::new(),
        };
        println!("  - {}{}{}", d.friendly_name, tag, vb);
        println!("    id: {}", d.id);
    }
    Ok(())
}

#[cfg(not(windows))]
fn list_devices() -> Result<()> {
    anyhow::bail!("--list-devices only works on Windows.")
}

#[cfg(windows)]
fn log_input_devices() {
    use audio_io::devices::DeviceList;
    let Ok(list) = DeviceList::enumerate() else {
        return;
    };
    for d in &list.capture {
        let default = if d.is_default { " (default)" } else { "" };
        info!(name = %d.friendly_name, id = %d.id, "input device available{}", default);
    }
}

#[cfg(windows)]
fn resolve_mic_by_substring(needle: &str) -> Result<String> {
    use audio_io::devices::DeviceList;
    let needle_lc = needle.to_ascii_lowercase();
    let list = DeviceList::enumerate()?;
    let matches: Vec<_> = list
        .capture
        .iter()
        .filter(|d| d.friendly_name.to_ascii_lowercase().contains(&needle_lc))
        .collect();
    match matches.as_slice() {
        [] => Err(anyhow::anyhow!("no capture device matched")),
        [d] => Ok(d.friendly_name.clone()),
        many => {
            let names: Vec<_> = many.iter().map(|d| d.friendly_name.as_str()).collect();
            Err(anyhow::anyhow!(
                "ambiguous: {} devices matched ({}). Refine the substring.",
                many.len(),
                names.join(", ")
            ))
        }
    }
}

#[cfg(not(windows))]
fn main() -> Result<()> {
    init_tracing();
    anyhow::bail!("RoomMute is Windows-only. Build for x86_64-pc-windows-msvc.");
}

#[cfg(windows)]
mod single_instance {
    use windows::core::w;
    use windows::Win32::Foundation::{GetLastError, ERROR_ALREADY_EXISTS, HANDLE};
    use windows::Win32::System::Threading::CreateMutexW;

    /// Another tray instance holds the lock. Not an error — see `acquire`.
    #[derive(Debug)]
    pub struct AlreadyRunning;

    pub struct Lock(HANDLE);

    impl Drop for Lock {
        fn drop(&mut self) {
            if self.0.is_invalid() {
                return; // Never opened; nothing to close.
            }
            unsafe {
                let _ = windows::Win32::Foundation::CloseHandle(self.0);
            }
        }
    }

    /// The name is deliberately session-local (`Local\`, not `Global\`): a
    /// `Global\` name is visible to every session on the machine, so any
    /// other user or process could squat it and permanently block startup —
    /// and creating one needs SeCreateGlobalPrivilege, which a standard user
    /// account doesn't have. One tray per login session is what we actually
    /// want anyway.
    pub fn acquire() -> Result<Lock, AlreadyRunning> {
        acquire_named(w!("Local\\RoomMute.SingleInstance"))
    }

    /// Split out so tests can contend on a name of their own rather than on
    /// the real one — which the user's actual tray may be holding.
    fn acquire_named(name: windows::core::PCWSTR) -> Result<Lock, AlreadyRunning> {
        unsafe {
            let h = match CreateMutexW(None, true, name) {
                Ok(h) => h,
                Err(e) => {
                    // We can't tell whether we're alone. Starting anyway is
                    // the lesser evil: refusing to run because a bookkeeping
                    // mutex failed would be a worse outcome than two trays
                    // briefly coexisting.
                    tracing::warn!(error = %e, "single-instance check unavailable; starting anyway");
                    return Ok(Lock(HANDLE::default()));
                }
            };
            if GetLastError() == ERROR_ALREADY_EXISTS {
                // We own this handle even when the mutex already existed.
                let _ = windows::Win32::Foundation::CloseHandle(h);
                return Err(AlreadyRunning);
            }
            Ok(Lock(h))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The whole point of the feature: the second tray must lose, and the
        /// name must free up once the winner exits.
        #[test]
        fn only_one_holder_at_a_time() {
            let name = w!("Local\\RoomMute.SingleInstance.test");

            let first = acquire_named(name).expect("nothing else holds the test name");
            assert!(
                acquire_named(name).is_err(),
                "a second instance must bow out"
            );

            drop(first);
            assert!(
                acquire_named(name).is_ok(),
                "the name must be reusable after the holder exits, or a crash \
                 would lock the user out until they log off"
            );
        }

        /// A null handle is what `acquire` returns when the mutex could not be
        /// created at all. Dropping it must not try to close it.
        #[test]
        fn dropping_a_lock_we_never_opened_is_harmless() {
            drop(Lock(HANDLE::default()));
        }
    }
}

/// Lightweight RwLock shim — we don't pull in parking_lot for a single
/// usage. std's RwLock is fine for config reads.
mod parking_lot_compat {
    pub use std::sync::RwLock;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("roommute-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    // `--denoise` used to pass `None` when there was no --model flag, which
    // sent it to RNNoise while the tray ran DeepFilterNet3. The numbers it
    // printed were therefore not the app's, and the README quoted them.
    // The README compares DeepFilterNet3 against RNNoise on the demo sample.
    // Since --denoise defaults to the configured model, producing the RNNoise
    // side of that comparison needs a way to say so, or the published table
    // is one a reader cannot reproduce.
    #[test]
    fn rnnoise_can_be_forced_for_comparison() {
        assert_eq!(
            offline_model(None, Some("configured.tar.gz".into()), true),
            None,
            "--rnnoise has to reach the built-in backend even when a model is configured"
        );
        assert!(parse_args_from(["--rnnoise"]).rnnoise);
    }

    #[test]
    fn denoise_falls_back_to_the_configured_model() {
        let configured = Some(std::path::PathBuf::from("beside-the-exe.tar.gz"));

        assert_eq!(
            offline_model(None, configured.clone(), false),
            configured,
            "without --model, offline denoising must use the same model as the tray"
        );
    }

    #[test]
    fn an_explicit_model_flag_wins_over_the_config() {
        assert_eq!(
            offline_model(Some("chosen.onnx"), Some("configured.tar.gz".into()), false),
            Some(std::path::PathBuf::from("chosen.onnx"))
        );
    }

    #[test]
    fn no_flag_and_no_config_means_the_built_in_fallback() {
        assert_eq!(offline_model(None, None, false), None);
    }

    #[test]
    fn small_logs_are_left_alone() {
        let dir = scratch_dir("small-log");
        let log = dir.join("roommute.log");
        std::fs::write(&log, b"a few lines of startup chatter").unwrap();

        rotate_log_if_large(&log, 1024);

        assert!(log.exists(), "small log should not be rotated");
        assert!(!log.with_extension("log.old").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_logs_are_rolled_over() {
        let dir = scratch_dir("big-log");
        let log = dir.join("roommute.log");
        std::fs::write(&log, vec![b'x'; 2048]).unwrap();

        rotate_log_if_large(&log, 1024);

        assert!(!log.exists(), "oversized log should have been moved aside");
        let rolled = log.with_extension("log.old");
        assert!(rolled.exists(), "expected roommute.log.old");
        assert_eq!(std::fs::metadata(&rolled).unwrap().len(), 2048);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_log_is_not_an_error() {
        let dir = scratch_dir("no-log");
        // First run: nothing to rotate, and nothing should blow up.
        rotate_log_if_large(&dir.join("roommute.log"), 1024);
        assert!(!dir.join("roommute.log.old").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cli_parses_mic_in_both_spellings() {
        // `--mic X` and `--mic=X` must resolve identically, and unknown flags
        // stay ignored rather than being swallowed as a device name.
        let split = parse_args_from(["--mic", "Yeti"]);
        let joined = parse_args_from(["--mic=Yeti"]);
        assert_eq!(split.mic.as_deref(), Some("Yeti"));
        assert_eq!(joined.mic.as_deref(), Some("Yeti"));

        let listed = parse_args_from(["--list-devices", "--nonsense"]);
        assert!(listed.list_devices);
        assert!(listed.mic.is_none());
    }

    #[test]
    fn cli_parses_the_two_argument_flags() {
        let a = parse_args_from(["--record", "60", "out.wav"]);
        assert_eq!(a.record, Some((60.0, "out.wav".to_string())));

        let b = parse_args_from(["--denoise", "in.wav", "out.wav"]);
        assert_eq!(b.denoise, Some(("in.wav".into(), "out.wav".into())));

        let c = parse_args_from(["--model", "m.onnx", "--atten", "12.5"]);
        assert_eq!(c.model.as_deref(), Some("m.onnx"));
        assert_eq!(c.atten, Some(12.5));

        for flag in ["-h", "--help"] {
            assert!(parse_args_from([flag]).help);
        }
    }

    /// Nothing here should ever consume the *next* flag as a value, and a
    /// duration that isn't a number must not silently become a no-op recording.
    #[test]
    fn cli_does_not_invent_values_for_incomplete_flags() {
        assert!(parse_args_from(["--record", "abc", "out.wav"])
            .record
            .is_none());
        assert!(parse_args_from(["--record", "60"]).record.is_none());
        assert!(parse_args_from(["--denoise", "only-one.wav"])
            .denoise
            .is_none());
        assert!(parse_args_from(["--atten", "loud"]).atten.is_none());
        assert!(parse_args_from(["--mic"]).mic.is_none());

        // Empty command line: the tray launch. Everything off.
        let none = parse_args_from(Vec::<String>::new());
        assert!(!none.help && !none.list_devices);
        assert!(none.mic.is_none() && none.record.is_none() && none.denoise.is_none());
    }

    /// The help text is the only documentation most people will read, and it
    /// once advertised config keys (`input_device_id`) that no longer existed.
    /// Tie it to the struct that actually gets parsed.
    #[test]
    fn help_describes_the_config_keys_that_exist() {
        let help = HELP;
        let cfg = toml::to_string(&config::Config::default()).unwrap();
        for key in ["microphones", "output_device", "enabled", "use_onnx"] {
            assert!(cfg.contains(key), "{key} is not a real config key any more");
            assert!(help.contains(key), "help does not mention `{key}`");
        }
        assert!(
            !help.contains("device_id"),
            "help still advertises id-based config; devices are named now"
        );
        for flag in ["--list-devices", "--mic", "--record", "--denoise"] {
            assert!(help.contains(flag), "help does not mention {flag}");
        }
    }

    /// `Instant::now() - d` panics when the machine has been up for less than
    /// `d`, and RoomMute starts at login, so boot is exactly when that
    /// happens. It shipped once as a "60 seconds ago" sentinel meaning "never
    /// logged".
    ///
    /// It cannot be tested by observation: `Instant` is opaque, has no
    /// constructor, and the only way to reach the panic is to run on a machine
    /// whose uptime is genuinely under the offset. A test asserting the
    /// pipeline builds passes on any developer machine and any CI runner,
    /// which makes it worse than no test — it looks like a guard and never
    /// fails. So check the property that is actually checkable: the pattern is
    /// not in the source. Use `Option<Instant>` (`None` = never) or
    /// `checked_sub` instead.
    #[test]
    fn no_time_arithmetic_on_a_fresh_instant() {
        let sources = [
            ("main.rs", include_str!("main.rs")),
            ("pipeline.rs", include_str!("pipeline.rs")),
            ("tray.rs", include_str!("tray.rs")),
            ("offline.rs", include_str!("offline.rs")),
            ("config.rs", include_str!("config.rs")),
            ("log_format.rs", include_str!("log_format.rs")),
        ];
        // Split so this test does not match its own source when it scans
        // main.rs — a self-match is a red for the wrong reason.
        let needle = concat!("Instant::now()", " - ");
        for (name, src) in sources {
            for (n, line) in src.lines().enumerate() {
                if line.trim_start().starts_with("//") {
                    continue; // Prose about the pattern, not the pattern.
                }
                assert!(
                    !line.contains(needle),
                    "{name}:{} subtracts from a fresh Instant, which panics on a \
                     machine that has just booted — and RoomMute starts at login:\n  {}",
                    n + 1,
                    line.trim()
                );
            }
        }
    }

    #[test]
    fn is_missing_cable_looks_through_the_whole_chain() {
        let buried = anyhow::Error::new(audio_io::AudioError::VirtualCableMissing)
            .context("opening the render device")
            .context("starting the audio pipeline");
        assert!(is_missing_cable(&buried), "must survive being wrapped");

        let other = anyhow::Error::new(audio_io::AudioError::DeviceInvalidated {
            context: "starting capture",
        })
        .context("starting the audio pipeline");
        assert!(!is_missing_cable(&other));
        assert!(!is_missing_cable(&anyhow::anyhow!("plain string error")));
    }
}
