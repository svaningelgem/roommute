use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Microphone to capture from, by the name Windows shows. Empty = follow
    /// the Windows default.
    ///
    /// A name rather than an endpoint id: ids are opaque GUIDs that nobody can
    /// edit by hand, and they don't survive replugging the device — Windows
    /// re-enumerates it with a fresh one. The name is stable, readable, and
    /// what the tray menu shows. Resolving name to id is our problem, not the
    /// config file's.
    #[serde(default, alias = "input_device_name")]
    pub input_device: String,
    /// Microphones in order of preference, by name, best first.
    ///
    /// A list rather than one choice because devices come and go: unplug the
    /// USB mic mid-call and the next one down takes over instead of the app
    /// stopping to ask. Windows' own default is always the final fallback, so
    /// the list can never strand you.
    #[serde(default)]
    pub microphones: Vec<String>,
    /// Where cleaned audio goes, by name. Empty = auto-detect a virtual cable.
    #[serde(default)]
    pub output_device: String,
    /// Master enable. When false, the pipeline runs in bypass mode (passes
    /// audio through without DSP) so toggling is instant.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Attenuation limit in dB for DeepFilterNet. 6.0 = subtle, 100.0 = max.
    #[serde(default = "default_atten")]
    pub attenuation_db: f32,
    // Start-at-login deliberately has no field here. It lives in
    // HKCU\...\CurrentVersion\Run, which Windows reads and the user can
    // inspect or remove without us. A config mirror could only ever drift
    // from it, and would silently do nothing if someone set it by hand.
    /// Path to an ONNX denoise model. Only honoured by builds made with
    /// `--features onnx`.
    #[serde(default)]
    pub model_path: String,
    /// Run DeepFilterNet3 instead of the built-in RNNoise. On by default:
    /// against a competing voice it gains 3.6 dB of SNR where RNNoise gains
    /// none at all, and in a room with several noise sources 9.6 dB against
    /// 6.8 dB. Falls back to RNNoise on its own if no model is present.
    ///
    /// The key is named for ONNX; the model runs through tract. Renaming it
    /// would silently reset the setting in every existing config.toml.
    #[serde(default = "default_true")]
    pub use_onnx: bool,
    /// Whether the welcome has been shown. Defaults to false so an existing
    /// config, written before this key existed, still gets it once.
    #[serde(default)]
    pub welcomed: bool,
}

fn default_true() -> bool {
    true
}
fn default_atten() -> f32 {
    100.0
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_device: String::new(),
            microphones: Vec::new(),
            output_device: String::new(),
            enabled: true,
            attenuation_db: default_atten(),
            model_path: String::new(),
            use_onnx: true,
            welcomed: false,
        }
    }
}

impl Config {
    pub fn load_or_default() -> Self {
        match Self::load() {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "config load failed; using defaults");
                Self::default()
            }
        }
    }

    pub fn load() -> anyhow::Result<Self> {
        Self::load_from(&config_path())
    }

    fn load_from(path: &std::path::Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        Self::parse(&std::fs::read_to_string(path)?)
    }

    fn parse(text: &str) -> anyhow::Result<Self> {
        let mut cfg: Self = toml::from_str(text)?;
        // Fold a single legacy choice into the priority list.
        if cfg.microphones.is_empty() && !cfg.input_device.is_empty() {
            cfg.microphones.push(std::mem::take(&mut cfg.input_device));
        }
        Ok(cfg)
    }

    /// Move `name` to the front, keeping the rest in order and capping the
    /// list so it doesn't grow forever with devices seen once.
    pub fn prefer_microphone(&mut self, name: &str) {
        const MAX_REMEMBERED: usize = 5;
        self.microphones.retain(|m| m != name);
        self.microphones.insert(0, name.to_string());
        self.microphones.truncate(MAX_REMEMBERED);
    }

    pub fn save(&self) -> anyhow::Result<()> {
        self.save_to(&config_path())
    }

    fn save_to(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }
}

impl Config {
    /// The model to offer in the tray, if there is one: whatever `model_path`
    /// points at, else the first known filename sitting next to the
    /// executable, so "drop the file in and pick it" works with no config
    /// editing at all.
    ///
    /// The `.tar.gz` is preferred because it is the one we build ourselves
    /// from the published checkpoint (`scripts/export_dfn3.py`) and the one
    /// the installer ships. `model.onnx` stays supported: it is what earlier
    /// installs have beside them, and pointing `model_path` at any single-file
    /// streaming export still works.
    pub fn available_model(&self) -> Option<PathBuf> {
        let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
        self.model_in(&dir)
    }

    /// The directory is a parameter for the same reason `load`/`save` take a
    /// path: otherwise every test shares one real location. These tests share
    /// the *test binary's* directory, and one of them has to create a model
    /// there while another asserts none is present — so they raced, and CI
    /// failed on whichever lost. A caller-supplied directory removes the
    /// shared state rather than papering over it with a lock.
    fn model_in(&self, dir: &Path) -> Option<PathBuf> {
        if !self.model_path.is_empty() {
            let configured = PathBuf::from(&self.model_path);
            if configured.exists() {
                return Some(configured);
            }
            // Deliberately fall through rather than give up. The path may
            // predate an upgrade or a moved install, and dropping silently to
            // RNNoise means no background-speech removal — the one thing the
            // app exists for — while a usable model sits beside the exe.
            tracing::warn!(
                path = %configured.display(),
                "configured model is missing; looking beside the executable"
            );
        }
        // `model/` first: the installer lays the graphs down as ordinary files
        // there, so anyone opening the install directory can see what the app
        // runs. Matched on a graph inside it rather than the folder alone, so
        // an unrelated directory called `model` is not mistaken for one.
        let loose = dir.join("model");
        if loose.join("enc.onnx").exists() {
            return Some(loose);
        }
        // Archives stay supported: upstream distributes the `.tar.gz`, and
        // `model.onnx` is what installs from before the switch have.
        ["dfn3_ours.tar.gz", "model.onnx"]
            .iter()
            .map(|name| dir.join(name))
            .find(|p| p.exists())
    }

    /// The model to actually load, honouring the on/off switch.
    pub fn active_model(&self) -> Option<PathBuf> {
        self.use_onnx.then(|| self.available_model()).flatten()
    }
}

pub fn config_path() -> PathBuf {
    base_dir().join("config.toml")
}

pub fn log_dir() -> PathBuf {
    base_dir().join("logs")
}

fn base_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("RoomMute")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory of this test's own. Tests run in parallel, so anything
    /// shared between them — including the directory the test binary sits in —
    /// is a race waiting to be blamed on something else.
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("roommute-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Shown once means once, including across a restart, so it has to be
    /// written down rather than held in memory.
    #[test]
    fn the_welcome_is_remembered_as_seen() {
        let dir = scratch("welcomed");
        let path = dir.join("config.toml");

        let mut c = Config::default();
        assert!(!c.welcomed, "a fresh install has not seen it");
        c.welcomed = true;
        c.save_to(&path).unwrap();

        assert!(
            Config::load_from(&path).unwrap().welcomed,
            "reloading must not show it a second time"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Someone upgrading has a config.toml with no such key. They have never
    /// seen the welcome, so they should get it.
    #[test]
    fn a_config_from_before_this_key_still_gets_the_welcome() {
        let c = Config::parse("microphones = [\"Yeti\"]").unwrap();
        assert!(!c.welcomed);
    }

    #[test]
    fn preferring_a_microphone_moves_it_to_the_front() {
        let mut c = Config::default();
        c.prefer_microphone("A");
        c.prefer_microphone("B");
        assert_eq!(c.microphones, vec!["B", "A"]);

        // Choosing an existing one promotes rather than duplicates.
        c.prefer_microphone("A");
        assert_eq!(c.microphones, vec!["A", "B"]);
    }

    #[test]
    fn the_remembered_list_does_not_grow_without_bound() {
        let mut c = Config::default();
        for i in 0..12 {
            c.prefer_microphone(&format!("mic {i}"));
        }
        assert_eq!(c.microphones.len(), 5);
        assert_eq!(c.microphones[0], "mic 11", "most recent first");
    }

    /// The CLI override and the tray both have to write where the pipeline
    /// reads. Writing to the legacy single field made --mic a no-op.
    #[test]
    fn a_legacy_single_choice_becomes_the_first_preference() {
        let cfg = Config::parse(
            r#"
            input_device = "Microphone (fifine Microphone)"
            enabled = true
        "#,
        )
        .unwrap();
        assert_eq!(cfg.microphones, vec!["Microphone (fifine Microphone)"]);
        assert!(cfg.input_device.is_empty(), "moved, not copied");

        // A list already present wins; the legacy field is not prepended on
        // top of it every time the file is read.
        let cfg = Config::parse(
            r#"
            input_device = "Old Mic"
            microphones = ["New Mic"]
        "#,
        )
        .unwrap();
        assert_eq!(cfg.microphones, vec!["New Mic"]);
    }

    /// Everything the tray writes has to come back on the next launch.
    #[test]
    fn settings_survive_a_save_and_reload() {
        let dir = std::env::temp_dir().join(format!("roommute-cfg-{}", std::process::id()));
        let path = dir.join("nested").join("config.toml");
        let _ = std::fs::remove_dir_all(&dir);

        let mut saved = Config {
            microphones: vec!["Yeti".into(), "Webcam".into()],
            output_device: "CABLE Input".into(),
            enabled: false,
            attenuation_db: 42.5,
            use_onnx: false,
            ..Config::default()
        };
        saved.prefer_microphone("Webcam");
        // Saving must create the directory: first run has no %APPDATA% folder.
        saved.save_to(&path).unwrap();

        let loaded = Config::load_from(&path).unwrap();
        assert_eq!(loaded.microphones, vec!["Webcam", "Yeti"]);
        assert_eq!(loaded.output_device, "CABLE Input");
        assert!(!loaded.enabled);
        assert!(!loaded.use_onnx);
        assert_eq!(loaded.attenuation_db, 42.5);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A first run, and a file someone edited into nonsense. Neither may stop
    /// the app starting — the tray is how you'd fix the config in the first
    /// place.
    #[test]
    fn a_missing_file_is_defaults_and_a_broken_one_is_an_error() {
        let missing = std::env::temp_dir().join("roommute-does-not-exist/config.toml");
        let cfg = Config::load_from(&missing).unwrap();
        assert!(cfg.enabled && cfg.use_onnx, "defaults are on");

        assert!(Config::parse("microphones = \"not a list\"").is_err());
    }

    /// Config files predate every field added since; missing keys must fall
    /// back rather than refusing to load.
    #[test]
    fn an_old_config_missing_new_keys_still_loads() {
        let cfg = Config::parse("enabled = true").unwrap();
        assert!(cfg.use_onnx, "ONNX is the default for anyone upgrading");
        assert_eq!(cfg.attenuation_db, default_atten());
        assert!(cfg.microphones.is_empty());
    }

    #[test]
    fn the_config_lives_somewhere_named_and_takes_the_logs_with_it() {
        let cfg = config_path();
        assert!(cfg.ends_with("RoomMute/config.toml") || cfg.ends_with(r"RoomMute\config.toml"));
        assert_eq!(
            log_dir().parent(),
            cfg.parent(),
            "logs sit beside the config"
        );
    }

    /// A `model_path` pointing at a file that has been moved or deleted must
    /// not disable the model outright. Before this fell through, an upgrade or
    /// a moved install left the user on RNNoise — no background-speech removal
    /// at all — with a perfectly good model sitting beside the executable.
    #[test]
    fn a_stale_model_path_falls_back_to_the_bundled_model() {
        let dir = scratch("stale");
        let beside = dir.join("dfn3_ours.tar.gz");
        std::fs::write(&beside, b"stand-in for the bundled model").unwrap();

        let c = Config {
            model_path: dir.join("moved-away.onnx").to_string_lossy().into_owned(),
            ..Config::default()
        };
        assert_eq!(
            c.model_in(&dir),
            Some(beside.clone()),
            "a configured path that no longer exists should fall through to \
             what is shipped, not leave the user with no model"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// What the installer actually lays down.
    #[test]
    fn a_model_directory_beside_the_exe_wins() {
        let dir = scratch("loose");
        std::fs::create_dir_all(dir.join("model")).unwrap();
        std::fs::write(dir.join("model/enc.onnx"), b"graph").unwrap();
        std::fs::write(dir.join("dfn3_ours.tar.gz"), b"archive").unwrap();

        assert_eq!(
            Config::default().model_in(&dir),
            Some(dir.join("model")),
            "loose files are what we ship; the archive is only for compatibility"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A bare `model` folder with nothing in it must not be mistaken for one.
    #[test]
    fn an_empty_model_folder_is_not_a_model() {
        let dir = scratch("hollow");
        std::fs::create_dir_all(dir.join("model")).unwrap();

        assert_eq!(Config::default().model_in(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The export we build ourselves wins over a single-file `model.onnx` left
    /// behind by an older install.
    #[test]
    fn the_bundled_export_is_preferred_over_a_legacy_model() {
        let dir = scratch("prefer");
        std::fs::write(dir.join("model.onnx"), b"older install").unwrap();
        std::fs::write(dir.join("dfn3_ours.tar.gz"), b"what we ship").unwrap();

        assert_eq!(
            Config::default().model_in(&dir),
            Some(dir.join("dfn3_ours.tar.gz"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_legacy_model_is_still_found_when_it_is_all_there_is() {
        let dir = scratch("legacy");
        std::fs::write(dir.join("model.onnx"), b"older install").unwrap();

        assert_eq!(
            Config::default().model_in(&dir),
            Some(dir.join("model.onnx")),
            "installs from before the switch must keep working"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_directory_offers_no_model() {
        let dir = scratch("empty");
        assert_eq!(Config::default().model_in(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The tray offers whatever `model_path` points at, but only if it is
    /// really there — a stale path must fall back to RNNoise, not fail.
    #[test]
    fn a_model_is_only_offered_when_the_file_exists() {
        let dir = scratch("model");
        let model = dir.join("dfn3.onnx");
        std::fs::write(&model, b"not really a model, but it exists").unwrap();

        let c = Config {
            model_path: model.to_string_lossy().into_owned(),
            ..Config::default()
        };
        assert_eq!(c.model_in(&dir), Some(model.clone()));
        assert_eq!(
            c.active_model(),
            Some(model.clone()),
            "use_onnx defaults on"
        );

        std::fs::remove_file(&model).unwrap();
        assert!(
            c.model_in(&dir).is_none(),
            "a stale path with nothing beside it offers nothing"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_onnx_model_is_only_active_when_switched_on() {
        // A real file, so this tests the switch and not whether some model
        // happens to be sitting beside the test binary.
        let dir = std::env::temp_dir().join(format!("roommute-switch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("present.onnx");
        std::fs::write(&model, b"stand-in").unwrap();

        let mut c = Config {
            model_path: model.to_string_lossy().into_owned(),
            use_onnx: false,
            ..Config::default()
        };
        assert!(c.active_model().is_none(), "off means off");

        c.use_onnx = true;
        assert_eq!(c.active_model(), Some(model), "on means the chosen model");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
