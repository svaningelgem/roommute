//! Start-at-login, via the per-user `Run` key.
//!
//! `HKCU\...\CurrentVersion\Run` rather than the machine-wide `HKLM` one or a
//! scheduled task: it needs no elevation, it's trivially inspectable by the
//! user, and uninstalling is deleting one value. RoomMute holds a microphone
//! open — it should not be quietly installing itself for every account on the
//! machine.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::ERROR_SUCCESS;
use windows::Win32::System::Registry::{
    RegCloseKey, RegCreateKeyExW, RegDeleteValueW, RegQueryValueExW, RegSetValueExW, HKEY,
    HKEY_CURRENT_USER, KEY_QUERY_VALUE, KEY_SET_VALUE, REG_OPTION_NON_VOLATILE, REG_SAM_FLAGS,
    REG_SZ,
};

const RUN_KEY: PCWSTR = w!(r"Software\Microsoft\Windows\CurrentVersion\Run");
const VALUE_NAME: PCWSTR = w!("RoomMute");

/// The command Windows will run at login. Quoted, because `C:\Program Files\…`
/// would otherwise be parsed as several arguments.
fn command_string(exe: &Path) -> String {
    format!("\"{}\"", exe.display())
}

fn exe_path() -> Result<PathBuf> {
    std::env::current_exe().map_err(Into::into)
}

struct Key(HKEY);

impl Drop for Key {
    fn drop(&mut self) {
        unsafe {
            let _ = RegCloseKey(self.0);
        }
    }
}

fn open_run_key(access: REG_SAM_FLAGS) -> Result<Key> {
    let mut key = HKEY::default();
    let status = unsafe {
        RegCreateKeyExW(
            HKEY_CURRENT_USER,
            RUN_KEY,
            None,
            None,
            REG_OPTION_NON_VOLATILE,
            access,
            None,
            &mut key,
            None,
        )
    };
    if status != ERROR_SUCCESS {
        bail!("opening the Run key failed: error {}", status.0);
    }
    Ok(Key(key))
}

/// Is RoomMute currently registered to start at login?
pub fn is_enabled() -> bool {
    is_entry_enabled(VALUE_NAME)
}

fn is_entry_enabled(name: PCWSTR) -> bool {
    let Ok(key) = open_run_key(KEY_QUERY_VALUE) else {
        return false;
    };
    // We only care whether the value exists, not what it holds — a stale path
    // from a moved binary still means "the user asked for autostart", and
    // `set(true)` will rewrite it.
    unsafe { RegQueryValueExW(key.0, name, None, None, None, None) == ERROR_SUCCESS }
}

/// Register or unregister start-at-login for the current user.
pub fn set(enabled: bool) -> Result<()> {
    set_entry(VALUE_NAME, &exe_path()?, enabled)
}

/// The value name and the path are arguments rather than baked in so tests can
/// round-trip against a name of their own. Writing the real one would mean
/// `cargo test` rewriting the user's autostart to point at the test binary.
fn set_entry(name: PCWSTR, exe: &Path, enabled: bool) -> Result<()> {
    let key = open_run_key(KEY_SET_VALUE)?;
    let status = if enabled {
        let value = command_string(exe);
        // REG_SZ wants NUL-terminated UTF-16, handed over as raw bytes. Built
        // little-endian by hand rather than reinterpreting a &[u16] — same
        // bytes on every target Windows runs on, and no unsafe.
        let bytes: Vec<u8> = value
            .encode_utf16()
            .chain(std::iter::once(0))
            .flat_map(u16::to_le_bytes)
            .collect();
        unsafe { RegSetValueExW(key.0, name, None, REG_SZ, Some(&bytes)) }
    } else {
        let status = unsafe { RegDeleteValueW(key.0, name) };
        // Deleting something that was never there is a success as far as the
        // caller is concerned.
        if status != ERROR_SUCCESS && !is_entry_enabled(name) {
            ERROR_SUCCESS
        } else {
            status
        }
    };
    if status != ERROR_SUCCESS {
        bail!("writing the Run key failed: error {}", status.0);
    }
    Ok(())
}

/// Read a Run-key value back as a string. Only the tests need this — the app
/// never cares what the value holds, only whether it exists.
#[cfg(test)]
fn read_entry(name: PCWSTR) -> Option<String> {
    let key = open_run_key(KEY_QUERY_VALUE).ok()?;
    let mut size: u32 = 0;
    unsafe {
        if RegQueryValueExW(key.0, name, None, None, None, Some(&mut size)) != ERROR_SUCCESS {
            return None;
        }
        let mut buf = vec![0u8; size as usize];
        if RegQueryValueExW(
            key.0,
            name,
            None,
            None,
            Some(buf.as_mut_ptr()),
            Some(&mut size),
        ) != ERROR_SUCCESS
        {
            return None;
        }
        // Indexed rather than `chunks_exact(2)`: clippy 1.98 wants
        // `as_chunks::<2>()` for a constant size, and that needs Rust 1.88 —
        // above this crate's declared rust-version. This reads the same and
        // compiles anywhere.
        let wide: Vec<u16> = (0..buf.len() / 2)
            .map(|i| u16::from_le_bytes([buf[2 * i], buf[2 * i + 1]]))
            .take_while(|&c| c != 0)
            .collect();
        Some(String::from_utf16_lossy(&wide))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_is_quoted_for_paths_with_spaces() {
        let cmd = command_string(Path::new(r"C:\Program Files\RoomMute\roommute.exe"));
        assert_eq!(cmd, "\"C:\\Program Files\\RoomMute\\roommute.exe\"");
        assert!(cmd.starts_with('"') && cmd.ends_with('"'));
    }

    /// Everything the suite does to the registry, against whichever name the
    /// caller owns. Tests run in parallel, so each needs its own — sharing one
    /// makes them race over the same value.
    fn round_trip(name: PCWSTR) {
        let exe = Path::new(r"C:\nowhere\roommute-selftest.exe");

        set_entry(name, exe, true).expect("enable");
        assert!(
            is_entry_enabled(name),
            "should be registered after enabling"
        );

        set_entry(name, exe, false).expect("disable");
        assert!(!is_entry_enabled(name), "should be gone after disabling");

        // Disabling something that was never there must not error.
        set_entry(name, exe, false).expect("disable again");
    }

    #[test]
    fn enabling_and_disabling_round_trips() {
        round_trip(w!("RoomMute-selftest-roundtrip"));
    }

    /// The round trip only asks whether the value exists. What Windows will
    /// actually run is the string, and that string is encoded to UTF-16 bytes
    /// by hand — so encode it, read it back through the registry, and require
    /// the command to survive intact.
    ///
    /// The path carries an accent and an emoji on purpose: the emoji is
    /// outside the BMP and therefore a surrogate pair, which is where a
    /// hand-rolled UTF-16 encoder goes wrong if it is going to.
    #[test]
    fn the_value_written_is_the_command_windows_will_run() {
        let name = w!("RoomMute-selftest-encoding");
        let exe = Path::new("C:\\nowhere\\Ro\u{00f4}mMute \u{1f3a4}\\roommute.exe");

        set_entry(name, exe, true).expect("enable");
        let written = read_entry(name);
        set_entry(name, exe, false).expect("disable");

        assert_eq!(
            written.as_deref(),
            Some(command_string(exe).as_str()),
            "the Run value has to be the quoted command, byte for byte"
        );
    }

    #[test]
    fn a_name_that_was_never_written_reads_as_nothing() {
        assert_eq!(read_entry(w!("RoomMute-selftest-absent")), None);
    }

    /// Read-only on purpose. `set(true)` writes `current_exe()`, which under
    /// `cargo test` is a temporary binary in target/debug/deps — so the public
    /// pair can only be half-tested, and this is the half that is safe.
    #[test]
    fn is_enabled_agrees_with_what_is_actually_in_the_key() {
        assert_eq!(
            is_enabled(),
            read_entry(VALUE_NAME).is_some(),
            "is_enabled() has to mean 'the value is present', however this \
             machine happens to be configured"
        );
    }

    #[test]
    fn the_command_points_at_a_real_file() {
        let exe = exe_path().expect("the running executable must be locatable");
        assert!(exe.is_file(), "{} is not a file", exe.display());
    }

    /// The round trip used to call `set(true)`, which writes
    /// `current_exe()` — under `cargo test` that is the test binary in
    /// target/debug/deps. Anyone who had autostart switched on and then ran
    /// the suite got their Run key pointed at a temporary file, so at the
    /// next login Windows launched nothing.
    /// `set_entry` must write only the value name it was handed.
    ///
    /// Checked *while* the entry exists, not before and after: the round trip
    /// enables and then disables, so a version that wrote the real name would
    /// leave no trace by the time it finished, and comparing endpoints would
    /// see nothing wrong on a machine with no entry of its own.
    ///
    /// This deliberately never writes `VALUE_NAME`. An earlier version seeded
    /// it to simulate a user with autostart on; a failure mid-flight left that
    /// seed behind, committing the exact harm it was written to prevent.
    #[test]
    fn writing_one_entry_never_touches_the_apps_own() {
        let name = w!("RoomMute-selftest-isolation");
        let exe = Path::new(r"C:\nowhere\roommute-selftest.exe");
        let before = read_entry(VALUE_NAME);

        set_entry(name, exe, true).expect("enable");
        let during = read_entry(VALUE_NAME);
        set_entry(name, exe, false).expect("disable");

        assert_eq!(
            during, before,
            "writing {:?} also wrote the app's own entry — via `set(true)` that \
             points the user's Run key at whatever binary is running, which under \
             `cargo test` is a temporary file in target/debug/deps",
            "RoomMute-selftest-isolation"
        );
    }
}
