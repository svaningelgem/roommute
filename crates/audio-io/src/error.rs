use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("device not found: {0}")]
    DeviceNotFound(String),

    #[error("no virtual audio cable is installed")]
    VirtualCableMissing,

    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("ambiguous device selection: {0}")]
    AmbiguousDevice(String),

    /// The endpoint went away underneath us — unplugged, re-enumerated, or
    /// reconfigured. Its own variant because it's the one WASAPI failure that
    /// is both common and recoverable: re-resolve the device and reopen.
    #[error("the audio device was disconnected or reconfigured ({context})")]
    DeviceInvalidated { context: &'static str },

    #[error("{context} failed: {detail}")]
    Wasapi {
        context: &'static str,
        detail: String,
    },

    #[error("the audio device stopped delivering audio")]
    Stalled,

    #[error("audio thread panicked or exited unexpectedly")]
    ThreadDied,

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl AudioError {
    /// Is this worth retrying with a freshly resolved device?
    pub fn is_recoverable(&self) -> bool {
        matches!(self, Self::DeviceInvalidated { .. } | Self::Stalled)
    }
}

/// Plain-English text for the HRESULTs users actually hit, because
/// "HRESULT 0x88890004" in a dialog box helps nobody. Returns `None` for
/// codes we have nothing better to say about than the number.
#[cfg(windows)]
fn explain(code: u32) -> Option<&'static str> {
    Some(match code {
        0x8889_0001 => "the audio client was not initialised",
        0x8889_0002 => "the audio device is already in use in exclusive mode",
        0x8889_0003 => "wrong endpoint type (a capture call on a playback device, or vice versa)",
        0x8889_0006 => "the audio buffer was too large for the device",
        0x8889_0008 => "the device does not support this audio format",
        0x8889_000A => "the audio device is in use by another application in exclusive mode",
        0x8889_0010 => "the audio endpoint was unplugged",
        0x8889_0015 => "Windows Audio (the audio service) is not running",
        0x8007_0005 => {
            "access denied — check Settings, Privacy & Security, Microphone, and that \
             'Let desktop apps access your microphone' is on"
        }
        0x8007_0490 => "no such device — it may have been unplugged",
        // ERROR_MOD_NOT_FOUND. Windows says "the specified module could not be
        // found", which reads as a missing RoomMute DLL and sends people
        // reinstalling this app. It is the endpoint's own effects chain: an
        // audio enhancement (Realtek, Nahimic, Waves, Dolby and friends
        // register these) whose DLL Windows cannot load. Opening the format
        // for the device pulls that chain in, which is why it surfaces here.
        0x8007_007E => {
            "this microphone could not be opened — Windows could not load one of its              effect DLLs. Pick a different microphone from the tray menu; turning off              that device's audio enhancements also fixes it"
        }
        _ => return None,
    })
}

#[cfg(windows)]
impl AudioError {
    pub(crate) fn wasapi(context: &'static str, e: windows::core::Error) -> Self {
        let code = e.code().0 as u32;
        // AUDCLNT_E_DEVICE_INVALIDATED — recoverable, so it gets its own variant.
        if code == 0x8889_0004 {
            return Self::DeviceInvalidated { context };
        }
        let detail = match explain(code) {
            Some(text) => format!("{text} (0x{code:08X})"),
            None => {
                // Windows often returns an empty message for these, which is
                // how the old formatting produced a trailing bare colon.
                let msg = e.message();
                let msg = msg.trim();
                if msg.is_empty() {
                    format!("HRESULT 0x{code:08X}")
                } else {
                    format!("{msg} (0x{code:08X})")
                }
            }
        };
        Self::Wasapi { context, detail }
    }
}

pub type Result<T> = std::result::Result<T, AudioError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalidation_and_stalls_are_worth_retrying() {
        assert!(AudioError::DeviceInvalidated { context: "x" }.is_recoverable());
        assert!(AudioError::Stalled.is_recoverable());
        assert!(!AudioError::VirtualCableMissing.is_recoverable());
        assert!(!AudioError::UnsupportedFormat("16-bit".into()).is_recoverable());
    }

    #[cfg(windows)]
    /// ERROR_MOD_NOT_FOUND out of GetMixFormat, reported from a real machine.
    /// Windows renders it as "The specified module could not be found", which
    /// sounds like RoomMute is missing a DLL of its own. It is not: that
    /// endpoint has an effect DLL Windows cannot load.
    ///
    /// The reporter fixed it by choosing another microphone, not by touching
    /// any driver, so that is what the text leads with. RoomMute now tries the
    /// other microphones itself before anyone sees this at all.
    #[test]
    fn a_missing_audio_effect_dll_says_so_rather_than_blaming_us() {
        let e = AudioError::wasapi(
            "GetMixFormat",
            windows::core::Error::from_hresult(windows::core::HRESULT(0x8007_007Eu32 as i32)),
        );
        let text = e.to_string();
        assert!(
            text.contains("different microphone"),
            "lead with the fix that actually worked — choosing another mic: {text}"
        );
        assert!(
            !text.contains("specified module could not be found"),
            "Windows' own wording reads as a missing RoomMute DLL: {text}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn known_codes_read_as_english_and_never_dangle() {
        // The exact failure from a re-enumerated USB mic.
        let e = AudioError::wasapi(
            "IMMDevice::Activate",
            windows::core::Error::from_hresult(windows::core::HRESULT(0x8889_0004u32 as i32)),
        );
        let text = e.to_string();
        assert!(text.contains("disconnected"), "{text}");
        assert!(!text.contains("0x88890004"), "raw HRESULT leaked: {text}");

        // An unmapped code still says something, and never ends in a bare colon.
        let odd = AudioError::wasapi(
            "GetBuffer",
            windows::core::Error::from_hresult(windows::core::HRESULT(0x8889_0099u32 as i32)),
        );
        let text = odd.to_string();
        assert!(text.contains("0x88890099"), "{text}");
        assert!(!text.trim_end().ends_with(':'), "dangling colon: {text}");
    }

    #[cfg(windows)]
    #[test]
    fn privacy_block_explains_where_to_look() {
        let e = AudioError::wasapi(
            "Activate",
            windows::core::Error::from_hresult(windows::core::HRESULT(0x8007_0005u32 as i32)),
        );
        assert!(e.to_string().contains("Microphone"), "{e}");
    }
}
