//! First-run conversations: the two things RoomMute can't do for the user.
//!
//! Both are deliberately plain-language. Someone installing a noise-cancelling
//! app has no reason to know what a virtual audio endpoint is, and telling
//! them "no virtual cable found" is the kind of message that gets an app
//! uninstalled.

use tracing::{info, warn};

/// Vendor page rather than a direct installer link: the licence terms and the
/// current signed build both live there.
pub const CABLE_URL: &str = "https://vb-audio.com/Cable/";

/// Where the ONNX model comes from, shown before anything is fetched.
pub const MODEL_SOURCE_URL: &str = "https://github.com/Rikorose/DeepFilterNet";

#[derive(Debug, PartialEq, Eq)]
pub enum CableChoice {
    OpenSite,
    Close,
}

/// Explain the missing cable without jargon, and offer the only two useful
/// actions.
///
/// "Close" must also switch off start-with-Windows. Otherwise someone who
/// isn't ready to install a driver gets this dialog on every single boot,
/// which is a popup loop they can only escape through the registry.
pub fn cable_missing() -> CableChoice {
    if crate::message_box_yes_no(CABLE_TEXT) {
        CableChoice::OpenSite
    } else {
        CableChoice::Close
    }
}

const CABLE_TEXT: &str = "RoomMute cleans up your microphone, but it needs one more \
free program to pass the cleaned sound on to Zoom, Teams, Discord and so on.\n\n\
That program is called VB-Cable. It acts like a second microphone that other \
apps can listen to. Without it, RoomMute has nowhere to send your voice.\n\n\
It takes about two minutes to install and it's free. Windows will ask for \
administrator permission while it installs, which is normal for anything that \
adds a microphone to your system. RoomMute itself never asks for that.\n\n\
    Yes  —  open the VB-Cable download page\n\
    No   —  close RoomMute for now\n\n\
(Choosing No also stops RoomMute starting with Windows, so you won't see \
this message every time you switch on your PC.)";

/// Ask before using the high-quality model, naming where it comes from.
///
/// Shown whenever the model is deliberately chosen, not once and remembered:
/// it's a licence acceptance, and it's a decision the user just made, so it
/// isn't a loop.
pub fn model_licence() -> bool {
    let accepted = crate::message_box_yes_no(&licence_text());
    info!(accepted, "model licence prompt answered");
    accepted
}

fn licence_text() -> String {
    format!(
        "The high-quality noise removal uses DeepFilterNet3, a speech model \
published by its authors at:\n\n    {MODEL_SOURCE_URL}\n\n\
It is a separate work from RoomMute, by Hendrik Schröter and contributors, \
and is included under its MIT / Apache-2.0 licence — see NOTICE.md next to the \
program. Nothing is downloaded.\n\n\
    Yes  —  use it (recommended)\n\
    No   —  stay on the simpler built-in noise removal\n\n\
You can change this later from the tray menu."
    )
}

/// Tell the user where to put the model, since we can't fetch it for them yet.
pub fn model_missing(expected: &std::path::Path) {
    warn!(path = %expected.display(), "ONNX model not present");
    crate::message_box(&missing_text(expected));
}

/// Shown once, after the first successful start, and from Help afterwards.
///
/// It exists because a tray app that is working perfectly still looks broken:
/// the audio goes somewhere the user has not pointed anything at yet. A
/// reporter hit exactly that — RoomMute was running fine and they had no idea
/// which microphone it had chosen or what to select in their meeting software.
///
/// `cable` must be the capture half. RoomMute renders *into* "CABLE Input";
/// naming that here would send people to the endpoint that produces silence.
pub fn welcome_text(mic: &str, cable: Option<&str>) -> String {
    let routing = match cable {
        Some(name) => format!(
            "To let other apps hear the cleaned sound, pick\n\n    {name}\n\n\
             as the microphone in Discord, Zoom, Teams, OBS or whatever you use. \
             Selecting RoomMute itself will not work — it is not a microphone."
        ),
        // Nothing to point anyone at, and the cable prompt has said its piece.
        None => "No virtual audio cable is installed yet, so other apps cannot hear the \
                 cleaned sound. RoomMute will offer to set that up."
            .to_string(),
    };
    format!(
        "Thanks for using RoomMute.\n\n\
         Your microphone is set to\n\n    {mic}\n\n\
         and you can change it any time from the RoomMute icon near the clock, at the \
         bottom right of your screen.\n\n\
         {routing}\n\n\
         This is shown once. You can read it again from Help in the same menu."
    )
}

/// Show it, and say whether it was actually shown, so the caller only records
/// "seen" when it was.
pub fn show_welcome(mic: &str, cable: Option<&str>) {
    info!(mic, cable, "showing the welcome");
    crate::message_box_info(&welcome_text(mic, cable));
}

fn missing_text(expected: &std::path::Path) -> String {
    format!(
        "The high-quality model isn't installed yet.\n\n\
         Download the DeepFilterNet3 ONNX export from:\n\n    {}\n\n\
         and unpack it into this folder, so enc.onnx, erb_dec.onnx, df_dec.onnx \
         and config.ini sit inside it:\n\n    {}\n\n\
         RoomMute will use it the next time you pick it from the Denoiser menu. \
         Until then it stays on the built-in noise removal.",
        MODEL_SOURCE_URL,
        expected.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two things this message exists to say, and the one it must not get
    /// wrong: RoomMute renders into "CABLE Input", so naming that half here
    /// would send someone to the endpoint that produces silence.
    #[test]
    fn the_welcome_names_the_mic_and_the_right_half_of_the_cable() {
        let text = welcome_text(
            "Microphone (Yeti)",
            Some("CABLE Output (VB-Audio Virtual Cable)"),
        );
        assert!(
            text.contains("Microphone (Yeti)"),
            "name the mic in use: {text}"
        );
        assert!(
            text.contains("CABLE Output"),
            "name what other apps pick: {text}"
        );
        assert!(
            !text.contains("CABLE Input"),
            "that half is where RoomMute writes; picking it gives silence: {text}"
        );
        assert!(
            text.contains("bottom right"),
            "someone who has never seen a tray icon needs to be told where: {text}"
        );
        assert!(text.contains("Help"), "say how to read it again: {text}");
    }

    /// With no cable there is nothing to point at. It must not name a device
    /// that does not exist or tell anyone to go picking one.
    #[test]
    fn the_welcome_admits_when_there_is_no_cable_yet() {
        let text = welcome_text("Microphone (Yeti)", None);
        assert!(text.contains("Microphone (Yeti)"), "{text}");
        assert!(text.contains("cable is installed"), "{text}");
        assert!(!text.contains("pick\n"), "nothing to pick yet: {text}");
    }

    #[test]
    fn links_point_where_we_say_they_do() {
        // A typo sends people somewhere arbitrary to download software.
        assert_eq!(CABLE_URL, "https://vb-audio.com/Cable/");
        assert!(MODEL_SOURCE_URL.starts_with("https://github.com/Rikorose/"));
        for url in [CABLE_URL, MODEL_SOURCE_URL] {
            assert!(url.starts_with("https://"), "{url} must be https");
        }
    }

    /// This dialog is shown to someone who has just installed an app that
    /// appears not to work. It has to explain itself without jargon, and it
    /// has to say what each button does — there is no cancel.
    #[test]
    fn the_cable_dialog_explains_itself_in_plain_language() {
        assert!(CABLE_TEXT.contains("VB-Cable"), "name the thing to install");
        assert!(CABLE_TEXT.contains("Yes") && CABLE_TEXT.contains("No"));
        for jargon in ["endpoint", "WASAPI", "driver", "virtual audio device"] {
            assert!(
                !CABLE_TEXT.contains(jargon),
                "'{jargon}' means nothing here"
            );
        }
    }

    /// RoomMute installs per-user and never asks for administrator rights,
    /// so someone who got this far has not seen a UAC prompt. VB-Cable is an
    /// audio driver and will raise one. Warn, rather than let it arrive as a
    /// surprise from a link this app just opened.
    #[test]
    fn the_cable_dialog_warns_that_windows_will_ask_for_permission() {
        assert!(
            CABLE_TEXT.contains("administrator"),
            "installing the cable needs admin rights; say so before opening the page"
        );
    }

    /// The text promises that "No" also turns off start-with-Windows. If that
    /// promise is ever removed from `real_main`, this dialog becomes a boot
    /// loop the user can only escape through the registry.
    #[test]
    fn the_cable_dialog_promises_to_stop_nagging() {
        assert!(
            CABLE_TEXT.contains("stops RoomMute starting with Windows"),
            "the No button's side effect must be stated up front"
        );
        let main_rs = include_str!("main.rs");
        assert!(
            main_rs.contains("autostart::set(false)"),
            "main no longer keeps the promise this dialog makes"
        );
    }

    /// We do not ship the model, and the dialog is a licence acceptance, so it
    /// has to say whose terms are being accepted and where it comes from.
    #[test]
    fn the_licence_dialog_names_the_model_and_its_source() {
        let text = licence_text();
        assert!(text.contains("DeepFilterNet3"));
        assert!(text.contains(MODEL_SOURCE_URL));
        assert!(text.contains("separate work"), "must not read as ours");
        // We bundle it under MIT/Apache-2.0 now, so the dialog must not claim
        // the user has to go and fetch it.
        assert!(
            !text.contains("does not include") && !text.contains("obtain it"),
            "the model ships with the app; the dialog says otherwise: {text}"
        );
        assert!(text.contains("Yes") && text.contains("No"));
    }

    /// "Model missing" is only useful if it says where to put the file.
    #[test]
    fn the_missing_model_dialog_gives_the_exact_path() {
        let text = missing_text(std::path::Path::new(r"C:\Apps\RoomMute\model.onnx"));
        assert!(text.contains(r"C:\Apps\RoomMute\model.onnx"));
        assert!(text.contains(MODEL_SOURCE_URL));
    }
}
