use crate::error::{AudioError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceDirection {
    Capture,
    Render,
}

#[derive(Debug, Clone)]
pub struct Device {
    /// Stable WASAPI endpoint ID (e.g. "{0.0.1.00000000}.{guid}"). Persisted
    /// in config so the user's chosen device survives reboots and unrelated
    /// device plug events.
    pub id: String,
    pub friendly_name: String,
    pub direction: DeviceDirection,
    pub is_default: bool,
}

/// Virtual audio cables we know how to drive.
///
/// A cable is two endpoints: a render one we push cleaned audio into, and a
/// capture one that other apps select as their microphone. Each entry lists
/// the substrings that must *all* appear in the friendly name — matching on
/// one word alone would let any device claim to be a cable, and this decides
/// where the microphone ends up.
const KNOWN_CABLES: &[Cable] = &[
    Cable {
        product: "VB-Cable",
        input: &["cable input", "vb-audio"],
        output: &["cable output", "vb-audio"],
    },
    Cable {
        product: "VoiceMeeter",
        input: &["voicemeeter", "input"],
        output: &["voicemeeter", "output"],
    },
    Cable {
        product: "Virtual Audio Cable",
        input: &["virtual audio cable", "line "],
        output: &["virtual audio cable", "line "],
    },
];

struct Cable {
    product: &'static str,
    /// Render side — where we write.
    input: &'static [&'static str],
    /// Capture side — where other apps read.
    output: &'static [&'static str],
}

/// Every product RoomMute can route through, for error messages.
pub fn known_cable_products() -> Vec<&'static str> {
    KNOWN_CABLES.iter().map(|c| c.product).collect()
}

/// Compare device names ignoring the instance prefix Windows adds when a
/// device is re-enumerated: "Microphone (fifine Microphone)" and
/// "Microphone (4- fifine Microphone)" are the same hardware.
pub fn same_device_name(a: &str, b: &str) -> bool {
    strip_instance_prefix(a) == strip_instance_prefix(b)
}

/// Drop a leading "<digits>- " from each parenthesised part.
fn strip_instance_prefix(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, part) in name.split('(').enumerate() {
        if i > 0 {
            out.push('(');
        }
        let trimmed = part.trim_start();
        let stripped = match trimmed.split_once("- ") {
            Some((head, tail)) if !head.is_empty() && head.chars().all(|c| c.is_ascii_digit()) => {
                tail
            }
            _ => trimmed,
        };
        out.push_str(stripped.trim());
    }
    out
}

fn matches_all(name: &str, needles: &[&str]) -> bool {
    let lower = name.to_ascii_lowercase();
    needles.iter().all(|n| lower.contains(n))
}

impl Device {
    /// The cable whose **input** side this is — the endpoint we render cleaned
    /// audio into. VB-Audio names it "CABLE Input (VB-Audio Virtual Cable)".
    pub fn virtual_cable_input(&self) -> Option<&'static str> {
        if self.direction != DeviceDirection::Render {
            return None;
        }
        KNOWN_CABLES
            .iter()
            .find(|c| matches_all(&self.friendly_name, c.input))
            .map(|c| c.product)
    }

    /// The cable whose **output** side this is — the endpoint other apps pick
    /// as their microphone.
    ///
    /// Worth detecting because Windows tends to make a freshly installed cable
    /// the default capture device. Recording from it while rendering into the
    /// same cable feeds the thing into itself.
    pub fn virtual_cable_output(&self) -> Option<&'static str> {
        if self.direction != DeviceDirection::Capture {
            return None;
        }
        KNOWN_CABLES
            .iter()
            .find(|c| matches_all(&self.friendly_name, c.output))
            .map(|c| c.product)
    }
}

#[derive(Debug, Default)]
pub struct DeviceList {
    pub capture: Vec<Device>,
    pub render: Vec<Device>,
}

impl DeviceList {
    pub fn enumerate() -> Result<Self> {
        #[cfg(windows)]
        {
            crate::wasapi_capture::enumerate_all()
        }
        #[cfg(not(windows))]
        {
            Err(AudioError::Other(anyhow::anyhow!(
                "device enumeration is only supported on Windows"
            )))
        }
    }

    /// The one virtual cable input to render into, if exactly one is present.
    pub fn find_virtual_cable_input(&self) -> Result<&Device> {
        let mut matches = self
            .render
            .iter()
            .filter(|d| d.virtual_cable_input().is_some());
        let first = matches.next().ok_or(AudioError::VirtualCableMissing)?;
        if matches.next().is_some() {
            // Don't guess which cable gets the microphone.
            return Err(AudioError::AmbiguousDevice(
                "several virtual cables are installed; set output_device in config.toml to the \
                 name of the one you want"
                    .into(),
            ));
        }
        Ok(first)
    }

    pub fn default_capture(&self) -> Option<&Device> {
        self.capture.iter().find(|d| d.is_default)
    }

    pub fn capture_by_id(&self, id: &str) -> Option<&Device> {
        self.capture.iter().find(|d| d.id == id)
    }

    /// Resolve a remembered capture device.
    ///
    /// Endpoint ids are not as stable as they look: replug a USB microphone
    /// and Windows re-enumerates it with a fresh GUID and a bumped instance
    /// prefix ("Microphone (4- fifine Microphone)"). The id we stored last
    /// week is then dead, even though the same physical microphone is sitting
    /// right there. So fall back to the friendly name before giving up.
    pub fn resolve_capture(&self, name: &str) -> Option<&Device> {
        if name.is_empty() {
            return self.default_capture();
        }
        // Exact first, then ignoring the instance prefix Windows adds on
        // re-enumeration. Two devices sharing a name is possible but rare;
        // taking the first is better than refusing to start.
        self.capture
            .iter()
            .find(|d| d.friendly_name == name)
            .or_else(|| {
                self.capture
                    .iter()
                    .find(|d| same_device_name(&d.friendly_name, name))
            })
    }

    /// First available microphone from an ordered preference list, else the
    /// Windows default.
    ///
    /// Walking the list rather than insisting on one device is what makes an
    /// unplugged microphone a non-event: the next one down takes over.
    pub fn resolve_capture_ranked(&self, preferences: &[String]) -> Option<&Device> {
        self.capture_candidates(preferences).into_iter().next()
    }

    /// Every microphone worth trying, best first: the preferences that exist,
    /// then the Windows default as the floor.
    ///
    /// A list rather than one choice, because being enumerated does not mean a
    /// device can be opened — an endpoint with a broken effects chain lists
    /// fine and then fails, and the caller needs somewhere to go next.
    /// Deduplicated, so the default is not tried twice when it is also a
    /// preference.
    pub fn capture_candidates(&self, preferences: &[String]) -> Vec<&Device> {
        let mut out: Vec<&Device> = Vec::new();
        let mut candidates: Vec<&Device> = preferences
            .iter()
            .filter_map(|n| self.resolve_capture(n))
            .collect();
        candidates.extend(self.default_capture());
        // Then every other microphone, as a last resort.
        //
        // Preferences and the Windows default are the *likely* answers, not
        // the only ones. On the install that ran away, both failed to open
        // while a working microphone sat unused; its owner fixed it by
        // picking that one from the menu, which is something the app can do
        // for itself before it starts retrying forever.
        candidates.extend(self.capture.iter());
        for d in candidates {
            if !out.iter().any(|seen| seen.id == d.id) {
                out.push(d);
            }
        }
        out
    }

    /// The capture endpoint other apps must select to hear cleaned audio.
    ///
    /// The cable has two halves and they are easy to confuse: RoomMute renders
    /// into "CABLE Input", so telling a user to pick that in Discord sends
    /// them to the wrong one and nothing works. This finds the real capture
    /// endpoint instead of deriving a name by swapping the word.
    pub fn virtual_cable_output_device(&self) -> Option<&Device> {
        self.capture
            .iter()
            .find(|d| d.virtual_cable_output().is_some())
    }

    /// Where to render: an explicitly named device, else the one virtual cable.
    pub fn resolve_render(&self, name: &str) -> Result<&Device> {
        if name.is_empty() {
            return self.find_virtual_cable_input();
        }
        self.render
            .iter()
            .find(|d| d.friendly_name == name)
            .or_else(|| {
                self.render
                    .iter()
                    .find(|d| same_device_name(&d.friendly_name, name))
            })
            .ok_or_else(|| AudioError::DeviceNotFound(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reported install had its preferred microphone and the Windows
    /// default both fail to open, while a working one sat unused. The owner
    /// fixed it by choosing that one from the menu; the app should get there
    /// on its own before it starts retrying forever.
    #[test]
    fn every_microphone_is_a_candidate_once_the_likely_ones_are_exhausted() {
        let list = DeviceList {
            capture: vec![
                capture("Microphone (Broken)"),
                capture("Microphone (Works)"),
            ],
            render: vec![render("CABLE Input (VB-Audio Virtual Cable)")],
        };
        let names: Vec<&str> = list
            .capture_candidates(&["Microphone (Broken)".to_string()])
            .iter()
            .map(|d| d.friendly_name.as_str())
            .collect();

        assert_eq!(
            names.first(),
            Some(&"Microphone (Broken)"),
            "the preference still leads: {names:?}"
        );
        assert!(
            names.contains(&"Microphone (Works)"),
            "a microphone nobody named is still better than giving up: {names:?}"
        );
    }

    /// Trying the same device twice wastes a model load and doubles the log.
    #[test]
    fn no_microphone_is_offered_twice() {
        let list = DeviceList {
            capture: vec![capture("Microphone (Yeti)")],
            render: vec![render("CABLE Input (VB-Audio Virtual Cable)")],
        };
        let c = list.capture_candidates(&["Microphone (Yeti)".to_string()]);
        assert_eq!(
            c.len(),
            1,
            "preference, default and the sweep are the same device"
        );
    }

    /// Naming the wrong half of the cable is the difference between working
    /// and silent, so this pins which one we hand to the user.
    #[test]
    fn the_endpoint_offered_to_other_apps_is_the_capture_half() {
        let list = DeviceList {
            capture: vec![
                capture("Microphone (Yeti)"),
                capture("CABLE Output (VB-Audio Virtual Cable)"),
            ],
            render: vec![render("CABLE Input (VB-Audio Virtual Cable)")],
        };
        let offered = list
            .virtual_cable_output_device()
            .expect("the cable is installed, so there is one");
        assert!(
            offered.friendly_name.starts_with("CABLE Output"),
            "apps record from the Output half; we render into the Input half: {}",
            offered.friendly_name
        );
    }

    #[test]
    fn no_cable_means_nothing_to_offer() {
        let list = DeviceList {
            capture: vec![capture("Microphone (Yeti)")],
            render: vec![render("Speakers")],
        };
        assert!(list.virtual_cable_output_device().is_none());
    }

    fn render(name: &str) -> Device {
        Device {
            id: format!("{{0.0.0.00000000}}.{name}"),
            friendly_name: name.into(),
            direction: DeviceDirection::Render,
            is_default: false,
        }
    }

    fn capture(name: &str) -> Device {
        Device {
            direction: DeviceDirection::Capture,
            ..render(name)
        }
    }

    /// The exact failure seen in the wild: a USB mic replugged mid-session
    /// comes back with a new GUID and a "4- " instance prefix, so the stored
    /// id is dead while the hardware is still sitting there.
    #[test]
    fn a_replugged_mic_is_found_by_name_when_its_id_dies() {
        let list = DeviceList {
            capture: vec![capture("Microphone (4- fifine Microphone)")],
            render: vec![],
        };
        let found = list
            .resolve_capture("Microphone (fifine Microphone)")
            .expect("should recover by name");
        assert_eq!(found.friendly_name, "Microphone (4- fifine Microphone)");
    }

    #[test]
    fn instance_prefixes_do_not_make_devices_different() {
        assert!(same_device_name(
            "Microphone (fifine Microphone)",
            "Microphone (4- fifine Microphone)"
        ));
        assert!(same_device_name(
            "Microphone (2- USB Audio Device)",
            "Microphone (7- USB Audio Device)"
        ));
        // Genuinely different hardware must not collapse together.
        assert!(!same_device_name(
            "Microphone (fifine Microphone)",
            "Microphone (Realtek Audio)"
        ));
        // A leading number that isn't an instance prefix is left alone.
        assert!(!same_device_name("Line 1 (VAC)", "Line 2 (VAC)"));
    }

    #[test]
    fn an_exact_name_wins_over_a_prefix_insensitive_one() {
        let list = DeviceList {
            capture: vec![
                capture("Microphone (4- fifine Microphone)"),
                capture("Microphone (fifine Microphone)"),
            ],
            render: vec![],
        };
        let got = list
            .resolve_capture("Microphone (fifine Microphone)")
            .unwrap();
        assert_eq!(got.friendly_name, "Microphone (fifine Microphone)");
    }

    #[test]
    fn an_empty_name_means_the_windows_default() {
        let mut chosen = capture("Microphone (Realtek Audio)");
        chosen.is_default = true;
        let list = DeviceList {
            capture: vec![capture("Microphone (fifine Microphone)"), chosen],
            render: vec![],
        };
        assert_eq!(
            list.resolve_capture("").unwrap().friendly_name,
            "Microphone (Realtek Audio)"
        );
    }

    #[test]
    fn a_name_that_matches_nothing_stays_unresolved() {
        let list = DeviceList {
            capture: vec![capture("Microphone (Realtek Audio)")],
            render: vec![],
        };
        assert!(list.resolve_capture("Microphone (Gone)").is_none());
    }

    #[test]
    fn matches_the_real_vb_cable_endpoint() {
        assert_eq!(
            render("CABLE Input (VB-Audio Virtual Cable)").virtual_cable_input(),
            Some("VB-Cable")
        );
    }

    #[test]
    fn ignores_lookalikes_and_the_wrong_direction() {
        // Name-squatting: anything can call itself "CABLE Input".
        assert_eq!(
            render("CABLE Input (Totally Not A Recorder)").virtual_cable_input(),
            None
        );
        // The side other apps capture from, not the side we render into.
        assert_eq!(
            render("CABLE Output (VB-Audio Virtual Cable)").virtual_cable_input(),
            None
        );
        // Right name, wrong direction.
        assert_eq!(
            capture("CABLE Input (VB-Audio Virtual Cable)").virtual_cable_input(),
            None
        );
    }

    /// The loop guard depends on this: installing a cable tends to make its
    /// output the default capture device, and recording from that while
    /// rendering into the same cable feeds it into itself.
    #[test]
    fn recognises_the_capture_side_of_a_cable() {
        assert_eq!(
            capture("CABLE Output (VB-Audio Virtual Cable)").virtual_cable_output(),
            Some("VB-Cable")
        );
        assert_eq!(
            capture("Microphone (fifine Microphone)").virtual_cable_output(),
            None
        );
        // Render side is not a capture side.
        assert_eq!(
            render("CABLE Output (VB-Audio Virtual Cable)").virtual_cable_output(),
            None
        );
    }

    /// VB-Cable also installs "CABLE In 16ch", which must not be confused for
    /// the endpoint we render into.
    #[test]
    fn ignores_the_multichannel_sibling_endpoint() {
        assert_eq!(
            render("CABLE In 16ch (VB-Audio Virtual Cable)").virtual_cable_input(),
            None
        );
    }

    /// Error text that names a setting is only useful if the setting exists.
    /// Devices moved from ids to names; `output_device_id` never came with
    /// them, so following this advice edited a key nothing reads.
    #[test]
    fn the_ambiguous_cable_error_names_a_config_key_that_exists() {
        let list = DeviceList {
            capture: vec![],
            render: vec![
                render("CABLE Input (VB-Audio Virtual Cable)"),
                render("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)"),
            ],
        };
        let msg = list
            .find_virtual_cable_input()
            .expect_err("two cables must be ambiguous")
            .to_string();

        assert!(
            msg.contains("output_device"),
            "should say what to set: {msg}"
        );
        assert!(
            !msg.contains("_id"),
            "config keys are device *names* now, not ids: {msg}"
        );
    }

    #[test]
    fn refuses_to_guess_between_two_cables() {
        let list = DeviceList {
            capture: vec![],
            render: vec![
                render("CABLE Input (VB-Audio Virtual Cable)"),
                render("CABLE Input (VB-Audio Virtual Cable) 2"),
            ],
        };
        assert!(list.find_virtual_cable_input().is_err());
    }

    #[test]
    fn reports_missing_cable() {
        let list = DeviceList {
            capture: vec![],
            render: vec![render("Speakers (Realtek Audio)")],
        };
        assert!(matches!(
            list.find_virtual_cable_input(),
            Err(AudioError::VirtualCableMissing)
        ));
    }
}

#[cfg(test)]
mod priority_tests {
    use super::*;

    fn capture(name: &str) -> Device {
        Device {
            id: format!("{{0.0.1.00000000}}.{name}"),
            friendly_name: name.into(),
            direction: DeviceDirection::Capture,
            is_default: false,
        }
    }

    fn list(names: &[&str], default: &str) -> DeviceList {
        DeviceList {
            capture: names
                .iter()
                .map(|n| {
                    let mut d = capture(n);
                    d.is_default = *n == default;
                    d
                })
                .collect(),
            render: vec![],
        }
    }

    fn prefs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn takes_the_highest_ranked_device_that_is_present() {
        let l = list(
            &["Headset (Sennheiser)", "Microphone (fifine)"],
            "Headset (Sennheiser)",
        );
        let got = l
            .resolve_capture_ranked(&prefs(&["Microphone (fifine)", "Headset (Sennheiser)"]))
            .unwrap();
        assert_eq!(
            got.friendly_name, "Microphone (fifine)",
            "rank must beat system default"
        );
    }

    /// The case from the wild: the preferred mic is unplugged mid-session.
    /// It should drop to the next one rather than stopping to ask.
    #[test]
    fn falls_through_to_the_next_when_the_preferred_one_is_gone() {
        let l = list(&["Headset (Sennheiser)"], "Headset (Sennheiser)");
        let got = l
            .resolve_capture_ranked(&prefs(&["Microphone (fifine)", "Headset (Sennheiser)"]))
            .unwrap();
        assert_eq!(got.friendly_name, "Headset (Sennheiser)");
    }

    #[test]
    fn windows_default_is_the_floor_when_nothing_ranked_is_present() {
        let l = list(&["Microphone (Realtek)"], "Microphone (Realtek)");
        let got = l
            .resolve_capture_ranked(&prefs(&["Microphone (fifine)"]))
            .unwrap();
        assert_eq!(
            got.friendly_name, "Microphone (Realtek)",
            "must not strand the user"
        );
    }

    #[test]
    fn an_empty_preference_list_means_the_windows_default() {
        let l = list(&["A (x)", "B (y)"], "B (y)");
        assert_eq!(
            l.resolve_capture_ranked(&[]).unwrap().friendly_name,
            "B (y)"
        );
    }

    #[test]
    fn ranking_survives_a_replug_that_renumbers_the_device() {
        // Stored as "fifine", comes back as "4- fifine".
        let l = list(
            &["Microphone (4- fifine Microphone)"],
            "Microphone (4- fifine Microphone)",
        );
        let got = l
            .resolve_capture_ranked(&prefs(&["Microphone (fifine Microphone)"]))
            .unwrap();
        assert_eq!(got.friendly_name, "Microphone (4- fifine Microphone)");
    }

    #[test]
    fn no_devices_at_all_resolves_to_nothing() {
        let l = DeviceList {
            capture: vec![],
            render: vec![],
        };
        assert!(l.resolve_capture_ranked(&prefs(&["anything"])).is_none());
    }
}
