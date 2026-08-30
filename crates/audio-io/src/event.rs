//! An auto-closing wrapper for the audio engine's buffer-ready event.
//!
//! Rust's guarantees are about memory safety, not resource cleanup: leaking is
//! a safe operation, and `Drop` only runs for types that implement it. The COM
//! interfaces in this crate look after themselves because the `windows` crate
//! implements `Drop` for them, but a raw `HANDLE` is a plain newtype over a
//! pointer — the crate cannot know whether it wants `CloseHandle`, `FindClose`,
//! `RegCloseKey` or something else, so it does nothing and the handle leaks.
//!
//! Both WASAPI loops create one of these per stream, and the tray watchdog
//! rebuilds the pipeline whenever a device goes quiet, so "once at startup" is
//! not the usage pattern: a device that keeps dropping out leaks a pair of
//! handles every few seconds for as long as the app runs.
//!
//! `autostart.rs` already does the same thing for `HKEY`.

use windows::core::PCWSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Threading::CreateEventW;

use crate::error::{AudioError, Result};

/// An event handle that closes itself.
pub(crate) struct Event(HANDLE);

impl Event {
    /// An auto-reset, initially unsignalled event, as WASAPI expects.
    pub fn new() -> Result<Self> {
        let h = unsafe { CreateEventW(None, false, false, PCWSTR::null()) }
            .map_err(|e| AudioError::wasapi("CreateEventW", e))?;
        Ok(Self(h))
    }

    pub fn handle(&self) -> HANDLE {
        self.0
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // Nothing useful to do if this fails, and it cannot fail for a handle
        // we created and never shared.
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use windows::Win32::Foundation::{GetHandleInformation, HANDLE};

    use super::*;

    /// Is this handle still open in our process?
    fn still_open(h: HANDLE) -> bool {
        let mut flags = 0u32;
        unsafe { GetHandleInformation(h, &mut flags).is_ok() }
    }

    /// A live event is a real, open handle.
    ///
    /// This deliberately stops before re-checking the handle *after* the drop.
    /// Windows recycles handle values: once closed, the same number can be
    /// handed straight back out, and the other tests in this binary open
    /// registry keys, files and threads throughout. A sibling landing on that
    /// value in the window between the drop and the check makes
    /// GetHandleInformation succeed and reports a leak that never happened —
    /// which is what failed one CI run while passing every other.
    ///
    /// That dropping really closes it is covered by the test below, across
    /// 500 handles. A leak shows there as +500 against a tolerance of 50, and
    /// no recycled value can fake that.
    #[test]
    fn a_live_event_is_an_open_handle() {
        let event = Event::new().expect("create an event");
        let raw = event.handle();
        assert!(!raw.is_invalid(), "a fresh event must be a real handle");
        assert!(still_open(raw), "and it must be open while the guard lives");
        drop(event);
    }

    /// The failure this guards is cumulative, so prove it does not creep:
    /// creating and dropping many events must not raise the process's handle
    /// count.
    #[test]
    fn creating_and_dropping_many_events_does_not_accumulate_handles() {
        use windows::Win32::System::Threading::GetCurrentProcess;

        let count = || -> u32 {
            let mut n = 0u32;
            unsafe {
                let _ = windows::Win32::System::Threading::GetProcessHandleCount(
                    GetCurrentProcess(),
                    &mut n,
                );
            }
            n
        };

        // The counter is process-wide, and the other tests in this binary run
        // in parallel and open handles of their own between the two samples.
        // So the guard cannot be a tight bound — it has to be a wide gap: a
        // real leak adds one handle per event, which is EVENTS, while the
        // noise from siblings is a handful. TOLERANCE sits an order of
        // magnitude below the leak it must catch and well above that noise.
        const EVENTS: u32 = 500;
        const TOLERANCE: u32 = 50;

        // Warm up so one-off allocations do not show as growth.
        for _ in 0..10 {
            drop(Event::new().unwrap());
        }
        let before = count();
        for _ in 0..EVENTS {
            drop(Event::new().unwrap());
        }
        let after = count();

        assert!(
            after <= before + TOLERANCE,
            "handle count went {before} -> {after} over {EVENTS} events; they are leaking"
        );
    }
}
