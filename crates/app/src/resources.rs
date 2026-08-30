//! What this process itself costs, measured rather than inferred.
//!
//! The tray meter answers a different question — how much of each 10 ms frame
//! the denoiser spends — and the two disagree by the number of cores, which
//! is confusing enough that one gets reported as a bug against the other. So
//! this reports both normalisations, explicitly, every minute:
//!
//! * **of one core** is comparable with the tray meter.
//! * **of the machine** is what Task Manager shows, and on a 32-thread box a
//!   perfectly healthy 7% of one core lands at 0.2% there and displays as 0.

use std::time::{Duration, Instant};

/// CPU as a share of one core and of the whole machine, both as percentages.
///
/// Split out from the Win32 calls so the arithmetic — the part that was
/// actually being misread — is testable without a process to measure.
fn percentages(cpu: Duration, wall: Duration, cores: f64) -> (f64, f64) {
    if wall.is_zero() || cores <= 0.0 {
        return (0.0, 0.0);
    }
    let of_one = cpu.as_secs_f64() / wall.as_secs_f64() * 100.0;
    (of_one, of_one / cores)
}

#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Against the whole machine, the basis Task Manager uses.
    ///
    /// `percentages` still computes the per-core figure, and its tests pin
    /// the relationship between the two, because that relationship is what
    /// made the tray meter look like it disagreed with Task Manager.
    pub cpu_of_machine: f64,
    pub working_set_mb: f64,
    pub over: Duration,
}

/// Samples this process, reporting the average since the previous sample.
pub struct ProcessMeter {
    last: Option<(Instant, Duration)>,
    cores: f64,
}

impl ProcessMeter {
    pub fn new() -> Self {
        Self {
            last: None,
            cores: std::thread::available_parallelism()
                .map(|n| n.get() as f64)
                .unwrap_or(1.0),
        }
    }

    /// `None` on the first call, which establishes the baseline: an average
    /// needs two readings, and reporting the process's whole lifetime as if
    /// it were the last minute would be its own kind of wrong.
    pub fn sample(&mut self, now: Instant) -> Option<Sample> {
        let cpu = process_cpu_time()?;
        let working_set_mb = working_set_bytes().unwrap_or(0) as f64 / (1024.0 * 1024.0);

        let out = match self.last {
            Some((then, before)) => {
                let wall = now.duration_since(then);
                let used = cpu.checked_sub(before).unwrap_or_default();
                let (_of_one_core, of_machine) = percentages(used, wall, self.cores);
                Some(Sample {
                    cpu_of_machine: of_machine,
                    working_set_mb,
                    over: wall,
                })
            }
            None => None,
        };
        self.last = Some((now, cpu));
        out
    }
}

#[cfg(windows)]
fn process_cpu_time() -> Option<Duration> {
    use windows::Win32::Foundation::FILETIME;
    use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    let (mut creation, mut exit, mut kernel, mut user) = (
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
    );
    // Safety: all four are valid out-params for the life of the call.
    unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    }
    .ok()?;

    // FILETIME counts 100-nanosecond ticks, split across two 32-bit halves.
    let ticks = |f: FILETIME| ((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64;
    Some(Duration::from_nanos((ticks(kernel) + ticks(user)) * 100))
}

#[cfg(windows)]
fn working_set_bytes() -> Option<usize> {
    use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows::Win32::System::Threading::GetCurrentProcess;

    let mut counters = PROCESS_MEMORY_COUNTERS::default();
    let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // Safety: `counters` outlives the call and `size` describes it exactly.
    unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) }.ok()?;
    Some(counters.WorkingSetSize)
}

#[cfg(not(windows))]
fn process_cpu_time() -> Option<Duration> {
    None
}

#[cfg(not(windows))]
fn working_set_bytes() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact confusion that prompted this: the tray said 7% and Task
    /// Manager said 0, on a 32-thread machine, and both were right.
    #[test]
    fn the_same_load_reads_differently_per_core_and_per_machine() {
        let (one, machine) =
            percentages(Duration::from_micros(700), Duration::from_millis(10), 32.0);
        assert!((one - 7.0).abs() < 1e-9, "7% of a core: {one}");
        assert!(
            (machine - 0.21875).abs() < 1e-9,
            "which Task Manager rounds to 0: {machine}"
        );
    }

    #[test]
    fn a_fully_busy_core_is_a_hundred_percent_of_one() {
        let (one, machine) = percentages(Duration::from_secs(1), Duration::from_secs(1), 8.0);
        assert!((one - 100.0).abs() < 1e-9);
        assert!((machine - 12.5).abs() < 1e-9);
    }

    /// Division by zero on the first tick, or on a machine reporting no
    /// parallelism, must not produce NaN in a log line.
    #[test]
    fn no_elapsed_time_is_zero_rather_than_not_a_number() {
        assert_eq!(
            percentages(Duration::from_secs(1), Duration::ZERO, 4.0),
            (0.0, 0.0)
        );
        assert_eq!(
            percentages(Duration::from_secs(1), Duration::from_secs(1), 0.0),
            (0.0, 0.0)
        );
    }

    /// An average needs two readings; the first only sets the baseline.
    #[test]
    fn the_first_sample_reports_nothing() {
        let mut m = ProcessMeter::new();
        let t = Instant::now();
        assert!(m.sample(t).is_none(), "nothing to average against yet");
    }
}
