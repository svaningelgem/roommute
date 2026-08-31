//! A log file that rotates while the program is running.
//!
//! Rotation used to happen once, at startup. A single long run could
//! therefore pass the limit and keep going: one machine reached 13.8 MB
//! against a 5 MB cap, because it had been up for a day and never restarted.
//! Checking only at startup means the check never fires for exactly the
//! processes that need it.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;

/// Keeps one previous file. Two generations is enough to catch a problem
/// that just happened, which is all anyone reads a log for here.
struct Inner {
    path: PathBuf,
    file: Option<File>,
    written: u64,
    max: u64,
}

impl Inner {
    fn rotate(&mut self) -> io::Result<()> {
        // Close before renaming. Windows refuses to rename a file that is
        // still open unless it was opened to allow it, and Rust's File is
        // not, so this is not merely tidy — it is the only order that works.
        self.file = None;
        let old = self.path.with_extension("log.old");
        let _ = std::fs::remove_file(&old);
        std::fs::rename(&self.path, &old)?;
        self.file = Some(open(&self.path)?);
        self.written = 0;
        Ok(())
    }
}

fn open(path: &Path) -> io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

#[derive(Clone)]
pub struct RotatingLog(Arc<Mutex<Inner>>);

impl RotatingLog {
    pub fn new(path: PathBuf, max: u64) -> io::Result<Self> {
        let file = open(&path)?;
        let written = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self(Arc::new(Mutex::new(Inner {
            path,
            file: Some(file),
            written,
            max,
        }))))
    }
}

pub struct Handle(Arc<Mutex<Inner>>);

impl Write for Handle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut inner = match self.0.lock() {
            Ok(g) => g,
            // A poisoned lock must not take logging down with it.
            Err(p) => p.into_inner(),
        };
        if inner.written + buf.len() as u64 > inner.max {
            // A failed rotation is not worth losing the line over: keep
            // writing to the file we have rather than returning an error into
            // the tracing layer.
            let _ = inner.rotate();
        }
        let n = match inner.file.as_mut() {
            Some(f) => f.write(buf)?,
            None => buf.len(),
        };
        inner.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut inner = match self.0.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        match inner.file.as_mut() {
            Some(f) => f.flush(),
            None => Ok(()),
        }
    }
}

impl<'a> MakeWriter<'a> for RotatingLog {
    type Writer = Handle;
    fn make_writer(&'a self) -> Self::Writer {
        Handle(self.0.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("roommute-log-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A small existing log is picked up and appended to, not rolled over.
    /// Rotating on every start would throw away the run that just crashed,
    /// which is the one worth reading.
    #[test]
    fn a_small_existing_file_is_kept_and_appended_to() {
        let dir = scratch("small");
        let path = dir.join("roommute.log");
        std::fs::write(&path, b"what happened last time\n").unwrap();

        let log = RotatingLog::new(path.clone(), 1024).unwrap();
        log.make_writer().write_all(b"and this time\n").unwrap();

        let live = std::fs::read_to_string(&path).unwrap();
        assert!(live.contains("what happened last time"), "history was lost");
        assert!(live.contains("and this time"));
        assert!(
            !path.with_extension("log.old").exists(),
            "nothing needed rotating"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A first run has no file at all, which must not be an error.
    #[test]
    fn a_first_run_creates_the_file() {
        let dir = scratch("first");
        let path = dir.join("roommute.log");
        let log = RotatingLog::new(path.clone(), 1024).expect("a missing log is normal");
        log.make_writer()
            .write_all(
                b"hello
",
            )
            .unwrap();

        assert!(path.exists());
        assert!(
            !path.with_extension("log.old").exists(),
            "nothing to roll over"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The failure this exists for: a process that runs for a day without
    /// restarting used to grow its log without limit.
    #[test]
    fn a_long_run_rotates_without_restarting() {
        let dir = scratch("rotate");
        let path = dir.join("roommute.log");
        let log = RotatingLog::new(path.clone(), 1024).unwrap();

        let line = vec![b'x'; 200];
        for _ in 0..20 {
            log.make_writer().write_all(&line).unwrap();
        }

        let live = std::fs::metadata(&path).unwrap().len();
        assert!(live <= 1024, "the live file passed its limit: {live}");
        assert!(
            path.with_extension("log.old").exists(),
            "the previous generation has to be kept, or rotation is deletion"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Rotation must not lose the line that triggered it.
    #[test]
    fn the_line_that_triggers_rotation_is_still_written() {
        let dir = scratch("keep");
        let path = dir.join("roommute.log");
        let log = RotatingLog::new(path.clone(), 64).unwrap();

        log.make_writer().write_all(&[b'a'; 60]).unwrap();
        log.make_writer().write_all(b"the important one\n").unwrap();

        let live = std::fs::read_to_string(&path).unwrap();
        assert!(live.contains("the important one"), "got: {live:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Picking up an existing file must account for what is already in it,
    /// or the first rotation comes a whole file too late.
    #[test]
    fn an_existing_file_counts_towards_the_limit() {
        let dir = scratch("resume");
        let path = dir.join("roommute.log");
        std::fs::write(&path, vec![b'y'; 900]).unwrap();

        let log = RotatingLog::new(path.clone(), 1024).unwrap();
        log.make_writer().write_all(&[b'z'; 200]).unwrap();

        let live = std::fs::metadata(&path).unwrap().len();
        assert!(live < 900, "should have rotated on the first write: {live}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
