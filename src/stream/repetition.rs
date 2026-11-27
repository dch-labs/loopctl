//! The stream repetition guard — a runaway-generation abort for
//! degenerate repetition loops.
//!
//! A quantized model can enter a repetition attractor and emit the same
//! window of text indefinitely — cheaply, when speculative decoding
//! amplifies it — until the provider's output cap stops it. The guard
//! watches the decoded assistant text (both content and thinking
//! lanes, appended in arrival order) and reports the loop as soon as
//! one byte window repeats verbatim enough consecutive times — or a run
//! of one character grows past a runaway line — so the consumer can
//! abort the request in seconds instead of decoding to the cap.
//!
//! # Detection rule
//!
//! A byte window of `min_period_bytes`..=`max_period_bytes` at the end
//! of the text, immediately preceded by `min_repeats - 1` byte-identical
//! copies of itself, is a repetition loop. The smallest qualifying
//! period wins, so the earliest signal surfaces. A run of text that
//! differs anywhere inside a window breaks the streak — near-repetitive
//! but non-identical content (CSV rows that vary, code lines with
//! differing indentation) never fires.
//!
//! A run of one character is a line, not a stanza, and the window rule
//! never fires on one. A box-drawing border shows why: its horizontal is
//! three bytes in UTF-8, so sixty-four of them are 192 bytes — four
//! 48-byte windows of identical text — and a byte-level comparison cannot
//! tell that border from a loop. A window that is one character repeated
//! is therefore skipped at every period, which exempts ASCII-art borders
//! and rules of dashes at any width. A model stuck on one character is
//! still stuck, so length decides instead: once the trailing run reaches
//! [`SINGLE_CHAR_RUNAWAY_BYTES`] — counted in whole copies of the
//! character — the guard reports a single-character runaway, with the
//! character's byte length as the period and the copies it took to reach
//! the line as the repeat count. A run of a multi-character unit
//! (`"0,0,0,0\n"`, a two-row alternation) is not one character and keeps
//! firing as a stanza loop.
//!
//! # Quick Start
//!
//! ```
//! use loopctl::stream::repetition::{
//!     RepetitionGuardConfig, detect_repetition,
//! };
//!
//! let config = RepetitionGuardConfig::default();
//! let stanza = "<tool_call><function=Bash></function></tool_call>\n";
//! let degenerate = stanza.repeat(4);
//! assert!(detect_repetition(degenerate.as_bytes(), &config).is_some());
//!
//! let varied = (0..50)
//!     .map(|row| format!("id={row},value={row}\n"))
//!     .collect::<String>();
//! assert!(detect_repetition(varied.as_bytes(), &config).is_none());
//!
//! let border = "\u{2500}".repeat(64);
//! assert!(detect_repetition(border.as_bytes(), &config).is_none());
//! ```

/// The largest rolling-buffer capacity a valid configuration may
/// demand, in bytes (`1 MiB`).
///
/// The product bound behind [`RepetitionGuardConfig::validate`]: a
/// period cap times its repeat headroom beyond this ceiling is
/// rejected, because the trim that keeps the guard's rolling buffer
/// and per-delta scan cost constant only fires below the buffer's
/// capacity — a configuration whose capacity saturates would let both
/// grow with the whole stream. The default configuration's capacity
/// is `20 KiB`, a factor of 51 under the ceiling.
pub const MAX_BUFFER_CAPACITY_BYTES: usize = 1 << 20;

/// The longest run of one character the guard tolerates at the text's
/// tail, in bytes (`4 KiB`).
///
/// The second half of the module's detection rule: a run of a single
/// character — a box-drawing border, a rule of dashes, a wall of
/// spaces — never fires the window rule, so length alone decides when
/// one is a runaway. The line is counted in whole copies of the
/// character, rounded down — three-byte glyphs reach it at the 1,365th
/// copy (4,095 bytes), ASCII at the 4,096th — and bounded by the
/// configuration's [`buffer_capacity`](RepetitionGuardConfig::buffer_capacity)
/// where that is smaller, but never below two copies: a capacity too
/// small to hold two of a wide glyph still condemns a pair, and a lone
/// glyph stays exempt. Counting in whole copies is what makes the
/// trimmed streaming path and the untrimmed completed-text path agree: a
/// rolling buffer trimmed to the capacity reaches exactly the whole
/// copies the line holds, whether the trim cut inside a glyph or on one,
/// and a buffer filled edge to edge with one character has already
/// answered.
pub const SINGLE_CHAR_RUNAWAY_BYTES: usize = 1 << 12;

/// The repetition guard's configuration.
///
/// Defaults are armed and conservative: a window of at least 48 bytes
/// repeated 4 consecutive times, or one character run
/// [`SINGLE_CHAR_RUNAWAY_BYTES`] long. Hosts with legitimate content that
/// repeats verbatim at those scales (identical CSV rows, generated
/// tables) can raise [`min_repeats`](Self::min_repeats) or
/// [`min_period_bytes`](Self::min_period_bytes), or disarm the guard
/// entirely with [`enabled`](Self::enabled) = `false` for
/// byte-identical legacy behavior.
///
/// # Example
///
/// ```
/// use loopctl::stream::repetition::RepetitionGuardConfig;
///
/// let strict = RepetitionGuardConfig {
///     min_repeats: 6,
///     ..RepetitionGuardConfig::default()
/// };
/// assert!(strict.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepetitionGuardConfig {
    /// Whether the guard is armed.
    ///
    /// `true` by default. `false` disables detection entirely — the
    /// consumer sees byte-identical behavior to a build that never
    /// knew the guard existed.
    pub enabled: bool,

    /// The smallest byte window treated as a repetition period.
    ///
    /// 48 by default — above the longest common boilerplate runs the
    /// guard means to protect (short license lines, closing-brace
    /// runs, short CSV rows), so only a genuinely long verbatim
    /// stanza can start a streak. A window that is one character
    /// repeated never starts one at any period; a single-character
    /// runaway is judged by [`SINGLE_CHAR_RUNAWAY_BYTES`] instead. Must
    /// be at least 1.
    pub min_period_bytes: usize,

    /// The consecutive verbatim repeats required to declare a loop.
    ///
    /// 4 by default — the conservative end of the 3–4 band: a
    /// degenerate loop is still caught within a few periods (a
    /// ~84-byte stanza trips at the fourth copy), while a single
    /// duplicated legitimate block does not fire. Must be at least 2.
    pub min_repeats: usize,

    /// The largest byte window scanned for a repetition period.
    ///
    /// 4096 by default. Periods longer than this can repeat forever
    /// without firing — the bound keeps the per-delta scan cost
    /// constant, and [`validate`](Self::validate) enforces that the
    /// product of this cap and the repeat headroom stays within
    /// [`MAX_BUFFER_CAPACITY_BYTES`], so an accepted configuration
    /// cannot defeat the bound. Must be at least
    /// [`min_period_bytes`](Self::min_period_bytes).
    pub max_period_bytes: usize,
}

impl Default for RepetitionGuardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_period_bytes: 48,
            min_repeats: 4,
            max_period_bytes: 4096,
        }
    }
}

impl RepetitionGuardConfig {
    /// Validate the configuration, returning the first problem found.
    ///
    /// # Errors
    ///
    /// Returns a human-readable message when
    /// [`min_period_bytes`](Self::min_period_bytes) is zero,
    /// [`min_repeats`](Self::min_repeats) is below two,
    /// [`max_period_bytes`](Self::max_period_bytes) is below
    /// [`min_period_bytes`](Self::min_period_bytes), or the implied
    /// [`buffer_capacity`](Self::buffer_capacity) exceeds
    /// [`MAX_BUFFER_CAPACITY_BYTES`] — the last because a capacity the
    /// trim can never reach would let the guard's memory and scan cost
    /// grow with the whole stream, the exact cost the period cap's doc
    /// promises stays constant.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::stream::repetition::RepetitionGuardConfig;
    ///
    /// assert!(RepetitionGuardConfig::default().validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        if self.min_period_bytes == 0 {
            return Err("min_period_bytes must be at least 1".to_string());
        }
        if self.min_repeats < 2 {
            return Err("min_repeats must be at least 2".to_string());
        }
        if self.max_period_bytes < self.min_period_bytes {
            return Err(format!(
                "max_period_bytes ({}) must be at least min_period_bytes ({})",
                self.max_period_bytes, self.min_period_bytes
            ));
        }
        if self.buffer_capacity() > MAX_BUFFER_CAPACITY_BYTES {
            return Err(format!(
                "max_period_bytes ({}) times the min_repeats ({}) headroom exceeds the \
                 {}-byte buffer ceiling — the guard's rolling buffer and per-delta scan \
                 would grow with the stream instead of staying bounded",
                self.max_period_bytes,
                self.min_repeats.saturating_add(1),
                MAX_BUFFER_CAPACITY_BYTES
            ));
        }
        Ok(())
    }

    /// The buffer capacity a consumer needs to retain, in bytes.
    ///
    /// One period beyond the largest scannable streak — enough for the
    /// detector to see `min_repeats` copies of the longest period plus
    /// headroom for the count to grow. Consumers trim their rolling
    /// buffer to this; a configuration whose implied capacity exceeds
    /// [`MAX_BUFFER_CAPACITY_BYTES`] fails
    /// [`validate`](Self::validate), so an accepted capacity is one
    /// the trim can actually enforce.
    #[must_use]
    pub fn buffer_capacity(&self) -> usize {
        self.max_period_bytes
            .saturating_mul(self.min_repeats.saturating_add(1))
    }
}

/// One detected repetition loop or single-character runaway.
///
/// The verdict a consumer aborts on: the byte window that repeated and
/// how many consecutive copies were visible at detection time — a
/// stanza's copies counted to the verdict threshold, a run of one
/// character's counted to the runaway line.
///
/// # Example
///
/// ```
/// use loopctl::stream::repetition::{RepetitionGuardConfig, detect_repetition};
///
/// let config = RepetitionGuardConfig::default();
/// let stanza = "<tool_call><function=Bash></function></tool_call>\n";
/// let hit = detect_repetition(stanza.repeat(4).as_bytes(), &config)
///     .expect("four verbatim copies are a loop");
/// assert_eq!(hit.period_bytes, stanza.len(), "a stanza's period is its own length");
/// let hit = detect_repetition("\u{2500}".repeat(1_365).as_bytes(), &config)
///     .expect("a run at the line is a runaway");
/// assert_eq!(hit.period_bytes, 3, "under the default floor of 48, one to four names a run");
///
/// let low_floor = RepetitionGuardConfig {
///     min_period_bytes: 2,
///     ..RepetitionGuardConfig::default()
/// };
/// let hit = detect_repetition("}\n".repeat(10).as_bytes(), &low_floor)
///     .expect("a two-byte stanza loop under a two-byte floor");
/// assert_eq!(hit.period_bytes, 2, "a period a two-byte runaway would also report");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepetitionHit {
    /// The repeating window's length, in bytes.
    ///
    /// For a stanza loop, the smallest qualifying period at detection
    /// time — the window that first accumulated [`repeats`](Self::repeats)
    /// consecutive verbatim copies. For a single-character runaway, the
    /// character's UTF-8 length (one to four). That tells the two shapes
    /// apart only while
    /// [`min_period_bytes`](RepetitionGuardConfig::min_period_bytes) is
    /// above four — the default floor — since a stanza's period is never
    /// below the floor; under a floor of four or less this field alone
    /// cannot, as the example shows.
    pub period_bytes: usize,

    /// The consecutive verbatim copies visible at detection time.
    ///
    /// For a stanza loop, counted up to
    /// [`min_repeats`](RepetitionGuardConfig::min_repeats) and no
    /// further — the verdict threshold, not a census of how far the
    /// loop had already run. For a single-character runaway, the copies
    /// it took to reach [`SINGLE_CHAR_RUNAWAY_BYTES`] — the same
    /// threshold, never a census of a longer run.
    pub repeats: usize,
}

/// The run of one repeated character ending a text, as the detector
/// measures it once per scan.
///
/// What lets the window rule tell a line from a stanza: the character's
/// UTF-8 length and how many consecutive copies of it end the text,
/// counted only as far as the runaway line — enough to decide both
/// rules without walking a run the guard has already condemned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrailingRun {
    /// The repeated character's UTF-8 length, in bytes.
    ///
    /// One to four: the smallest unit a run repeats at, and the period
    /// a runaway verdict reports.
    char_len: usize,

    /// The runaway line, in copies of the character.
    ///
    /// The line in bytes divided by the character's length, rounded
    /// down: the whole copies that fit inside it — never fewer than
    /// two. A capacity holding under two whole copies of a wide glyph
    /// would otherwise leave the rule unreachable, the run unmeasurable
    /// past the one glyph the exemption spares. Rounding down is what
    /// lets a trimmed rolling buffer answer — one trimmed to a capacity
    /// that is not a multiple of the character's length starts inside a
    /// glyph and holds exactly this many whole copies, so a run that
    /// fills it has reached the line.
    line_copies: usize,

    /// The consecutive copies ending the text, in characters.
    ///
    /// Counted only up to the line, so a long run costs the detector a
    /// bounded walk while the window rule still sees every whole-glyph
    /// window inside it.
    copies: usize,
}

impl TrailingRun {
    /// Measure the run of one character at `buffer`'s end, counting
    /// copies only up to the line `limit` bytes holds — two at the
    /// least, so a line under two whole copies still measures a pair.
    ///
    /// `None` when the buffer ends inside a character or holds only
    /// continuation bytes, so there is no character to repeat.
    fn measure(buffer: &[u8], limit: usize) -> Option<Self> {
        let char_len = trailing_char_len(buffer)?;
        let glyph = buffer.get(buffer.len().checked_sub(char_len)?..)?;
        let line_copies = limit.checked_div(char_len)?.max(2);
        let mut copies = 1_usize;
        while copies < line_copies {
            let run_start = buffer.len().saturating_sub(copies.saturating_mul(char_len));
            let Some(previous_start) = run_start.checked_sub(char_len) else {
                break;
            };
            if buffer.get(previous_start..run_start) != Some(glyph) {
                break;
            }
            copies = copies.saturating_add(1);
        }
        Some(Self {
            char_len,
            line_copies,
            copies,
        })
    }

    /// The run's length in bytes.
    ///
    /// The figure the window rule compares against: it asks whether a
    /// period fits inside the run.
    fn len_bytes(self) -> usize {
        self.copies.saturating_mul(self.char_len)
    }

    /// Whether the `period`-byte window at the text's end is one
    /// character repeated — whole copies of the run's glyph, edge to edge.
    fn covers(self, period: usize) -> bool {
        period <= self.len_bytes() && period.checked_rem(self.char_len) == Some(0)
    }

    /// The runaway verdict once the run has reached the line.
    ///
    /// A single copy is a character, not a run, so it condemns nothing —
    /// the first clause states the exemption outright; the line never
    /// falls below two copies, so the second alone would already imply it.
    fn runaway(self) -> Option<RepetitionHit> {
        if self.copies >= 2 && self.copies >= self.line_copies {
            return Some(RepetitionHit {
                period_bytes: self.char_len,
                repeats: self.copies,
            });
        }
        None
    }
}

/// The UTF-8 length of the character `buffer` ends with, if it ends on
/// a character boundary.
///
/// UTF-8 is self-synchronizing: among the suffixes one to four bytes
/// long, the shortest that is valid UTF-8 is exactly the final
/// character — a continuation byte cannot open a sequence, so no
/// shorter suffix of a multibyte character parses, and an ASCII final
/// byte parses at length one. A buffer that ends inside a character or
/// holds only continuation bytes has no valid suffix and answers `None`.
fn trailing_char_len(buffer: &[u8]) -> Option<usize> {
    (1..=4).find(|&len| {
        buffer
            .len()
            .checked_sub(len)
            .and_then(|start| buffer.get(start..))
            .is_some_and(|bytes| std::str::from_utf8(bytes).is_ok())
    })
}

/// Whether the text's tail is a repetition loop or a single-character
/// runaway.
///
/// Two rules over the text's tail, the runaway rule first. The window
/// rule scans periods from
/// [`min_period_bytes`](RepetitionGuardConfig::min_period_bytes) up to
/// the smaller of [`max_period_bytes`](RepetitionGuardConfig::max_period_bytes)
/// and what the text length can hold `min_repeats` copies of; the
/// first period whose trailing window is preceded by enough
/// byte-identical copies of itself fires — unless that window is one
/// character repeated, which is a line (a border, a rule of dashes)
/// rather than a stanza and passes at every period. The runaway rule
/// answers for what the exemption lets through: a trailing run of one
/// character [`SINGLE_CHAR_RUNAWAY_BYTES`] long, counted in whole copies
/// of the character and bounded by the configuration's buffer capacity
/// where that is smaller, never below two copies — a model stuck on one
/// character, reported with
/// the character's byte length as the period and the copies it took to
/// reach the line as the repeat count. Disarmed
/// ([`enabled`](RepetitionGuardConfig::enabled) = `false`)
/// or self-inconsistent configurations never fire — an invalid config
/// disables rather than reports. Comparison is byte-level: equal-length
/// byte slices compare exactly regardless of UTF-8 boundary placement,
/// and every reported period is in bytes.
#[must_use]
pub fn detect_repetition(buffer: &[u8], config: &RepetitionGuardConfig) -> Option<RepetitionHit> {
    if !config.enabled
        || config.min_period_bytes == 0
        || config.min_repeats < 2
        || config.max_period_bytes < config.min_period_bytes
        || config.buffer_capacity() > MAX_BUFFER_CAPACITY_BYTES
    {
        return None;
    }
    let runaway_line = SINGLE_CHAR_RUNAWAY_BYTES.min(config.buffer_capacity());
    let run = TrailingRun::measure(buffer, runaway_line);
    if let Some(hit) = run.and_then(TrailingRun::runaway) {
        return Some(hit);
    }
    let upper = config
        .max_period_bytes
        .min(buffer.len().checked_div(config.min_repeats)?);
    for period in config.min_period_bytes..=upper {
        if run.is_some_and(|run| run.covers(period)) {
            continue;
        }
        let tail_start = buffer.len().checked_sub(period)?;
        let tail = buffer.get(tail_start..)?;
        let mut repeats = 1_usize;
        let mut cursor = tail_start;
        while cursor >= period && repeats < config.min_repeats {
            let previous_start = cursor.checked_sub(period)?;
            let previous = buffer.get(previous_start..cursor)?;
            if previous != tail {
                break;
            }
            repeats = repeats.saturating_add(1);
            cursor = previous_start;
        }
        if repeats >= config.min_repeats {
            return Some(RepetitionHit {
                period_bytes: period,
                repeats,
            });
        }
    }
    None
}

/// The assistant text the guard scans, both lanes concatenated.
///
/// Content and thinking text in part order — the closest a completed
/// [`Message`](crate::message::Message) comes to the arrival-order byte
/// stream the live detector watches. Tool-call and tool-result parts
/// carry no generated prose and never enter the buffer. Feeds both the
/// handler's non-streaming fallback guard and the engine's
/// always-compiled non-streaming turn guard.
#[must_use]
pub(crate) fn concat_lane_text(message: &crate::message::Message) -> String {
    use crate::message::MessagePart;
    let mut text = String::new();
    for part in &message.parts {
        match part {
            MessagePart::Text { text: fragment } | MessagePart::Thinking { text: fragment, .. } => {
                text.push_str(fragment);
            }
            _ => {}
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_armed_and_conservative() {
        let config = RepetitionGuardConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_period_bytes, 48);
        assert_eq!(config.min_repeats, 4);
        assert_eq!(config.max_period_bytes, 4096);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_inconsistent_bounds() {
        assert!(
            RepetitionGuardConfig {
                min_period_bytes: 0,
                ..RepetitionGuardConfig::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            RepetitionGuardConfig {
                min_repeats: 1,
                ..RepetitionGuardConfig::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            RepetitionGuardConfig {
                max_period_bytes: 8,
                min_period_bytes: 48,
                ..RepetitionGuardConfig::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn a_config_over_the_buffer_ceiling_never_fires() {
        // Detection disables on every rule validate enforces — the
        // capacity rule included. The same degenerate tail fires under
        // knobs inside the ceiling and stays silent above it, so a
        // hand-built config whose buffer would grow with the stream
        // never aborts anything. The stanza is not a run of one
        // character: a run passes the window rule at any capacity, so it
        // cannot stand in for the degenerate tail here.
        let stanza = b"row\n";
        let mut buffer = stanza.to_vec();
        buffer.extend_from_slice(stanza);
        buffer.extend_from_slice(stanza);
        let over_ceiling = RepetitionGuardConfig {
            min_period_bytes: 4,
            min_repeats: 3,
            max_period_bytes: 600_000,
            ..RepetitionGuardConfig::default()
        };
        assert!(over_ceiling.validate().is_err());
        assert!(
            detect_repetition(&buffer, &over_ceiling).is_none(),
            "a config whose implied capacity exceeds the ceiling never fires"
        );
        let inside = RepetitionGuardConfig {
            max_period_bytes: 1_024,
            ..over_ceiling
        };
        assert!(inside.validate().is_ok());
        assert!(
            detect_repetition(&buffer, &inside).is_some(),
            "the same tail fires under knobs inside the ceiling — the pin cannot pass vacuously"
        );
    }

    #[test]
    fn validate_rejects_knobs_that_defeat_the_buffer_bound() {
        // A huge period cap with the trim never firing is the footgun:
        // the guard buffer would grow with the whole stream and the
        // per-delta scan with it. The product bound holds the promise
        // the max-period doc makes.
        assert!(
            RepetitionGuardConfig {
                max_period_bytes: usize::MAX,
                ..RepetitionGuardConfig::default()
            }
            .validate()
            .is_err(),
            "the period cap alone cannot exceed the buffer ceiling"
        );
        assert!(
            RepetitionGuardConfig {
                max_period_bytes: 262_145,
                min_repeats: 3,
                ..RepetitionGuardConfig::default()
            }
            .validate()
            .is_err(),
            "period cap times repeat headroom over the ceiling is rejected"
        );
        let at_ceiling = RepetitionGuardConfig {
            max_period_bytes: 262_144,
            min_repeats: 3,
            ..RepetitionGuardConfig::default()
        };
        assert!(
            at_ceiling.validate().is_ok(),
            "a config whose capacity lands exactly on the ceiling is accepted"
        );
        assert_eq!(
            at_ceiling.buffer_capacity(),
            super::MAX_BUFFER_CAPACITY_BYTES,
            "the accepted boundary config's capacity is the ceiling itself"
        );
    }

    #[test]
    fn identical_stanza_repeats_fire_at_the_fourth_copy() {
        let config = RepetitionGuardConfig::default();
        let stanza = "<tool_call><function=Bash><parameter=command>true\n</parameter></function></tool_call>";
        assert!(
            detect_repetition(stanza.repeat(3).as_bytes(), &config).is_none(),
            "three copies are one short of the conservative default"
        );
        let hit = detect_repetition(stanza.repeat(4).as_bytes(), &config)
            .expect("four verbatim copies are a loop");
        assert_eq!(hit.period_bytes, stanza.len());
        assert_eq!(hit.repeats, 4);
    }

    #[test]
    fn long_single_block_repeats_fire_past_the_minimum() {
        // The block is longer than the minimum window and periodic
        // nowhere inside itself, so the smallest period the text repeats
        // at is the block's own length: a period far above the minimum
        // still fires, with the repeat count stopping at the threshold.
        let config = RepetitionGuardConfig::default();
        let block = (0..100).fold(String::new(), |mut joined, segment| {
            push_line(
                &mut joined,
                format_args!("segment-{segment:04}-of-a-long-distinct-block"),
            );
            joined
        });
        assert!(
            block.len() > config.min_period_bytes && block.len() <= config.max_period_bytes,
            "the fixture spans the scannable range: {} bytes",
            block.len()
        );
        let hit = detect_repetition(block.repeat(6).as_bytes(), &config)
            .expect("a period above the minimum still fires");
        assert_eq!(
            hit.period_bytes,
            block.len(),
            "the block's own length is the smallest period the text repeats at"
        );
        assert_eq!(hit.repeats, config.min_repeats);
    }

    /// Append one formatted line to `text`, panicking never — writing
    /// to a `String` cannot fail.
    fn push_line(text: &mut String, line: std::fmt::Arguments<'_>) {
        std::fmt::Write::write_fmt(text, line).expect("writing to a String cannot fail");
    }

    #[test]
    fn periods_above_the_cap_never_fire() {
        let config = RepetitionGuardConfig::default();
        let mut block = String::new();
        for segment in 0..200 {
            push_line(
                &mut block,
                format_args!("segment-{segment:04}-of-a-long-distinct-block\n"),
            );
        }
        assert_eq!(block.len(), 7600);
        assert!(
            detect_repetition(block.repeat(3).as_bytes(), &config).is_none(),
            "a non-self-periodic block above max_period_bytes repeating is outside the contract"
        );
    }

    #[test]
    fn two_row_alternation_is_periodic_and_fires() {
        // Perfectly alternating rows repeat verbatim at a two-row
        // period, so the rule fires there too — a stream locked into
        // an a/b alternation is as degenerate as a single-row loop,
        // and the knob workload is the same.
        let config = RepetitionGuardConfig::default();
        let sensor_a = "2026-11-19T00:00:00,sensor-a,ok,0.512345678901234,0.49\n";
        let sensor_b = "2026-11-19T00:00:00,sensor-b,ok,0.512345678901234,0.51\n";
        let mut text = String::new();
        for index in 0..40 {
            let line = if index % 2 == 0 { sensor_a } else { sensor_b };
            text.push_str(line);
        }
        assert!(detect_repetition(text.as_bytes(), &config).is_some());
    }

    #[test]
    fn fully_identical_long_rows_fire_as_documented() {
        // The rule is consecutive verbatim windows, so a run of four or
        // more byte-identical rows at or above the 48-byte floor does
        // fire — machine-generated data with fully identical rows is
        // the documented knob workload (raise min_repeats, raise
        // min_period_bytes, or disarm), not a case the default can
        // tell apart from a degenerate loop.
        let config = RepetitionGuardConfig::default();
        let row = "2026-11-19T00:00:00,sensor-a,ok,0.512345678901234,0.49\n";
        assert!(detect_repetition(row.repeat(6).as_bytes(), &config).is_some());
        let forgiving = RepetitionGuardConfig {
            min_repeats: 8,
            ..RepetitionGuardConfig::default()
        };
        assert!(
            detect_repetition(row.repeat(6).as_bytes(), &forgiving).is_none(),
            "the min_repeats knob moves the line for this workload"
        );
    }

    #[test]
    fn varied_csv_rows_never_fire() {
        let config = RepetitionGuardConfig::default();
        let text = (0..1_000)
            .map(|row| format!("id={row},name=item-{row},value={row}.25,flag=true\n"))
            .fold(String::new(), |mut joined, line| {
                push_line(&mut joined, format_args!("{line}"));
                joined
            });
        assert!(detect_repetition(text.as_bytes(), &config).is_none());
    }

    #[test]
    fn a_markdown_table_never_fires() {
        let config = RepetitionGuardConfig::default();
        let mut text = String::from("| id | name | value |\n|---|---|---|\n");
        for row in 0..200 {
            push_line(
                &mut text,
                format_args!("| {row} | item-{row} | {row}.50 |\n"),
            );
        }
        assert!(detect_repetition(text.as_bytes(), &config).is_none());
    }

    #[test]
    fn similar_code_lines_with_indentation_differences_never_fire() {
        let config = RepetitionGuardConfig::default();
        let mut text = String::new();
        for depth in 0..60 {
            push_line(
                &mut text,
                format_args!(
                    "{}if (item{depth}.value > 0) {{ total += item{depth}.value; }}\n",
                    " ".repeat(depth % 4)
                ),
            );
        }
        assert!(detect_repetition(text.as_bytes(), &config).is_none());
    }

    #[test]
    fn short_units_among_varied_content_never_fire() {
        let config = RepetitionGuardConfig::default();
        let mut text = String::new();
        for row in 0..200 {
            push_line(&mut text, format_args!("item {row}: value {row}.25\n}}\n"));
        }
        assert!(
            detect_repetition(text.as_bytes(), &config).is_none(),
            "sub-window boilerplate (closing braces, short separators) between varying rows is not a loop"
        );
    }

    #[test]
    fn a_long_uniform_run_of_short_units_fires() {
        // "0,0,0,0\n" repeated is 8-periodic text, hence also
        // 48-periodic — a long uniform run of a short unit is
        // indistinguishable from a loop and fires at the smallest
        // scannable window.
        let config = RepetitionGuardConfig::default();
        assert!(detect_repetition("0,0,0,0\n".repeat(500).as_bytes(), &config).is_some());
        assert!(detect_repetition("}\n".repeat(500).as_bytes(), &config).is_some());
    }

    #[test]
    fn a_single_glyph_border_never_fires_the_window_rule() {
        // A box-drawing horizontal is three bytes in UTF-8, so 64 of
        // them are 192 bytes — four minimum windows of identical text.
        // The guard scans every prefix of a stream, so every width up
        // to the line's last whole copy (1_364 glyphs, 4_092 bytes) has
        // to pass, not only the finished border.
        let config = RepetitionGuardConfig::default();
        assert_eq!("\u{2500}".repeat(64).len(), 192);
        let mut border = String::new();
        for _ in 0..1_364 {
            border.push('\u{2500}');
            assert!(
                detect_repetition(border.as_bytes(), &config).is_none(),
                "a border {} glyphs wide is a line, not a loop",
                border.chars().count()
            );
        }
        assert!(
            detect_repetition("-".repeat(4_095).as_bytes(), &config).is_none(),
            "an ASCII rule of dashes is the same line at a third of the glyph width"
        );
    }

    #[test]
    fn a_single_character_runaway_fires_at_the_line() {
        let config = RepetitionGuardConfig::default();
        assert!(
            detect_repetition("x".repeat(4_095).as_bytes(), &config).is_none(),
            "one byte under the line is still only a long run"
        );
        let hit = detect_repetition("x".repeat(4_096).as_bytes(), &config)
            .expect("one character run past 4 KiB is a runaway");
        assert_eq!(
            hit.period_bytes, 1,
            "the runaway's period is the character's byte length"
        );
        assert_eq!(hit.repeats, 4_096, "the copies it took to cross the line");
    }

    #[test]
    fn a_multibyte_runaway_reports_the_glyph_length_and_its_copies() {
        let config = RepetitionGuardConfig::default();
        // The line is whole copies of the character: 4_096 / 3 rounds
        // down to 1_365 copies (4_095 bytes), so the 1_365th copy reaches
        // it and 1_364 (4_092 bytes) is the last width under.
        assert!(detect_repetition("\u{2500}".repeat(1_364).as_bytes(), &config).is_none());
        let hit = detect_repetition("\u{2500}".repeat(1_365).as_bytes(), &config)
            .expect("a three-byte glyph reaches the line at the 1_365th copy");
        assert_eq!(hit.period_bytes, 3);
        assert_eq!(hit.repeats, 1_365);
    }

    #[test]
    fn a_runaway_line_never_exceeds_the_buffer_capacity() {
        // A shrunken period cap shrinks the rolling buffer with it, and
        // the line follows the capacity down. On the untrimmed completed
        // text the line is the whole copies of the character the capacity
        // holds: 1_280 for a one-byte glyph, 426 (1_278 bytes) for a
        // three-byte one. The trimmed streaming buffer is a different
        // shape — a_trimmed_rolling_buffer_reaches_the_line drives it.
        let config = RepetitionGuardConfig {
            max_period_bytes: 256,
            ..RepetitionGuardConfig::default()
        };
        assert_eq!(config.buffer_capacity(), 1_280);
        assert!(detect_repetition("x".repeat(1_279).as_bytes(), &config).is_none());
        let hit = detect_repetition("x".repeat(1_280).as_bytes(), &config)
            .expect("the whole buffer one character is a runaway");
        assert_eq!(hit.period_bytes, 1);
        assert_eq!(hit.repeats, 1_280);
        assert!(
            detect_repetition("\u{2500}".repeat(425).as_bytes(), &config).is_none(),
            "425 copies are 1_275 bytes — under the line"
        );
        let hit = detect_repetition("\u{2500}".repeat(426).as_bytes(), &config)
            .expect("426 copies are the whole copies the capacity holds");
        assert_eq!(hit.period_bytes, 3);
        assert_eq!(hit.repeats, 426);
    }

    #[test]
    fn a_capacity_under_two_whole_copies_still_condemns_a_pair() {
        // The smallest capacity a valid configuration implies is three
        // bytes (a one-byte period cap with two repeats), which holds
        // one whole copy of a two-to-four-byte glyph and none of the
        // second. Left at the capacity's whole copies, the line sits
        // under two and the rule is unreachable: the run never measures
        // past the lone glyph the exemption spares, and any run of it
        // passes in silence. Floored at two copies, a pair condemns at
        // the glyph's own period while a lone glyph stays exempt. A
        // one-byte glyph's line is already the whole capacity, and the
        // floor leaves it where it is.
        let config = RepetitionGuardConfig {
            min_period_bytes: 1,
            min_repeats: 2,
            max_period_bytes: 1,
            ..RepetitionGuardConfig::default()
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.buffer_capacity(), 3);
        for (glyph, char_len) in [("é", 2_usize), ("─", 3), ("😀", 4)] {
            assert_eq!(
                detect_repetition(glyph.as_bytes(), &config),
                None,
                "{glyph:?}: one copy is a character, not a run"
            );
            assert_eq!(
                detect_repetition(glyph.repeat(2).as_bytes(), &config),
                Some(RepetitionHit {
                    period_bytes: char_len,
                    repeats: 2,
                }),
                "{glyph:?}: a pair reaches the two-copy floor"
            );
        }
        assert_eq!(
            detect_repetition("xx".as_bytes(), &config),
            None,
            "a one-byte glyph's line is the whole capacity — two copies are under it"
        );
        assert_eq!(
            detect_repetition("xxx".as_bytes(), &config),
            Some(RepetitionHit {
                period_bytes: 1,
                repeats: 3,
            }),
            "three one-byte copies fill the capacity and reach the line"
        );
    }

    #[test]
    fn a_trimmed_rolling_buffer_reaches_the_line_for_every_character_length() {
        // The streaming handler trims its rolling buffer to exactly the
        // capacity — on no character boundary — and rescans after every
        // delta. A capacity that is not a multiple of the character's
        // length leaves the buffer starting inside a glyph, holding one
        // copy fewer than a whole number of lines; a line counted in
        // whole copies (rounded down) is still reached when the run fills
        // the buffer, whether the trim cut inside a glyph or on one, at
        // the same copy count the untrimmed completed text reports.
        let config = RepetitionGuardConfig {
            max_period_bytes: 256,
            ..RepetitionGuardConfig::default()
        };
        let capacity = config.buffer_capacity();
        assert_eq!(capacity, 1_280);
        for (glyph, char_len) in [("x", 1_usize), ("é", 2), ("─", 3), ("😀", 4)] {
            let line_copies = capacity / char_len;
            let expected = RepetitionHit {
                period_bytes: char_len,
                repeats: line_copies,
            };
            let run = glyph.repeat(line_copies + 40);
            let run = run.as_bytes();
            let byte_trimmed = &run[run.len() - capacity..];
            assert_eq!(byte_trimmed.len(), capacity);
            assert_eq!(
                detect_repetition(byte_trimmed, &config),
                Some(expected),
                "{glyph:?}: a trim to exactly the capacity, inside a glyph"
            );
            let aligned = capacity - (capacity % char_len);
            let aligned_trimmed = &run[run.len() - aligned..];
            assert_eq!(
                detect_repetition(aligned_trimmed, &config),
                Some(expected),
                "{glyph:?}: a trim on a character boundary holds the same copies"
            );
            assert_eq!(
                detect_repetition(glyph.repeat(line_copies - 1).as_bytes(), &config),
                None,
                "{glyph:?}: one copy under the line is still only a long run"
            );
            assert_eq!(
                detect_repetition(glyph.repeat(line_copies).as_bytes(), &config),
                Some(expected),
                "{glyph:?}: the untrimmed text answers at the same copy count"
            );
        }
    }

    #[test]
    fn a_stanza_loop_ending_in_a_run_still_fires_at_the_stanza_period() {
        // The exemption is per window, never a short-circuit on the
        // tail: every window up to the 200-byte dash run is a run and
        // passes, and the 204-byte window — the stanza — still
        // accumulates its four copies.
        let config = RepetitionGuardConfig::default();
        let stanza = format!("row {}", "-".repeat(200));
        assert_eq!(stanza.len(), 204);
        let hit = detect_repetition(stanza.repeat(4).as_bytes(), &config)
            .expect("a stanza that happens to end in a run is still a stanza");
        assert_eq!(hit.period_bytes, stanza.len());
        assert_eq!(hit.repeats, 4);
    }

    #[test]
    fn disabled_config_never_fires() {
        let config = RepetitionGuardConfig {
            enabled: false,
            ..RepetitionGuardConfig::default()
        };
        let stanza = "the same thing over and over and over and over again, verbatim.";
        assert!(detect_repetition(stanza.repeat(20).as_bytes(), &config).is_none());
    }

    #[test]
    fn inconsistent_config_never_fires_rather_than_dividing_by_zero() {
        let zero_repeats = RepetitionGuardConfig {
            min_repeats: 0,
            ..RepetitionGuardConfig::default()
        };
        assert!(detect_repetition(b"anything at all", &zero_repeats).is_none());
    }

    #[test]
    fn multibyte_text_compares_by_bytes_without_panicking() {
        let config = RepetitionGuardConfig::default();
        let stanza = "ключ-значение-повтор-цикл-повтор-цикл-ещё-раз-снова";
        let hit = detect_repetition(stanza.repeat(4).as_bytes(), &config)
            .expect("a multibyte stanza repeating verbatim still fires");
        assert_eq!(hit.period_bytes, stanza.len());
    }

    #[test]
    fn a_prefix_of_distinct_text_does_not_mask_the_loop() {
        let config = RepetitionGuardConfig::default();
        let stanza = "a completely ordinary sentence that simply repeats. ";
        let mut text = String::from("an ordinary opening paragraph with its own content.\n\n");
        text.push_str(&stanza.repeat(5));
        let hit =
            detect_repetition(text.as_bytes(), &config).expect("the trailing loop still fires");
        assert_eq!(hit.period_bytes, stanza.len());
    }
}
