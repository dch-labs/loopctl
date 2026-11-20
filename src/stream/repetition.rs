//! The stream repetition guard — a runaway-generation abort for
//! degenerate repetition loops.
//!
//! A quantized model can enter a repetition attractor and emit the same
//! window of text indefinitely — cheaply, when speculative decoding
//! amplifies it — until the provider's output cap stops it. The guard
//! watches the decoded assistant text (both content and thinking
//! lanes, appended in arrival order) and reports the loop as soon as
//! one byte window repeats verbatim enough consecutive times, so the
//! consumer can abort the request in seconds instead of decoding to
//! the cap.
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

/// The repetition guard's configuration.
///
/// Defaults are armed and conservative: a window of at least 48 bytes
/// repeated 4 consecutive times. Hosts with legitimate content that
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
    /// stanza can start a streak. Must be at least 1.
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

/// One detected repetition loop.
///
/// The verdict a consumer aborts on: the byte window that repeated and
/// how many consecutive copies were visible at detection time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepetitionHit {
    /// The repeating window's length, in bytes.
    ///
    /// The smallest qualifying period at detection time — the window
    /// that first accumulated [`repeats`](Self::repeats) consecutive
    /// verbatim copies.
    pub period_bytes: usize,

    /// The consecutive verbatim copies visible at detection time.
    ///
    /// Counted up to [`min_repeats`](RepetitionGuardConfig::min_repeats)
    /// and no further — the verdict threshold, not a census of how far
    /// the loop had already run.
    pub repeats: usize,
}

/// Whether the text's tail is a verbatim repetition loop.
///
/// Scans periods from [`min_period_bytes`](RepetitionGuardConfig::min_period_bytes)
/// up to the smaller of [`max_period_bytes`](RepetitionGuardConfig::max_period_bytes)
/// and what the text length can hold `min_repeats` copies of; the
/// first period whose trailing window is preceded by enough
/// byte-identical copies of itself wins. Disarmed
/// ([`enabled`](RepetitionGuardConfig::enabled) = `false`) or
/// self-inconsistent configurations never fire — an invalid config
/// disables rather than reports. Comparison is byte-level: equal-length
/// byte slices compare exactly regardless of UTF-8 boundary placement,
/// and the reported period is in bytes.
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
    let upper = config
        .max_period_bytes
        .min(buffer.len().checked_div(config.min_repeats)?);
    for period in config.min_period_bytes..=upper {
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
        // never aborts anything.
        let stanza = vec![b'x'; 4];
        let mut buffer = stanza.clone();
        buffer.extend_from_slice(&stanza);
        buffer.extend_from_slice(&stanza);
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
        let config = RepetitionGuardConfig::default();
        let block = "x".repeat(3000);
        let hit = detect_repetition(block.repeat(6).as_bytes(), &config)
            .expect("a period above the minimum still fires");
        assert_eq!(
            hit.period_bytes, config.min_period_bytes,
            "uniform text is periodic at the smallest scannable window, and the smallest period wins"
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
