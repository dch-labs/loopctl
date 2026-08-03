//! Ratios of integer counts as floats in `[0.0, 1.0]`, computed without
//! tripping `cast_precision_loss`.
//!
//! Clippy's `cast_precision_loss` lint (denied via the crate's pedantic config)
//! rejects direct `usize as f32` / `usize as f64` / `f64 as f32` casts because
//! a float mantissa cannot hold a full 64-bit integer. The only fully
//! lint-clean paths from a count to a float go through an integer that fits the
//! mantissa: `u32` for `f64` (lossless), `u16` for `f32` (lossless). This
//! module wraps those paths behind a single concept — a unit-interval ratio of
//! two counts — so the narrowing, scaling, and zero-denominator handling live
//! in one place instead of being inlined (and silently drifted) per call site.

/// Ratio of two integer counts as an `f64` in `[0.0, 1.0]`.
///
/// Each count is narrowed to `f64` via `u32` (whose 32 bits fit inside `f64`'s
/// 52-bit mantissa) with a saturating fallback to `u32::MAX` for counts above
/// `u32::MAX`. Accepts any integer that narrows to `u32` (`usize`, `u64`,
/// `u32`, `u16`, ...). Returns `0.0` when `denominator == 0`. Use this for
/// utilization, success-rate, and similar fractions derived from token counts,
/// call counts, or other magnitudes realistically far below `u32::MAX`.
#[must_use]
pub fn unit_ratio_f64(numerator: impl TryInto<u32>, denominator: impl TryInto<u32>) -> f64 {
    let numerator = numerator.try_into().unwrap_or(u32::MAX);
    let denominator = denominator.try_into().unwrap_or(u32::MAX);
    if denominator == 0 {
        return 0.0;
    }
    f64::from(numerator) / f64::from(denominator)
}

/// Ratio of two `usize` counts as an `f32` in `[0.0, 1.0]`.
///
/// Both counts are scaled by a shared divisor before each is narrowed to `u16`
/// (the widest integer that losslessly converts to `f32`), so the quotient is
/// preserved even when both exceed `u16::MAX`. Without the shared scaling,
/// independent saturation would collapse any partial overlap above `u16::MAX`
/// to `1.0`. Returns `0.0` when `denominator == 0`. Use this for similarity,
/// match-fraction, and other `[0, 1]` ratios over word or set sizes.
#[must_use]
pub fn unit_ratio_f32(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        return 0.0;
    }
    let scale = scale_to_u16(denominator.max(numerator));
    let numerator_scaled = numerator.checked_div(scale).unwrap_or(numerator);
    let denominator_scaled = denominator.checked_div(scale).unwrap_or(denominator);
    let numerator_f = f32::from(u16::try_from(numerator_scaled).unwrap_or(u16::MAX));
    let denominator_f = f32::from(u16::try_from(denominator_scaled).unwrap_or(u16::MAX));
    numerator_f / denominator_f
}

/// Divisor that brings a `usize` count at or below `u16::MAX`.
///
/// Returns `1` when `count` already fits in `u16`; otherwise a divisor that
/// scales it (and any sibling count divided by the same value) into range.
/// Dividing both operands of a ratio by this shared value preserves the
/// quotient — see [`unit_ratio_f32`].
fn scale_to_u16(count: usize) -> usize {
    const U16_RANGE: usize = u16::MAX as usize + 1;
    count
        .checked_div(U16_RANGE)
        .unwrap_or(usize::MAX)
        .saturating_add(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_ratio_f64_normal_range() {
        assert!((unit_ratio_f64(1, 4) - 0.25).abs() < 0.001);
        assert!((unit_ratio_f64(3, 3) - 1.0).abs() < f64::EPSILON);
        assert!(unit_ratio_f64(0, 3).abs() < f64::EPSILON);
    }

    #[test]
    fn unit_ratio_f64_accepts_u64_counts() {
        assert!((unit_ratio_f64(50u64, 100u64) - 0.5).abs() < 0.001);
    }

    #[test]
    fn unit_ratio_f64_is_zero_for_zero_denominator() {
        assert!(unit_ratio_f64(5, 0).abs() < f64::EPSILON);
    }

    #[test]
    fn unit_ratio_f64_saturates_above_u32_max() {
        assert!(unit_ratio_f64(u32::MAX as usize + 1, u32::MAX as usize + 1) >= 0.99);
    }

    #[test]
    fn unit_ratio_f32_identity_and_disjoint_in_normal_range() {
        assert!((unit_ratio_f32(3, 3) - 1.0).abs() < f32::EPSILON);
        assert!(unit_ratio_f32(0, 3).abs() < f32::EPSILON);
    }

    #[test]
    fn unit_ratio_f32_preserves_fraction_in_normal_range() {
        assert!((unit_ratio_f32(1, 4) - 0.25).abs() < 0.01);
        assert!((unit_ratio_f32(2, 3) - 0.66).abs() < 0.01);
    }

    #[test]
    fn unit_ratio_f32_is_zero_for_zero_denominator() {
        assert!(unit_ratio_f32(5, 0).abs() < f32::EPSILON);
    }

    #[test]
    fn unit_ratio_f32_large_partial_overlap_does_not_collapse_to_one() {
        let ratio = unit_ratio_f32(70_000, 100_000);
        assert!((ratio - 0.7_f32).abs() < 0.01, "expected ~0.7, got {ratio}");
        assert!(
            (ratio - 1.0_f32).abs() > 0.01,
            "partial overlap must not collapse to 1.0"
        );
    }

    #[test]
    fn scale_to_u16_is_one_below_threshold_and_grows_above() {
        assert_eq!(scale_to_u16(0), 1);
        assert_eq!(scale_to_u16(1), 1);
        assert_eq!(scale_to_u16(u16::MAX as usize), 1);
        assert_eq!(scale_to_u16(u16::MAX as usize + 1), 2);
        assert!(scale_to_u16(u16::MAX as usize * 4) > 1);
    }
}
