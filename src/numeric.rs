//! Ratios of integer counts as floats, computed without tripping
//! `cast_precision_loss`.
//!
//! Clippy's `cast_precision_loss` lint — denied via the crate's pedantic
//! config — rejects direct `usize as f32`, `usize as f64`, and `f64 as f32`
//! casts, because a float mantissa cannot hold a full 64-bit integer. The only
//! fully lint-clean paths from a count to a float go through an integer that
//! fits the mantissa: `u32` for `f64` (lossless), `u16` for `f32` (lossless).
//!
//! This module wraps those paths behind a single concept — the ratio of two
//! counts — so the narrowing, shared scaling, and zero-denominator handling
//! live in one place instead of being inlined (and silently drifted) per call
//! site. The generic [`unit_ratio`] is the entry point; pass two integer counts
//! of the same type and Rust infers the family (`u64` → `f64`, `usize` → `f32`)
//! from the argument types via the [`RatioInt`] implementations.

/// Integer family usable as the operands of a [`unit_ratio`] computation.
///
/// Implemented for the wide unsigned integer that holds the input counts
/// (`u64` for `f64`, `usize` for `f32`). The associated [`Narrow`](Self::Narrow)
/// type is the matching integer that losslessly converts to the family's float,
/// and [`Float`](Self::Float) is the float itself.
///
/// The associated constants and functions exist because the integer operations
/// the ratio needs — `checked_div` and `saturating_add` — are *inherent*
/// methods on each integer rather than trait methods, so no standard-library
/// generic bound can reach them. The same applies to float `/`, which is kept
/// behind [`float_div`](Self::float_div): a generic `T::Float / T::Float` would
/// trip the crate's `arithmetic_side_effects` lint because clippy cannot prove
/// the operands are floats. Burying the concrete division inside the impl keeps
/// the lint satisfied while letting the generic [`unit_ratio`] stay uniform.
pub(crate) trait RatioInt: Copy + PartialOrd + Sized {
    /// Narrow unsigned integer that converts losslessly into [`Float`](Self::Float).
    ///
    /// The widest integer that fits the float's mantissa — `u32` for `f64`,
    /// `u16` for `f32`. Counts are scaled into this type's range before the
    /// final float conversion so the conversion never trips
    /// `cast_precision_loss`.
    type Narrow: Copy;

    /// Float type the ratio is returned as.
    ///
    /// Bounded by `Default` (so the generic can produce `0.0` without a
    /// literal) and `From<Narrow>` (the lossless conversion this family
    /// revolves around).
    type Float: Copy + Default + From<Self::Narrow>;

    /// Representation of zero on the wide integer.
    ///
    /// Used both for the zero-denominator guard and to detect a scaled
    /// numerator that has floored to zero (see [`unit_ratio`]).
    const ZERO: Self;

    /// Representation of one on the wide integer.
    ///
    /// The minimum value a clamped scaled numerator is raised to, and the
    /// `+1` added to the scale so a count equal to `NARROW_MAX` is left
    /// unscaled.
    const ONE: Self;

    /// `Self::MAX` — the saturating fallback when an integer op would overflow.
    ///
    /// Used inside the scale computation when the divisor itself overflows;
    /// in practice the inputs are far smaller, so this is purely defensive.
    const MAX: Self;

    /// [`Narrow`](Self::Narrow)::`MAX`.
    ///
    /// Used as the saturating fallback when a scaled count still does not fit
    /// — a defensive guard, since the shared scaling is sized to prevent this
    /// in practice.
    const NARROW_MAX: Self::Narrow;

    /// One past [`Narrow`](Self::Narrow)::`MAX` — the divisor threshold.
    ///
    /// This is the smallest divisor that guarantees a scaled count fits in
    /// `Narrow`. Using exactly `NARROW_MAX + 1` (rather than `NARROW_MAX`) is
    /// what keeps a count equal to `NARROW_MAX` on the unscaled path:
    /// `NARROW_MAX / (NARROW_MAX + 1) == 0`, so the computed scale is `1` and
    /// the count survives intact. The boundary test in this module pins this
    /// invariant, which was the site of an off-by-one bug during development.
    const RANGE: Self;

    /// Maximum of two values.
    ///
    /// Used to pick the larger of the numerator and denominator when sizing the
    /// shared scale, so a count that fits never forces the other operand to be
    /// scaled away.
    #[must_use]
    fn max(self, other: Self) -> Self;

    /// Checked division.
    ///
    /// Returns `None` on divide-by-zero so the caller can supply a fallback
    /// rather than panicking — required by the no-panic policy.
    fn checked_div(self, rhs: Self) -> Option<Self>;

    /// Saturating addition.
    ///
    /// Used only on the `+1` term when computing the scale, so the result can
    /// never overflow; saturating is the lint-clean spelling.
    #[must_use]
    fn saturating_add(self, rhs: Self) -> Self;

    /// Fallible narrowing to [`Narrow`](Self::Narrow).
    ///
    /// Returns `None` if the value exceeds `NARROW_MAX`; the caller falls back
    /// to [`NARROW_MAX`](Self::NARROW_MAX). In practice the shared scaling has
    /// already brought the value into range, so the fallback is defensive.
    fn try_into_narrow(self) -> Option<Self::Narrow>;

    /// Divide two floats to form the ratio.
    ///
    /// A trait method rather than a generic `T::Float / T::Float` solely to
    /// satisfy the crate's `arithmetic_side_effects` lint — see the trait-level
    /// docs.
    fn float_div(numerator: Self::Float, denominator: Self::Float) -> Self::Float;
}

/// [`RatioInt`] for `f64` ratios: counts in `u64`, narrowed via `u32`.
///
/// `u32` fits losslessly into `f64`'s 52-bit mantissa, so this is the highest
/// family the crate needs. Counts above `u32::MAX` are scaled before
/// narrowing — see [`unit_ratio`].
impl RatioInt for u64 {
    type Narrow = u32;
    type Float = f64;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = u64::MAX;
    const NARROW_MAX: Self::Narrow = u32::MAX;
    const RANGE: Self = u32::MAX as u64 + 1;

    fn max(self, other: Self) -> Self {
        core::cmp::max(self, other)
    }

    fn checked_div(self, rhs: Self) -> Option<Self> {
        u64::checked_div(self, rhs)
    }

    fn saturating_add(self, rhs: Self) -> Self {
        u64::saturating_add(self, rhs)
    }

    fn try_into_narrow(self) -> Option<Self::Narrow> {
        u32::try_from(self).ok()
    }

    fn float_div(numerator: Self::Float, denominator: Self::Float) -> Self::Float {
        numerator / denominator
    }
}

/// [`RatioInt`] for `f32` ratios: counts in `usize`, narrowed via `u16`.
///
/// `u16` fits losslessly into `f32`'s 24-bit mantissa, so this is the family
/// for similarity and match-fraction scores that return `f32`. Counts above
/// `u16::MAX` are scaled before narrowing — see [`unit_ratio`].
impl RatioInt for usize {
    type Narrow = u16;
    type Float = f32;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = usize::MAX;
    const NARROW_MAX: Self::Narrow = u16::MAX;
    const RANGE: Self = u16::MAX as usize + 1;

    fn max(self, other: Self) -> Self {
        core::cmp::max(self, other)
    }

    fn checked_div(self, rhs: Self) -> Option<Self> {
        usize::checked_div(self, rhs)
    }

    fn saturating_add(self, rhs: Self) -> Self {
        usize::saturating_add(self, rhs)
    }

    fn try_into_narrow(self) -> Option<Self::Narrow> {
        u16::try_from(self).ok()
    }

    fn float_div(numerator: Self::Float, denominator: Self::Float) -> Self::Float {
        numerator / denominator
    }
}

/// Ratio of two counts as a float, computed without tripping
/// `cast_precision_loss`.
///
/// Both counts are first scaled by a shared divisor and then each is narrowed
/// through the family's [`Narrow`](RatioInt::Narrow) integer (the widest
/// integer that losslessly converts to the family's [`Float`](RatioInt::Float)).
/// Scaling *both* operands by the same factor preserves the quotient, so even
/// when both counts exceed `Narrow::MAX` the ratio is exact rather than
/// collapsing to `1.0` via independent saturation.
///
/// The divisor is sized off the larger of the two counts: `count / RANGE + 1`,
/// where `RANGE` is one past `Narrow::MAX`. The `+1` keeps a count equal to
/// `Narrow::MAX` on the unscaled path (`NARROW_MAX / (NARROW_MAX + 1) == 0`,
/// so the divisor is `1`); the boundary test in this module pins that
/// invariant, which was the site of an off-by-one bug during development.
///
/// Two further edge cases are handled explicitly:
///
/// - **Zero denominator** returns `0.0` (rather than `NaN` or `inf`), matching
///   what every current caller wants for an empty set or window.
/// - **Scaled operand that floors to `0`** is clamped up to `1` — applied
///   symmetrically to numerator and denominator. For the numerator this keeps
///   a small positive value over a large one (e.g. `1 / 70_000`) from
///   collapsing to `0.0`; for the denominator it keeps a large numerator over a
///   small one (e.g. `70_000 / 1`) from producing `inf`. Either way, a positive
///   denominator guarantees a finite result.
///
/// The result is not clamped to `[0, 1]`: when the numerator exceeds the
/// denominator it rises above `1.0` (e.g. context-window overflow reports a
/// utilization greater than one).
/// Divide `value` by `scale`, keeping any positive input strictly positive.
///
/// The integer division floors, so a small positive `value` divided by a large
/// `scale` can drop to zero — collapsing a genuine operand of the ratio. This
/// clamps such a floored result back up to one so that a positive numerator or
/// denominator never vanishes (which would otherwise yield `0.0` or `inf`
/// respectively). Used symmetrically on both operands of [`unit_ratio`].
fn scale_operand<T: RatioInt>(value: T, scale: T) -> T {
    let scaled = T::checked_div(value, scale).unwrap_or(value);
    if value > T::ZERO && scaled == T::ZERO {
        T::ONE
    } else {
        scaled
    }
}

#[must_use]
pub(crate) fn unit_ratio<T: RatioInt>(numerator: T, denominator: T) -> T::Float {
    if denominator == T::ZERO {
        return T::Float::default();
    }
    let scale = T::saturating_add(
        T::checked_div(T::max(denominator, numerator), T::RANGE).unwrap_or(T::MAX),
        T::ONE,
    );
    let numerator_scaled = scale_operand(numerator, scale);
    let denominator_scaled = scale_operand(denominator, scale);
    let numerator_narrow = numerator_scaled.try_into_narrow().unwrap_or(T::NARROW_MAX);
    let denominator_narrow = denominator_scaled
        .try_into_narrow()
        .unwrap_or(T::NARROW_MAX);
    T::float_div(
        T::Float::from(numerator_narrow),
        T::Float::from(denominator_narrow),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_ratio_f64_normal_range() {
        assert!((unit_ratio::<u64>(1, 4) - 0.25).abs() < 0.001);
        assert!((unit_ratio::<u64>(3, 3) - 1.0).abs() < f64::EPSILON);
        assert!(unit_ratio::<u64>(0, 3).abs() < f64::EPSILON);
    }

    #[test]
    fn unit_ratio_f64_is_zero_for_zero_denominator() {
        assert!(unit_ratio::<u64>(5, 0).abs() < f64::EPSILON);
    }

    #[test]
    fn unit_ratio_f64_exceeds_one_for_overflow() {
        let ratio = unit_ratio::<u64>(150, 100);
        assert!(
            (ratio - 1.5).abs() < 0.001,
            "overflow must exceed 1.0, got {ratio}"
        );
    }

    #[test]
    fn unit_ratio_f64_large_partial_overlap_does_not_collapse_to_one() {
        let over = u64::from(u32::MAX) + 1;
        let ratio = unit_ratio::<u64>(over * 7 / 10, over);
        assert!(
            (ratio - 0.7).abs() < 0.01,
            "partial overlap above u32::MAX must not collapse to 1.0, got {ratio}"
        );
    }

    #[test]
    fn unit_ratio_f64_small_positive_numerator_stays_positive_at_large_scale() {
        // numerator=1 over a denominator above u32::MAX triggers shared
        // scaling; the integer division 1/scale would floor to 0, collapsing a
        // genuine match to 0.0. The result must stay strictly positive.
        let over = u64::from(u32::MAX) + 1;
        let ratio = unit_ratio::<u64>(1, over);
        assert!(
            ratio > 0.0,
            "positive numerator must yield positive ratio, got {ratio}"
        );
    }

    #[test]
    fn unit_ratio_f64_large_overflow_exceeds_one() {
        let over = u64::from(u32::MAX) + 1;
        let ratio = unit_ratio::<u64>(over + over / 2, over);
        assert!(
            ratio > 1.0,
            "overflow above u32::MAX must stay above 1.0, got {ratio}"
        );
    }

    #[test]
    fn unit_ratio_f32_identity_and_disjoint_in_normal_range() {
        assert!((unit_ratio::<usize>(3, 3) - 1.0).abs() < f32::EPSILON);
        assert!(unit_ratio::<usize>(0, 3).abs() < f32::EPSILON);
    }

    #[test]
    fn unit_ratio_f32_preserves_fraction_in_normal_range() {
        assert!((unit_ratio::<usize>(1, 4) - 0.25).abs() < 0.01);
        assert!((unit_ratio::<usize>(2, 3) - 0.66).abs() < 0.01);
    }

    #[test]
    fn unit_ratio_f32_is_zero_for_zero_denominator() {
        assert!(unit_ratio::<usize>(5, 0).abs() < f32::EPSILON);
    }

    #[test]
    fn unit_ratio_f32_large_partial_overlap_does_not_collapse_to_one() {
        let ratio = unit_ratio::<usize>(70_000, 100_000);
        assert!((ratio - 0.7_f32).abs() < 0.01, "expected ~0.7, got {ratio}");
        assert!(
            (ratio - 1.0_f32).abs() > 0.01,
            "partial overlap must not collapse to 1.0"
        );
    }

    #[test]
    fn unit_ratio_f32_small_positive_numerator_stays_positive_at_large_scale() {
        // numerator=1 over a denominator above u16::MAX triggers shared
        // scaling; the integer division 1/2 would floor to 0, collapsing a
        // genuine match to 0.0. The result must stay strictly positive.
        let ratio = unit_ratio::<usize>(1, 70_000);
        assert!(
            ratio > 0.0,
            "positive numerator must yield positive ratio, got {ratio}"
        );
    }

    #[test]
    fn unit_ratio_at_narrow_threshold_does_not_overscale() {
        // A count equal to Narrow::MAX must stay on the unscaled path: the
        // divisor is count / RANGE + 1, and RANGE == NARROW_MAX + 1 keeps it
        // at 1. A near-identical pair (differing by 1) must therefore yield a
        // result just under 1.0, not collapse to exactly 1.0 (which a wrong
        // RANGE == NARROW_MAX would cause by scaling both operands to equal
        // floored values).
        let near_max = u16::MAX as usize;
        let ratio = unit_ratio::<usize>(near_max.saturating_sub(1), near_max);
        assert!(
            ratio < 1.0,
            "near-identical counts must not collapse to 1.0 at the threshold, got {ratio}"
        );
        assert!(
            (ratio - 0.999_97_f32).abs() < 0.001,
            "expected ~0.99997, got {ratio}"
        );
    }

    #[test]
    fn unit_ratio_f32_small_positive_denominator_stays_finite_at_large_scale() {
        // A large numerator (above u16::MAX) with a small nonzero denominator
        // triggers shared scaling; the integer division den/scale would floor
        // to 0, producing inf (or NaN) without a denominator clamp. The result
        // must stay finite and reflect the true large ratio.
        let ratio = unit_ratio::<usize>(70_000, 1);
        assert!(
            ratio.is_finite(),
            "positive denominator must yield a finite ratio, got {ratio}"
        );
        assert!(ratio > 1.0, "70_000/1 must exceed 1.0, got {ratio}");
    }

    #[test]
    fn unit_ratio_f64_small_positive_denominator_stays_finite_at_large_scale() {
        let over = u64::from(u32::MAX) + 1;
        let ratio = unit_ratio::<u64>(over, 1);
        assert!(
            ratio.is_finite(),
            "positive denominator must yield a finite ratio, got {ratio}"
        );
        assert!(ratio > 1.0, "over/1 must exceed 1.0, got {ratio}");
    }
}
