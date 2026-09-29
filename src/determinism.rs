//! Injectable clock, id, and randomness seams for deterministic replay.
//!
//! While it runs, the engine reads three implicit nondeterminism sources:
//! the monotonic clock for start instants and every duration it reports,
//! UUID v4 minting for session and run identity, and a random draw for
//! retry-backoff jitter. Unpinned, two runs of the same conversation
//! produce different ids, timestamps, and jitter — harmless for live use,
//! fatal for anything that must agree byte-for-byte across runs (cassette
//! replay, recorded request comparison, diff suites). The clock trait
//! also carries a wall half; the engine never reads it — that half is the
//! adoption surface for host components that stamp `SystemTime`.
//!
//! Each source gets an object-safe trait here with two implementations: a
//! default that performs today's exact read, and a seedable pinned one a
//! host installs when it needs reproducibility. Install points sit at each
//! seam's consumer — the clock and id generator on
//! [`LoopManagers`](crate::managers::LoopManagers), because the engine mints
//! the session id inside its constructors; the rng on
//! [`StreamHandler`](crate::stream::handler::StreamHandler), because the
//! retry jitter is its only draw.
//!
//! # Example
//!
//! Two engines built from the same seeds mint the same ids, in order —
//! a fresh generator with the same seed predicts the engine's sequence:
//!
//! ```
//! use std::sync::Arc;
//! use std::time::UNIX_EPOCH;
//! use loopctl::determinism::{FixedClock, IdGen, SeededIdGen};
//! use loopctl::managers::LoopManagers;
//!
//! let managers = LoopManagers::new()
//!     .with_clock(Arc::new(FixedClock::new(UNIX_EPOCH)))
//!     .with_id_gen(Arc::new(SeededIdGen::new(42)));
//!
//! let oracle = SeededIdGen::new(42);
//! assert_eq!(managers.id_gen().next_id(), oracle.next_id());
//! ```
//!
//! Components outside the engine (memory stores, the trajectory observer,
//! provider clients) keep their own reads; the traits here are the public
//! surface such a component adopts when it needs pinned time or ids of its
//! own.

use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use uuid::Uuid;

/// The clock seam: both time halves behind one injectable source.
///
/// The engine reads the monotonic half for span measurement — run and
/// session start instants and every duration it reports — so that is the
/// half a pinned implementation must freeze for replay. The wall half
/// exists for host components that stamp `SystemTime` onto artifacts a
/// consumer may render or persist; the engine itself never reads it.
/// Implementations must be cheap to call and free of panics; the default
/// [`SystemClock`] performs the live reads, and a pinned implementation
/// (see [`FixedClock`]) freezes both halves.
pub trait Clock: Send + Sync + fmt::Debug {
    /// The current wall-clock reading.
    ///
    /// Use for timestamps that land on artifacts a host may render or
    /// persist. The default implementation returns `SystemTime::now()`.
    fn now(&self) -> SystemTime;

    /// The current monotonic reading.
    ///
    /// Use as the anchor for span measurement via
    /// [`elapsed_since`](Self::elapsed_since). Monotonic instants are only
    /// meaningful as differences — their absolute value never reaches an
    /// artifact — so a pinned clock may return any instant it captured,
    /// provided every read returns the same one.
    fn monotonic(&self) -> Instant;

    /// Duration from `start` to the current monotonic reading.
    ///
    /// Provided so callers never compute `start.elapsed()` on a
    /// clock-sourced instant: `elapsed()` always reads the real monotonic
    /// clock and would bypass the seam. The default subtracts with
    /// saturation, so a start after the current reading reports a zero
    /// duration rather than panicking — which is exactly what a frozen
    /// clock's callers see.
    fn elapsed_since(&self, start: Instant) -> Duration {
        self.monotonic().saturating_duration_since(start)
    }
}

/// The default [`Clock`]: the live system reads.
///
/// `now()` returns `SystemTime::now()` and `monotonic()` returns
/// `Instant::now()`, so durations measured through it advance in real time.
/// This is what an unpinned engine behaves identically to.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }

    fn monotonic(&self) -> Instant {
        Instant::now()
    }
}

/// A pinned [`Clock`]: both halves frozen for replay.
///
/// The wall half is the constructor argument, so timestamps rendered from
/// it are exactly reproducible. The monotonic half is a real instant
/// captured at construction — the type is opaque, so no synthetic instant
/// exists — but only differences between monotonic reads ever reach an
/// artifact, and every read returns the same captured instant, so every
/// duration measured through [`elapsed_since`](Clock::elapsed_since) is
/// zero.
///
/// # Example
///
/// ```
/// use std::time::{Duration, UNIX_EPOCH};
/// use loopctl::determinism::{Clock, FixedClock};
///
/// let clock = FixedClock::new(UNIX_EPOCH);
/// assert_eq!(clock.now(), UNIX_EPOCH);
/// assert_eq!(clock.elapsed_since(clock.monotonic()), Duration::ZERO);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FixedClock {
    /// The frozen wall-clock reading.
    ///
    /// Returned verbatim by every `now()` call; constructed by the caller,
    /// so replayed timestamps are exactly the recorded ones.
    wall: SystemTime,

    /// The frozen monotonic anchor.
    ///
    /// A real instant captured at construction; every `monotonic()` call
    /// returns it, so every measured span collapses to zero. Its absolute
    /// value never reaches an artifact.
    mono: Instant,
}

impl FixedClock {
    /// Freeze both halves at `wall`.
    ///
    /// The monotonic half captures a real instant at construction time; the
    /// wall half is exactly `wall`, so this is the value replayed artifacts
    /// carry.
    #[must_use]
    pub fn new(wall: SystemTime) -> Self {
        Self {
            wall,
            mono: Instant::now(),
        }
    }
}

impl Clock for FixedClock {
    fn now(&self) -> SystemTime {
        self.wall
    }

    fn monotonic(&self) -> Instant {
        self.mono
    }
}

/// The id seam: session and run identity behind one injectable source.
///
/// The engine mints a session id in its constructors and a run id at every
/// `run()` call. The default [`UuidIdGen`] mints fresh UUID v4 values —
/// today's behavior — while a seeded implementation replays a fixed
/// sequence, letting two engines built from the same seed agree on every id
/// they emit.
pub trait IdGen: Send + Sync + fmt::Debug {
    /// Mint the next identifier.
    ///
    /// Implementations must produce distinct values across successive calls
    /// on the same generator (a pinned generator replays a *sequence*, not
    /// one constant), and should shape them as UUID v4 so every consumer of
    /// today's ids — serde, cassettes, session tags — keeps parsing them.
    fn next_id(&self) -> Uuid;
}

/// The default [`IdGen`]: fresh UUID v4 values.
///
/// Delegates to `Uuid::new_v4()` — exactly what the engine minted before
/// the seam existed, so an unpinned engine is byte-identical to the old
/// behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct UuidIdGen;

impl IdGen for UuidIdGen {
    fn next_id(&self) -> Uuid {
        Uuid::new_v4()
    }
}

/// A pinned [`IdGen`]: a deterministic v4-shaped sequence from one seed.
///
/// Two generators built from the same seed produce the same id sequence,
/// element for element, and successive ids within one generator differ — a
/// replay, not a constant. The state is the seed alone (exposed by
/// [`seed`](Self::seed)), so a recorded run's ids are reproducible from the
/// seed that started it; ids carry the v4 version and variant bits, keeping
/// them parse-compatible with everything that handles minted v4 ids.
#[derive(Debug)]
pub struct SeededIdGen {
    /// The seed this generator replays from.
    ///
    /// The generator's entire serializable state: constructing a fresh
    /// generator with this seed reproduces the id sequence from its start.
    seed: u64,

    /// The running xorshift state.
    ///
    /// Advanced by every draw under a mutex; single-operation data, so a
    /// poisoned lock recovers rather than propagates (the state is still a
    /// valid xorshift word whatever the interleaving was).
    state: Mutex<u64>,
}

impl SeededIdGen {
    /// Start a generator that replays from `seed`.
    ///
    /// Any seed value works, including zero — the seed is mixed before use,
    /// so the zero seed still yields a full-period stream rather than a
    /// degenerate one.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            state: Mutex::new(mix_seed(seed)),
        }
    }

    /// The seed this generator replays from.
    ///
    /// The generator's serializable state: `SeededIdGen::new(gen.seed())`
    /// rebuilds a generator whose next ids match a fresh replay of the
    /// original from its start.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl IdGen for SeededIdGen {
    fn next_id(&self) -> Uuid {
        let mut state = crate::error::recover_guard(self.state.lock());
        let msb = shape_version_bits(xorshift_next(&mut state));
        let lsb = shape_variant_bits(xorshift_next(&mut state));
        Uuid::from_u64_pair(msb, lsb)
    }
}

/// The rng seam: the engine's random draws behind one injectable source.
///
/// The only engine-adjacent draw is the retry-backoff jitter, so this trait
/// is minimal — one uniform `f64`. The default [`ThreadLocalRng`] delegates
/// to `fastrand`'s thread-local generator (the exact draw the jitter used
/// before the seam existed); a seeded implementation replays a fixed
/// sequence.
pub trait Rng: Send + Sync + fmt::Debug {
    /// Draw the next uniform value in `[0, 1)`.
    ///
    /// Implementations must return values in the half-open unit interval
    /// and produce distinct values across successive calls on the same
    /// generator.
    fn next_f64(&self) -> f64;
}

/// The default [`Rng`]: `fastrand`'s thread-local generator.
///
/// Delegates to `fastrand::f64()` — the exact draw the retry jitter made
/// before the seam existed, so an unpinned handler backs off identically to
/// the old behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThreadLocalRng;

impl Rng for ThreadLocalRng {
    fn next_f64(&self) -> f64 {
        fastrand::f64()
    }
}

/// A pinned [`Rng`]: a deterministic draw sequence from one seed.
///
/// Wraps a `fastrand` generator seeded once at construction; two rngs built
/// from the same seed produce the same draw sequence, element for element,
/// and successive draws within one rng differ. The state is the seed alone
/// (exposed by [`seed`](Self::seed)).
#[derive(Debug)]
pub struct SeededRng {
    /// The seed this rng replays from.
    ///
    /// The rng's entire serializable state: a fresh rng built from it
    /// reproduces the draw sequence from its start.
    seed: u64,

    /// The underlying seeded generator.
    ///
    /// Advanced by every draw under a mutex; single-operation data, so a
    /// poisoned lock recovers rather than propagates.
    rng: Mutex<fastrand::Rng>,
}

impl SeededRng {
    /// Start an rng that replays from `seed`.
    ///
    /// Any seed value works, including zero — `fastrand` mixes the seed
    /// internally, so the zero seed still yields a working stream.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            rng: Mutex::new(fastrand::Rng::with_seed(seed)),
        }
    }

    /// The seed this rng replays from.
    ///
    /// The rng's serializable state: `SeededRng::new(rng.seed())` rebuilds
    /// an rng whose next draws match a fresh replay of the original from
    /// its start.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl Rng for SeededRng {
    fn next_f64(&self) -> f64 {
        let mut rng = crate::error::recover_guard(self.rng.lock());
        rng.f64()
    }
}

/// Mix a caller seed into a nonzero xorshift state.
///
/// `SplitMix64`'s finalizer: any input, including zero, maps to a
/// well-distributed word, with a nonzero fallback for the one input that
/// could map to zero — a zero xorshift state never advances, and the zero
/// seed must still work.
fn mix_seed(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let mixed = z ^ (z >> 31);
    if mixed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        mixed
    }
}

/// Advance a xorshift64* state and return the next draw.
///
/// Mutates `state` in place; the multiply-free shift triple keeps the state
/// moving and the final multiply spreads the high bits, so consecutive
/// draws are uncorrelated for id- and jitter-scale use.
fn xorshift_next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Stamp the UUID v4 version bits onto the most-significant half.
///
/// The version nibble is bits 12..16 of the high u64 (the high nibble of
/// the uuid's sixth byte); setting it to `0x4` keeps seeded ids
/// parse-compatible with minted v4 ids.
fn shape_version_bits(msb: u64) -> u64 {
    (msb & !(0xF_u64 << 12)) | (0x4_u64 << 12)
}

/// Stamp the RFC 4122 variant bits onto the least-significant half.
///
/// The variant field is the top two bits of the low u64's most significant
/// byte (bits 62..64); setting them to `0b10` marks the id as an RFC 4122
/// variant, as every minted v4 id carries.
fn shape_variant_bits(lsb: u64) -> u64 {
    (lsb & !(0b11_u64 << 62)) | (0b10_u64 << 62)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn seeded_id_gen_replays_the_same_sequence_from_a_seed() {
        let gen_a = SeededIdGen::new(0x00C0_FFEE);
        let gen_b = SeededIdGen::new(0x00C0_FFEE);
        let left: Vec<Uuid> = (0..8).map(|_| gen_a.next_id()).collect();
        let right: Vec<Uuid> = (0..8).map(|_| gen_b.next_id()).collect();
        assert_eq!(
            left, right,
            "two generators from one seed must replay the identical sequence"
        );
        let distinct: HashSet<&Uuid> = left.iter().collect();
        assert_eq!(
            distinct.len(),
            left.len(),
            "successive ids within one stream differ: {left:?}"
        );
        assert!(
            left.iter().all(|id| id.get_version_num() == 4),
            "seeded ids carry the v4 version bits: {left:?}"
        );

        let zero_seed = SeededIdGen::new(0);
        let first = zero_seed.next_id();
        let second = zero_seed.next_id();
        assert_ne!(
            first, second,
            "seed 0 must yield a working stream, not a degenerate one"
        );
        assert_eq!(
            first.get_version_num(),
            4,
            "the zero seed's ids are v4-shaped too"
        );
    }

    #[test]
    fn seeded_rng_replays_the_same_draws_from_a_seed() {
        let rng_a = SeededRng::new(7);
        let rng_b = SeededRng::new(7);
        let left: Vec<f64> = (0..8).map(|_| rng_a.next_f64()).collect();
        let right: Vec<f64> = (0..8).map(|_| rng_b.next_f64()).collect();
        assert_eq!(
            left, right,
            "two rngs from one seed must replay the identical draws"
        );
        assert!(
            left.iter().all(|v| (0.0..1.0).contains(v)),
            "draws are uniform in [0,1): {left:?}"
        );
        assert_ne!(
            left[0].to_bits(),
            left[1].to_bits(),
            "successive draws differ: {left:?}"
        );

        let zero_seed = SeededRng::new(0);
        let first_draw = zero_seed.next_f64();
        let second_draw = zero_seed.next_f64();
        assert_ne!(
            first_draw.to_bits(),
            second_draw.to_bits(),
            "seed 0 is a usable stream"
        );
    }

    #[test]
    fn fixed_clock_freezes_both_halves_and_elapses_zero() {
        let wall = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let clock = FixedClock::new(wall);
        assert_eq!(
            clock.now(),
            wall,
            "the wall half replays the constructor argument"
        );
        assert_eq!(clock.now(), clock.now(), "the wall half is frozen");
        let first = clock.monotonic();
        let second = clock.monotonic();
        assert_eq!(first, second, "the monotonic half is frozen");
        assert_eq!(
            clock.elapsed_since(first),
            Duration::ZERO,
            "a frozen clock elapses nothing"
        );
        assert_eq!(
            clock.elapsed_since(Instant::now()),
            Duration::ZERO,
            "a start after the frozen anchor saturates to zero, never panics"
        );
    }

    #[test]
    fn system_clock_advances_and_elapses_forward() {
        let clock = SystemClock;
        let first = clock.monotonic();
        let second = clock.monotonic();
        assert!(second >= first, "monotonic reads never go backwards");
        assert!(
            clock.now() >= SystemTime::UNIX_EPOCH,
            "wall reads are at or after the epoch"
        );
        assert!(
            clock.elapsed_since(first) <= Duration::from_secs(1),
            "elapsed from a just-captured instant is a tiny real span"
        );
    }
}
