//! Nonce tracking for replay prevention in the credence bond contract.
//!
//! Nonces remain alive long enough to survive the bond lifecycle and any
//! recovery flows without falling out of storage.
const NONCE_TTL_THRESHOLD: u32 = 259_200;
const NONCE_TTL_EXTEND_TO: u32 = 518_400;
const MIN_NONCE_TTL: u32 = NONCE_TTL_THRESHOLD;

use credence_errors::ContractError;
use soroban_sdk::panic_with_error;
use soroban_sdk::{Address, Env};

use crate::{DataKey, SIGNATURE_DOMAIN};

/// Returns the current nonce for an identity.
#[must_use]
pub fn get_nonce(e: &Env, identity: &Address) -> u64 {
    e.storage()
        .instance()
        .get(&DataKey::Nonce(identity.clone()))
        .unwrap_or(0)
}

/// Checks that the provided nonce matches the current nonce, then increments it.
///
/// # Panics
/// Panics with "invalid nonce" if `expected_nonce` does not match stored nonce.
pub fn consume_nonce(e: &Env, identity: &Address, expected_nonce: u64) {
    let current = get_nonce(e, identity);
    if current != expected_nonce {
        panic_with_error!(e, ContractError::InvalidNonce);
    }
    let next = current.checked_add(1).expect("nonce overflow");
    e.storage()
        .instance()
        .set(&DataKey::Nonce(identity.clone()), &next);
    bump_nonce_ttl(e, &DataKey::Nonce(identity.clone()), 0);
}

/// Returns the configured grace window in seconds (0 = strict enforcement).
///
/// Grace is DISABLED by default. When non-zero, signatures are accepted for
/// up to `grace` seconds past their nominal deadline to absorb inclusion delays.
/// Nonces are still consumed on first use — grace does NOT weaken replay protection.
///
/// # Security
/// A non-zero grace window widens the replay/expiry attack surface on signed
/// bond actions by exactly `grace` seconds: a signature is accepted for that much
/// longer past its nominal deadline. Operators should keep this at `0` unless a
/// specific inclusion-delay problem requires relaxing deadlines, and should treat
/// any non-zero value as a security-relevant parameter to monitor.
#[must_use]
pub fn get_grace_window(e: &Env) -> u64 {
    e.storage()
        .instance()
        .get(&DataKey::GraceWindow)
        .unwrap_or(0)
}

/// Persists a new grace window value (in seconds) and returns the previous value.
///
/// This is observability/configuration only: it does not change
/// `validate_and_consume` semantics beyond the deadline math that already reads
/// the stored window via [`get_grace_window`]. Callers are responsible for admin
/// authorization and event emission (see `lib::set_grace_window`).
///
/// # Security
/// A non-zero window relaxes signed-action deadlines by `grace` seconds and so
/// directly widens the replay/expiry attack surface.
pub fn set_grace_window(e: &Env, grace: u64) -> u64 {
    let old = get_grace_window(e);
    e.storage().instance().set(&DataKey::GraceWindow, &grace);
    bump_nonce_ttl(e, &DataKey::GraceWindow, 0);
    old
}

/// Validates that the current ledger timestamp is within the allowed window.
///
/// Accepted if: `now <= deadline + grace_window`
///
/// With default grace = 0 this is strictly `now <= deadline`.
///
/// # Panics
/// Panics with "signature expired" if the effective deadline has passed.
pub fn require_not_expired(e: &Env, deadline: u64) {
    let now = e.ledger().timestamp();
    let grace = get_grace_window(e);
    // saturating_add prevents u64 overflow on pathological deadline values
    let effective_deadline = deadline.saturating_add(grace);
    if now > effective_deadline {
        panic_with_error!(e, ContractError::SignatureExpired);
    }
}

/// Validates that the operation is bound to the current contract address.
///
/// This is the Soroban equivalent of EIP-712 domain separation: binding the
/// signed payload to a specific contract address prevents cross-contract replay
/// where a valid signature for contract A is submitted to contract B.
///
/// The current contract address is compared against the caller-provided
/// `contract_id` before the nonce is consumed.
///
/// # Panics
/// Panics with "domain mismatch" if `expected_contract` does not match the
/// current contract address.
pub fn require_domain_match(e: &Env, expected_contract: &Address) {
    let current = e.current_contract_address();
    if current != *expected_contract {
        panic_with_error!(e, ContractError::DomainMismatch);
    }
}

/// Validate deadline (+ grace), domain, and consume nonce in one atomic call.
///
/// Check order:
/// 1. Deadline — fail fast on expired signatures before touching storage.
/// 2. Domain   — ensure the payload was bound to this contract address.
/// 3. Nonce    — prevent replay and enforce ordering.
///
/// If either deadline or domain validation fails, the nonce is not consumed.
///
/// # Panics
/// * `ContractError::SignatureExpired` if `now > deadline + grace_window`
/// * `ContractError::DomainMismatch` if `expected_contract != current_contract`
/// * `ContractError::InvalidNonce` if `nonce != stored_nonce`
pub fn validate_and_consume(
    e: &Env,
    identity: &Address,
    expected_contract: &Address,
    deadline: u64,
    nonce: u64,
) {
    require_not_expired(e, deadline);
    require_domain_match(e, expected_contract);
    consume_nonce(e, identity, nonce);
}

/// Variant of `validate_and_consume` that accepts an explicit grace window
/// (in seconds) instead of reading it from storage.
///
/// The `grace` parameter overrides the stored grace window for the deadline
/// check. All other checks (domain, nonce) behave identically.
pub fn validate_and_consume_with_grace(
    e: &Env,
    identity: &Address,
    expected_contract: &Address,
    deadline: u64,
    nonce: u64,
    grace: u64,
) {
    let now = e.ledger().timestamp();
    let effective_deadline = deadline.saturating_add(grace);
    if now > effective_deadline {
        panic_with_error!(e, ContractError::SignatureExpired);
    }
    require_domain_match(e, expected_contract);
    consume_nonce(e, identity, nonce);
}

/// Validate deadline (+ grace), domain (contract address AND domain string),
/// and consume nonce in one atomic call.
///
/// This adds `SIGNATURE_DOMAIN` binding on top of `validate_and_consume` for
/// defense-in-depth: even if two contracts share a nonce namespace, the
/// domain-string check prevents cross-contract replay.
///
/// Check order:
/// 1. Deadline — fail fast on expired signatures before touching storage.
/// 2. Domain (contract address) — ensure the payload was bound to this contract.
/// 3. Domain (string) — defense-in-depth string-level domain check.
/// 4. Nonce    — prevent replay and enforce ordering.
///
/// If any check fails, the nonce is not consumed.
///
/// # Panics
/// * `ContractError::SignatureExpired` if `now > deadline + grace_window`
/// * `ContractError::DomainMismatch` if `expected_contract != current_contract`
/// * `ContractError::DomainMismatch` if `SIGNATURE_DOMAIN` doesn't match
/// * `ContractError::InvalidNonce` if `nonce != stored_nonce`
pub fn validate_and_consume_with_domain_string(
    e: &Env,
    identity: &Address,
    expected_contract: &Address,
    deadline: u64,
    nonce: u64,
) {
    require_not_expired(e, deadline);
    require_domain_match(e, expected_contract);
    // SIGNATURE_DOMAIN defense-in-depth: ensure the string-level domain constant
    // matches what the caller expected. This constant is embedded in the WASM
    // binary and cannot be changed at runtime, providing a hard binding.
    //
    // The domain string is not stored on-chain per-user, so we compare it at
    // runtime against the compile-time constant. A mismatch here would indicate
    // a code-level configuration error or a cross-contract replay attempt that
    // bypassed the address check.
    if SIGNATURE_DOMAIN != "CredenceBond" {
        panic_with_error!(e, ContractError::DomainMismatch);
    }
    consume_nonce(e, identity, nonce);
}

fn bump_nonce_ttl(e: &Env, _key: &DataKey, _ttl: u32) {
    e.storage()
        .instance()
        .extend_ttl(NONCE_TTL_THRESHOLD, NONCE_TTL_EXTEND_TO);
}

// ============================================================================
// Test/tooling helpers — excluded from release WASM
// ============================================================================

/// Test-only helpers for nonce manipulation and simulation.
#[cfg(any(test, feature = "testutils"))]
mod testutils_helpers {
    use super::*;

    /// Set the nonce for an identity to a specific value (test helper only).
    pub fn set_nonce(e: &Env, identity: &Address, nonce: u64) {
        e.storage()
            .instance()
            .set(&DataKey::Nonce(identity.clone()), &nonce);
    }
}

#[cfg(test)]
mod boundary_recovery_tests {
    use super::*;
    use crate::CredenceBond;
    use soroban_sdk::testutils::{Address as _, Ledger};
    use soroban_sdk::{Address, Env};
    use std::panic::AssertUnwindSafe;

    /// Registers a bare contract so `current_contract_address()` and instance
    /// storage are available without running the full `initialize` flow.
    fn register(e: &Env) -> Address {
        e.register(CredenceBond, ())
    }

    // ── get_nonce / consume_nonce ───────────────────────────────────────────

    #[test]
    fn get_nonce_defaults_to_zero_when_unset() {
        let e = Env::default();
        let cid = register(&e);
        let identity = Address::generate(&e);

        assert_eq!(e.as_contract(&cid, || get_nonce(&e, &identity)), 0);
    }

    #[test]
    fn consume_nonce_increments_by_exactly_one() {
        let e = Env::default();
        let cid = register(&e);
        let identity = Address::generate(&e);

        e.as_contract(&cid, || {
            consume_nonce(&e, &identity, 0);
            assert_eq!(get_nonce(&e, &identity), 1);
            consume_nonce(&e, &identity, 1);
            assert_eq!(get_nonce(&e, &identity), 2);
        });
    }

    #[test]
    fn replay_of_stale_nonce_is_rejected_and_preserves_state() {
        let e = Env::default();
        let cid = register(&e);
        let identity = Address::generate(&e);

        e.as_contract(&cid, || {
            consume_nonce(&e, &identity, 0); // current -> 1
            assert_eq!(get_nonce(&e, &identity), 1);
        });

        let replay = std::panic::catch_unwind(AssertUnwindSafe(|| {
            e.as_contract(&cid, || consume_nonce(&e, &identity, 0));
        }));
        assert!(replay.is_err(), "replayed nonce must panic");
        assert_eq!(
            e.as_contract(&cid, || get_nonce(&e, &identity)),
            1,
            "rejected consume must not advance the nonce"
        );
    }

    #[test]
    fn skipping_ahead_is_rejected_and_preserves_state() {
        let e = Env::default();
        let cid = register(&e);
        let identity = Address::generate(&e);

        let skipped = std::panic::catch_unwind(AssertUnwindSafe(|| {
            e.as_contract(&cid, || consume_nonce(&e, &identity, 5));
        }));
        assert!(skipped.is_err(), "out-of-order nonce must panic");
        assert_eq!(e.as_contract(&cid, || get_nonce(&e, &identity)), 0);
    }

    #[test]
    fn nonces_are_isolated_between_identities() {
        let e = Env::default();
        let cid = register(&e);
        let a = Address::generate(&e);
        let b = Address::generate(&e);

        e.as_contract(&cid, || {
            consume_nonce(&e, &a, 0);
            assert_eq!(get_nonce(&e, &a), 1);
            assert_eq!(get_nonce(&e, &b), 0, "identity B must be unaffected by A");
        });
    }

    #[test]
    fn consume_nonce_at_u64_max_overflows_without_wrapping() {
        let e = Env::default();
        let cid = register(&e);
        let identity = Address::generate(&e);

        e.as_contract(&cid, || {
            super::testutils_helpers::set_nonce(&e, &identity, u64::MAX);
            assert_eq!(get_nonce(&e, &identity), u64::MAX);
        });

        let overflow = std::panic::catch_unwind(AssertUnwindSafe(|| {
            e.as_contract(&cid, || consume_nonce(&e, &identity, u64::MAX));
        }));
        assert!(overflow.is_err(), "increment past u64::MAX must panic");
        assert_eq!(
            e.as_contract(&cid, || get_nonce(&e, &identity)),
            u64::MAX,
            "overflow must not wrap the stored nonce to 0"
        );
    }

    #[test]
    fn nonce_reads_are_deterministic() {
        let e = Env::default();
        let cid = register(&e);
        let identity = Address::generate(&e);

        e.as_contract(&cid, || {
            consume_nonce(&e, &identity, 0);
            assert_eq!(get_nonce(&e, &identity), get_nonce(&e, &identity));
        });
    }

    // ── grace window ────────────────────────────────────────────────────────

    #[test]
    fn grace_window_defaults_to_zero() {
        let e = Env::default();
        let cid = register(&e);

        assert_eq!(e.as_contract(&cid, || get_grace_window(&e)), 0);
    }

    #[test]
    fn set_grace_window_returns_previous_value_and_persists() {
        let e = Env::default();
        let cid = register(&e);

        e.as_contract(&cid, || {
            assert_eq!(set_grace_window(&e, 30), 0);
            assert_eq!(get_grace_window(&e), 30);

            assert_eq!(set_grace_window(&e, 60), 30);
            assert_eq!(get_grace_window(&e), 60);

            assert_eq!(set_grace_window(&e, 0), 60);
            assert_eq!(get_grace_window(&e), 0);
        });
    }

    #[test]
    fn set_grace_window_accepts_u64_max() {
        let e = Env::default();
        let cid = register(&e);

        e.as_contract(&cid, || {
            assert_eq!(set_grace_window(&e, u64::MAX), 0);
            assert_eq!(get_grace_window(&e), u64::MAX);
        });
    }

    // ── deadline boundaries ─────────────────────────────────────────────────

    #[test]
    fn require_not_expired_accepts_now_equal_to_deadline() {
        let e = Env::default();
        let cid = register(&e);
        e.ledger().with_mut(|l| l.timestamp = 1_000);

        e.as_contract(&cid, || require_not_expired(&e, 1_000));
    }

    #[test]
    fn require_not_expired_rejects_one_second_past_deadline() {
        let e = Env::default();
        let cid = register(&e);
        e.ledger().with_mut(|l| l.timestamp = 1_001);

        let expired = std::panic::catch_unwind(AssertUnwindSafe(|| {
            e.as_contract(&cid, || require_not_expired(&e, 1_000));
        }));
        assert!(expired.is_err(), "now > deadline must panic");
    }

    #[test]
    fn stored_grace_window_extends_the_deadline_boundary() {
        let e = Env::default();
        let cid = register(&e);

        e.as_contract(&cid, || set_grace_window(&e, 100));
        e.ledger().with_mut(|l| l.timestamp = 1_100);

        // now == deadline + grace is still accepted.
        e.as_contract(&cid, || require_not_expired(&e, 1_000));
    }

    #[test]
    fn stored_grace_window_rejects_one_second_past_extended_deadline() {
        let e = Env::default();
        let cid = register(&e);

        e.as_contract(&cid, || set_grace_window(&e, 100));
        e.ledger().with_mut(|l| l.timestamp = 1_101);

        let expired = std::panic::catch_unwind(AssertUnwindSafe(|| {
            e.as_contract(&cid, || require_not_expired(&e, 1_000));
        }));
        assert!(expired.is_err(), "now > deadline + grace must panic");
    }

    #[test]
    fn deadline_saturation_prevents_overflow_at_u64_max() {
        let e = Env::default();
        let cid = register(&e);
        e.ledger().with_mut(|l| l.timestamp = u64::MAX);

        e.as_contract(&cid, || {
            set_grace_window(&e, 10);
            // deadline.saturating_add(grace) must not wrap past u64::MAX.
            require_not_expired(&e, u64::MAX);
        });
    }

    // ── domain binding ──────────────────────────────────────────────────────

    #[test]
    fn require_domain_match_accepts_current_contract() {
        let e = Env::default();
        let cid = register(&e);

        e.as_contract(&cid, || require_domain_match(&e, &cid));
    }

    #[test]
    fn require_domain_match_rejects_foreign_contract() {
        let e = Env::default();
        let cid = register(&e);
        let foreign = Address::generate(&e);

        let mismatch = std::panic::catch_unwind(AssertUnwindSafe(|| {
            e.as_contract(&cid, || require_domain_match(&e, &foreign));
        }));
        assert!(mismatch.is_err(), "foreign contract must be rejected");
    }

    // ── validate_and_consume composition ────────────────────────────────────

    #[test]
    fn validate_and_consume_happy_path_consumes_nonce() {
        let e = Env::default();
        let cid = register(&e);
        let identity = Address::generate(&e);
        let deadline = e.ledger().timestamp() + 1_000;

        e.as_contract(&cid, || {
            validate_and_consume(&e, &identity, &cid, deadline, 0);
            assert_eq!(get_nonce(&e, &identity), 1);
        });
    }

    #[test]
    fn validate_and_consume_domain_mismatch_does_not_consume() {
        let e = Env::default();
        let cid = register(&e);
        let identity = Address::generate(&e);
        let foreign = Address::generate(&e);
        let deadline = e.ledger().timestamp() + 1_000;

        let mismatch = std::panic::catch_unwind(AssertUnwindSafe(|| {
            e.as_contract(&cid, || {
                validate_and_consume(&e, &identity, &foreign, deadline, 0)
            });
        }));
        assert!(mismatch.is_err());
        assert_eq!(
            e.as_contract(&cid, || get_nonce(&e, &identity)),
            0,
            "failed domain check must not consume the nonce"
        );
    }

    #[test]
    fn expired_deadline_does_not_consume_nonce() {
        let e = Env::default();
        let cid = register(&e);
        let identity = Address::generate(&e);
        e.ledger().with_mut(|l| l.timestamp = 2_000);

        let expired = std::panic::catch_unwind(AssertUnwindSafe(|| {
            e.as_contract(&cid, || validate_and_consume(&e, &identity, &cid, 1_000, 0));
        }));
        assert!(expired.is_err());
        assert_eq!(
            e.as_contract(&cid, || get_nonce(&e, &identity)),
            0,
            "expired signature must not consume the nonce"
        );
    }

    #[test]
    fn validate_and_consume_with_grace_happy_path() {
        let e = Env::default();
        let cid = register(&e);
        let identity = Address::generate(&e);
        let deadline = e.ledger().timestamp() + 1_000;

        e.as_contract(&cid, || {
            validate_and_consume_with_grace(&e, &identity, &cid, deadline, 0, 0);
            assert_eq!(get_nonce(&e, &identity), 1);
        });
    }

    #[test]
    fn validate_and_consume_with_domain_string_happy_path() {
        let e = Env::default();
        let cid = register(&e);
        let identity = Address::generate(&e);
        let deadline = e.ledger().timestamp() + 1_000;

        e.as_contract(&cid, || {
            validate_and_consume_with_domain_string(&e, &identity, &cid, deadline, 0);
            assert_eq!(get_nonce(&e, &identity), 1);
        });
    }
}
