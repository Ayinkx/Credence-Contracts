use soroban_sdk::{contracttype, Address, Env, Vec};

/// Instance-storage keys owned by this module.
///
/// This enum shares a Rust name (`DataKey`) with the crate-root enum in
/// `lib.rs`, which also writes to instance storage. Soroban keys
/// `#[contracttype]` enums by variant name + field shape, not by Rust type
/// name, so a future unit variant added to `lib.rs::DataKey` with the exact
/// name `AcceptedTokens` would silently alias this key. See
/// `docs/STORAGE_KEY_LAYOUT.md` for the full collision-safety rules before
/// adding variants to either enum.
#[contracttype]
pub enum DataKey {
    AcceptedTokens,
}

pub fn get_accepted_tokens(e: &Env) -> Vec<Address> {
    e.storage()
        .instance()
        .get(&DataKey::AcceptedTokens)
        .unwrap_or_else(|| Vec::new(e))
}

pub fn set_accepted_tokens(e: &Env, tokens: &Vec<Address>) {
    e.storage().instance().set(&DataKey::AcceptedTokens, tokens);
}

pub fn is_token_accepted(e: &Env, token: &Address) -> bool {
    let accepted = get_accepted_tokens(e);
    accepted.iter().any(|t| t == *token)
}

pub fn is_locked(e: &Env) -> bool {
    e.storage()
        .instance()
        .get(&crate::DataKey::SettlingFlag)
        .unwrap_or(false)
}

pub fn set_lock(e: &Env, value: bool) {
    e.storage()
        .instance()
        .set(&crate::DataKey::SettlingFlag, &value);
}

pub fn get_admin(e: &Env) -> Option<Address> {
    e.storage()
        .instance()
        .get(&crate::DataKey::Admin)
}

#[cfg(test)]
mod boundary_recovery_tests {
    use super::*;
    use crate::CredenceBond;
    use soroban_sdk::testutils::Address as _;
    use soroban_sdk::{Address, Env, Vec};

    /// Registers a bare contract so instance storage can be exercised without
    /// running the full `initialize` flow.
    fn register(e: &Env) -> Address {
        e.register(CredenceBond, ())
    }

    // ── get/set_accepted_tokens ─────────────────────────────────────────────

    #[test]
    fn accepted_tokens_default_to_empty_when_unset() {
        let e = Env::default();
        let cid = register(&e);

        let tokens = e.as_contract(&cid, || get_accepted_tokens(&e));
        assert_eq!(
            tokens.len(),
            0,
            "unset key must read back as empty, not panic"
        );
    }

    #[test]
    fn accepted_tokens_round_trip_preserves_order_and_duplicates() {
        let e = Env::default();
        let cid = register(&e);
        let a = Address::generate(&e);
        let b = Address::generate(&e);

        e.as_contract(&cid, || {
            let mut input = Vec::new(&e);
            input.push_back(a.clone());
            input.push_back(b.clone());
            input.push_back(a.clone());
            set_accepted_tokens(&e, &input);

            let stored = get_accepted_tokens(&e);
            assert_eq!(stored.len(), 3);
            assert_eq!(stored.get(0).unwrap(), a);
            assert_eq!(stored.get(1).unwrap(), b);
            assert_eq!(stored.get(2).unwrap(), a);
        });
    }

    #[test]
    fn setting_accepted_tokens_to_empty_clears_previous_state() {
        let e = Env::default();
        let cid = register(&e);
        let token = Address::generate(&e);

        e.as_contract(&cid, || {
            let mut input = Vec::new(&e);
            input.push_back(token.clone());
            set_accepted_tokens(&e, &input);
            assert_eq!(get_accepted_tokens(&e).len(), 1);

            set_accepted_tokens(&e, &Vec::new(&e));
            assert_eq!(
                get_accepted_tokens(&e).len(),
                0,
                "empty write must fully clear"
            );
            assert!(!is_token_accepted(&e, &token));
        });
    }

    #[test]
    fn accepted_tokens_reads_are_deterministic() {
        let e = Env::default();
        let cid = register(&e);
        let token = Address::generate(&e);

        e.as_contract(&cid, || {
            let mut input = Vec::new(&e);
            input.push_back(token.clone());
            set_accepted_tokens(&e, &input);

            assert_eq!(get_accepted_tokens(&e), get_accepted_tokens(&e));
        });
    }

    // ── is_token_accepted ───────────────────────────────────────────────────

    #[test]
    fn is_token_accepted_is_false_when_unset() {
        let e = Env::default();
        let cid = register(&e);
        let token = Address::generate(&e);

        assert!(!e.as_contract(&cid, || is_token_accepted(&e, &token)));
    }

    #[test]
    fn is_token_accepted_matches_exact_membership() {
        let e = Env::default();
        let cid = register(&e);
        let allowed = Address::generate(&e);
        let other = Address::generate(&e);

        e.as_contract(&cid, || {
            let mut input = Vec::new(&e);
            input.push_back(allowed.clone());
            set_accepted_tokens(&e, &input);

            assert!(is_token_accepted(&e, &allowed));
            assert!(
                !is_token_accepted(&e, &other),
                "non-member must be rejected"
            );
        });
    }

    #[test]
    fn is_token_accepted_handles_duplicate_entries() {
        let e = Env::default();
        let cid = register(&e);
        let token = Address::generate(&e);

        e.as_contract(&cid, || {
            let mut input = Vec::new(&e);
            input.push_back(token.clone());
            input.push_back(token.clone());
            set_accepted_tokens(&e, &input);

            assert!(is_token_accepted(&e, &token));
        });
    }

    // ── is_locked / set_lock ────────────────────────────────────────────────

    #[test]
    fn is_locked_defaults_false() {
        let e = Env::default();
        let cid = register(&e);

        assert!(!e.as_contract(&cid, || is_locked(&e)));
    }

    #[test]
    fn set_lock_round_trips_true_then_false() {
        let e = Env::default();
        let cid = register(&e);

        e.as_contract(&cid, || {
            set_lock(&e, true);
            assert!(is_locked(&e), "lock must engage");

            set_lock(&e, false);
            assert!(!is_locked(&e), "lock must release (recovery path)");
        });
    }

    // ── get_admin ───────────────────────────────────────────────────────────

    #[test]
    fn get_admin_returns_none_when_unset() {
        let e = Env::default();
        let cid = register(&e);

        assert!(e.as_contract(&cid, || get_admin(&e)).is_none());
    }

    #[test]
    fn get_admin_returns_stored_admin() {
        let e = Env::default();
        let cid = register(&e);
        let admin = Address::generate(&e);

        e.as_contract(&cid, || {
            e.storage().instance().set(&crate::DataKey::Admin, &admin);
            assert_eq!(get_admin(&e), Some(admin));
        });
    }

    #[test]
    fn get_admin_reflects_last_write() {
        let e = Env::default();
        let cid = register(&e);
        let first = Address::generate(&e);
        let second = Address::generate(&e);

        e.as_contract(&cid, || {
            e.storage().instance().set(&crate::DataKey::Admin, &first);
            e.storage().instance().set(&crate::DataKey::Admin, &second);
            assert_eq!(get_admin(&e), Some(second), "later admin write must win");
        });
    }

    // ── cross-key isolation (regression) ────────────────────────────────────

    #[test]
    fn lock_and_accepted_tokens_do_not_interfere() {
        let e = Env::default();
        let cid = register(&e);
        let token = Address::generate(&e);

        e.as_contract(&cid, || {
            let mut input = Vec::new(&e);
            input.push_back(token.clone());
            set_accepted_tokens(&e, &input);
            set_lock(&e, true);

            assert!(is_token_accepted(&e, &token));
            assert!(is_locked(&e));
            assert_eq!(get_accepted_tokens(&e).len(), 1);

            set_lock(&e, false);
            assert!(
                is_token_accepted(&e, &token),
                "lock writes must not touch token list"
            );
        });
    }

    #[test]
    fn admin_and_accepted_tokens_do_not_interfere() {
        let e = Env::default();
        let cid = register(&e);
        let admin = Address::generate(&e);
        let token = Address::generate(&e);

        e.as_contract(&cid, || {
            e.storage().instance().set(&crate::DataKey::Admin, &admin);
            let mut input = Vec::new(&e);
            input.push_back(token.clone());
            set_accepted_tokens(&e, &input);

            assert_eq!(get_admin(&e), Some(admin));
            assert!(is_token_accepted(&e, &token));
        });
    }
}
