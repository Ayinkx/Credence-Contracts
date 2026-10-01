#![no_std]
#![deny(clippy::float_arithmetic)]
#![allow(
    deprecated,
    unused_imports,
    unused_variables,
    dead_code,
    unused_assignments,
    unused_mut,
    mismatched_lifetime_syntaxes,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::restriction
)]
// Must come AFTER `#![allow(clippy::restriction, ...)]` above: the
// `clippy::disallowed_macros` lint belongs to the `restriction` group, so
// a later allow would re-silence it. cargo build --release / WASM build
// is the only mode where this deny fires (tests
// stay free to use format!/write! for diagnostics).
#![cfg_attr(not(test), deny(clippy::disallowed_macros))]

//! # Serialization and retry contract
//!
//! Soroban executes every invocation against a consistent ledger snapshot and
//! commits its storage changes atomically: a call that panics writes nothing,
//! and invocations that touch the same contract state are serialised by the
//! ledger in sequence order. This module makes that model explicit for the
//! privileged configuration, pause, and ownership entrypoints:
//!
//! * **Serialization** — all state transitions in this contract are
//!   read-modify-write transactions. The ledger serialises concurrent requests
//!   to the same contract, so two conflicting requests cannot interleave
//!   half-way through a mutation.
//! * **Conflict detection** — every committed privileged mutation advances the
//!   monotonic [`DataKey::ConfigEpoch`] counter exactly once
//!   ([`AdminContract::get_config_epoch`]). A client that read governance
//!   state at epoch *N* can detect that a concurrent request committed at
//!   epoch *N+1* and retry against fresh state.
//! * **Atomicity** — rejected, stale, repeated, and failed operations never
//!   advance the epoch and never leave partial state behind; a panic rolls the
//!   entire invocation back.
//! * **Idempotency** — repeated operations that would not change state (same
//!   role update, duplicate pause/unpause, duplicate proposal approval) are
//!   no-ops: they mutate nothing, emit no events, and do not advance the
//!   epoch.
//!
//! ## Client retry contract
//!
//! 1. Read the current epoch and the governance state you depend on.
//! 2. Submit the privileged request.
//! 3. If the request is rejected (auth, paused, stale epoch, insufficient
//!    approvals, …) or the epoch has advanced since your read, re-read the
//!    state and retry with the fresh snapshot. Because failed calls roll back
//!    atomically, retrying can never double-apply a partial change.

pub mod pausable;

#[cfg(test)]
mod test_events_schema;
#[cfg(test)]
mod test_ownership_transfer;
#[cfg(test)]
mod test_execute_pause_proposal_enhanced;

use credence_errors::{ContractError, Role};
use soroban_sdk::panic_with_error;
use soroban_sdk::{
    contract, contractimpl, contracttype, Address, Env, IntoVal, String, Symbol, Vec,
};

/// Signature domain identifier for the Admin contract.
///
/// This constant binds signatures to this specific contract, preventing
/// cross-contract replay attacks where a signature intended for one contract
/// could be replayed against another. Each contract in the Credence system
/// has a unique signature domain constant.
///
/// # Security
///
/// Without domain separation, a signature created for contract A could be
/// replayed against contract B if both contracts share the same nonce namespace
/// and signature verification logic. By including this domain in the signed
/// payload hash, we ensure signatures are only valid for their intended contract.
///
/// # Value
///
/// The domain is a human-readable string that uniquely identifies this contract
/// within the Credence system. It should be included in the signed payload hash
/// along with other payload fields (nonce, deadline, etc.).
#[allow(dead_code)]
const SIGNATURE_DOMAIN: &str = "Admin";

/// Admin role hierarchy levels
#[contracttype]
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Copy)]
pub enum AdminRole {
    /// Can perform all operations including managing other admins
    SuperAdmin = 3,
    /// Can manage operators and perform most administrative tasks
    Admin = 2,
    /// Can perform limited operational tasks
    Operator = 1,
}

/// Admin role information
#[contracttype]
#[derive(Clone, Debug)]
pub struct AdminInfo {
    /// The admin address
    pub address: Address,
    /// The assigned role
    pub role: AdminRole,
    /// Timestamp when this role was assigned
    pub assigned_at: u64,
    /// Address of the admin who assigned this role
    pub assigned_by: Address,
    /// Whether this admin is currently active
    pub active: bool,
    /// Unix timestamp until which this admin is suspended (0 = not suspended).
    /// While `e.ledger().timestamp() < suspended_until` the admin is treated
    /// as inactive; the suspension expires automatically — no second transaction
    /// is required.
    pub suspended_until: u64,
}

/// Storage keys for the admin contract
#[contracttype]
#[derive(Clone)]
pub enum DataKey {
    /// List of all admin addresses
    AdminList,
    /// Admin information by address: Address -> AdminInfo
    AdminInfo(Address),
    /// Role-based admin lists: AdminRole -> Vec<Address>
    RoleAdmins(AdminRole),
    /// Contract initialization flag
    Initialized,
    /// Minimum number of admins required
    MinAdmins,
    /// Maximum number of admins allowed
    MaxAdmins,
    // Pause mechanism
    Paused,
    PauseSigner(Address),
    PauseSignerCount,
    PauseThreshold,
    PauseProposalCounter,
    PauseProposal(u64),
    PauseApproval(u64, Address),
    PauseApprovalCount(u64),
    /// Current contract owner
    Owner,
    /// Pending owner for two-step ownership transfer
    PendingOwner,
    /// Timestamp (ledger seconds) when the current ownership transfer was proposed.
    /// Used to enforce a timelock delay before acceptance.
    TransferProposedAt,
    /// Monotonic configuration epoch — incremented once per committed privileged
    /// mutation so clients can detect concurrent conflicts and retry.
    ConfigEpoch,
}

/// Minimum delay (in ledger seconds) between `transfer_ownership` and `accept_ownership`.
/// 86_400 seconds ≈ 24 hours at the one-second-per-ledger cadence.
const OWNERSHIP_TRANSFER_TIMELOCK: u64 = 86_400;

/// The zero/invalid address sentinel.
///
/// In Soroban the all-zero Ed25519 public key encodes to this strkey.
/// Assigning a governance role to (or transferring ownership to) this
/// address can permanently strand administration, so every privileged
/// entrypoint that accepts a target `Address` MUST reject it.
const INVALID_ADDRESS_SENTINEL: &str = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF";

/// Hard cap on the page size accepted by paginated admin reads.
///
/// A caller requesting a larger `limit` is silently clamped to this value so a
/// single read can never exceed the gas/read budget regardless of the argument.
const MAX_PAGE_LIMIT: u32 = 200;

const STORAGE_TTL_EXTEND_TO: u32 = 31_536_000;

fn bump_instance_ttl(e: &Env) {
    e.storage()
        .instance()
        .extend_ttl(STORAGE_TTL_EXTEND_TO / 2, STORAGE_TTL_EXTEND_TO);
}

/// Increment the monotonic configuration epoch.
///
/// This is the conflict-detection counter of the admin contract's
/// serialization contract: it is advanced exactly once per *committed*
/// privileged mutation and is never advanced by a rejected, repeated (no-op),
/// or failed operation. Clients can use [`AdminContract::get_config_epoch`] to
/// detect that their snapshot of governance state is stale and retry their
/// flow against the latest state.
fn bump_config_epoch(e: &Env) {
    let current: u64 = e
        .storage()
        .instance()
        .get(&DataKey::ConfigEpoch)
        .unwrap_or(0);
    let next = current
        .checked_add(1)
        .unwrap_or_else(|| panic_with_error!(e, ContractError::Overflow));
    e.storage().instance().set(&DataKey::ConfigEpoch, &next);
}

#[contract]
pub struct AdminContract;

#[contractimpl]
impl AdminContract {
    /// Return the contract version.
    pub fn version(e: Env) -> String {
        String::from_str(&e, credence_errors::VERSION)
    }

/// Return whether `address` is currently an active admin.
    ///
    /// # Determinism and failure boundaries
    ///
    /// This is a pure read: it never mutates storage, never advances
    /// [`DataKey::ConfigEpoch`], and never emits events. Given the same ledger
    /// snapshot it always returns the same value, so it is safe to call from
    /// other contracts and from off-chain simulations.
    ///
    /// An address is considered an admin if and only if **all** of the
    /// following hold:
    ///
    /// 1. An [`AdminInfo`] record exists for the address.
    /// 2. The record's `active` flag is `true`.
    /// 3. The record is not currently suspended, i.e.
    ///    `suspended_until == 0 || e.ledger().timestamp() >= suspended_until`.
    ///
    /// Suspension expires automatically once the ledger timestamp reaches
    /// `suspended_until`, so no second transaction is required to restore
    /// admin status.
    ///
    /// # Boundary cases
    ///
    /// * Uninitialized contract — returns `false` (no panic, no partial read).
    /// * Unknown address — returns `false`.
    /// * Deactivated admin — returns `false`.
    /// * Suspended admin — returns `false` until the suspension expires.
    /// * Suspension boundary — at exactly `suspended_until` the admin is
    ///   active again (`>=` comparison).
    ///
    /// # Security
    ///
    /// This function performs no authorization check and exposes no sensitive
    /// data: it only reveals whether a public address currently holds admin
    /// privileges, which is already observable through privileged entrypoints.
    pub fn is_admin(e: Env, address: Address) -> bool {
        let info: Option<AdminInfo> = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(address));

        match info {
            None => false,
            Some(info) => {
                if !info.active {
                    return false;
                }
                if info.suspended_until != 0
                    && e.ledger().timestamp() < info.suspended_until
                {
                    return false;
                }
                true
            }
        }
    }

    /// Deterministically check whether `address` holds `role` as of the current
    /// ledger snapshot.
    ///
    /// This is the read-only authorization probe used by callers that need to
    /// decide whether to attempt a privileged mutation. It is intentionally
    /// side-effect free: it never mutates storage, never advances
    /// [`DataKey::ConfigEpoch`], and never emits events, so it is safe to call
    /// from a simulation or a retry loop.
    ///
    /// # Determinism and failure boundaries
    ///
    /// The result is a pure function of the committed ledger state:
    ///
    /// * **Uninitialized contract** — returns `false` rather than panicking, so
    ///   a caller probing a not-yet-deployed configuration gets a stable
    ///   negative answer instead of an opaque trap.
    /// * **Unknown address** — returns `false`; no entry is ever created.
    /// * **Inactive or suspended admin** — returns `false` while
    ///   `e.ledger().timestamp() < suspended_until`, and `true` again once the
    ///   suspension expires, with no second transaction required.
    /// * **Role mismatch** — returns `false`; roles are compared by exact
    ///   equality, not by hierarchy, so an `Operator` does not satisfy a
    ///   `SuperAdmin` probe.
    /// * **Invalid sentinel address** — returns `false`; the zero address can
    ///   never hold a role.
    ///
    /// Because the function only reads, concurrent invocations cannot observe
    /// a torn state: Soroban executes each invocation against a consistent
    /// ledger snapshot and serialises conflicting writes, so a caller that
    /// reads `true` here and then submits a mutation will either succeed or be
    /// rejected atomically (and can retry against a fresh snapshot).
    ///
    /// # Arguments
    /// * `address` - The address whose role membership is being probed.
    /// * `role` - The exact role to test for.
    ///
    /// # Returns
    /// `true` iff `address` is an active, non-suspended admin holding exactly
    /// `role` in the current ledger snapshot.
    pub fn check_role_at_ledger(e: Env, address: Address, role: AdminRole) -> bool {
        bump_instance_ttl(&e);

        // An uninitialized contract has no governance state; report a stable
        // negative instead of trapping so probes are deterministic.
        if !e.storage().instance().has(&DataKey::Initialized) {
            return false;
        }

        // The zero/invalid sentinel can never hold a governance role.
        if address.to_string() == String::from_str(&e, INVALID_ADDRESS_SENTINEL) {
            return false;
        }

        let info: AdminInfo = match e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(address.clone()))
        {
            Some(info) => info,
            None => return false,
        };

        // Suspension is time-bounded and expires automatically.
        if info.suspended_until != 0 && e.ledger().timestamp() < info.suspended_until {
            return false;
        }

        info.active && info.role == role
    }
    /// Initialize the admin contract with a super admin.
    ///
    /// # Arguments
    /// * `super_admin` - Address that will have super admin privileges
    /// * `min_admins` - Minimum number of admins required (default: 1)
    /// * `max_admins` - Maximum number of admins allowed (default: 100)
    ///
    /// # Panics
    /// * If contract is already initialized
    /// * If min_admins is 0 or greater than max_admins
    ///
    /// # Events
    /// Emits `admin_initialized` with the super admin address
    pub fn initialize(e: Env, super_admin: Address, min_admins: u32, max_admins: u32) {
        bump_instance_ttl(&e);
        credence_errors::require_contract_uninitialized(
            &e,
            e.storage().instance().has(&DataKey::Initialized),
        );

        if min_admins == 0 {
            panic_with_error!(&e, ContractError::InvalidPauseAction);
        }

        if min_admins > max_admins {
            panic_with_error!(&e, ContractError::InvalidPauseAction);
        }

        super_admin
            .require_auth_for_args((super_admin.clone(), min_admins, max_admins).into_val(&e));

        // Set configuration
        e.storage().instance().set(&DataKey::Initialized, &true);
        e.storage().instance().set(&DataKey::MinAdmins, &min_admins);
        e.storage().instance().set(&DataKey::MaxAdmins, &max_admins);

        // Initialize pause state
        e.storage().instance().set(&DataKey::Paused, &false);
        e.storage()
            .instance()
            .set(&DataKey::PauseSignerCount, &0_u32);
        e.storage().instance().set(&DataKey::PauseThreshold, &0_u32);
        e.storage()
            .instance()
            .set(&DataKey::PauseProposalCounter, &0_u64);

        // Create initial super admin
        let admin_info = AdminInfo {
            address: super_admin.clone(),
            role: AdminRole::SuperAdmin,
            assigned_at: e.ledger().timestamp(),
            assigned_by: super_admin.clone(), // Self-assigned for initialization
            active: true,
            suspended_until: 0,
        };

        // Store admin info
        e.storage()
            .instance()
            .set(&DataKey::AdminInfo(super_admin.clone()), &admin_info);

        // Initialize admin list
        let mut admin_list: Vec<Address> = Vec::new(&e);
        admin_list.push_back(super_admin.clone());
        e.storage().instance().set(&DataKey::AdminList, &admin_list);

        // Initialize role-based admin list
        let super_admins = Vec::from_array(&e, [super_admin.clone()]);
        e.storage()
            .instance()
            .set(&DataKey::RoleAdmins(AdminRole::SuperAdmin), &super_admins);

        // Initialize empty lists for other roles
        e.storage().instance().set(
            &DataKey::RoleAdmins(AdminRole::Admin),
            &Vec::<Address>::new(&e),
        );
        e.storage().instance().set(
            &DataKey::RoleAdmins(AdminRole::Operator),
            &Vec::<Address>::new(&e),
        );

        // Set the initial owner as the super admin
        e.storage().instance().set(&DataKey::Owner, &super_admin);

        e.events()
            .publish((Symbol::new(&e, "admin_initialized"),), super_admin);
    }

    /// Add a new admin with the specified role.
    ///
    /// Callers SHOULD gate this entrypoint on
    /// [`AdminContract::check_role_at_ledger`] to avoid submitting a mutation
    /// that will be rejected for lack of authorization.
    ///
    /// # Arguments
    /// * `caller` - Address of the caller making the assignment
    /// * `new_admin` - Address of the new admin to add
    /// * `role` - Role to assign to the new admin
    ///
    /// # Returns
    /// The created `AdminInfo`
    ///
    /// # Panics
    /// * If caller is not authorized to assign this role
    /// * If new_admin is already an admin
    /// * If maximum admin limit would be exceeded
    /// * If caller is trying to assign equal or higher role to themselves
    ///
    /// # Events
    /// Emits `admin_added` with the new admin information
    pub fn add_admin(e: Env, caller: Address, new_admin: Address, role: AdminRole) -> AdminInfo {
        bump_instance_ttl(&e);
        pausable::require_not_paused(&e);
        caller.require_auth_for_args((caller.clone(), new_admin.clone(), role).into_val(&e));

        Self::require_valid_admin_address(&e, &new_admin);

        // Verify caller authorization
        Self::require_role_at_least(&e, &caller, Self::get_required_role_to_assign(role))
            .unwrap_or_else(|_| panic_with_error!(&e, ContractError::NotAdmin));

        // Check if new admin already exists
        if e.storage()
            .instance()
            .has(&DataKey::AdminInfo(new_admin.clone()))
        {
            panic_with_error!(&e, ContractError::AlreadyActive);
        }

        // Prevent self-assignment of equal or higher role
        if caller == new_admin && Self::get_role(e.clone(), caller.clone()) >= role {
            panic_with_error!(&e, ContractError::NotAdmin);
        }

        // Check admin limit
        let current_count = Self::get_admin_count(e.clone());
        let max_admins: u32 = e
            .storage()
            .instance()
            .get(&DataKey::MaxAdmins)
            .unwrap_or(100);
        if current_count >= max_admins {
            panic_with_error!(&e, ContractError::ThresholdExceedsSigners);
        }

        bump_config_epoch(&e);

        // Create admin info
        let admin_info = AdminInfo {
            address: new_admin.clone(),
            role,
            assigned_at: e.ledger().timestamp(),
            assigned_by: caller.clone(),
            active: true,
            suspended_until: 0,
        };

        // Store admin info
        e.storage()
            .instance()
            .set(&DataKey::AdminInfo(new_admin.clone()), &admin_info.clone());

        // Update admin list
        let mut admin_list: Vec<Address> = e
            .storage()
            .instance()
            .get(&DataKey::AdminList)
            .unwrap_or(Vec::new(&e));
        admin_list.push_back(new_admin.clone());
        e.storage().instance().set(&DataKey::AdminList, &admin_list);

        // Update role-based admin list
        let mut role_admins: Vec<Address> = e
            .storage()
            .instance()
            .get(&DataKey::RoleAdmins(role))
            .unwrap_or(Vec::new(&e));
        role_admins.push_back(new_admin.clone());
        e.storage()
            .instance()
            .set(&DataKey::RoleAdmins(role), &role_admins);

        e.events()
            .publish((Symbol::new(&e, "admin_added"),), admin_info.clone());

        e.events().publish(
            (Symbol::new(&e, "ROLE_ASSIGNED"), new_admin),
            (role, caller),
        );

        admin_info
    }

    /// Remove an admin from the system.
    ///
    /// # Arguments
    /// * `caller` - Address of the caller making the removal
    /// * `admin_to_remove` - Address of the admin to remove
    ///
    /// # Panics
    /// * If caller is not authorized to remove this admin
    /// * If admin_to_remove is not an admin
    /// * If removing would violate minimum admin requirements
    /// * If admin is trying to remove themselves and they're the last admin of their role
    ///
    /// # Events
    /// Emits `admin_removed` with the removed admin information
    pub fn remove_admin(e: Env, caller: Address, admin_to_remove: Address) {
        bump_instance_ttl(&e);
        pausable::require_not_paused(&e);
        caller.require_auth_for_args((caller.clone(), admin_to_remove.clone()).into_val(&e));

        Self::require_valid_admin_address(&e, &admin_to_remove);

        // Get admin info
        let admin_info: AdminInfo = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(admin_to_remove.clone()))
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotAdmin));

        // Verify caller authorization
        let caller_role = Self::get_role(e.clone(), caller.clone());
        if caller_role <= admin_info.role {
            panic_with_error!(&e, ContractError::NotAdmin);
        }

        // Check minimum admin requirements
        let role_admins: Vec<Address> = e
            .storage()
            .instance()
            .get(&DataKey::RoleAdmins(admin_info.role))
            .unwrap_or(Vec::new(&e));

        let min_admins: u32 = e.storage().instance().get(&DataKey::MinAdmins).unwrap_or(1);

        // Special protection for super admins
        if admin_info.role == AdminRole::SuperAdmin && role_admins.len() <= min_admins {
            panic_with_error!(&e, ContractError::InvalidPauseAction);
        }

        bump_config_epoch(&e);

        // Remove from admin info storage
        e.storage()
            .instance()
            .remove(&DataKey::AdminInfo(admin_to_remove.clone()));

        // Remove from admin list
        let mut admin_list: Vec<Address> = e
            .storage()
            .instance()
            .get(&DataKey::AdminList)
            .unwrap_or(Vec::new(&e));
        let admin_index = admin_list.iter().position(|x| x == admin_to_remove);
        if let Some(index) = admin_index {
            admin_list.remove(
                index
                    .try_into()
                    .unwrap_or_else(|_| panic_with_error!(&e, ContractError::Overflow)),
            );
            e.storage().instance().set(&DataKey::AdminList, &admin_list);
        }

        // Remove from role-based admin list
        let mut role_admins: Vec<Address> = e
            .storage()
            .instance()
            .get(&DataKey::RoleAdmins(admin_info.role))
            .unwrap_or(Vec::new(&e));
        let role_index = role_admins.iter().position(|x| x == admin_to_remove);
        if let Some(index) = role_index {
            role_admins.remove(
                index
                    .try_into()
                    .unwrap_or_else(|_| panic_with_error!(&e, ContractError::Overflow)),
            );
            e.storage()
                .instance()
                .set(&DataKey::RoleAdmins(admin_info.role), &role_admins);
        }

        e.events()
            .publish((Symbol::new(&e, "admin_removed"),), admin_info);

        e.events().publish(
            (Symbol::new(&e, "ROLE_REVOKED"), admin_to_remove),
            (caller,),
        );
    }

    /// Update an admin's role.
    ///
    /// # Arguments
    /// * `caller` - Address of the caller making the change
    /// * `admin_address` - Address of the admin to update
    /// * `new_role` - New role to assign
    ///
    /// # Returns
    /// The updated `AdminInfo`
    ///
    /// # Panics
    /// * If caller is not authorized to change to this role
    /// * If admin_address is not an admin
    /// * If caller is trying to assign equal or higher role to themselves
    ///
    /// # Events
    /// Emits `admin_role_updated` with the updated admin information
    pub fn update_admin_role(
        e: Env,
        caller: Address,
        admin_address: Address,
        new_role: AdminRole,
    ) -> AdminInfo {
        bump_instance_ttl(&e);
        pausable::require_not_paused(&e);
        caller
            .require_auth_for_args((caller.clone(), admin_address.clone(), new_role).into_val(&e));

        Self::require_valid_admin_address(&e, &admin_address);

        // Get current admin info
        let mut admin_info: AdminInfo = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(admin_address.clone()))
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotAdmin));

        // Verify caller authorization
        Self::require_role_at_least(&e, &caller, Self::get_required_role_to_assign(new_role))
            .unwrap_or_else(|_| panic_with_error!(&e, ContractError::NotAdmin));

        // Prevent self-assignment of equal or higher role
        if caller == admin_address && Self::get_role(e.clone(), caller.clone()) >= new_role {
            panic_with_error!(&e, ContractError::NotAdmin);
        }

        let old_role = admin_info.role;

        // Repeated/no-op role update: leave state and events untouched so
        // concurrent observers never see a spurious mutation or epoch bump.
        if new_role == old_role {
            return admin_info;
        }

        bump_config_epoch(&e);

        // Remove from old role list
        let mut old_role_admins: Vec<Address> = e
            .storage()
            .instance()
            .get(&DataKey::RoleAdmins(old_role))
            .unwrap_or(Vec::new(&e));
        let old_index = old_role_admins.iter().position(|x| x == admin_address);
        if let Some(index) = old_index {
            old_role_admins.remove(
                index
                    .try_into()
                    .unwrap_or_else(|_| panic_with_error!(&e, ContractError::Overflow)),
            );
            e.storage()
                .instance()
                .set(&DataKey::RoleAdmins(old_role), &old_role_admins);
        }

        // Add to new role list
        let mut new_role_admins: Vec<Address> = e
            .storage()
            .instance()
            .get(&DataKey::RoleAdmins(new_role))
            .unwrap_or(Vec::new(&e));
        new_role_admins.push_back(admin_address.clone());
        e.storage()
            .instance()
            .set(&DataKey::RoleAdmins(new_role), &new_role_admins);

        // Update admin info
        admin_info.role = new_role;
        admin_info.assigned_at = e.ledger().timestamp();
        admin_info.assigned_by = caller.clone();

        // Store updated admin info
        e.storage().instance().set(
            &DataKey::AdminInfo(admin_address.clone()),
            &admin_info.clone(),
        );

        e.events().publish(
            (Symbol::new(&e, "admin_role_updated"),),
            (admin_address.clone(), old_role, new_role),
        );

        e.events().publish(
            (Symbol::new(&e, "ROLE_ASSIGNED"), admin_address),
            (new_role, caller),
        );

        admin_info
    }

    /// Deactivate an admin (can be reactivated later).
    ///
    /// # Arguments
    /// * `caller` - Address of the caller making the change
    /// * `admin_address` - Address of the admin to deactivate
    ///
    /// # Panics
    /// * If caller is not authorized to deactivate this admin
    /// * If admin_address is not an admin
    /// * If admin is already deactivated
    ///
    /// # Events
    /// Emits `admin_deactivated` with the deactivated admin information
    pub fn deactivate_admin(e: Env, caller: Address, admin_address: Address) {
        bump_instance_ttl(&e);
        pausable::require_not_paused(&e);
        caller.require_auth_for_args((caller.clone(), admin_address.clone()).into_val(&e));

        Self::require_valid_admin_address(&e, &admin_address);

        let mut admin_info: AdminInfo = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(admin_address.clone()))
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotAdmin));

        // Verify caller authorization: caller must strictly outrank the target.
        let caller_role = Self::get_role(e.clone(), caller.clone());
        if caller_role <= admin_info.role {
            panic_with_error!(&e, ContractError::NotAdmin);
        }

        if !admin_info.active {
            panic_with_error!(&e, ContractError::AlreadyDeactivated);
        }

        bump_config_epoch(&e);

        admin_info.active = false;
        e.storage().instance().set(
            &DataKey::AdminInfo(admin_address.clone()),
            &admin_info.clone(),
        );

        e.events()
            .publish((Symbol::new(&e, "admin_deactivated"),), admin_info);

        e.events()
            .publish((Symbol::new(&e, "ROLE_REVOKED"), admin_address), (caller,));
    }

    /// Reactivate a previously deactivated admin.
    ///
    /// # Arguments
    /// * `caller` - Address of the caller making the change
    /// * `admin_address` - Address of the admin to reactivate
    ///
    /// # Panics
    /// * If caller is not authorized to reactivate this admin
    /// * If admin_address is not an admin
    /// * If admin is already active
    ///
    /// # Events
    /// Emits `admin_reactivated` with the reactivated admin information
    pub fn reactivate_admin(e: Env, caller: Address, admin_address: Address) {
        bump_instance_ttl(&e);
        pausable::require_not_paused(&e);
        caller.require_auth_for_args((caller.clone(), admin_address.clone()).into_val(&e));

        Self::require_valid_admin_address(&e, &admin_address);

        let mut admin_info: AdminInfo = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(admin_address.clone()))
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotAdmin));

        // Verify caller authorization
        let caller_role = Self::get_role(e.clone(), caller.clone());
        // Allow reactivation when caller has the same role as the target.
        if caller_role < admin_info.role {
            panic_with_error!(&e, ContractError::NotAdmin);
        }

        if admin_info.active {
            panic_with_error!(&e, ContractError::AlreadyActive);
        }

        bump_config_epoch(&e);

        admin_info.active = true;
        e.storage().instance().set(
            &DataKey::AdminInfo(admin_address.clone()),
            &admin_info.clone(),
        );

        e.events()
            .publish((Symbol::new(&e, "admin_reactivated"),), admin_info.clone());

        e.events().publish(
            (Symbol::new(&e, "ROLE_ASSIGNED"), admin_address),
            (admin_info.role, caller),
        );
    }

    /// Suspend an admin until a future ledger timestamp.
    ///
    /// While `e.ledger().timestamp() < until_ts` the admin is treated as
    /// inactive by `is_admin` and `has_role_at_least`.  Once the timestamp
    /// passes the admin is **automatically** effective again — no second
    /// transaction is needed.
    ///
    /// Suspension is distinct from `deactivate_admin`: deactivation is
    /// indefinite and requires an explicit `reactivate_admin` call, whereas
    /// suspension is self-expiring.
    ///
    /// # Arguments
    /// * `caller`    - Address authorising the suspension (must have higher or
    ///                 equal role to the target, same rules as `deactivate_admin`)
    /// * `admin`     - Address of the admin to suspend
    /// * `until_ts`  - Unix timestamp (seconds) after which the suspension
    ///                 expires; must be strictly greater than the current ledger
    ///                 timestamp
    ///
    /// # Panics
    /// * `NotAdmin`          — caller or target is not a known admin
    /// * `NotAdmin`          — caller role is strictly lower than target role
    /// * `AdminSuspended`    — `until_ts` is not in the future
    /// * `AdminSuspended`    — target admin is already suspended at or beyond `until_ts`
    ///                         (re-suspension must strictly extend the window)
    /// * `AdminUnchanged`    — target admin is the caller (self-suspension is rejected)
    /// * `InvalidPauseAction` — suspending would drop active admins below `MinAdmins`
    /// * `AlreadyDeactivated` — target admin is permanently deactivated
    ///
    /// # Events
    /// Emits `admin_suspended` with `(admin_address, until_ts)`
    pub fn suspend_admin(e: Env, caller: Address, admin: Address, until_ts: u64) {
        bump_instance_ttl(&e);
        pausable::require_not_paused(&e);
        caller.require_auth_for_args((caller.clone(), admin.clone(), until_ts).into_val(&e));

        // Self-suspension is rejected: an admin must not be able to lock
        // themselves out, which would otherwise strand governance when the
        // caller is the only effective admin of their role.
        if caller == admin {
            panic_with_error!(&e, ContractError::AdminUnchanged);
        }

        // until_ts must be in the future
        if until_ts <= e.ledger().timestamp() {
            panic_with_error!(&e, ContractError::AdminSuspended);
        }

        let mut admin_info: AdminInfo = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(admin.clone()))
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotAdmin));

        // Cannot suspend a permanently deactivated admin
        if !admin_info.active {
            panic_with_error!(&e, ContractError::AlreadyDeactivated);
        }

        // Re-suspension must strictly extend the existing window. A repeated
        // or shorter suspension is a no-op rejection so observers never see a
        // spurious epoch bump or event for a state that did not change.
        let now = e.ledger().timestamp();
        if admin_info.suspended_until > now && until_ts <= admin_info.suspended_until {
            panic_with_error!(&e, ContractError::AdminSuspended);
        }

        // Caller must have a role >= target's role (same rule as deactivate_admin)
        let caller_info: AdminInfo = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(caller.clone()))
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotAdmin));
        if caller_info.role < admin_info.role {
            panic_with_error!(&e, ContractError::NotAdmin);
        }

        // MinAdmins guard: count currently-effective active admins
        let min_admins: u32 = e.storage().instance().get(&DataKey::MinAdmins).unwrap_or(1);
        let all_admins: Vec<Address> = e
            .storage()
            .instance()
            .get(&DataKey::AdminList)
            .unwrap_or(Vec::new(&e));
        let mut effective_active: u32 = 0;
        for addr in all_admins.iter() {
            if addr == admin {
                continue; // exclude the target — they'll be suspended
            }
            if let Some(info) = e
                .storage()
                .instance()
                .get::<_, AdminInfo>(&DataKey::AdminInfo(addr))
            {
                if info.active && now >= info.suspended_until {
                    effective_active += 1;
                }
            }
        }
        if effective_active < min_admins {
            panic_with_error!(&e, ContractError::InvalidPauseAction);
        }

        bump_config_epoch(&e);

        admin_info.suspended_until = until_ts;
        e.storage()
            .instance()
            .set(&DataKey::AdminInfo(admin.clone()), &admin_info);

        e.events()
            .publish((Symbol::new(&e, "admin_suspended"),), (admin, until_ts));
    }

    /// Propose a new owner for the contract (two-step ownership transfer).
    /// * `new_owner` - Address of the proposed new owner
    ///
    /// # Panics
    /// * If caller is not the current owner
    /// * If new_owner is the same as current owner
    /// * If new_owner is not a SuperAdmin
    ///
    /// # Events
    /// Emits `ownership_transfer_initiated` with current owner and pending owner
    ///
    /// # Notes
    /// The ownership remains with the current owner until the new owner calls `accept_ownership`.
    pub fn transfer_ownership(e: Env, caller: Address, new_owner: Address) {
        bump_instance_ttl(&e);
        pausable::require_not_paused(&e);
        caller.require_auth_for_args((caller.clone(), new_owner.clone()).into_val(&e));

        Self::require_valid_admin_address(&e, &new_owner);

        // Get current owner
        let current_owner: Address = e
            .storage()
            .instance()
            .get(&DataKey::Owner)
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotInitialized));

        // Verify caller is the current owner
        if caller != current_owner {
            panic_with_error!(&e, ContractError::NotAdmin);
        }

        // Verify new owner is different from current owner
        if new_owner == current_owner {
            panic_with_error!(&e, ContractError::AdminUnchanged);
        }

        // A suspended or deactivated admin must not be able to receive durable
        // ownership. This check is repeated by `accept_ownership`, because the
        // candidate's status can change during the timelock.
        Self::require_effective_super_admin(&e, &new_owner);

        bump_config_epoch(&e);

        // Store pending owner and proposal timestamp for timelock
        e.storage()
            .instance()
            .set(&DataKey::PendingOwner, &new_owner.clone());
        e.storage()
            .instance()
            .set(&DataKey::TransferProposedAt, &e.ledger().timestamp());

        e.events().publish(
            (Symbol::new(&e, "ownership_transfer_initiated"),),
            (current_owner, new_owner),
        );
    }

    /// Accept ownership transfer (two-step acceptance with timelock).
    ///
    /// # Arguments
    /// * `caller` - Address of the pending owner accepting the transfer
    ///
    /// # Panics
    /// * `NoPendingAdmin` — no ownership transfer has been proposed
    /// * `NotAdmin` — caller is not the pending owner, or the pending
    ///   candidate is no longer a SuperAdmin
    /// * `TimelockNotReady` — the minimum delay since proposal has not elapsed
    /// * `AlreadyDeactivated` — the pending candidate was deactivated
    /// * `AdminSuspended` — the pending candidate is currently suspended
    ///
    /// # Events
    /// Emits `ownership_transfer_accepted` with previous owner and new owner
    ///
    /// # Notes
    /// This function completes the two-step ownership transfer process.
    /// The caller must be the address that was previously set as pending owner.
    /// A minimum delay of `OWNERSHIP_TRANSFER_TIMELOCK` seconds must elapse
    /// between `transfer_ownership` and `accept_ownership` to protect against
    /// compromised-owner takeovers.
    ///
    /// The pending candidate is revalidated against *current* state immediately
    /// before the ownership write. A candidate who was removed, demoted,
    /// deactivated, or suspended during the timelock cannot accept; the
    /// reverted call changes no state and emits no events, so the current owner
    /// can recover by replacing the proposal.
    pub fn accept_ownership(e: Env, caller: Address) {
        bump_instance_ttl(&e);
        pausable::require_not_paused(&e);
        caller.require_auth_for_args((caller.clone(),).into_val(&e));

        // Get pending owner
        let pending_owner: Address = e
            .storage()
            .instance()
            .get(&DataKey::PendingOwner)
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NoPendingAdmin));

        // Verify caller is the pending owner
        if caller != pending_owner {
            panic_with_error!(&e, ContractError::NotAdmin);
        }

        // Enforce timelock: the transfer proposal must have aged past the minimum delay
        let proposed_at: u64 = e
            .storage()
            .instance()
            .get(&DataKey::TransferProposedAt)
            .unwrap_or(0);
        let now = e.ledger().timestamp();
        let eligible_at = proposed_at
            .checked_add(OWNERSHIP_TRANSFER_TIMELOCK)
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::Overflow));
        if now < eligible_at {
            panic_with_error!(&e, ContractError::TimelockNotReady);
        }

// Revalidate the pending owner's effective SuperAdmin status at
        // acceptance time. The candidate's role, activation, or suspension
        // state may have changed during the timelock window; a stale proposal
        // must never grant durable ownership to an admin who is no longer an
        // effective SuperAdmin. This mirrors the check performed by
        // `transfer_ownership` and preserves the two-step transfer invariant.
        Self::require_effective_super_admin(&e, &pending_owner);
        bump_config_epoch(&e);

        // Get current owner for event emission
        let previous_owner: Address = e
            .storage()
            .instance()
            .get(&DataKey::Owner)
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotInitialized));

        // Transfer ownership
        e.storage()
            .instance()
            .set(&DataKey::Owner, &pending_owner.clone());

        // Clear pending owner and transfer timestamp
        e.storage().instance().remove(&DataKey::PendingOwner);
        e.storage().instance().remove(&DataKey::TransferProposedAt);

        // Emit admin rotated event with ledger sequence
        let ledger_seq: u32 = e.ledger().sequence();
        e.events().publish(
            (
                Symbol::new(&e, "admin_rotated"),
                previous_owner.clone(),
                pending_owner.clone(),
            ),
            ledger_seq,
        );

        // Emit original ownership transfer accepted event
        e.events().publish(
            (Symbol::new(&e, "ownership_transfer_accepted"),),
            (previous_owner.clone(), pending_owner.clone()),
        );
    }

    /// Get the current owner of the contract.
    ///
    /// # Returns
    /// The address of the current owner
    ///
    /// # Panics
    /// * If owner has not been set (contract not initialized)
    pub fn get_owner(e: Env) -> Address {
        bump_instance_ttl(&e);
        e.storage()
            .instance()
            .get(&DataKey::Owner)
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotInitialized))
    }

    /// Get the pending owner (if any) for the current ownership transfer.
    ///
    /// # Returns
    /// `Some(address)` if there is a pending owner, `None` otherwise
    pub fn get_pending_owner(e: Env) -> Option<Address> {
        bump_instance_ttl(&e);
        e.storage().instance().get(&DataKey::PendingOwner)
    }

    /// Return the ledger timestamp at which the current pending ownership
    /// transfer becomes eligible for acceptance, if a transfer is pending.
    ///
    /// Returns `None` when no transfer has been proposed. Otherwise returns
    /// `Some(proposed_at + OWNERSHIP_TRANSFER_TIMELOCK)` — the earliest ledger
    /// timestamp at which `accept_ownership` will succeed. Clients can use
    /// this to schedule retries deterministically without guessing.
    pub fn get_pending_owner_eligible_at(e: Env) -> Option<u64> {
        bump_instance_ttl(&e);
        let proposed_at: u64 = e
            .storage()
            .instance()
            .get(&DataKey::TransferProposedAt)?;
        Some(
            proposed_at
                .checked_add(OWNERSHIP_TRANSFER_TIMELOCK)
                .unwrap_or_else(|| panic_with_error!(&e, ContractError::Overflow)),
        )
    }

    /// Get information about a specific admin.
    ///
    /// # Arguments
    /// * `admin_address` - Address of the admin to query
    ///
    /// # Returns
    /// The `AdminInfo` for the specified admin
    ///
    /// # Panics
    /// * If admin_address is not an admin
    pub fn get_admin_info(e: Env, admin_address: Address) -> AdminInfo {
        bump_instance_ttl(&e);
        e.storage()
            .instance()
            .get(&DataKey::AdminInfo(admin_address))
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotAdmin))
    }

    /// Check if an address is an admin and return their role.
    ///
    /// # Arguments
    /// * `address` - Address to check
    ///
    /// # Returns
    /// The admin role if the address is an admin, panics otherwise
    pub fn get_admin_role(e: Env, address: Address) -> AdminRole {
        bump_instance_ttl(&e);
        // Failure-boundary invariant: `get_admin_role` is a read-only query
        // that MUST NOT mutate state, advance the config epoch, or emit
        // events. It is deterministic for all inputs:
        //   * known admin (active, suspended, or deactivated) -> stored role
        //   * unknown / never-registered address              -> NotAdmin panic
        //   * zero/invalid sentinel address                   -> NotAdmin panic
        // A suspended or deactivated admin still resolves to their stored
        // role here; callers that need effective-authority semantics must
        // use `is_admin` / `has_role_at_least` instead. This separation is
        // intentional and covered by focused failure-boundary tests.
        let admin_info: AdminInfo = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(address))
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotAdmin));
        admin_info.role
    }

    /// Check if an address is an active admin.
    ///
    /// # Arguments
    /// * `address` - Address to check
    ///
    /// # Returns
    /// `Role::Admin` if the address is an active admin, `Role::User` otherwise.
    pub fn is_admin(e: Env, address: Address) -> Role {
        match e
            .storage()
            .instance()
            .get::<_, AdminInfo>(&DataKey::AdminInfo(address))
        {
            Some(admin_info) => {
                if admin_info.active && e.ledger().timestamp() >= admin_info.suspended_until {
                    Role::Admin
                } else {
                    Role::User
                }
            }
            None => Role::User,
        }
    }

    /// Check if an address has at least the specified role level.
    ///
    /// # Arguments
    /// * `address` - Address to check
    /// * `required_role` - Minimum required role
    ///
    /// # Returns
    /// `true` if the address has at least the required role, `false` otherwise.
    /// A suspended admin fails this check until `suspended_until` has passed.
    pub fn has_role_at_least(e: Env, address: Address, required_role: AdminRole) -> bool {
        bump_instance_ttl(&e);
        match e
            .storage()
            .instance()
            .get::<_, AdminInfo>(&DataKey::AdminInfo(address))
        {
            Some(admin_info) => {
                admin_info.active
                    && e.ledger().timestamp() >= admin_info.suspended_until
                    && admin_info.role >= required_role
            }
            None => false,
        }
    }

    /// Historical role check: assert that `actor` held at least `role` at
    /// ledger timestamp `at_ledger`.
    ///
    /// Panics with [`ContractError::NotAdmin`] when `actor` is not a registered
    /// admin or when their role is below the required level.
    /// Panics with [`ContractError::RoleNotHeldAtLedger`] when the actor's role
    /// was granted **after** `at_ledger`, meaning they were not authorised at
    /// the time the signed action was created.
    ///
    /// See [`Self::require_role_at_ledger`] (private) for the full threat model.
    ///
    /// # Arguments
    /// * `role`      - Minimum `AdminRole` that must have been held
    /// * `actor`     - Address whose historical role is checked
    /// * `at_ledger` - Unix timestamp (seconds) of the signed action
    pub fn check_role_at_ledger(e: Env, role: AdminRole, actor: Address, at_ledger: u64) {
        bump_instance_ttl(&e);
        Self::require_role_at_ledger(e, role, actor, at_ledger);
    }

    /// Get all admin addresses.
    ///
    /// # Deprecated
    /// This function returns an unbounded list that will eventually exceed
    /// Soroban's per-transaction resource limits as the admin set grows.
    ///
    /// Use [`get_all_admins_page`] instead for bounded, paginated access.
    /// For event-based discovery, listen to `admin_added` / `admin_removed` events.
    ///
    /// # Returns
    /// A `Vec` of all admin addresses
    #[deprecated(note = "Use get_all_admins_page for bounded pagination")]
    pub fn get_all_admins(e: Env) -> Vec<Address> {
        bump_instance_ttl(&e);
        e.storage()
            .instance()
            .get(&DataKey::AdminList)
            .unwrap_or(Vec::new(&e))
    }

    /// Get all admins with a specific role.
    ///
    /// # Deprecated
    /// This function returns an unbounded list that will eventually exceed
    /// Soroban's per-transaction resource limits as the admin set grows.
    ///
    /// Use [`get_admins_by_role_page`] instead for bounded, paginated access.
    ///
    /// # Arguments
    /// * `role` - Role to filter by
    ///
    /// # Returns
    /// A `Vec` of admin addresses with the specified role
    #[deprecated(note = "Use get_admins_by_role_page for bounded pagination")]
    pub fn get_admins_by_role(e: Env, role: AdminRole) -> Vec<Address> {
        bump_instance_ttl(&e);
        e.storage()
            .instance()
            .get(&DataKey::RoleAdmins(role))
            .unwrap_or(Vec::new(&e))
    }

    /// Get a bounded, cursor-paginated page of all admin addresses.
    ///
    /// # Cursor contract
    /// * `cursor` — the 0-based index to start from. Pass `0` for the first page.
    /// * `limit` — maximum number of admins to return. Hard-capped at
    ///   [`MAX_PAGE_LIMIT`]; a larger value is clamped, never honoured.
    /// * Returns `(page, next_cursor)` where `next_cursor` is `Some(next_index)`
    ///   when more results remain, or `None` when the page exhausted the set.
    ///
    /// Feeding `next_cursor` back as `cursor` walks the full set in bounded,
    /// resumable pages; concatenating every page reproduces [`get_all_admins`].
    ///
    /// # Arguments
    /// * `e` - Soroban environment
    /// * `cursor` - 0-based start index
    /// * `limit` - Maximum items to return (clamped to [`MAX_PAGE_LIMIT`])
    ///
    /// # Returns
    /// A tuple of `(Vec<Address>, Option<u32>)` — the page and optional next cursor.
    pub fn get_all_admins_page(e: Env, cursor: u32, limit: u32) -> (Vec<Address>, Option<u32>) {
        bump_instance_ttl(&e);
        let admin_list: Vec<Address> = e
            .storage()
            .instance()
            .get(&DataKey::AdminList)
            .unwrap_or(Vec::new(&e));

        let total = admin_list.len();

        if cursor >= total {
            return (Vec::new(&e), None);
        }

        let effective_limit = if limit == 0 {
            MAX_PAGE_LIMIT
        } else {
            limit.min(MAX_PAGE_LIMIT)
        };

        let end = (cursor + effective_limit).min(total);
        let mut page = Vec::new(&e);
        for i in cursor..end {
            if let Some(addr) = admin_list.get(i) {
                page.push_back(addr);
            }
        }

        let next_cursor = if end >= total { None } else { Some(end) };

        (page, next_cursor)
    }

    /// Get a bounded, cursor-paginated page of admin addresses with a specific role.
    ///
    /// # Cursor contract
    /// * `cursor` — the 0-based index to start from. Pass `0` for the first page.
    /// * `limit` — maximum number of admins to return. Hard-capped at
    ///   [`MAX_PAGE_LIMIT`]; a larger value is clamped, never honoured.
    /// * Returns `(page, next_cursor)` where `next_cursor` is `Some(next_index)`
    ///   when more results remain, or `None` when the page exhausted the set.
    ///
    /// # Arguments
    /// * `e` - Soroban environment
    /// * `role` - Role to filter by
    /// * `cursor` - 0-based start index
    /// * `limit` - Maximum items to return (clamped to [`MAX_PAGE_LIMIT`])
    ///
    /// # Returns
    /// A tuple of `(Vec<Address>, Option<u32>)` — the page and optional next cursor.
    pub fn get_admins_by_role_page(
        e: Env,
        role: AdminRole,
        cursor: u32,
        limit: u32,
    ) -> (Vec<Address>, Option<u32>) {
        bump_instance_ttl(&e);
        let role_admins: Vec<Address> = e
            .storage()
            .instance()
            .get(&DataKey::RoleAdmins(role))
            .unwrap_or(Vec::new(&e));

        let total = role_admins.len();

        if cursor >= total {
            return (Vec::new(&e), None);
        }

        let effective_limit = if limit == 0 {
            MAX_PAGE_LIMIT
        } else {
            limit.min(MAX_PAGE_LIMIT)
        };

        let end = (cursor + effective_limit).min(total);
        let mut page = Vec::new(&e);
        for i in cursor..end {
            if let Some(addr) = role_admins.get(i) {
                page.push_back(addr);
            }
        }

        let next_cursor = if end >= total { None } else { Some(end) };

        (page, next_cursor)
    }

    /// Get the total number of admins.
    ///
    /// # Returns
    /// The total count of admins
    pub fn get_admin_count(e: Env) -> u32 {
        bump_instance_ttl(&e);
        #[allow(deprecated)]
        Self::get_all_admins(e).len()
    }

    /// Get the number of active admins.
    ///
    /// # Returns
    /// The count of active admins
    pub fn get_active_admin_count(e: Env) -> u32 {
        bump_instance_ttl(&e);
        #[allow(deprecated)]
        let all_admins = Self::get_all_admins(e.clone());
        let mut active_count = 0;
        for admin in all_admins.iter() {
            if let Some(admin_info) = e
                .storage()
                .instance()
                .get::<_, AdminInfo>(&DataKey::AdminInfo(admin.clone()))
            {
                if admin_info.active {
                    active_count += 1;
                }
            }
        }
        active_count
    }

    /// Get the number of currently-effective active admins.
    ///
    /// Unlike [`get_active_admin_count`], this counts only admins that are
    /// both `active == true` **and** not currently suspended
    /// (`e.ledger().timestamp() >= suspended_until`).  It is the count used
    /// by the `MinAdmins` guard in [`suspend_admin`] and matches the
    /// effective-admin semantics of [`is_admin`] and [`has_role_at_least`].
    ///
    /// # Determinism
    /// The result is a pure function of the persisted `AdminList` and the
    /// per-admin `AdminInfo` records at the current ledger timestamp.  It
    /// performs no mutation, advances no epoch, and emits no events, so it is
    /// safe to call from read-only paths and from within other entrypoints.
    ///
    /// # Boundary behaviour
    /// * Empty / uninitialised `AdminList` → `0`.
    /// * An `AdminList` entry with no matching `AdminInfo` (dangling entry)
    ///   is skipped, never counted, and never panics.
    /// * Suspension expiry is inclusive: at exactly `suspended_until` the
    ///   admin is effective again.
    ///
    /// # Returns
    /// The count of currently-effective active admins.
    pub fn get_effective_active_admin_count(e: Env) -> u32 {
        bump_instance_ttl(&e);
        #[allow(deprecated)]
        let all_admins = Self::get_all_admins(e.clone());
        let now = e.ledger().timestamp();
        let mut active_count: u32 = 0;
        for admin in all_admins.iter() {
            if let Some(admin_info) = e
                .storage()
                .instance()
                .get::<_, AdminInfo>(&DataKey::AdminInfo(admin.clone()))
            {
                if admin_info.active && now >= admin_info.suspended_until {
                    active_count = active_count
                        .checked_add(1)
                        .unwrap_or_else(|| panic_with_error!(&e, ContractError::Overflow));
                }
            }
        }
        active_count
    }

    /// Get contract configuration.
    ///
    /// # Determinism and failure boundaries
    ///
    /// This is a pure read: it never mutates storage, never advances
    /// [`DataKey::ConfigEpoch`], and never emits events. Given the same ledger
    /// snapshot it always returns the same `(min_admins, max_admins)` pair.
    ///
    /// Failure boundary: on an uninitialized contract (or one whose config
    /// keys were never written) this panics with
    /// [`ContractError::NotInitialized`] rather than returning a defaulted
    /// `(0, 0)` tuple. Returning a fabricated default would let callers
    /// silently proceed against a contract that has no enforced admin
    /// bounds, so the failure is surfaced explicitly and atomically.
    ///
    /// # Returns
    /// A tuple of (min_admins, max_admins)
    pub fn get_config(e: Env) -> (u32, u32) {
        bump_instance_ttl(&e);
        let min_admins: u32 = e
            .storage()
            .instance()
            .get(&DataKey::MinAdmins)
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotInitialized));
        let max_admins: u32 = e
            .storage()
            .instance()
            .get(&DataKey::MaxAdmins)
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotInitialized));
        (min_admins, max_admins)
    }

    /// Return the current configuration epoch.
    ///
    /// The epoch is a monotonic counter that advances exactly once per
    /// *committed* privileged mutation (admin role changes, suspension,
    /// ownership transfer, pause configuration, pause state transitions, and
    /// pause-proposal approvals). It never advances on rejected, repeated
    /// (no-op), or failed operations.
    ///
    /// Clients should read this value together with the governance state they
    /// depend on, and retry against fresh state whenever a privileged request
    /// is rejected or the epoch has advanced since their read. See the module
    /// documentation for the full serialization and retry contract.
    pub fn get_config_epoch(e: Env) -> u64 {
        bump_instance_ttl(&e);
        e.storage()
            .instance()
            .get(&DataKey::ConfigEpoch)
            .unwrap_or(0)
    }

    // Helper functions

    /// Get the role of an address (panics if not admin).
    pub fn get_role(e: Env, address: Address) -> AdminRole {
        bump_instance_ttl(&e);
        let admin_info: AdminInfo = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(address))
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotAdmin));
        admin_info.role
    }

    /// Get the minimum role required to assign a specific role.
    pub fn get_required_role_to_assign(role: AdminRole) -> AdminRole {
        match role {
            AdminRole::SuperAdmin => AdminRole::SuperAdmin,
            AdminRole::Admin => AdminRole::SuperAdmin,
            AdminRole::Operator => AdminRole::Admin,
        }
    }

    /// Require that the target address is not the zero/invalid sentinel
    /// and is not the contract's own address.
    ///
    /// # Policy
    /// An address is considered invalid when:
    ///
    /// 1. Its strkey encoding matches the all-zero Ed25519 public key
    ///    (`GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABZKY`).
    ///    This catches uninitialised/garbage strkeys.
    /// 2. It equals the contract's own address.  Assigning a governance
    ///    role to the contract itself can cause invariants to break.
    ///
    /// # Panics
    /// * `ContractError::InvalidAdminAddress` if the address fails either check.
    fn require_valid_admin_address(e: &Env, address: &Address) {
        if address.to_string() == String::from_str(e, INVALID_ADDRESS_SENTINEL) {
            panic_with_error!(e, ContractError::InvalidAdminAddress);
        }
        if address == &e.current_contract_address() {
            panic_with_error!(e, ContractError::InvalidAdminAddress);
        }
    }

    /// Require that the caller has at least the specified role.
    fn require_role_at_least(
        e: &Env,
        caller: &Address,
        required_role: AdminRole,
    ) -> Result<(), ()> {
        let admin_info: AdminInfo = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(caller.clone()))
            .unwrap_or_else(|| panic_with_error!(e, ContractError::NotAdmin));
        if admin_info.active
            && e.ledger().timestamp() >= admin_info.suspended_until
            && admin_info.role >= required_role
        {
            Ok(())
        } else {
            Err(())
        }
    }

    /// Require an owner candidate to be an effective SuperAdmin now.
    ///
    /// Ownership proposals are intentionally revalidated at acceptance, not
    /// trusted based on the state at proposal time. This preserves the
    /// two-step transfer API while preventing a stale proposal from granting
    /// durable authority to an inactive, suspended, removed, or demoted admin.
    fn require_effective_super_admin(e: &Env, candidate: &Address) {
        let admin_info: AdminInfo = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(candidate.clone()))
            .unwrap_or_else(|| panic_with_error!(e, ContractError::NotAdmin));

        if admin_info.role != AdminRole::SuperAdmin {
            panic_with_error!(e, ContractError::NotAdmin);
        }
        if !admin_info.active {
            panic_with_error!(e, ContractError::AlreadyDeactivated);
        }
        if e.ledger().timestamp() < admin_info.suspended_until {
            panic_with_error!(e, ContractError::AdminSuspended);
        }
    }

    /// Historical role check: verify that `actor` held at least `role` at
    /// ledger timestamp `at_ledger`.
    ///
    /// This is a **defence-in-depth** guard for delegated actions that carry
    /// an off-chain signature produced at a past ledger. Without it an attacker
    /// could:
    ///
    /// 1. Obtain a signature from an address that was *not yet* an admin at
    ///    the time the signature was created.
    /// 2. Wait until that address is later elevated, then replay the signed
    ///    payload to authorise a historical action as if the signer had been
    ///    an admin all along.
    ///
    /// By asserting `admin_info.assigned_at <= at_ledger` we ensure the role
    /// assignment predates (or coincides with) the moment the signature covers.
    ///
    /// # Arguments
    /// * `e`          - Soroban environment
    /// * `role`       - Minimum `AdminRole` that must have been held
    /// * `actor`      - Address whose historical role is checked
    /// * `at_ledger`  - Ledger timestamp of the signed action
    ///
    /// # Errors
    /// * [`ContractError::NotAdmin`] — `actor` is not a known admin at all.
    /// * [`ContractError::RoleNotHeldAtLedger`] — `actor`'s role was assigned
    ///   after `at_ledger`, meaning they were not yet authorised at that time.
    ///
    /// # Panics
    /// Panics via [`panic_with_error!`] — compatible with Soroban's error
    /// propagation model.
    pub fn require_role_at_ledger(e: Env, role: AdminRole, actor: Address, at_ledger: u64) {
        let admin_info: AdminInfo = e
            .storage()
            .instance()
            .get(&DataKey::AdminInfo(actor))
            .unwrap_or_else(|| panic_with_error!(&e, ContractError::NotAdmin));

        // The actor must have the required role level.
        if admin_info.role < role {
            panic_with_error!(&e, ContractError::NotAdmin);
        }

        // The role must have been assigned at or before the ledger under review.
        // If `assigned_at > at_ledger` the actor was not yet an admin when the
        // action was signed, so the authorisation is invalid.
        if admin_info.assigned_at > at_ledger {
            panic_with_error!(&e, ContractError::RoleNotHeldAtLedger);
        }
    }
}

#[cfg(test)]
mod test;

#[cfg(test)]
mod test_require_role_at_ledger;

// Pause mechanism entrypoints
#[contractimpl]
impl AdminContract {
    pub fn is_paused(e: Env) -> bool {
        bump_instance_ttl(&e);
        pausable::is_paused(&e)
    }

    pub fn pause(e: Env, caller: Address) -> Option<u64> {
        bump_instance_ttl(&e);
        pausable::pause(&e, &caller)
    }

    pub fn unpause(e: Env, caller: Address) -> Option<u64> {
        bump_instance_ttl(&e);
        pausable::unpause(&e, &caller)
    }

    pub fn set_pause_signer(e: Env, admin: Address, signer: Address, enabled: bool) {
        bump_instance_ttl(&e);
        pausable::set_pause_signer(&e, &admin, &signer, enabled)
    }

    pub fn set_pause_threshold(e: Env, admin: Address, threshold: u32) {
        bump_instance_ttl(&e);
        pausable::set_pause_threshold(&e, &admin, threshold)
    }

    pub fn approve_pause_proposal(e: Env, signer: Address, proposal_id: u64) {
        bump_instance_ttl(&e);
        pausable::approve_pause_proposal(&e, &signer, proposal_id)
    }

    /// Execute a pause/unpause proposal once approvals meet the threshold.
    ///
    /// # Deterministic failure boundaries
    ///
    /// This function implements comprehensive error boundaries to ensure deterministic
    /// behavior across all input scenarios, state conditions, and concurrent execution:
    ///
    /// ## Input validation
    /// * Rejects zero or invalid proposal IDs before any state reads
    /// * Validates proposal ID bounds to prevent overflow/underflow conditions
    ///
    /// ## State consistency
    /// * Verifies contract initialization and configuration integrity
    /// * Ensures pause threshold and signer configuration consistency
    /// * Validates proposal state integrity before execution
    ///
    /// ## Concurrent execution safety
    /// * Uses monotonic epoch tracking to detect concurrent state mutations
    /// * Implements idempotent execution when proposal state is unchanged
    /// * Prevents double-execution through deterministic state checks
    ///
    /// ## Authorization boundaries
    /// * Validates proposal creation by authorized signers
    /// * Ensures signer approvals remain valid at execution time
    /// * Verifies threshold configuration hasn't been compromised
    ///
    /// ## Error classification
    /// * **Transient errors** (retryable): `StaleAdminEpoch`, `InsufficientApprovals`
    /// * **Permanent errors** (not retryable): `ProposalNotFound`, `InvalidPauseAction`
    /// * **System errors** (require investigation): Configuration inconsistencies
    ///
    /// ## Observability
    /// * Comprehensive event logging for all execution phases
    /// * Diagnostic metrics for performance monitoring
    /// * Error context logging without sensitive data exposure
    /// * State transition tracking for audit trails
    ///
    /// # Arguments
    /// * `proposal_id` - The proposal ID to execute
    ///
    /// # Panics
    /// * `ProposalNotFound` - No proposal exists for the given ID
    /// * `StaleAdminEpoch` - Proposal ID derived from stale epoch (retryable)
    /// * `InsufficientApprovals` - Approval threshold not met (retryable)
    /// * `InvalidPauseAction` - Proposal action value is invalid
    /// * `NotInitialized` - Contract not properly initialized
    /// * `InvalidAdminAddress` - Configuration contains invalid addresses
    /// * `ThresholdExceedsSigners` - Invalid threshold configuration
    ///
    /// # Events
    /// Emits execution attempt events for observability without exposing sensitive data
    pub fn execute_pause_proposal(e: Env, proposal_id: u64) {
        bump_instance_ttl(&e);

        // ── Execution Context Logging ───────────────────────────────────────────
        // Log execution start with context for monitoring and debugging
        e.events().publish(
            (Symbol::new(&e, "pause_proposal_execution_started"),),
            (proposal_id, e.ledger().sequence(), e.ledger().timestamp()),
        );

        // ── Input Validation ─────────────────────────────────────────────────────
        // Reject invalid proposal IDs before any state operations to ensure
        // deterministic failure boundaries independent of storage state
        match Self::validate_proposal_id_with_logging(&e, proposal_id) {
            Ok(()) => {
                e.events().publish(
                    (Symbol::new(&e, "pause_proposal_validation_passed"),),
                    proposal_id,
                );
            }
            Err(error_code) => {
                e.events().publish(
                    (Symbol::new(&e, "pause_proposal_validation_failed"),),
                    (proposal_id, error_code),
                );
                panic_with_error!(&e, ContractError::InvalidPauseAction);
            }
        }

        // ── State Consistency Verification ──────────────────────────────────────
        // Verify contract and configuration integrity before proposal execution
        match Self::validate_pause_configuration_with_logging(&e) {
            Ok(config) => {
                e.events().publish(
                    (Symbol::new(&e, "pause_config_validation_passed"),),
                    (config.threshold, config.signer_count, config.initialized),
                );
            }
            Err(error_code) => {
                e.events().publish(
                    (Symbol::new(&e, "pause_config_validation_failed"),),
                    (proposal_id, error_code),
                );
                // The specific error will be panicked by the validation function
                Self::require_valid_pause_configuration(&e);
            }
        }

        // ── Delegate to Enhanced Pausable Implementation ────────────────────────
        // The pausable module handles the core execution logic with enhanced
        // error boundaries and deterministic state transitions
        pausable::execute_pause_proposal(&e, proposal_id);
        
        // ── Final Success Logging ───────────────────────────────────────────────
        e.events().publish(
            (Symbol::new(&e, "pause_proposal_execution_completed"),),
            (proposal_id, e.ledger().sequence()),
        );
    }

    /// Validate that a proposal ID is within acceptable bounds and not a sentinel value.
    ///
    /// # Deterministic validation
    /// * Rejects zero proposal ID (invalid sentinel value)
    /// * Ensures proposal ID is within reasonable bounds
    /// * Provides consistent error behavior independent of storage state
    ///
    /// # Arguments
    /// * `proposal_id` - The proposal ID to validate
    ///
    /// # Panics
    /// * `InvalidPauseAction` - Proposal ID is zero or invalid
    fn require_valid_proposal_id(e: &Env, proposal_id: u64) {
        if proposal_id == 0 {
            panic_with_error!(e, ContractError::InvalidPauseAction);
        }
        // Additional bounds checking to prevent potential overflow in future operations
        if proposal_id == u64::MAX {
            panic_with_error!(e, ContractError::Overflow);
        }
    }

    /// Validate pause configuration consistency and contract initialization.
    ///
    /// # Deterministic validation
    /// * Ensures contract is properly initialized
    /// * Validates pause threshold doesn't exceed signer count
    /// * Verifies signer configuration integrity
    /// * Checks for configuration inconsistencies that could lead to deadlock
    ///
    /// # Panics
    /// * `NotInitialized` - Contract not initialized
    /// * `ThresholdExceedsSigners` - Invalid threshold configuration
    fn require_valid_pause_configuration(e: &Env) {
        // Verify contract initialization
        let initialized: bool = e
            .storage()
            .instance()
            .get(&DataKey::Initialized)
            .unwrap_or(false);
        if !initialized {
            panic_with_error!(e, ContractError::NotInitialized);
        }

        // Validate pause threshold configuration consistency
        let threshold: u32 = e
            .storage()
            .instance()
            .get(&DataKey::PauseThreshold)
            .unwrap_or(0);
        let signer_count: u32 = e
            .storage()
            .instance()
            .get(&DataKey::PauseSignerCount)
            .unwrap_or(0);

        // Prevent deadlock: threshold must never exceed available signers
        if threshold > 0 && threshold > signer_count {
            panic_with_error!(e, ContractError::ThresholdExceedsSigners);
        }

        // If threshold is configured (> 0), ensure we have signers
        if threshold > 0 && signer_count == 0 {
            panic_with_error!(e, ContractError::ThresholdExceedsSigners);
        }
    }

    /// Validate that an address is not a sentinel or invalid value.
    ///
    /// # Security
    /// Rejects addresses that could cause permanent loss of admin control:
    /// * Zero/invalid address sentinel that cannot be controlled
    /// * Contract's own address which would create circular dependencies
    ///
    /// # Arguments
    /// * `address` - The address to validate
    ///
    /// # Panics
    /// * `InvalidAdminAddress` - Address is invalid or a sentinel value
    fn require_valid_admin_address(e: &Env, address: &Address) {
        // Reject the zero/invalid address sentinel
        if address.to_string() == String::from_str(e, INVALID_ADDRESS_SENTINEL) {
            panic_with_error!(e, ContractError::InvalidAdminAddress);
        }
        // Reject self-reference to prevent circular dependencies
        if *address == e.current_contract_address() {
            panic_with_error!(e, ContractError::InvalidAdminAddress);
        }
    }

    /// Configuration structure for observability
    #[derive(Clone, Debug)]
    struct PauseConfiguration {
        threshold: u32,
        signer_count: u32,
        initialized: bool,
    }

    /// Enhanced proposal ID validation with detailed logging.
    ///
    /// # Arguments
    /// * `proposal_id` - The proposal ID to validate
    ///
    /// # Returns
    /// * `Ok(())` if validation passes
    /// * `Err(error_code)` with diagnostic error code if validation fails
    fn validate_proposal_id_with_logging(e: &Env, proposal_id: u64) -> Result<(), u32> {
        if proposal_id == 0 {
            return Err(1); // Error code 1: zero proposal ID
        }
        if proposal_id == u64::MAX {
            return Err(2); // Error code 2: overflow boundary
        }
        Ok(())
    }

    /// Enhanced configuration validation with detailed logging.
    ///
    /// # Returns
    /// * `Ok(PauseConfiguration)` if validation passes
    /// * `Err(error_code)` with diagnostic error code if validation fails
    fn validate_pause_configuration_with_logging(e: &Env) -> Result<PauseConfiguration, u32> {
        let initialized: bool = e
            .storage()
            .instance()
            .get(&DataKey::Initialized)
            .unwrap_or(false);
        
        if !initialized {
            return Err(1); // Error code 1: not initialized
        }

        let threshold: u32 = e
            .storage()
            .instance()
            .get(&DataKey::PauseThreshold)
            .unwrap_or(0);
        let signer_count: u32 = e
            .storage()
            .instance()
            .get(&DataKey::PauseSignerCount)
            .unwrap_or(0);

        if threshold > 0 && threshold > signer_count {
            return Err(2); // Error code 2: threshold exceeds signers
        }

        if threshold > 0 && signer_count == 0 {
            return Err(3); // Error code 3: threshold set but no signers
        }

        Ok(PauseConfiguration {
            threshold,
            signer_count,
            initialized,
        })
    }
}

#[cfg(test)]
mod test_pausable;

#[cfg(test)]
mod test_unpause_failure_boundaries;

#[cfg(test)]
mod test_pause_failure_boundaries;

/// Deterministic failure-boundary coverage for `set_pause_signer` (issue #1409).
/// Covers: authorization hierarchy, zero-address rejection, self-assignment
/// rejection, idempotency, epoch monotonicity, count/threshold invariants,
/// concurrent execution safety, and full lifecycle regression scenarios.
#[cfg(test)]
mod test_set_pause_signer_boundaries;

#[cfg(test)]
mod test_zero_address_working;

#[cfg(test)]
mod test_admin_epoch_guard;

#[cfg(test)]
mod test_basic;

#[cfg(test)]
mod test_zero_address;

#[cfg(test)]
mod test_zero_address_simple;

#[cfg(test)]
mod test_immutable_config_simple;

#[cfg(test)]
mod test_authorization;

#[cfg(test)]
mod test_set_pause_threshold_boundaries;

#[cfg(test)]
mod test_suspension;

#[cfg(test)]
mod test_deactivate_admin_boundaries;

#[cfg(test)]
mod test_auth_entrypoints;

#[cfg(test)]
mod test_require_role_at_least;

#[cfg(test)]
mod test_emergency;

#[cfg(test)]
mod test_role_events;

#[cfg(test)]
mod test_reactivate_admin_boundaries;

#[cfg(test)]
mod test_concurrency_race_safety;

#[cfg(test)]
mod test_get_admin_info_boundaries;

#[cfg(test)]
mod test_atomic_rollback;

#[cfg(test)]
mod test_required_role_to_assign;

#[cfg(test)]
mod test_require_valid_admin_address;

#[cfg(test)]
mod test_pause_boundary_recovery;
