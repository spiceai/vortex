// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recording the schemes a compression chose, and replaying them on other arrays.
//!
//! Choosing a scheme compresses a sample of the input once per candidate, and every child a
//! winning scheme produces is chosen for the same way. For an input of at most
//! `SAMPLE_SIZE * SAMPLE_COUNT` rows the sample is the whole input, so the search costs one full
//! compression per candidate. When many small arrays are drawn from the same data, such as the
//! row blocks of one column, the choices made for one of them hold for the rest, and replaying
//! them skips the search.

use std::sync::Arc;

use parking_lot::Mutex;
use vortex_utils::aliases::hash_map::HashMap;

use crate::scheme::SchemeId;

/// What [`CascadingCompressor`](crate::CascadingCompressor) decided at one point in a cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// Compress with this scheme.
    Scheme(SchemeId),
    /// No scheme was chosen: keep the canonical array.
    Canonical,
}

/// Whether a [`PlanState`] collects decisions or hands them out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanMode {
    /// Decisions are stored as the compression makes them.
    Record,
    /// Stored decisions replace the search.
    Replay,
}

/// The decisions of one compression, keyed by cascade path (the `(scheme, child index)` chain
/// from the root to the array being compressed).
///
/// A path that was never decided falls back to the full search when replayed, so a plan
/// recorded on one array can be replayed on any other array of the same dtype.
#[derive(Debug, Clone, Default)]
pub struct CompressionPlan {
    /// The decision at each cascade path.
    decisions: HashMap<Vec<(SchemeId, usize)>, Decision>,
}

impl CompressionPlan {
    /// The number of cascade paths this plan decides.
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Whether this plan decides nothing.
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }
}

/// A plan being recorded or replayed, shared by every context in one compression.
#[derive(Debug)]
pub(crate) struct PlanState {
    /// Whether this state records or replays.
    mode: PlanMode,
    /// The decisions recorded so far, or the ones to replay.
    plan: Mutex<CompressionPlan>,
}

impl PlanState {
    /// A state that records the decisions of the compression it is passed to.
    pub(crate) fn recording() -> Arc<Self> {
        Arc::new(Self {
            mode: PlanMode::Record,
            plan: Mutex::new(CompressionPlan::default()),
        })
    }

    /// A state that replays `plan`.
    pub(crate) fn replaying(plan: CompressionPlan) -> Arc<Self> {
        Arc::new(Self {
            mode: PlanMode::Replay,
            plan: Mutex::new(plan),
        })
    }

    /// The recorded decision for `path`, when replaying.
    pub(crate) fn replayed(&self, path: &[(SchemeId, usize)]) -> Option<Decision> {
        if self.mode != PlanMode::Replay {
            return None;
        }
        self.plan.lock().decisions.get(path).copied()
    }

    /// Records `decision` for `path`, when recording.
    pub(crate) fn record(&self, path: &[(SchemeId, usize)], decision: Decision) {
        if self.mode == PlanMode::Record {
            self.plan.lock().decisions.insert(path.to_vec(), decision);
        }
    }

    /// The decisions this state holds.
    pub(crate) fn into_plan(self: Arc<Self>) -> CompressionPlan {
        match Arc::try_unwrap(self) {
            Ok(state) => state.plan.into_inner(),
            Err(shared) => shared.plan.lock().clone(),
        }
    }
}
