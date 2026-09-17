// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bounded memoization for expressions evaluated by reusable layout readers.

use std::convert::Infallible;

use parking_lot::Mutex;
use vortex_array::expr::ExactExpr;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::fns::dynamic::DynamicComparison;
use vortex_utils::aliases::hash_map::HashMap;

const MAX_ENTRIES: usize = 32;

/// An entry-bounded cache that never retains dynamic expression state.
///
/// A full cache is cleared on insertion. Initializers and evicted-value destructors run
/// outside its lock; concurrent misses may initialize the same key more than once.
/// The entry bound does not account for array buffers retained by cached values.
pub struct ExpressionCache<V> {
    entries: Mutex<HashMap<ExactExpr, V>>,
}

impl<V> Default for ExpressionCache<V> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::default()),
        }
    }
}

impl<V: Clone> ExpressionCache<V> {
    /// Return a cached value or initialize it, without caching dynamic expressions.
    pub fn get_or_insert_with(&self, expr: &Expression, init: impl FnOnce() -> V) -> V {
        match self.get_or_try_insert_with(expr, || Ok::<_, Infallible>(init())) {
            Ok(value) => value,
            Err(never) => match never {},
        }
    }

    /// Return a cached value or initialize it, without caching errors or dynamic expressions.
    pub fn get_or_try_insert_with<E>(
        &self,
        expr: &Expression,
        init: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        if contains_dynamic(expr) {
            return init();
        }
        let key = ExactExpr(expr.clone());
        if let Some(value) = self.entries.lock().get(&key).cloned() {
            return Ok(value);
        }
        let value = init()?;
        let mut entries = self.entries.lock();
        if let Some(existing) = entries.get(&key) {
            return Ok(existing.clone());
        }
        let retired = (entries.len() >= MAX_ENTRIES).then(|| std::mem::take(&mut *entries));
        entries.insert(key, value.clone());
        drop(entries);
        drop(retired);
        Ok(value)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.lock().len()
    }
}

fn contains_dynamic(expr: &Expression) -> bool {
    expr.is::<DynamicComparison>() || expr.children().iter().any(contains_dynamic)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::expr::lit;

    use super::*;

    #[test]
    fn concurrent_insertions_stay_bounded() {
        let cache = ExpressionCache::default();
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let cache = &cache;
                scope.spawn(move || {
                    for value in 0..1_000i32 {
                        let key = lit(worker * 1_000 + value);
                        assert_eq!(cache.get_or_insert_with(&key, || value), value);
                        assert!(cache.len() <= MAX_ENTRIES);
                    }
                });
            }
        });
    }

    #[test]
    fn eviction_releases_cached_values() {
        let cache = ExpressionCache::default();
        let value = Arc::new(0);
        let weak = Arc::downgrade(&value);
        cache.get_or_insert_with(&lit(0i32), || value);
        assert!(weak.upgrade().is_some());
        for key in 1..=MAX_ENTRIES {
            cache.get_or_insert_with(&lit(key as u64), || Arc::new(key));
        }
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn failed_initialization_is_retried() {
        let cache = ExpressionCache::default();
        let key = lit(1i32);
        assert_eq!(
            cache.get_or_try_insert_with(&key, || Err::<i32, _>("error")),
            Err("error")
        );
        assert_eq!(
            cache.get_or_try_insert_with(&key, || Ok::<_, &str>(7)),
            Ok(7)
        );
        assert_eq!(cache.get_or_insert_with(&key, || 8), 7);
    }
}
