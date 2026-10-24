// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Where compression time goes when a wide string column is written in chunks smaller than the
//! default 8,192 rows. Compresses the same ~3 KB JSON column as 8,192-row chunks and as
//! sub-chunks, with the full scheme search and with only the scheme that wins, and times
//! OnPair's dictionary training and parsing on their own.
//!
//! `cargo bench -p vortex-btrblocks --bench small_chunk_compress`
//! Env: `SCC_ROWS` (default 98,304), `SCC_REPS` (default 5).
//!
//! It measures CPU time with Unix APIs, so on other platforms it only says so.

#[cfg(unix)]
mod unix {
    #![expect(clippy::unwrap_used)]
    #![expect(clippy::cast_precision_loss)]

    use std::sync::LazyLock;

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_btrblocks::BtrBlocksCompressor;
    use vortex_btrblocks::BtrBlocksCompressorBuilder;
    use vortex_btrblocks::SchemeExt;
    use vortex_btrblocks::schemes::string;
    use vortex_session::VortexSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    /// Deterministic xorshift so every run compresses identical data.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// A ~3 KB JSON document per row, like a cached HTTP response body.
    fn json_row(rng: &mut Rng, row: usize) -> String {
        let mut out = String::with_capacity(3200);
        out.push_str(&format!("{{\"id\":{row},\"items\":["));
        let mut first = true;
        while out.len() < 3000 {
            if !first {
                out.push(',');
            }
            first = false;
            let value = rng.next();
            out.push_str(&format!(
                "{{\"sku\":\"SKU-{:08x}\",\"price\":{}.{:02},\"qty\":{},\"name\":\"item {:x}\",\"tags\":[\"t{}\",\"t{}\"]}}",
                value >> 32,
                (value >> 32) % 1000,
                (value >> 40) % 100,
                (value >> 48) % 50,
                value >> 20,
                (value >> 8) % 17,
                (value >> 12) % 23
            ));
        }
        out.push_str("]}");
        out
    }

    fn compress_chunks(compressor: &BtrBlocksCompressor, array: &ArrayRef, rows: usize) -> u64 {
        let mut ctx = SESSION.create_execution_ctx();
        let mut bytes = 0;
        let mut start = 0;
        while start < array.len() {
            let end = (start + rows).min(array.len());
            let chunk = array.slice(start..end).unwrap();
            bytes += compressor.compress(&chunk, &mut ctx).unwrap().nbytes();
            start = end;
        }
        bytes
    }

    /// CPU time of the calling thread, in milliseconds. Everything timed here runs on one
    /// thread, so this is far less sensitive to other load on the machine than wall time.
    fn thread_cpu_ms() -> f64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `ts` is a valid out-pointer.
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &raw mut ts) };
        ts.tv_sec as f64 * 1e3 + ts.tv_nsec as f64 / 1e6
    }

    /// Exclusive (self) thread-CPU time per span, keyed by span name and its scheme/kind field.
    mod self_time {
        use std::cell::RefCell;
        use std::collections::BTreeMap;
        use std::fmt::Debug;

        use tracing::Subscriber;
        use tracing::field::Field;
        use tracing::field::Visit;
        use tracing::span::Attributes;
        use tracing::span::Id;
        use tracing_subscriber::Layer;
        use tracing_subscriber::layer::Context;
        use tracing_subscriber::registry::LookupSpan;

        // Everything this bench times runs on one thread.
        thread_local! {
            static TOTALS: RefCell<BTreeMap<String, (f64, u64)>> = const { RefCell::new(BTreeMap::new()) };
            static STACK: RefCell<Vec<(f64, f64)>> = const { RefCell::new(Vec::new()) };
        }

        struct Key(String);
        struct KeyVisitor<'a>(&'a mut String);
        impl Visit for KeyVisitor<'_> {
            fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
                if matches!(
                    field.name(),
                    "scheme_candidate" | "scheme_chosen" | "stats_kind"
                ) {
                    self.0.push_str(&format!("[{value:?}]"));
                }
            }
        }

        pub struct SelfTime;
        impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for SelfTime {
            fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
                let mut key = attrs.metadata().name().to_string();
                attrs.record(&mut KeyVisitor(&mut key));
                if let Some(span) = ctx.span(id) {
                    span.extensions_mut().insert(Key(key));
                }
            }
            fn on_enter(&self, _id: &Id, _ctx: Context<'_, S>) {
                STACK.with(|s| s.borrow_mut().push((super::thread_cpu_ms(), 0.0)));
            }
            fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
                let Some((start, children)) = STACK.with(|s| s.borrow_mut().pop()) else {
                    return;
                };
                let total = super::thread_cpu_ms() - start;
                STACK.with(|s| {
                    if let Some(parent) = s.borrow_mut().last_mut() {
                        parent.1 += total;
                    }
                });
                if let Some(span) = ctx.span(id)
                    && let Some(Key(key)) = span.extensions().get::<Key>()
                {
                    TOTALS.with(|totals| {
                        let mut totals = totals.borrow_mut();
                        let entry = totals.entry(key.clone()).or_insert((0.0, 0));
                        entry.0 += total - children;
                        entry.1 += 1;
                    });
                }
            }
        }

        pub fn take() -> BTreeMap<String, (f64, u64)> {
            TOTALS.with(|totals| std::mem::take(&mut *totals.borrow_mut()))
        }
    }

    fn compress_planned(
        compressor: &BtrBlocksCompressor,
        array: &ArrayRef,
        rows: usize,
        chunk_rows: usize,
        from_sample: bool,
    ) -> u64 {
        let mut ctx = SESSION.create_execution_ctx();
        let mut bytes = 0;
        for block in (0..rows).step_by(8192) {
            let block_end = (block + 8192).min(rows);
            // The writer's planning: schemes chosen on the sample `compress` would draw from
            // the whole block.
            let mut plan = from_sample.then(|| {
                let whole = array.slice(block..block_end).unwrap();
                compressor.plan_for(&whole, &mut ctx).unwrap()
            });
            for start in (block..block_end).step_by(chunk_rows) {
                let end = (start + chunk_rows).min(block_end);
                let chunk = array.slice(start..end).unwrap();
                let compressed = match &plan {
                    None => {
                        let (c, p) = compressor.compress_recording(&chunk, &mut ctx).unwrap();
                        plan = Some(p);
                        c
                    }
                    Some(p) => compressor.compress_with_plan(&chunk, p, &mut ctx).unwrap(),
                };
                bytes += compressed.nbytes();
            }
        }
        bytes
    }

    pub fn main() {
        if std::env::var_os("SCC_SPANS").is_some() {
            use tracing_subscriber::layer::SubscriberExt;
            let subscriber = tracing_subscriber::registry().with(self_time::SelfTime);
            tracing::subscriber::set_global_default(subscriber).unwrap();
        }
        let rows: usize = std::env::var("SCC_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(98_304);
        let reps: usize = std::env::var("SCC_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let strings: Vec<String> = (0..rows).map(|i| json_row(&mut rng, i)).collect();
        let raw_bytes: usize = strings.iter().map(String::len).sum();
        let array =
            &VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();

        let full = &BtrBlocksCompressor::default();
        let onpair_only = &BtrBlocksCompressorBuilder::default()
            .exclude_schemes([string::FSSTScheme.id(), string::StringDictScheme.id()])
            .build();

        type Arm<'a> = (String, Box<dyn FnMut() -> u64 + 'a>);
        let mut arms: Vec<Arm<'_>> = vec![(
            "default: full search, 8192 rows".to_string(),
            Box::new(move || compress_chunks(full, array, 8192)),
        )];
        for chunk_rows in [1024_usize, 339] {
            arms.push((
                format!("full search, {chunk_rows} rows"),
                Box::new(move || compress_chunks(full, array, chunk_rows)),
            ));
            arms.push((
                format!("onpair only, {chunk_rows} rows"),
                Box::new(move || compress_chunks(onpair_only, array, chunk_rows)),
            ));
            arms.push((
                format!("plan P1 first sub-chunk, {chunk_rows} rows"),
                Box::new(move || compress_planned(full, array, rows, chunk_rows, false)),
            ));
            arms.push((
                format!("plan_for on the whole block, {chunk_rows} rows"),
                Box::new(move || compress_planned(full, array, rows, chunk_rows, true)),
            ));
        }

        // Interleaved: every rep runs each arm once, so load on the machine falls on all arms alike.
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        let mut outs = vec![0_u64; arms.len()];
        let mut spans: Vec<std::collections::BTreeMap<String, (f64, u64)>> =
            vec![Default::default(); arms.len()];
        for _ in 0..reps {
            for (i, (_, arm)) in arms.iter_mut().enumerate() {
                self_time::take();
                let t = thread_cpu_ms();
                outs[i] = arm();
                times[i].push(thread_cpu_ms() - t);
                for (k, (ms, n)) in self_time::take() {
                    let e = spans[i].entry(k).or_insert((0.0, 0));
                    e.0 += ms;
                    e.1 += n;
                }
            }
        }
        println!(
            "rows={rows} raw={:.1} MiB reps={reps} interleaved (thread CPU ms)",
            raw_bytes as f64 / f64::from(1 << 20)
        );
        println!(
            "{:<36} {:>9} {:>9} {:>9} {:>10}",
            "arm", "median", "min", "max", "out_MiB"
        );
        for (i, (label, _)) in arms.iter().enumerate() {
            let t = &mut times[i];
            t.sort_by(f64::total_cmp);
            println!(
                "{:<36} {:>9.1} {:>9.1} {:>9.1} {:>10.2}",
                label,
                t[t.len() / 2],
                t[0],
                t[t.len() - 1],
                outs[i] as f64 / f64::from(1 << 20)
            );
            let mut s: Vec<_> = spans[i].iter().collect();
            s.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
            for (key, (ms, n)) in s.iter().take(8) {
                println!(
                    "    {:>9.1} ms/rep {:>8} calls/rep  {key}",
                    ms / reps as f64,
                    *n / reps as u64
                );
            }
        }
    }
}

fn main() {
    #[cfg(unix)]
    unix::main();
    #[cfg(not(unix))]
    eprintln!("this benchmark measures CPU time with Unix APIs; it does not run on this platform");
}
