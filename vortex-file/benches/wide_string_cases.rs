// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cases where a data-block byte maximum could cost more than it saves, each written with and
//! without the maximum in one process and read with the arms interleaved:
//!
//! - `json3k`: ~3 KB JSON rows; full scans, a `LIKE` filter over the wide column, a range filter
//!   projecting it, and one-row point reads.
//! - `shift_lo_hi` / `shift_hi_lo`: each 8,192-row block starts (or ends) with 1,024 rows drawn
//!   from 8 distinct values and is otherwise unique JSON. The writer dictionary-encodes this
//!   column, so the maximum does not apply; kept as a control.
//! - `rand_then_json` / `json_then_rand`: each 8,192-row block starts (or ends) with 1,024
//!   incompressible rows and is otherwise JSON, so the encodings chosen on a block's first
//!   chunk do not fit the rest of it.
//! - `wide100k` / `wide1m`: 100 KB and 1 MB rows.
//!
//! Reads go through a counting reader that coalesces like an object store (1 MiB gap, 16 MiB
//! max), so the request counts are what an object-store tier would issue.
//!
//! `cargo bench -p vortex-file --bench wide_string_cases`; env `WSC_CASES` (comma-separated
//! dataset names, default all), `WSC_REPS` (scan repetitions per arm, default 11),
//! `WSC_POINTS` (point reads per repetition, default 1,000), `WSC_MAX_MB` (default 4).
//!
//! It measures CPU time with Unix APIs, so on other platforms it only says so.

#[cfg(unix)]
#[path = "../tests/common/mod.rs"]
mod common;

#[cfg(unix)]
mod unix {
    #![expect(clippy::unwrap_used)]
    #![expect(clippy::cast_precision_loss)]
    #![expect(clippy::cast_possible_truncation)]
    #![expect(clippy::cast_possible_wrap)]
    #![expect(clippy::many_single_char_names)]
    #![expect(clippy::panic, reason = "an unknown dataset name is a usage error")]

    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    use futures::FutureExt;
    use futures::StreamExt;
    use futures::future::BoxFuture;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::dtype::FieldNames;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::like;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::root;
    use vortex_array::validity::Validity;
    use vortex_buffer::Alignment;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_file::OpenOptionsSessionExt;
    use vortex_file::VortexFile;
    use vortex_file::WriteOptionsSessionExt;
    use vortex_file::WriteStrategyBuilder;
    use vortex_io::CoalesceConfig;
    use vortex_io::VortexReadAt;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::session::LayoutSession;
    use vortex_scan::selection::Selection;
    use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
    use vortex_session::VortexSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();
        vortex_file::register_default_encodings(&session);
        super::common::enable_all_registered_array_encodings(&session);
        session
    });

    const NEEDLE: &str = "needle-7f3a9c";

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn cpu_ms() -> f64 {
        // SAFETY: getrusage fills the zeroed struct.
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: valid pointer to a rusage.
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut ru) };
        let tv = |t: libc::timeval| t.tv_sec as f64 * 1e3 + t.tv_usec as f64 / 1e3;
        tv(ru.ru_utime) + tv(ru.ru_stime)
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    const WORDS: [&str; 16] = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    ];

    /// A JSON document of about `target` bytes: repeated objects of words and numbers, so it
    /// compresses like real documents rather than like random bytes or a single repeated string.
    fn json(rng: &mut Rng, i: usize, target: usize) -> String {
        let mut s = format!("{{\"id\":{i},\"items\":[");
        while s.len() < target {
            let w = WORDS[(rng.next() % 16) as usize];
            let v = WORDS[(rng.next() % 16) as usize];
            s.push_str(&format!(
                "{{\"k\":\"{w}-{}\",\"n\":{},\"t\":\"{v} {w} {}\"}},",
                rng.next() % 10_000,
                rng.next() % 1_000_000,
                rng.next() % 97
            ));
        }
        s.push_str("]}");
        s
    }

    struct Dataset {
        name: &'static str,
        rows: Vec<String>,
    }

    fn dataset(name: &'static str) -> Dataset {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ name.len() as u64);
        let needle = |i: usize, s: String| {
            if i.is_multiple_of(997) {
                format!("{s}{NEEDLE}")
            } else {
                s
            }
        };
        let rows: Vec<String> = match name {
            "json3k" => (0..98_304)
                .map(|i| needle(i, json(&mut rng, i, 3_000)))
                .collect(),
            "shift_lo_hi" | "shift_hi_lo" => {
                let fixed: Vec<String> = (0..8).map(|i| json(&mut Rng(7 + i), 0, 3_000)).collect();
                (0..65_536)
                    .map(|i| {
                        let in_block = i % 8_192;
                        let low = if name == "shift_lo_hi" {
                            in_block < 1_024
                        } else {
                            in_block >= 7_168
                        };
                        let s = if low {
                            fixed[i % 8].clone()
                        } else {
                            json(&mut rng, i, 3_000)
                        };
                        needle(i, s)
                    })
                    .collect()
            }
            // Incompressible rows (hex of random words) for 1,024 rows of each 8,192-row block
            // and JSON for the rest, in either order. 3 KB rows under a 4 MiB maximum split into
            // 1,024-row chunks, so the encodings are chosen on exactly the odd-one-out part (or on
            // the JSON, and replayed on it).
            "rand_then_json" | "json_then_rand" => (0..65_536)
                .map(|i| {
                    let in_block = i % 8_192;
                    let odd = if name == "rand_then_json" {
                        in_block < 1_024
                    } else {
                        in_block >= 7_168
                    };
                    let s = if odd {
                        (0..188)
                            .map(|_| format!("{:016x}", rng.next()))
                            .collect::<String>()
                    } else {
                        json(&mut rng, i, 3_000)
                    };
                    needle(i, s)
                })
                .collect(),
            "wide100k" => (0..2_048)
                .map(|i| needle(i, json(&mut rng, i, 100_000)))
                .collect(),
            "wide1m" => (0..256)
                .map(|i| needle(i, json(&mut rng, i, 1_000_000)))
                .collect(),
            other => panic!("unknown dataset {other}"),
        };
        Dataset { name, rows }
    }

    fn table(rows: &[String]) -> ArrayRef {
        let ids = PrimitiveArray::from_iter(0..rows.len() as i64).into_array();
        let wide = VarBinViewArray::from_iter_str(rows.iter().map(String::as_str)).into_array();
        StructArray::new(
            FieldNames::from(["id", "wide"]),
            vec![ids, wide],
            rows.len(),
            Validity::NonNullable,
        )
        .into_array()
    }

    /// A reader over an in-memory file that coalesces like an object store and counts what it is
    /// asked for.
    #[derive(Clone)]
    struct CountingReadAt {
        inner: ByteBuffer,
        requests: Arc<AtomicU64>,
        bytes: Arc<AtomicU64>,
    }

    impl CountingReadAt {
        fn take(&self) -> (u64, u64) {
            (
                self.requests.swap(0, Ordering::Relaxed),
                self.bytes.swap(0, Ordering::Relaxed),
            )
        }
    }

    impl VortexReadAt for CountingReadAt {
        fn coalesce_config(&self) -> Option<CoalesceConfig> {
            Some(CoalesceConfig::object_storage())
        }

        fn concurrency(&self) -> usize {
            16
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            let len = self.inner.len() as u64;
            async move { Ok(len) }.boxed()
        }

        fn read_at(
            &self,
            offset: u64,
            length: usize,
            alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            self.requests.fetch_add(1, Ordering::Relaxed);
            self.bytes.fetch_add(length as u64, Ordering::Relaxed);
            self.inner.read_at(offset, length, alignment)
        }
    }

    struct Arm {
        label: &'static str,
        file: VortexFile,
        reader: CountingReadAt,
    }

    async fn write(data: &ArrayRef, max: Option<u64>) -> Vec<u8> {
        let strategy = WriteStrategyBuilder::default()
            .with_data_block_max_bytes(max)
            .build();
        let mut bytes = Vec::new();
        SESSION
            .write_options()
            .with_strategy(strategy)
            .write(&mut bytes, data.to_array_stream())
            .await
            .unwrap();
        bytes
    }

    /// Runs a scan; returns (rows, a hash of every output value) so the arms can be checked equal.
    async fn scan(
        file: &VortexFile,
        projection: &Expression,
        filter: Option<&Expression>,
        selection: Option<Selection>,
        hash: bool,
    ) -> (usize, u64) {
        let dtype = file.dtype().clone();
        let mut builder = file
            .scan()
            .unwrap()
            .with_projection(projection.clone().bind(&dtype).unwrap());
        if let Some(f) = filter {
            builder = builder.with_filter(f.clone().bind(&dtype).unwrap());
        }
        if let Some(s) = selection {
            builder = builder.with_selection(s);
        }
        let mut stream = builder.into_array_stream().unwrap();
        let mut ctx = SESSION.create_execution_ctx();
        let (mut rows, mut h) = (0, 0xcbf2_9ce4_8422_2325_u64);
        while let Some(chunk) = stream.next().await {
            let canonical = chunk.unwrap().execute::<Canonical>(&mut ctx).unwrap();
            rows += canonical.len();
            if hash {
                match &canonical {
                    Canonical::VarBinView(v) => {
                        for i in 0..v.len() {
                            for b in v.bytes_at(i).as_slice() {
                                h = (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3);
                            }
                        }
                    }
                    other => {
                        let ids = other.clone().into_primitive();
                        for v in ids.as_slice::<i64>() {
                            h = (h ^ *v as u64).wrapping_mul(0x100_0000_01b3);
                        }
                    }
                }
            }
        }
        (rows, h)
    }

    fn summarize(samples: &mut [f64]) -> String {
        samples.sort_by(f64::total_cmp);
        let n = samples.len();
        format!(
            "med {:8.2}  min {:8.2}  max {:8.2}",
            samples[n / 2],
            samples[0],
            samples[n - 1]
        )
    }

    fn tail(samples: &mut [f64]) -> String {
        samples.sort_by(f64::total_cmp);
        let n = samples.len();
        let p = |q: f64| samples[((n as f64 * q) as usize).min(n - 1)];
        let p999 = if n >= 10_000 {
            format!("{:7.3}", p(0.999))
        } else {
            "   n/a".to_owned()
        };
        format!(
            "n={n} p50 {:7.3} p99 {:7.3} p99.9 {p999} ms",
            p(0.5),
            p(0.99)
        )
    }

    pub fn main() {
        let reps = env_usize("WSC_REPS", 11);
        let points = env_usize("WSC_POINTS", 1_000);
        let max = env_usize("WSC_MAX_MB", 4) as u64 * 1024 * 1024;
        let cases = std::env::var("WSC_CASES")
            .unwrap_or_else(|_| "json3k,shift_lo_hi,shift_hi_lo,wide100k,wide1m".to_owned());
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(8)
            .enable_all()
            .build()
            .unwrap();
        println!("reps={reps} points={points} max_bytes={max}");

        for name in cases.split(',') {
            let ds = dataset(match name {
                "json3k" => "json3k",
                "shift_lo_hi" => "shift_lo_hi",
                "shift_hi_lo" => "shift_hi_lo",
                "rand_then_json" => "rand_then_json",
                "json_then_rand" => "json_then_rand",
                "wide100k" => "wide100k",
                "wide1m" => "wide1m",
                other => panic!("unknown dataset {other}"),
            });
            let data = table(&ds.rows);
            let n = ds.rows.len();
            let raw: usize = ds.rows.iter().map(String::len).sum();
            drop(ds.rows);
            println!(
                "\n### {} rows={n} raw={:.1}MiB",
                ds.name,
                raw as f64 / 1048576.0
            );

            // Writes, interleaved A B B A, three of each.
            let mut write_cpu = [Vec::new(), Vec::new()];
            let mut files: [Option<Vec<u8>>; 2] = [None, None];
            for arm in [0, 1, 1, 0, 0, 1, 1, 0, 0, 1] {
                let c0 = cpu_ms();
                let bytes = rt.block_on(write(&data, (arm == 1).then_some(max)));
                write_cpu[arm].push(cpu_ms() - c0);
                files[arm] = Some(bytes);
            }
            drop(data);
            let arms: Vec<Arm> = files
                .into_iter()
                .enumerate()
                .map(|(i, bytes)| {
                    let bytes = bytes.unwrap();
                    let reader = CountingReadAt {
                        inner: ByteBuffer::from(bytes),
                        requests: Arc::default(),
                        bytes: Arc::default(),
                    };
                    let file = rt
                        .block_on(SESSION.open_options().open_read(reader.clone()))
                        .unwrap();
                    Arm {
                        label: if i == 0 { "no max" } else { "max" },
                        file,
                        reader,
                    }
                })
                .collect();
            for (i, arm) in arms.iter().enumerate() {
                println!(
                    "{:7} write_cpu_ms {}  file {:.2}MiB  segments {}",
                    arm.label,
                    summarize(&mut write_cpu[i].clone()),
                    arm.reader.inner.len() as f64 / 1048576.0,
                    arm.file.footer().segment_map().len()
                );
            }

            let wide = get_item("wide", root());
            let id = get_item("id", root());
            let lo = (n as i64) / 2;
            let hi = lo + (n as i64) / 10;
            let range = and(gt(id.clone(), lit(lo)), lt(id.clone(), lit(hi)));
            let pattern = like(wide.clone(), lit(format!("%{NEEDLE}%")));
            let scans: [(&str, &Expression, Option<&Expression>); 3] = [
                ("full scan of wide", &wide, None),
                ("LIKE on wide, project id", &id, Some(&pattern)),
                ("10% id range, project wide", &wide, Some(&range)),
            ];

            for (label, projection, filter) in scans {
                // Untimed check that both arms return the same values, and the request counts.
                let mut seen = Vec::new();
                for arm in &arms {
                    arm.reader.take();
                    let out = rt.block_on(scan(&arm.file, projection, filter, None, true));
                    seen.push((out, arm.reader.take()));
                }
                assert_eq!(
                    seen[0].0, seen[1].0,
                    "{label}: the arms returned different values"
                );
                let mut cpu = [Vec::new(), Vec::new()];
                let mut wall = [Vec::new(), Vec::new()];
                for _ in 0..reps {
                    for i in [0, 1, 1, 0] {
                        let (c0, w0) = (cpu_ms(), Instant::now());
                        rt.block_on(scan(&arms[i].file, projection, filter, None, false));
                        cpu[i].push(cpu_ms() - c0);
                        wall[i].push(w0.elapsed().as_secs_f64() * 1e3);
                    }
                }
                println!("  {label}  (rows {})", seen[0].0.0);
                for i in 0..2 {
                    println!(
                        "    {:7} cpu_ms {}  wall_ms {}  requests {:6}  bytes {:.2}MiB",
                        arms[i].label,
                        summarize(&mut cpu[i]),
                        summarize(&mut wall[i]),
                        seen[i].1.0,
                        seen[i].1.1 as f64 / 1048576.0
                    );
                }
            }

            // One-row point reads, arms interleaved per batch.
            let rows_to_read: Vec<u64> = {
                let mut rng = Rng(42);
                (0..points).map(|_| rng.next() % n as u64).collect()
            };
            let mut lat = [Vec::new(), Vec::new()];
            let mut cpu = [0.0, 0.0];
            let mut reqs = [(0, 0), (0, 0)];
            let batches = 10;
            for b in 0..batches {
                for i in if b % 2 == 0 { [0, 1] } else { [1, 0] } {
                    arms[i].reader.take();
                    let c0 = cpu_ms();
                    for &row in &rows_to_read {
                        let w0 = Instant::now();
                        let sel = Selection::IncludeByIndex(
                            StrictSortedBuffer::try_new(Buffer::from_iter([row])).unwrap(),
                        );
                        let (r, _) =
                            rt.block_on(scan(&arms[i].file, &wide, None, Some(sel), false));
                        assert_eq!(r, 1);
                        lat[i].push(w0.elapsed().as_secs_f64() * 1e3);
                    }
                    cpu[i] += cpu_ms() - c0;
                    let (q, by) = arms[i].reader.take();
                    reqs[i] = (reqs[i].0 + q, reqs[i].1 + by);
                }
            }
            println!("  one-row point reads of wide");
            let total = (points * batches) as f64;
            for i in 0..2 {
                println!(
                    "    {:7} cpu_ms/read {:.3}  wall {}  requests/read {:.2}  bytes/read {:.0}",
                    arms[i].label,
                    cpu[i] / total,
                    tail(&mut lat[i]),
                    reqs[i].0 as f64 / total,
                    reqs[i].1 as f64 / total
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
