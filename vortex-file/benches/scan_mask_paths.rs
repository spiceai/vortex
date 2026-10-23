// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Read-path cost of scans whose row masks are known up front (a row selection) or computed (a
//! filter), over a multi-chunk file of narrow columns. Run against two builds to compare them.
//!
//! `cargo bench -p vortex-file --bench scan_mask_paths`; env `SMP_ROWS` (default 4,194,304),
//! `SMP_REPS` (default 9).
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

    use std::sync::LazyLock;
    use std::time::Instant;

    use futures::StreamExt;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::FieldNames;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::root;
    use vortex_array::expr::select;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_file::OpenOptionsSessionExt;
    use vortex_file::VortexFile;
    use vortex_file::WriteOptionsSessionExt;
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

    fn table(rows: usize) -> ArrayRef {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let k = PrimitiveArray::from_iter(0..rows as i64).into_array();
        let a =
            PrimitiveArray::from_iter((0..rows).map(|_| (rng.next() % 1_000_000) as i64 - 500_000))
                .into_array();
        let b = PrimitiveArray::from_iter((0..rows).map(|_| (rng.next() % 100_000) as f64 / 100.0))
            .into_array();
        let d = PrimitiveArray::from_iter((0..rows).map(|i| (i / 3000) as i32)).into_array();
        let s = VarBinViewArray::from_iter_str(
            (0..rows).map(|i| format!("dist-info-{:014}", i % 97_003)),
        )
        .into_array();
        StructArray::new(
            FieldNames::from(["k", "a", "b", "d", "s"]),
            vec![k, a, b, d, s],
            rows,
            Validity::NonNullable,
        )
        .into_array()
    }

    async fn scan(
        file: &VortexFile,
        projection: Expression,
        filter: Option<Expression>,
        selection: Option<Selection>,
    ) -> (usize, u64) {
        let dtype = file.dtype().clone();
        let mut builder = file
            .scan()
            .unwrap()
            .with_projection(projection.bind(&dtype).unwrap());
        if let Some(f) = filter {
            builder = builder.with_filter(f.bind(&dtype).unwrap());
        }
        if let Some(s) = selection {
            builder = builder.with_selection(s);
        }
        let mut stream = builder.into_array_stream().unwrap();
        let mut ctx = SESSION.create_execution_ctx();
        let (mut rows, mut bytes) = (0, 0);
        while let Some(chunk) = stream.next().await {
            let chunk = chunk
                .unwrap()
                .execute::<vortex_array::Canonical>(&mut ctx)
                .unwrap()
                .into_array();
            rows += chunk.len();
            bytes += chunk.nbytes();
        }
        (rows, bytes)
    }

    pub fn main() {
        let rows: usize = std::env::var("SMP_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4_194_304);
        let reps: usize = std::env::var("SMP_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(9);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        let file = rt.block_on(async {
            let mut bytes = Vec::new();
            SESSION
                .write_options()
                .write(&mut bytes, table(rows).to_array_stream())
                .await
                .unwrap();
            let digest = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
                (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
            });
            println!(
                "rows={rows} file_bytes={} fnv64={digest:016x} reps={reps}",
                bytes.len()
            );
            SESSION
                .open_options()
                .open_buffer(ByteBuffer::from(bytes))
                .unwrap()
        });

        // A deletion-style selection: exclude 1% of rows, spread across every chunk.
        let excluded: Buffer<u64> = (0..rows as u64).step_by(100).collect();
        let point_rows: Vec<u64> = {
            let mut rng = Rng(42);
            (0..200).map(|_| rng.next() % rows as u64).collect()
        };
        let lo = (rows as i64) / 2;
        let hi = lo + (rows as i64) / 10;
        type Case<'a> = (&'a str, Box<dyn Fn() -> (usize, u64) + 'a>);
        let cases: Vec<Case<'_>> = vec![
            (
                "full projection a,b,s",
                Box::new(|| rt.block_on(scan(&file, select(["a", "b", "s"], root()), None, None))),
            ),
            (
                "filter 10% range, project a,b,s",
                Box::new(|| {
                    rt.block_on(scan(
                        &file,
                        select(["a", "b", "s"], root()),
                        Some(and(
                            gt(get_item("k", root()), lit(lo)),
                            lt(get_item("k", root()), lit(hi)),
                        )),
                        None,
                    ))
                }),
            ),
            (
                "exclude 1% + filter a>0, project b,s",
                Box::new(|| {
                    rt.block_on(scan(
                        &file,
                        select(["b", "s"], root()),
                        Some(gt(get_item("a", root()), lit(0_i64))),
                        Some(Selection::ExcludeByIndex(
                            StrictSortedBuffer::try_new(excluded.clone()).unwrap(),
                        )),
                    ))
                }),
            ),
            (
                "200 one-row point reads, project s",
                Box::new(|| {
                    let mut total = (0, 0);
                    for &row in &point_rows {
                        let (r, b) = rt.block_on(scan(
                            &file,
                            select(["s"], root()),
                            None,
                            Some(Selection::IncludeByIndex(
                                StrictSortedBuffer::try_new(Buffer::from_iter([row])).unwrap(),
                            )),
                        ));
                        total = (total.0 + r, total.1 + b);
                    }
                    total
                }),
            ),
        ];
        for (_, case) in &cases {
            case(); // warm-up
        }
        let mut cpu: Vec<Vec<f64>> = vec![Vec::new(); cases.len()];
        let mut wall: Vec<Vec<f64>> = vec![Vec::new(); cases.len()];
        let mut outs = vec![(0, 0); cases.len()];
        for _ in 0..reps {
            for (i, (_, case)) in cases.iter().enumerate() {
                let (c0, w0) = (cpu_ms(), Instant::now());
                outs[i] = case();
                cpu[i].push(cpu_ms() - c0);
                wall[i].push(w0.elapsed().as_secs_f64() * 1e3);
            }
        }
        println!(
            "{:<40} {:>10} {:>10} {:>10} {:>10} {:>9} {:>12}",
            "case", "cpu_med", "cpu_min", "cpu_max", "wall_med", "rows", "bytes"
        );
        for (i, (name, _)) in cases.iter().enumerate() {
            cpu[i].sort_by(f64::total_cmp);
            wall[i].sort_by(f64::total_cmp);
            let m = reps / 2;
            println!(
                "{name:<40} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>9} {:>12}",
                cpu[i][m],
                cpu[i][0],
                cpu[i][reps - 1],
                wall[i][m],
                outs[i].0,
                outs[i].1
            );
        }
    }
}

fn main() {
    #[cfg(unix)]
    unix::main();
    #[cfg(not(unix))]
    eprintln!("this benchmark measures CPU time with Unix APIs; it does not run on this platform");
}
