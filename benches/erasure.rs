use constellation_parameterised_encoder::{demo_pslice, simulate_loss, Encoder, ErasureParams};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::SeedableRng;

fn param_sets() -> Vec<(&'static str, ErasureParams)> {
    vec![
        ("agave_32_32", ErasureParams::agave_default()),
        ("constellation_64", ErasureParams::constellation(64)),
        ("constellation_128", ErasureParams::constellation(128)),
        ("constellation_256", ErasureParams::constellation(256)),
    ]
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    for (label, params) in param_sets() {
        for payload_size in [1024usize, 8192, 65536] {
            let encoder = Encoder::new(params);
            let pslice = demo_pslice(payload_size);

            for _ in 0..10 {
                let _ = encoder.encode(&pslice).unwrap();
            }

            group.throughput(Throughput::Bytes(payload_size as u64));
            group.bench_with_input(
                BenchmarkId::new(label, payload_size),
                &pslice,
                |b, pslice| {
                    b.iter(|| {
                        let encoded = encoder.encode(black_box(pslice)).unwrap();
                        black_box(encoded.shreds)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    for (label, params) in param_sets() {
        let encoder = Encoder::new(params);
        let payload_size = 8192usize;
        let pslice = demo_pslice(payload_size);
        let encoded = encoder.encode(&pslice).unwrap();

        let loss_counts = [
            (1, "loss_1"),
            (params.parity_shards / 4, "loss_25pct"),
            (params.parity_shards / 2, "loss_50pct"),
            (params.parity_shards, "loss_max"),
        ];

        for (drop_count, loss_label) in loss_counts {
            if drop_count == 0 {
                continue;
            }

            let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
            let received = simulate_loss(encoded.shreds.clone(), drop_count, &mut rng);

            group.throughput(Throughput::Bytes(payload_size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{label}/{loss_label}"), drop_count),
                &received,
                |b, received| {
                    b.iter(|| {
                        let mut work = received.clone();
                        let recovered = encoder.decode(&mut work, encoded.original_len).unwrap();
                        black_box(recovered)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("round_trip");

    for (label, params) in param_sets() {
        for payload_size in [1024usize, 8192, 65536] {
            let encoder = Encoder::new(params);
            let pslice = demo_pslice(payload_size);

            group.throughput(Throughput::Bytes(payload_size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{label}/max_loss"), payload_size),
                &pslice,
                |b, pslice| {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(99999);
                    b.iter(|| {
                        let encoded = encoder.encode(black_box(pslice)).unwrap();
                        let mut received =
                            simulate_loss(encoded.shreds, params.max_loss(), &mut rng);
                        let recovered = encoder.decode(&mut received, encoded.original_len).unwrap();
                        black_box(recovered)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_recover_shreds_pr5695_style(c: &mut Criterion) {
    let mut group = c.benchmark_group("recover_pr5695");

    let params_list = [
        ErasureParams::agave_default(),
        ErasureParams::constellation(256),
    ];

    for params in params_list {
        let label = if params == ErasureParams::agave_default() {
            "agave"
        } else {
            "constellation_256"
        };

        let encoder = Encoder::new(params);

        for num_packets in [28, 32, 48, 56] {
            let payload_size = num_packets * 1232;
            let pslice = demo_pslice(payload_size);
            let encoded = encoder.encode(&pslice).unwrap();

            for num_lost in [1, 8, 16, 32].iter().filter(|&&n| n <= params.max_loss()) {
                let mut rng = rand::rngs::StdRng::seed_from_u64(54321);
                let received = simulate_loss(encoded.shreds.clone(), *num_lost, &mut rng);

                let name = format!("{label}_{num_packets}_{num_lost}");
                group.throughput(Throughput::Bytes(payload_size as u64));
                group.bench_function(&name, |b| {
                    b.iter(|| {
                        let mut work = received.clone();
                        let recovered = encoder.decode(&mut work, encoded.original_len).unwrap();
                        black_box(recovered)
                    })
                });
            }
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_round_trip,
    bench_recover_shreds_pr5695_style,
);
criterion_main!(benches);
