use constellation_parameterised_encoder::{demo_pslice, simulate_loss, Encoder, ErasureParams};
use rand::SeedableRng;

fn main() {
    let params_list = [
        ("agave_32_32", ErasureParams::agave_default()),
        ("constellation_256", ErasureParams::constellation(256)),
    ];

    let payload_sizes = [1024, 8192, 65536];
    let iterations = 1000;

    for (name, params) in &params_list {
        let encoder = Encoder::new(*params);

        for &payload_size in &payload_sizes {
            let pslice = demo_pslice(payload_size);
            let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

            for _ in 0..iterations {
                let encoded = encoder.encode(&pslice).unwrap();

                let mut received = simulate_loss(encoded.shreds, params.max_loss(), &mut rng);

                let recovered = encoder.decode(&mut received, encoded.original_len).unwrap();

                assert_eq!(recovered.len(), pslice.len());
            }

            eprintln!("{name} @ {payload_size}B: {iterations} round-trips OK");
        }
    }
}
