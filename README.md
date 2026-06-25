# constellation-parameterised-encoder

A Constellation-parameterised Reed-Solomon erasure encoder using the same `reed-solomon-erasure` crate as Agave's shredder, with configurable FEC parameters for pslice/pshred encoding.

## Features

- **Configurable erasure parameters**: Agave-compatible (32,32) or Constellation (q/4, 3q/4) configurations
- **ReedSolomonCache**: LRU cache following Agave's pattern for efficient codec reuse
- **Wire framing**: Self-describing frame format for network transmission
- **Comprehensive testing**: Round-trip encode→decode verification with simulated packet loss

## Usage

```rust
use constellation_parameterised_encoder::{Encoder, ErasureParams, simulate_loss};

// Agave-compatible (32 data, 32 parity)
let encoder = Encoder::new(ErasureParams::agave_default());

// Constellation for 256 attesters (64 data, 192 parity)
let encoder = Encoder::new(ErasureParams::constellation(256));

// Encode
let payload = b"transaction data...";
let encoded = encoder.encode(payload)?;

// Simulate 75% packet loss (max tolerable for constellation_256)
let mut received = simulate_loss(encoded.shreds, 192, &mut rng);

// Recover original data
let recovered = encoder.decode(&mut received, encoded.original_len)?;
assert_eq!(recovered, payload);
```

## Erasure Configurations

| Config | Data | Parity | Total | Max Loss | Use Case |
|--------|------|--------|-------|----------|----------|
| `agave_default()` | 32 | 32 | 64 | 50% | Solana mainnet |
| `constellation(64)` | 16 | 48 | 64 | 75% | Small attester set |
| `constellation(128)` | 32 | 96 | 128 | 75% | Medium attester set |
| `constellation(256)` | 64 | 192 | 256 | 75% | Large attester set |

## Testing

```bash
cargo test
```

All 11 tests verify round-trip correctness: encode → random loss → decode → verify original recovered.

## Benchmarks

```bash
cargo bench
```

Benchmark groups (extending PR #5695's structure):
- `encode/` - Encoding throughput
- `decode/` - Recovery throughput with variable loss
- `round_trip/` - Full encode→loss→decode cycle
- `recover_pr5695/` - PR #5695-style benchmarks

## License

MIT
