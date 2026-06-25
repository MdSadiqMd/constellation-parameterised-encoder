use std::sync::{Arc, OnceLock, RwLock};

use reed_solomon_erasure::galois_8::ReedSolomon;
use thiserror::Error;

pub use reed_solomon_erasure::Error as RsError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Reed-Solomon error: {0}")]
    ReedSolomon(#[from] RsError),
    #[error("Invalid header: insufficient bytes")]
    InvalidHeader,
    #[error("Shard length mismatch: expected {expected}, got {actual}")]
    ShardLengthMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ErasureParams {
    pub data_shards: usize,
    pub parity_shards: usize,
}

impl ErasureParams {
    pub fn new(data_shards: usize, parity_shards: usize) -> Self {
        Self {
            data_shards,
            parity_shards,
        }
    }

    /// Agave's default FEC configuration: 32 data, 32 parity (1:1 ratio)
    pub fn agave_default() -> Self {
        Self::new(32, 32)
    }

    /// Constellation configuration for q attesters: q/4 data, 3q/4 parity (4x redundancy)
    /// This allows reconstruction from any q/4 of the q shreds.
    pub fn constellation(q: usize) -> Self {
        let data = q / 4;
        Self::new(data, q - data)
    }

    pub fn total_shards(&self) -> usize {
        self.data_shards + self.parity_shards
    }

    /// Maximum number of shreds that can be lost while still recovering original data
    pub fn max_loss(&self) -> usize {
        self.parity_shards
    }
}

type CacheEntry = Arc<OnceLock<Result<Arc<ReedSolomon>, RsError>>>;

/// LRU cache for ReedSolomon instances, following Agave's pattern.
/// Keyed by (data_shards, parity_shards) tuple.
pub struct ReedSolomonCache {
    entries: RwLock<lru::LruCache<(usize, usize), CacheEntry>>,
}

impl ReedSolomonCache {
    const CAPACITY: usize = 128;

    pub fn new() -> Self {
        Self {
            entries: RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(Self::CAPACITY).unwrap(),
            )),
        }
    }

    pub fn get(&self, params: ErasureParams) -> Result<Arc<ReedSolomon>, Error> {
        let key = (params.data_shards, params.parity_shards);

        let entry = {
            let mut cache = self.entries.write().unwrap();
            cache
                .get_or_insert(key, || Arc::new(OnceLock::new()))
                .clone()
        };

        entry
            .get_or_init(|| ReedSolomon::new(params.data_shards, params.parity_shards).map(Arc::new))
            .clone()
            .map_err(Error::from)
    }
}

impl Default for ReedSolomonCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Encoded pshreds output from the encoder
pub struct Pshreds {
    pub params: ErasureParams,
    pub shreds: Vec<Vec<u8>>,
    pub shard_len: usize,
    pub original_len: usize,
}

/// Encoder for creating and recovering pshreds using Reed-Solomon erasure coding
pub struct Encoder {
    params: ErasureParams,
    cache: ReedSolomonCache,
}

impl Encoder {
    pub fn new(params: ErasureParams) -> Self {
        Self {
            params,
            cache: ReedSolomonCache::default(),
        }
    }

    pub fn params(&self) -> ErasureParams {
        self.params
    }

    /// Encode a pslice (payload) into pshreds
    pub fn encode(&self, pslice: &[u8]) -> Result<Pshreds, Error> {
        let ErasureParams {
            data_shards,
            parity_shards,
        } = self.params;

        let shard_len = pslice.len().div_ceil(data_shards).max(1);

        let mut buf = pslice.to_vec();
        buf.resize(shard_len * data_shards, 0);

        let mut shreds: Vec<Vec<u8>> = buf.chunks(shard_len).map(|c| c.to_vec()).collect();
        shreds.extend((0..parity_shards).map(|_| vec![0u8; shard_len]));

        let rs = self.cache.get(self.params)?;
        rs.encode(&mut shreds)?;

        Ok(Pshreds {
            params: self.params,
            shreds,
            shard_len,
            original_len: pslice.len(),
        })
    }

    /// Decode received shreds back to original pslice.
    /// `received` contains `Some(shard)` for shreds that arrived, `None` for lost ones.
    /// Requires at least `data_shards` shreds to be present.
    pub fn decode(
        &self,
        received: &mut [Option<Vec<u8>>],
        original_len: usize,
    ) -> Result<Vec<u8>, Error> {
        let rs = self.cache.get(self.params)?;
        rs.reconstruct(received)?;

        let mut pslice = Vec::with_capacity(original_len);
        for shard in received.iter().take(self.params.data_shards) {
            pslice.extend_from_slice(shard.as_ref().unwrap());
        }
        pslice.truncate(original_len);
        Ok(pslice)
    }
}

pub const HEADER_LEN: usize = 2 + 2 + 2 + 4; // data_shards, parity_shards, index, original_len

/// Frame a single pshred for network transmission
pub fn frame_pshred(
    params: ErasureParams,
    index: usize,
    original_len: usize,
    shard: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(HEADER_LEN + shard.len());
    buf.extend_from_slice(&(params.data_shards as u16).to_le_bytes());
    buf.extend_from_slice(&(params.parity_shards as u16).to_le_bytes());
    buf.extend_from_slice(&(index as u16).to_le_bytes());
    buf.extend_from_slice(&(original_len as u32).to_le_bytes());
    buf.extend_from_slice(shard);
    buf
}

/// Parsed pshred frame from network
pub struct PshredFrame {
    pub params: ErasureParams,
    pub index: usize,
    pub original_len: usize,
    pub shard: Vec<u8>,
}

/// Parse a pshred frame received from network
pub fn parse_pshred(bytes: &[u8]) -> Result<PshredFrame, Error> {
    if bytes.len() < HEADER_LEN {
        return Err(Error::InvalidHeader);
    }
    let data_shards = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
    let parity_shards = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
    let index = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
    let original_len = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
    Ok(PshredFrame {
        params: ErasureParams::new(data_shards, parity_shards),
        index,
        original_len,
        shard: bytes[HEADER_LEN..].to_vec(),
    })
}

/// Generate a deterministic test payload
pub fn demo_pslice(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

/// Simulates network loss by randomly dropping `drop_count` shreds.
/// Returns a Vec of Option<Vec<u8>> where None indicates a lost shard.
pub fn simulate_loss<R: rand::Rng>(
    shreds: Vec<Vec<u8>>,
    drop_count: usize,
    rng: &mut R,
) -> Vec<Option<Vec<u8>>> {
    use rand::seq::SliceRandom;

    let mut received: Vec<Option<Vec<u8>>> = shreds.into_iter().map(Some).collect();
    let mut indices: Vec<usize> = (0..received.len()).collect();
    indices.shuffle(rng);

    for &i in indices.iter().take(drop_count) {
        received[i] = None;
    }
    received
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_params() -> Vec<(&'static str, ErasureParams)> {
        vec![
            ("agave_32_32", ErasureParams::agave_default()),
            ("constellation_64", ErasureParams::constellation(64)),
            ("constellation_128", ErasureParams::constellation(128)),
            ("constellation_256", ErasureParams::constellation(256)),
        ]
    }

    #[test]
    fn erasure_params_calculations() {
        let agave = ErasureParams::agave_default();
        assert_eq!(agave.data_shards, 32);
        assert_eq!(agave.parity_shards, 32);
        assert_eq!(agave.total_shards(), 64);
        assert_eq!(agave.max_loss(), 32);

        let c256 = ErasureParams::constellation(256);
        assert_eq!(c256.data_shards, 64);
        assert_eq!(c256.parity_shards, 192);
        assert_eq!(c256.total_shards(), 256);
        assert_eq!(c256.max_loss(), 192);
    }

    #[test]
    fn round_trip_no_loss() {
        for (name, params) in test_params() {
            for payload_size in [64, 1024, 8192, 65536] {
                let encoder = Encoder::new(params);
                let original = demo_pslice(payload_size);

                let encoded = encoder.encode(&original).expect("encode failed");
                assert_eq!(encoded.shreds.len(), params.total_shards());

                let mut received: Vec<Option<Vec<u8>>> =
                    encoded.shreds.into_iter().map(Some).collect();
                let recovered = encoder
                    .decode(&mut received, encoded.original_len)
                    .expect("decode failed");

                assert_eq!(
                    recovered, original,
                    "{name}: round-trip failed for payload size {payload_size}"
                );
            }
        }
    }

    #[test]
    fn round_trip_max_loss() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

        for (name, params) in test_params() {
            for payload_size in [64, 1024, 8192] {
                let encoder = Encoder::new(params);
                let original = demo_pslice(payload_size);

                let encoded = encoder.encode(&original).expect("encode failed");

                let mut received = simulate_loss(encoded.shreds, params.max_loss(), &mut rng);

                let surviving = received.iter().filter(|s| s.is_some()).count();
                assert_eq!(
                    surviving,
                    params.data_shards,
                    "should have exactly data_shards surviving"
                );

                let recovered = encoder
                    .decode(&mut received, encoded.original_len)
                    .expect("decode failed");

                assert_eq!(
                    recovered, original,
                    "{name}: max-loss round-trip failed for payload size {payload_size}"
                );
            }
        }
    }

    #[test]
    fn round_trip_partial_loss() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(67890);

        for (name, params) in test_params() {
            let encoder = Encoder::new(params);
            let original = demo_pslice(4096);
            let encoded = encoder.encode(&original).expect("encode failed");

            for loss_fraction in [0.25, 0.5, 0.75] {
                let drop_count = (params.parity_shards as f64 * loss_fraction) as usize;

                let mut received = simulate_loss(encoded.shreds.clone(), drop_count, &mut rng);
                let recovered = encoder
                    .decode(&mut received, encoded.original_len)
                    .expect("decode failed");

                assert_eq!(
                    recovered, original,
                    "{name}: partial-loss ({loss_fraction}) round-trip failed"
                );
            }
        }
    }

    #[test]
    fn round_trip_many_random_patterns() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11111);

        let params = ErasureParams::constellation(256);
        let encoder = Encoder::new(params);
        let original = demo_pslice(8192);

        for trial in 0..100 {
            let encoded = encoder.encode(&original).expect("encode failed");
            let mut received = simulate_loss(encoded.shreds, params.max_loss(), &mut rng);

            let recovered = encoder
                .decode(&mut received, encoded.original_len)
                .unwrap_or_else(|e| panic!("trial {trial}: decode failed: {e}"));

            assert_eq!(recovered, original, "trial {trial}: data mismatch");
        }
    }

    #[test]
    fn round_trip_small_payload() {
        for (name, params) in test_params() {
            let encoder = Encoder::new(params);

            for payload_size in [1, 7, 15, 31] {
                let original = demo_pslice(payload_size);
                let encoded = encoder.encode(&original).expect("encode failed");

                let mut received: Vec<Option<Vec<u8>>> =
                    encoded.shreds.into_iter().map(Some).collect();
                let recovered = encoder
                    .decode(&mut received, encoded.original_len)
                    .expect("decode failed");

                assert_eq!(
                    recovered, original,
                    "{name}: small payload {payload_size} failed"
                );
            }
        }
    }

    #[test]
    fn round_trip_only_data_shards_survive() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(22222);

        for (name, params) in test_params() {
            let encoder = Encoder::new(params);
            let original = demo_pslice(2048);
            let encoded = encoder.encode(&original).expect("encode failed");

            let mut received: Vec<Option<Vec<u8>>> =
                encoded.shreds.into_iter().map(Some).collect();

            use rand::seq::SliceRandom;
            let mut parity_indices: Vec<usize> =
                (params.data_shards..params.total_shards()).collect();
            parity_indices.shuffle(&mut rng);

            for &i in &parity_indices {
                received[i] = None;
            }

            let surviving = received.iter().filter(|s| s.is_some()).count();
            assert_eq!(surviving, params.data_shards);

            let recovered = encoder
                .decode(&mut received, encoded.original_len)
                .expect("decode failed");

            assert_eq!(
                recovered, original,
                "{name}: only-data-shards recovery failed"
            );
        }
    }

    #[test]
    fn round_trip_only_parity_shards_survive() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(33333);

        for (name, params) in test_params() {
            if params.parity_shards < params.data_shards {
                continue;
            }

            let encoder = Encoder::new(params);
            let original = demo_pslice(2048);
            let encoded = encoder.encode(&original).expect("encode failed");

            let mut received: Vec<Option<Vec<u8>>> =
                encoded.shreds.into_iter().map(Some).collect();

            for i in 0..params.data_shards {
                received[i] = None;
            }

            use rand::seq::SliceRandom;
            let mut parity_indices: Vec<usize> =
                (params.data_shards..params.total_shards()).collect();
            parity_indices.shuffle(&mut rng);

            let excess_parity = params.parity_shards - params.data_shards;
            for &i in parity_indices.iter().take(excess_parity) {
                received[i] = None;
            }

            let surviving = received.iter().filter(|s| s.is_some()).count();
            assert_eq!(surviving, params.data_shards);

            let recovered = encoder
                .decode(&mut received, encoded.original_len)
                .expect("decode failed");

            assert_eq!(
                recovered, original,
                "{name}: only-parity-shards recovery failed"
            );
        }
    }

    #[test]
    fn wire_frame_round_trip() {
        let params = ErasureParams::constellation(256);
        let encoder = Encoder::new(params);
        let original = demo_pslice(1024);
        let encoded = encoder.encode(&original).expect("encode failed");

        for (index, shard) in encoded.shreds.iter().enumerate() {
            let framed = frame_pshred(params, index, encoded.original_len, shard);
            let parsed = parse_pshred(&framed).expect("parse failed");

            assert_eq!(parsed.params, params);
            assert_eq!(parsed.index, index);
            assert_eq!(parsed.original_len, encoded.original_len);
            assert_eq!(&parsed.shard, shard);
        }
    }

    #[test]
    fn wire_frame_reassembly() {
        let params = ErasureParams::constellation(64);
        let encoder = Encoder::new(params);
        let original = demo_pslice(512);
        let encoded = encoder.encode(&original).expect("encode failed");

        let frames: Vec<Vec<u8>> = encoded
            .shreds
            .iter()
            .enumerate()
            .map(|(i, shard)| frame_pshred(params, i, encoded.original_len, shard))
            .collect();

        let mut rng = rand::rngs::StdRng::seed_from_u64(44444);
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..frames.len()).collect();
        indices.shuffle(&mut rng);

        let drop_count = params.max_loss();
        let surviving_indices: Vec<usize> = indices.into_iter().skip(drop_count).collect();

        let mut received: Vec<Option<Vec<u8>>> = vec![None; params.total_shards()];
        let mut original_len = 0;

        for &i in &surviving_indices {
            let parsed = parse_pshred(&frames[i]).expect("parse failed");
            original_len = parsed.original_len;
            received[parsed.index] = Some(parsed.shard);
        }

        let recovered = encoder
            .decode(&mut received, original_len)
            .expect("decode failed");

        assert_eq!(recovered, original);
    }

    #[test]
    fn reed_solomon_cache_reuse() {
        let cache = ReedSolomonCache::new();

        let params = ErasureParams::constellation(256);

        let rs1 = cache.get(params).expect("first get failed");
        let rs2 = cache.get(params).expect("second get failed");

        assert!(Arc::ptr_eq(&rs1, &rs2), "cache should return same instance");
    }
}
