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
