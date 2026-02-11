use std::f64::consts::PI;
use crate::wave::{Phase, WavePacket};

pub struct ResonanceField {
    pub packets: Vec<WavePacket>,
}

impl ResonanceField {
    pub fn new() -> Self {
        ResonanceField { packets: vec![] }
    }

    pub fn add(&mut self, packet: WavePacket) {
        self.packets.push(packet);
    }

    /// Exact match: find all packets where attr coherence > threshold
    pub fn query_exact(
        &self,
        attr: &str,
        target: &Phase,
        threshold: f64,
    ) -> Vec<(&WavePacket, f64)> {
        self.packets
            .iter()
            .filter_map(|p| {
                p.phase_for(attr).and_then(|phase| {
                    let c = phase.coherence(target);
                    if c >= threshold {
                        Some((p, c))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Harmonic scan: find all packets at nth harmonic of target
    pub fn query_harmonic(
        &self,
        attr: &str,
        target: &Phase,
        harmonic: u32,
        threshold: f64,
    ) -> Vec<(&WavePacket, f64)> {
        self.packets
            .iter()
            .filter_map(|p| {
                p.phase_for(attr).and_then(|phase| {
                    let c = phase.harmonic_coherence(target, harmonic);
                    if c >= threshold {
                        Some((p, c))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Fuzzy harmonic scan: find packets near a target angle with tolerance
    pub fn query_fuzzy(
        &self,
        attr: &str,
        target: &Phase,
        target_angle_deg: f64,
        orb_deg: f64,
    ) -> Vec<(&WavePacket, f64)> {
        self.packets
            .iter()
            .filter_map(|p| {
                p.phase_for(attr).and_then(|phase| {
                    let score = target.fuzzy_match(phase, target_angle_deg, orb_deg);
                    if score > 0.0 {
                        Some((p, score))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Typed reach scan: query using entity-type-specific visible angles
    /// Uses directed distance (0-360°) because reach is asymmetric/one-directional
    pub fn query_typed_reach(
        &self,
        attr: &str,
        target: &Phase,
        visible_angles: &[f64],
        orb_deg: f64,
    ) -> Vec<(&WavePacket, f64)> {
        self.packets
            .iter()
            .filter_map(|p| {
                p.phase_for(attr).and_then(|phase| {
                    let best_score = visible_angles
                        .iter()
                        .map(|&angle| target.directed_fuzzy_match(phase, angle, orb_deg))
                        .fold(0.0_f64, f64::max);
                    if best_score > 0.0 {
                        Some((p, best_score))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
}

/// Bucket-indexed wave field: position on the circle IS the index.
/// No separate index structure to maintain — insertion determines position,
/// position determines bucket, query checks only relevant buckets.
pub struct BucketIndex {
    attr: String,
    bucket_count: u32,
    buckets: Vec<Vec<usize>>,
    packets: Vec<WavePacket>,
}

impl BucketIndex {
    pub fn new(attr: &str, bucket_count: u32) -> Self {
        BucketIndex {
            attr: attr.to_string(),
            bucket_count,
            buckets: vec![vec![]; bucket_count as usize],
            packets: vec![],
        }
    }

    pub fn insert(&mut self, packet: WavePacket) {
        let idx = self.packets.len();
        let bucket = packet.phase_for(&self.attr).map(|phase| self.phase_to_bucket(phase));
        if let Some(b) = bucket {
            self.buckets[b].push(idx);
        }
        self.packets.push(packet);
    }

    fn phase_to_bucket(&self, phase: &Phase) -> usize {
        let angle = phase.0.rem_euclid(2.0 * PI);
        let bucket = (angle / (2.0 * PI) * self.bucket_count as f64).floor() as usize;
        bucket % self.bucket_count as usize
    }

    fn exact_spread(&self, threshold: f64) -> usize {
        let bucket_angle = 2.0 * PI / self.bucket_count as f64;
        let max_delta = threshold.clamp(-1.0, 1.0).acos();
        (max_delta / bucket_angle).ceil() as usize
    }

    /// Exact query: check only buckets within angular range where cos(δ) >= threshold
    pub fn query_exact(&self, target: &Phase, threshold: f64) -> (Vec<(&WavePacket, f64)>, usize) {
        let center = self.phase_to_bucket(target);
        let spread = self.exact_spread(threshold);
        let b = self.bucket_count as usize;

        let mut results = Vec::new();
        let mut examined = 0usize;

        for offset in 0..=(2 * spread) {
            let bucket_idx = (center + b - spread + offset) % b;
            for &pkt_idx in &self.buckets[bucket_idx] {
                examined += 1;
                let p = &self.packets[pkt_idx];
                if let Some(phase) = p.phase_for(&self.attr) {
                    let c = phase.coherence(target);
                    if c >= threshold {
                        results.push((p, c));
                    }
                }
            }
        }
        (results, examined)
    }

    /// Harmonic query: check n regions around the circle at 360°/n intervals,
    /// each region narrowed by factor n.
    pub fn query_harmonic(
        &self,
        target: &Phase,
        harmonic: u32,
        threshold: f64,
    ) -> (Vec<(&WavePacket, f64)>, usize) {
        let b = self.bucket_count as usize;
        let bucket_angle = 2.0 * PI / self.bucket_count as f64;
        let max_delta = threshold.clamp(-1.0, 1.0).acos() / harmonic as f64;
        let spread = (max_delta / bucket_angle).ceil() as usize;

        let mut checked_buckets = vec![false; b];
        let mut results = Vec::new();
        let mut examined = 0usize;

        for k in 0..harmonic {
            let region_offset = 2.0 * PI * k as f64 / harmonic as f64;
            let region_angle = (target.0 + region_offset).rem_euclid(2.0 * PI);
            let region_phase = Phase(region_angle);
            let center = self.phase_to_bucket(&region_phase);

            for offset in 0..=(2 * spread) {
                let bucket_idx = (center + b - spread + offset) % b;
                if checked_buckets[bucket_idx] {
                    continue;
                }
                checked_buckets[bucket_idx] = true;
                for &pkt_idx in &self.buckets[bucket_idx] {
                    examined += 1;
                    let p = &self.packets[pkt_idx];
                    if let Some(phase) = p.phase_for(&self.attr) {
                        let c = phase.harmonic_coherence(target, harmonic);
                        if c >= threshold {
                            results.push((p, c));
                        }
                    }
                }
            }
        }
        (results, examined)
    }
}
