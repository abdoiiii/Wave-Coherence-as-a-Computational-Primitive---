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
    removed: Vec<bool>,
}

impl BucketIndex {
    pub fn new(attr: &str, bucket_count: u32) -> Self {
        BucketIndex {
            attr: attr.to_string(),
            bucket_count,
            buckets: vec![vec![]; bucket_count as usize],
            packets: vec![],
            removed: vec![],
        }
    }

    pub fn insert(&mut self, packet: WavePacket) {
        let idx = self.packets.len();
        let bucket = packet.phase_for(&self.attr).map(|phase| self.phase_to_bucket(phase));
        if let Some(b) = bucket {
            self.buckets[b].push(idx);
        }
        self.packets.push(packet);
        self.removed.push(false);
    }

    /// Remove an entity by ID. Returns true if found and removed.
    pub fn remove_by_id(&mut self, id: &str) -> bool {
        let idx = match self.packets.iter().position(|p| p.id == id) {
            Some(i) => i,
            None => return false,
        };
        if self.removed[idx] {
            return false;
        }
        self.removed[idx] = true;
        let angle_opt = self.packets[idx]
            .phase_for(&self.attr)
            .map(|phase| phase.0);
        if let Some(angle) = angle_opt {
            let normalized = angle.rem_euclid(2.0 * PI);
            let b = (normalized / (2.0 * PI) * self.bucket_count as f64).floor() as usize
                % self.bucket_count as usize;
            self.buckets[b].retain(|&i| i != idx);
        }
        true
    }

    /// Update an entity: remove old, insert new. Returns true if old was found.
    pub fn update(&mut self, id: &str, new_packet: WavePacket) -> bool {
        let found = self.remove_by_id(id);
        self.insert(new_packet);
        found
    }

    /// Count active (non-removed) entities.
    pub fn active_count(&self) -> usize {
        self.removed.iter().filter(|&&r| !r).count()
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

/// Multi-attribute bucket index on a 2D torus (B×B grid).
/// Two attributes, each mapped to a bucket on their own circle.
/// Compound queries check only the rectangular neighborhood on the torus.
pub struct MultiAttrBucketIndex {
    attrs: [String; 2],
    bucket_count: u32,
    grid: Vec<Vec<usize>>, // B*B cells, flat indexing: grid[a * B + b]
    packets: Vec<WavePacket>,
}

impl MultiAttrBucketIndex {
    pub fn new(attr_a: &str, attr_b: &str, bucket_count: u32) -> Self {
        let size = (bucket_count * bucket_count) as usize;
        MultiAttrBucketIndex {
            attrs: [attr_a.to_string(), attr_b.to_string()],
            bucket_count,
            grid: vec![vec![]; size],
            packets: vec![],
        }
    }

    pub fn insert(&mut self, packet: WavePacket) {
        let idx = self.packets.len();
        let ba = packet.phase_for(&self.attrs[0]).map(|p| self.phase_to_bucket(p));
        let bb = packet.phase_for(&self.attrs[1]).map(|p| self.phase_to_bucket(p));
        if let (Some(a), Some(b)) = (ba, bb) {
            let cell = self.cell_index(a, b);
            self.grid[cell].push(idx);
        }
        self.packets.push(packet);
    }

    fn phase_to_bucket(&self, phase: &Phase) -> usize {
        let angle = phase.0.rem_euclid(2.0 * PI);
        let bucket = (angle / (2.0 * PI) * self.bucket_count as f64).floor() as usize;
        bucket % self.bucket_count as usize
    }

    fn cell_index(&self, bucket_a: usize, bucket_b: usize) -> usize {
        bucket_a * self.bucket_count as usize + bucket_b
    }

    fn exact_spread(&self, threshold: f64) -> usize {
        let bucket_angle = 2.0 * PI / self.bucket_count as f64;
        let max_delta = threshold.clamp(-1.0, 1.0).acos();
        (max_delta / bucket_angle).ceil() as usize
    }

    /// Exact match on both attributes. Returns (results, examined_count).
    pub fn query_exact_both(
        &self,
        target_a: &Phase,
        target_b: &Phase,
        threshold: f64,
    ) -> (Vec<(&WavePacket, f64)>, usize) {
        let b = self.bucket_count as usize;
        let spread = self.exact_spread(threshold);
        let center_a = self.phase_to_bucket(target_a);
        let center_b = self.phase_to_bucket(target_b);

        let mut results = Vec::new();
        let mut examined = 0usize;

        for off_a in 0..=(2 * spread) {
            let ba = (center_a + b - spread + off_a) % b;
            for off_b in 0..=(2 * spread) {
                let bb = (center_b + b - spread + off_b) % b;
                let cell = self.cell_index(ba, bb);
                for &pkt_idx in &self.grid[cell] {
                    examined += 1;
                    let p = &self.packets[pkt_idx];
                    let ca = p.phase_for(&self.attrs[0])
                        .map(|ph| ph.coherence(target_a)).unwrap_or(-1.0);
                    let cb = p.phase_for(&self.attrs[1])
                        .map(|ph| ph.coherence(target_b)).unwrap_or(-1.0);
                    if ca >= threshold && cb >= threshold {
                        results.push((p, ca * cb));
                    }
                }
            }
        }
        (results, examined)
    }

    /// Mixed query: exact on attr_a, harmonic on attr_b.
    pub fn query_exact_harmonic(
        &self,
        target_a: &Phase,
        threshold_a: f64,
        target_b: &Phase,
        harmonic_b: u32,
        threshold_b: f64,
    ) -> (Vec<(&WavePacket, f64)>, usize) {
        let b = self.bucket_count as usize;
        let spread_a = self.exact_spread(threshold_a);
        let center_a = self.phase_to_bucket(target_a);

        let bucket_angle = 2.0 * PI / self.bucket_count as f64;
        let max_delta_b = threshold_b.clamp(-1.0, 1.0).acos() / harmonic_b as f64;
        let spread_b = (max_delta_b / bucket_angle).ceil() as usize;

        let mut checked_cells = vec![false; b * b];
        let mut results = Vec::new();
        let mut examined = 0usize;

        for off_a in 0..=(2 * spread_a) {
            let ba = (center_a + b - spread_a + off_a) % b;

            for k in 0..harmonic_b {
                let region_offset = 2.0 * PI * k as f64 / harmonic_b as f64;
                let region_angle = (target_b.0 + region_offset).rem_euclid(2.0 * PI);
                let region_phase = Phase(region_angle);
                let center_b = self.phase_to_bucket(&region_phase);

                for off_b in 0..=(2 * spread_b) {
                    let bb = (center_b + b - spread_b + off_b) % b;
                    let cell = self.cell_index(ba, bb);
                    if checked_cells[cell] { continue; }
                    checked_cells[cell] = true;

                    for &pkt_idx in &self.grid[cell] {
                        examined += 1;
                        let p = &self.packets[pkt_idx];
                        let ca = p.phase_for(&self.attrs[0])
                            .map(|ph| ph.coherence(target_a)).unwrap_or(-1.0);
                        let cb = p.phase_for(&self.attrs[1])
                            .map(|ph| ph.harmonic_coherence(target_b, harmonic_b)).unwrap_or(-1.0);
                        if ca >= threshold_a && cb >= threshold_b {
                            results.push((p, ca * cb));
                        }
                    }
                }
            }
        }
        (results, examined)
    }
}
