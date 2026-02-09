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
