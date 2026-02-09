use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Phase(pub f64); // angle in radians [0, 2π)

impl Phase {
    pub fn from_value(value: u64, bucket_count: u32) -> Self {
        let angle = (value % bucket_count as u64) as f64 * 2.0 * PI / bucket_count as f64;
        Phase(angle)
    }

    pub fn from_degrees(degrees: f64) -> Self {
        Phase(degrees * PI / 180.0)
    }

    /// Core operation: coherence = cos(angle_a - angle_b)
    /// Returns: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
    pub fn coherence(&self, other: &Phase) -> f64 {
        (self.0 - other.0).cos()
    }

    /// Harmonic coherence: detects nth-harmonic relationships
    /// n=1: exact match. n=2: opposition. n=3: trine/120°. etc.
    pub fn harmonic_coherence(&self, other: &Phase, n: u32) -> f64 {
        (n as f64 * (self.0 - other.0)).cos()
    }

    /// Angular distance in degrees (always positive, 0-180, shortest path)
    pub fn distance_degrees(&self, other: &Phase) -> f64 {
        let diff = (self.0 - other.0).abs() % (2.0 * PI);
        let d = if diff > PI { 2.0 * PI - diff } else { diff };
        d * 180.0 / PI
    }

    /// Directed angular distance in degrees (0-360, counterclockwise from self to other)
    pub fn directed_distance_degrees(&self, other: &Phase) -> f64 {
        let diff = (other.0 - self.0).rem_euclid(2.0 * PI);
        diff * 180.0 / PI
    }

    /// Fuzzy match with tolerance (orb) using shortest distance (0-180°)
    /// Returns 0.0 if outside orb, smooth falloff from 1.0 at exact to 0.0 at edge
    pub fn fuzzy_match(&self, other: &Phase, target_angle_deg: f64, orb_deg: f64) -> f64 {
        let delta = (self.distance_degrees(other) - target_angle_deg).abs();
        if delta > orb_deg {
            0.0
        } else {
            (delta * PI / (2.0 * orb_deg)).cos()
        }
    }

    /// Directed fuzzy match using directed distance (0-360°) for asymmetric reach
    pub fn directed_fuzzy_match(&self, other: &Phase, target_angle_deg: f64, orb_deg: f64) -> f64 {
        let delta = (self.directed_distance_degrees(other) - target_angle_deg).abs();
        if delta > orb_deg {
            0.0
        } else {
            (delta * PI / (2.0 * orb_deg)).cos()
        }
    }
}

#[derive(Clone, Debug)]
pub struct WavePacket {
    pub id: String,
    pub phases: Vec<(String, Phase)>, // (attribute_name, encoded_phase)
}

impl WavePacket {
    pub fn new(id: &str) -> Self {
        WavePacket {
            id: id.to_string(),
            phases: vec![],
        }
    }

    pub fn with_attr(mut self, name: &str, phase: Phase) -> Self {
        self.phases.push((name.to_string(), phase));
        self
    }

    /// Get phase for a specific attribute
    pub fn phase_for(&self, attr: &str) -> Option<&Phase> {
        self.phases.iter().find(|(n, _)| n == attr).map(|(_, p)| p)
    }
}
