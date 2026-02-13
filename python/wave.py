"""
Wave mechanics core: Phase encoding and WavePacket structures.

Translates src/wave.rs — the foundational types for phase-angle encoding.
Uses only the math standard library (zero external dependencies).
"""

import math


class Phase:
    """Phase angle in radians [0, 2pi). The atomic unit of wave encoding."""

    def __init__(self, angle: float):
        self.angle = angle

    @classmethod
    def from_value(cls, value: int, bucket_count: int) -> "Phase":
        angle = (value % bucket_count) * 2.0 * math.pi / bucket_count
        return cls(angle)

    @classmethod
    def from_degrees(cls, degrees: float) -> "Phase":
        return cls(degrees * math.pi / 180.0)

    def coherence(self, other: "Phase") -> float:
        """Core operation: cos(angle_a - angle_b).
        Returns: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite."""
        return math.cos(self.angle - other.angle)

    def harmonic_coherence(self, other: "Phase", n: int) -> float:
        """Harmonic coherence: detects nth-harmonic relationships.
        n=1: exact match. n=2: opposition. n=3: trine/120 deg. etc."""
        return math.cos(n * (self.angle - other.angle))

    def distance_degrees(self, other: "Phase") -> float:
        """Angular distance in degrees (always positive, 0-180, shortest path)."""
        diff = abs(self.angle - other.angle) % (2.0 * math.pi)
        d = (2.0 * math.pi - diff) if diff > math.pi else diff
        return d * 180.0 / math.pi

    def directed_distance_degrees(self, other: "Phase") -> float:
        """Directed angular distance in degrees (0-360, counterclockwise from self to other)."""
        diff = (other.angle - self.angle) % (2.0 * math.pi)
        return diff * 180.0 / math.pi

    def fuzzy_match(self, other: "Phase", target_angle_deg: float, orb_deg: float) -> float:
        """Fuzzy match with tolerance (orb) using shortest distance (0-180 deg).
        Returns 0.0 if outside orb, smooth falloff from 1.0 at exact to 0.0 at edge."""
        delta = abs(self.distance_degrees(other) - target_angle_deg)
        if delta > orb_deg:
            return 0.0
        return math.cos(delta * math.pi / (2.0 * orb_deg))

    def directed_fuzzy_match(self, other: "Phase", target_angle_deg: float, orb_deg: float) -> float:
        """Directed fuzzy match using directed distance (0-360 deg) for asymmetric reach."""
        delta = abs(self.directed_distance_degrees(other) - target_angle_deg)
        if delta > orb_deg:
            return 0.0
        return math.cos(delta * math.pi / (2.0 * orb_deg))


class WavePacket:
    """Multi-attribute phase-encoded entity."""

    def __init__(self, id: str):
        self.id = id
        self.phases: list[tuple[str, Phase]] = []

    def with_attr(self, name: str, phase: Phase) -> "WavePacket":
        """Builder method: add an attribute phase. Returns self for chaining."""
        self.phases.append((name, phase))
        return self

    def phase_for(self, attr: str):
        """Get phase for a specific attribute. Returns Phase or None."""
        for name, phase in self.phases:
            if name == attr:
                return phase
        return None
