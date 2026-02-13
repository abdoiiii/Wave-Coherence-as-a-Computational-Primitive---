"""
Resonance field structures: linear scan, bucket-indexed, and multi-attribute torus.

Translates src/field.rs — the query engines that operate on Phase-encoded data.
"""

import math
from wave import Phase, WavePacket


class ResonanceField:
    """Linear-scan resonance field. Ground truth for all queries."""

    def __init__(self):
        self.packets: list[WavePacket] = []

    def add(self, packet: WavePacket):
        self.packets.append(packet)

    def query_exact(self, attr: str, target: Phase, threshold: float) -> list[tuple[WavePacket, float]]:
        """Exact match: find all packets where attr coherence >= threshold."""
        results = []
        for p in self.packets:
            phase = p.phase_for(attr)
            if phase is not None:
                c = phase.coherence(target)
                if c >= threshold:
                    results.append((p, c))
        return results

    def query_harmonic(self, attr: str, target: Phase, harmonic: int,
                       threshold: float) -> list[tuple[WavePacket, float]]:
        """Harmonic scan: find all packets at nth harmonic of target."""
        results = []
        for p in self.packets:
            phase = p.phase_for(attr)
            if phase is not None:
                c = phase.harmonic_coherence(target, harmonic)
                if c >= threshold:
                    results.append((p, c))
        return results

    def query_fuzzy(self, attr: str, target: Phase, target_angle_deg: float,
                    orb_deg: float) -> list[tuple[WavePacket, float]]:
        """Fuzzy harmonic scan: find packets near a target angle with tolerance."""
        results = []
        for p in self.packets:
            phase = p.phase_for(attr)
            if phase is not None:
                score = target.fuzzy_match(phase, target_angle_deg, orb_deg)
                if score > 0.0:
                    results.append((p, score))
        return results

    def query_typed_reach(self, attr: str, target: Phase, visible_angles: list[float],
                          orb_deg: float) -> list[tuple[WavePacket, float]]:
        """Typed reach scan: query using entity-type-specific visible angles.
        Uses directed distance (0-360 deg) because reach is asymmetric."""
        results = []
        for p in self.packets:
            phase = p.phase_for(attr)
            if phase is not None:
                best_score = max(
                    (target.directed_fuzzy_match(phase, angle, orb_deg) for angle in visible_angles),
                    default=0.0
                )
                if best_score > 0.0:
                    results.append((p, best_score))
        return results


class BucketIndex:
    """Bucket-indexed wave field: position on the circle IS the index.
    No separate index structure to maintain — insertion determines position,
    position determines bucket, query checks only relevant buckets."""

    def __init__(self, attr: str, bucket_count: int):
        self.attr = attr
        self.bucket_count = bucket_count
        self.buckets: list[list[int]] = [[] for _ in range(bucket_count)]
        self.packets: list[WavePacket] = []
        self.removed: list[bool] = []

    def insert(self, packet: WavePacket):
        idx = len(self.packets)
        phase = packet.phase_for(self.attr)
        if phase is not None:
            b = self._phase_to_bucket(phase)
            self.buckets[b].append(idx)
        self.packets.append(packet)
        self.removed.append(False)

    def remove_by_id(self, id: str) -> bool:
        """Remove an entity by ID. Returns True if found and removed."""
        idx = None
        for i, p in enumerate(self.packets):
            if p.id == id:
                idx = i
                break
        if idx is None:
            return False
        if self.removed[idx]:
            return False
        self.removed[idx] = True
        phase = self.packets[idx].phase_for(self.attr)
        if phase is not None:
            normalized = phase.angle % (2.0 * math.pi)
            b = int(normalized / (2.0 * math.pi) * self.bucket_count) % self.bucket_count
            self.buckets[b] = [i for i in self.buckets[b] if i != idx]
        return True

    def update(self, id: str, new_packet: WavePacket) -> bool:
        """Update an entity: remove old, insert new. Returns True if old was found."""
        found = self.remove_by_id(id)
        self.insert(new_packet)
        return found

    def active_count(self) -> int:
        """Count active (non-removed) entities."""
        return sum(1 for r in self.removed if not r)

    def _phase_to_bucket(self, phase: Phase) -> int:
        angle = phase.angle % (2.0 * math.pi)
        bucket = int(angle / (2.0 * math.pi) * self.bucket_count)
        return bucket % self.bucket_count

    def _exact_spread(self, threshold: float) -> int:
        bucket_angle = 2.0 * math.pi / self.bucket_count
        clamped = max(-1.0, min(1.0, threshold))
        max_delta = math.acos(clamped)
        return math.ceil(max_delta / bucket_angle)

    def query_exact(self, target: Phase, threshold: float) -> tuple[list[tuple[WavePacket, float]], int]:
        """Exact query: check only buckets within angular range where cos(d) >= threshold.
        Returns (results, examined_count)."""
        center = self._phase_to_bucket(target)
        spread = self._exact_spread(threshold)
        b = self.bucket_count

        results = []
        examined = 0

        for offset in range(2 * spread + 1):
            bucket_idx = (center + b - spread + offset) % b
            for pkt_idx in self.buckets[bucket_idx]:
                examined += 1
                p = self.packets[pkt_idx]
                phase = p.phase_for(self.attr)
                if phase is not None:
                    c = phase.coherence(target)
                    if c >= threshold:
                        results.append((p, c))
        return results, examined

    def query_harmonic(self, target: Phase, harmonic: int,
                       threshold: float) -> tuple[list[tuple[WavePacket, float]], int]:
        """Harmonic query: check n regions around the circle at 360 deg/n intervals,
        each region narrowed by factor n. Returns (results, examined_count)."""
        b = self.bucket_count
        bucket_angle = 2.0 * math.pi / self.bucket_count
        clamped = max(-1.0, min(1.0, threshold))
        max_delta = math.acos(clamped) / harmonic
        spread = math.ceil(max_delta / bucket_angle)

        checked_buckets = [False] * b
        results = []
        examined = 0

        for k in range(harmonic):
            region_offset = 2.0 * math.pi * k / harmonic
            region_angle = (target.angle + region_offset) % (2.0 * math.pi)
            region_phase = Phase(region_angle)
            center = self._phase_to_bucket(region_phase)

            for offset in range(2 * spread + 1):
                bucket_idx = (center + b - spread + offset) % b
                if checked_buckets[bucket_idx]:
                    continue
                checked_buckets[bucket_idx] = True
                for pkt_idx in self.buckets[bucket_idx]:
                    examined += 1
                    p = self.packets[pkt_idx]
                    phase = p.phase_for(self.attr)
                    if phase is not None:
                        c = phase.harmonic_coherence(target, harmonic)
                        if c >= threshold:
                            results.append((p, c))
        return results, examined


class MultiAttrBucketIndex:
    """Multi-attribute bucket index on a 2D torus (B x B grid).
    Two attributes, each mapped to a bucket on their own circle.
    Compound queries check only the rectangular neighborhood on the torus."""

    def __init__(self, attr_a: str, attr_b: str, bucket_count: int):
        self.attrs = [attr_a, attr_b]
        self.bucket_count = bucket_count
        size = bucket_count * bucket_count
        self.grid: list[list[int]] = [[] for _ in range(size)]
        self.packets: list[WavePacket] = []

    def insert(self, packet: WavePacket):
        idx = len(self.packets)
        ba = self._phase_to_bucket(packet.phase_for(self.attrs[0]))
        bb = self._phase_to_bucket(packet.phase_for(self.attrs[1]))
        if ba is not None and bb is not None:
            cell = self._cell_index(ba, bb)
            self.grid[cell].append(idx)
        self.packets.append(packet)

    def _phase_to_bucket(self, phase) -> int | None:
        if phase is None:
            return None
        angle = phase.angle % (2.0 * math.pi)
        bucket = int(angle / (2.0 * math.pi) * self.bucket_count)
        return bucket % self.bucket_count

    def _cell_index(self, bucket_a: int, bucket_b: int) -> int:
        return bucket_a * self.bucket_count + bucket_b

    def _exact_spread(self, threshold: float) -> int:
        bucket_angle = 2.0 * math.pi / self.bucket_count
        clamped = max(-1.0, min(1.0, threshold))
        max_delta = math.acos(clamped)
        return math.ceil(max_delta / bucket_angle)

    def query_exact_both(self, target_a: Phase, target_b: Phase,
                         threshold: float) -> tuple[list[tuple[WavePacket, float]], int]:
        """Exact match on both attributes. Returns (results, examined_count)."""
        b = self.bucket_count
        spread = self._exact_spread(threshold)
        center_a = self._phase_to_bucket(target_a)
        center_b = self._phase_to_bucket(target_b)

        results = []
        examined = 0

        for off_a in range(2 * spread + 1):
            ba = (center_a + b - spread + off_a) % b
            for off_b in range(2 * spread + 1):
                bb = (center_b + b - spread + off_b) % b
                cell = self._cell_index(ba, bb)
                for pkt_idx in self.grid[cell]:
                    examined += 1
                    p = self.packets[pkt_idx]
                    ph_a = p.phase_for(self.attrs[0])
                    ph_b = p.phase_for(self.attrs[1])
                    ca = ph_a.coherence(target_a) if ph_a else -1.0
                    cb = ph_b.coherence(target_b) if ph_b else -1.0
                    if ca >= threshold and cb >= threshold:
                        results.append((p, ca * cb))
        return results, examined

    def query_exact_harmonic(self, target_a: Phase, threshold_a: float,
                             target_b: Phase, harmonic_b: int,
                             threshold_b: float) -> tuple[list[tuple[WavePacket, float]], int]:
        """Mixed query: exact on attr_a, harmonic on attr_b."""
        b = self.bucket_count
        spread_a = self._exact_spread(threshold_a)
        center_a = self._phase_to_bucket(target_a)

        bucket_angle = 2.0 * math.pi / self.bucket_count
        clamped_b = max(-1.0, min(1.0, threshold_b))
        max_delta_b = math.acos(clamped_b) / harmonic_b
        spread_b = math.ceil(max_delta_b / bucket_angle)

        checked_cells = [False] * (b * b)
        results = []
        examined = 0

        for off_a in range(2 * spread_a + 1):
            ba = (center_a + b - spread_a + off_a) % b

            for k in range(harmonic_b):
                region_offset = 2.0 * math.pi * k / harmonic_b
                region_angle = (target_b.angle + region_offset) % (2.0 * math.pi)
                region_phase = Phase(region_angle)
                center_b = self._phase_to_bucket(region_phase)

                for off_b in range(2 * spread_b + 1):
                    bb = (center_b + b - spread_b + off_b) % b
                    cell = self._cell_index(ba, bb)
                    if checked_cells[cell]:
                        continue
                    checked_cells[cell] = True

                    for pkt_idx in self.grid[cell]:
                        examined += 1
                        p = self.packets[pkt_idx]
                        ph_a = p.phase_for(self.attrs[0])
                        ph_b = p.phase_for(self.attrs[1])
                        ca = ph_a.coherence(target_a) if ph_a else -1.0
                        cb = ph_b.harmonic_coherence(target_b, harmonic_b) if ph_b else -1.0
                        if ca >= threshold_a and cb >= threshold_b:
                            results.append((p, ca * cb))
        return results, examined
