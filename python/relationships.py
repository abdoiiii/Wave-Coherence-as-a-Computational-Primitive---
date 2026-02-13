"""
Structural relationship primitives: directed cycles and pair tables.

Translates src/relationships.rs — non-geometric relationship structures.
"""


class DirectedCycle:
    """Directed cycle traversal on N nodes."""

    def __init__(self, size: int):
        self.size = size

    def step(self, position: int, step: int) -> int:
        """Single step: (position + step) mod size."""
        return (position + step) % self.size

    def chain(self, start: int, step: int, depth: int) -> list[int]:
        """Follow a chain of steps from a starting position."""
        result = [start]
        current = start
        for _ in range(depth):
            current = self.step(current, step)
            result.append(current)
        return result


class PairTable:
    """Structural pair table: explicit pairings not based on geometry."""

    def __init__(self, pairs: list[tuple[int, int]]):
        self.pairs = pairs

    def partner(self, position: int) -> int | None:
        """Find the structural partner of a given position."""
        for a, b in self.pairs:
            if a == position:
                return b
            if b == position:
                return a
        return None

    def are_paired(self, a: int, b: int) -> bool:
        """Check if two positions are structurally paired."""
        return any(
            (x == a and y == b) or (x == b and y == a)
            for x, y in self.pairs
        )
