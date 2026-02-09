/// Directed cycle traversal on N nodes
pub struct DirectedCycle {
    pub size: usize,
}

impl DirectedCycle {
    pub fn new(size: usize) -> Self {
        DirectedCycle { size }
    }

    /// Single step: (position + step) mod size
    pub fn step(&self, position: usize, step: i32) -> usize {
        let s = self.size as i32;
        ((position as i32 + step).rem_euclid(s)) as usize
    }

    /// Follow a chain of steps from a starting position
    pub fn chain(&self, start: usize, step: i32, depth: usize) -> Vec<usize> {
        let mut result = vec![start];
        let mut current = start;
        for _ in 0..depth {
            current = self.step(current, step);
            result.push(current);
        }
        result
    }
}

/// Structural pair table: explicit pairings not based on geometry
pub struct PairTable {
    pairs: Vec<(usize, usize)>,
}

impl PairTable {
    pub fn new(pairs: Vec<(usize, usize)>) -> Self {
        PairTable { pairs }
    }

    /// Find the structural partner of a given position
    pub fn partner(&self, position: usize) -> Option<usize> {
        for &(a, b) in &self.pairs {
            if a == position {
                return Some(b);
            }
            if b == position {
                return Some(a);
            }
        }
        None
    }

    /// Check if two positions are structurally paired
    pub fn are_paired(&self, a: usize, b: usize) -> bool {
        self.pairs
            .iter()
            .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
    }
}
