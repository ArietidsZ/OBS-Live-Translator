//! Beam search decoding for sequence generation

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A hypothesis in the beam search
#[derive(Clone, Debug)]
pub struct Hypothesis {
    /// Token sequence so far
    pub tokens: Vec<u32>,

    /// Log probability of this sequence
    pub log_prob: f32,

    /// Is this sequence finished (hit EOS token)?
    pub finished: bool,
}

impl Default for Hypothesis {
    fn default() -> Self {
        Self::new()
    }
}

impl Hypothesis {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            log_prob: 0.0,
            finished: false,
        }
    }

    /// Average log probability per token
    pub fn score(&self) -> f32 {
        if self.tokens.is_empty() {
            self.log_prob
        } else {
            self.log_prob / self.tokens.len() as f32
        }
    }
}

impl PartialEq for Hypothesis {
    fn eq(&self, other: &Self) -> bool {
        self.score() == other.score()
    }
}

impl Eq for Hypothesis {}

impl PartialOrd for Hypothesis {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hypothesis {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score().partial_cmp(&other.score()).unwrap_or(Ordering::Equal)
    }
}

/// Beam search decoder
pub struct BeamSearch {
    beam_size: usize,
    max_length: usize,
    eos_token_id: u32,
}

impl BeamSearch {
    pub fn new(beam_size: usize, max_length: usize, eos_token_id: u32) -> Self {
        Self {
            beam_size,
            max_length,
            eos_token_id,
        }
    }

    /// Perform beam search decoding
    ///
    /// # Arguments
    /// * `compute_next_logprobs` - Function that takes current hypotheses and returns log probabilities for next tokens
    ///
    /// # Returns
    /// Best hypothesis found
    pub fn search<F>(&self, mut compute_next_logprobs: F) -> Hypothesis
    where
        F: FnMut(&[Hypothesis]) -> Vec<Vec<(u32, f32)>>,
    {
        let mut beam = BinaryHeap::new();
        beam.push(Hypothesis::new());

        let mut finished = Vec::new();

        for _step in 0..self.max_length {
            if beam.is_empty() {
                break;
            }

            // Get current hypotheses
            let current: Vec<_> = beam.drain().collect();

            // Compute next token probabilities for all hypotheses
            let next_logprobs = compute_next_logprobs(&current);

            let mut candidates = BinaryHeap::new();

            // Extend each hypothesis with each possible next token
            for (hyp_idx, hypothesis) in current.iter().enumerate() {
                if hypothesis.finished {
                    finished.push(hypothesis.clone());
                    continue;
                }

                // Get top-k next tokens for this hypothesis
                for &(token_id, log_prob) in &next_logprobs[hyp_idx] {
                    let mut new_hyp = hypothesis.clone();
                    new_hyp.tokens.push(token_id);
                    new_hyp.log_prob += log_prob;

                    if token_id == self.eos_token_id {
                        new_hyp.finished = true;
                        finished.push(new_hyp);
                    } else {
                        candidates.push(new_hyp);
                    }
                }
            }

            // Keep top beam_size candidates
            beam = candidates
                .into_sorted_vec()
                .into_iter()
                .rev()
                .take(self.beam_size)
                .collect();

            // Early stopping if all beams finished
            if beam.iter().all(|h| h.finished) {
                break;
            }
        }

        // Return best hypothesis (finished or in beam)
        finished.extend(beam.into_sorted_vec());
        finished.into_iter().max().unwrap_or_else(Hypothesis::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_search_simple() {
        let beam_search = BeamSearch::new(3, 10, 2); // beam=3, max_len=10, eos=2

        // Simple mock: always prefer token 0, then 1, then EOS (2)
        let result = beam_search.search(|hyps| {
            hyps.iter()
                .map(|h| {
                    let len = h.tokens.len();
                    if len == 0 {
                        vec![(0, -0.1), (1, -0.5)]
                    } else if len == 1 {
                        vec![(1, -0.1), (0, -0.5)]
                    } else {
                        vec![(2, -0.1)] // EOS
                    }
                })
                .collect()
        });

        assert_eq!(result.tokens, vec![0, 1, 2]);
        assert!(result.finished);
    }

    #[test]
    fn test_hypothesis_scoring() {
        let mut h1 = Hypothesis::new();
        h1.tokens = vec![1, 2, 3];
        h1.log_prob = -3.0;

        let mut h2 = Hypothesis::new();
        h2.tokens = vec![1, 2];
        h2.log_prob = -1.5;

        // h1: -3.0 / 3 = -1.0
        // h2: -1.5 / 2 = -0.75
        assert!(h2.score() > h1.score());
    }
}
