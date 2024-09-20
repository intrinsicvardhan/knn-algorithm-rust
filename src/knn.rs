// src/knn.rs

use std::collections::HashMap;

pub struct KNN {
    k: usize,
    data: Vec<(Vec<Option<f64>>, String)>,
}

impl KNN {
    pub fn new(k: usize) -> Self {
        KNN { k, data: vec![] }
    }

    pub fn fit(&mut self, data: Vec<(Vec<Option<f64>>, String)>) {
        self.data = data;
    }

    pub fn predict(&self, input: Vec<Option<f64>>) -> String {
        let input = self.handle_missing_data(&input);

        let mut distances: Vec<(f64, String)> = self.data.iter()
            .map(|(features, label)| {
                let features = self.handle_missing_data(features);
                let dist = euclidean_distance(&features, &input);
                (dist, label.clone())
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut votes: HashMap<String, usize> = HashMap::new();
        for (_, label) in distances.iter().take(self.k) {
            *votes.entry(label.clone()).or_insert(0) += 1;
        }

        votes.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .unwrap()
    }

    fn handle_missing_data(&self, data: &[Option<f64>]) -> Vec<f64> {
        let mut means = vec![0.0; data.len()];
        let mut counts = vec![0; data.len()];

        for (features, _) in &self.data {
            for (i, value) in features.iter().enumerate() {
                if let Some(v) = value {
                    means[i] += v;
                    counts[i] += 1;
                }
            }
        }

        means.iter_mut().zip(counts.iter()).for_each(|(mean, &count)| {
            if count > 0 {
                *mean /= count as f64;
            }
        });

        data.iter().enumerate().map(|(i, value)| {
            value.unwrap_or(means[i])
        }).collect()
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}