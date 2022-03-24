use std::collections::HashMap;
use nalgebra::{DVector};

use crate::filter::KalmanFilter;
use crate::model::Model;

fn get_kalman_object_tracker<F, DimX, DimZ, DimU>(model: &Model, x0: Option<DVector<f64>>) -> KalmanFilter<f64>
{
    let mut tracker = KalmanFilter::<f64>::new(1, 1, 1);

    tracker.F = model.build_F();
    tracker.Q = model.build_Q();
    tracker.H = model.build_H();
    tracker.R = model.build_R();
    tracker.P = model.build_P();

    if x0.is_some() {
        tracker.x = x0.unwrap();
    }

    tracker
}

fn exponential_moving_average_fn(gamma: f64) -> Box<dyn Fn(DVector<f64>, DVector<f64>) -> DVector<f64>> {
    Box::new(move |old, new| -> DVector<f64> {
        gamma * old + (1.0 - gamma) * new
    })
}

struct SingleObjectTracker {
    id: String,
    steps_alive: i64,
    steps_positive: i64,
    staleness: f64,
    max_staleness: f64,

    update_score_fn: Box<dyn Fn(DVector<f64>, DVector<f64>) -> DVector<f64>>,
    update_feature_fn: Box<dyn Fn(DVector<f64>, DVector<f64>) -> DVector<f64>>,

    score: Option<f64>,
    feature: Option<DVector<f64>>,

    class_id_counts: HashMap<i64, i64>,
    class_id: Option<i64>,
}

impl SingleObjectTracker {
    pub fn new(
        max_staleness: f64,
        smooth_score_gamma: f64,
        smooth_feature_gamma: f64,
        score0: Option<f64>,
        class_id0: Option<i64>,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_hyphenated().to_string();
        let steps_alive = 1;
        let steps_positive= 1;
        let staleness = 0.0;
        let max_staleness = max_staleness;

        let score = score0;
        let feature = None;
        let class_id_counts = HashMap::new();
        let class_id = None;

        let mut tracker = Self {
            id,
            steps_alive,
            steps_positive,
            staleness,
            max_staleness,
            score,
            feature,
            class_id,
            class_id_counts,
            update_score_fn: exponential_moving_average_fn(smooth_score_gamma),
            update_feature_fn: exponential_moving_average_fn(smooth_feature_gamma),
        };

        let class_id = tracker.update_class_id(class_id0);
        tracker.class_id = class_id;

        tracker
    }

    fn update_class_id(&mut self, class_id: Option<i64>) -> Option<i64> {
        if class_id.is_none() {
            return None;
        }

        let class_id = class_id.unwrap();
        let entry = self.class_id_counts.entry(class_id).or_insert(1);
        *entry += 1;

        self.class_id_counts
            .iter()
            .max_by_key(|entry | entry.1)
            .map(|i| *i.0)
    }
}
