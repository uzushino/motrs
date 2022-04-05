use std::collections::HashMap;
use std::hash::Hash;
use nalgebra::{DMatrix};

use crate::filter::KalmanFilter;
use crate::model::{Model, self, ModelPreset};

fn get_kalman_object_tracker<F>(model: &Model, x0: Option<DMatrix<f64>>) -> KalmanFilter<f64>
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

fn exponential_moving_average_fn(gamma: f64) -> Box<dyn Fn(DMatrix<f64>, DMatrix<f64>) -> DMatrix<f64>> {
    Box::new(move |old, new| -> DMatrix<f64> {
        gamma * old + (1.0 - gamma) * new
    })
}

struct SingleObjectTracker {
    id: String,
    steps_alive: i64,
    steps_positive: i64,
    staleness: f64,
    max_staleness: f64,

    update_score_fn: Box<dyn Fn(DMatrix<f64>, DMatrix<f64>) -> DMatrix<f64>>,
    update_feature_fn: Box<dyn Fn(DMatrix<f64>, DMatrix<f64>) -> DMatrix<f64>>,

    score: Option<f64>,
    feature: Option<DMatrix<f64>>,

    class_id_counts: HashMap<i64, i64>,
    class_id: Option<i64>,
}

type TrackBox = DMatrix<f64>;

trait Tracker {
    fn _box(self) -> TrackBox;
    fn is_invalid(self) -> bool;
    fn _predict(&mut self);
}

impl SingleObjectTracker {
    pub fn new(kwargs : HashMap<String, f64>) -> Self {
        let id = uuid::Uuid::new_v4().to_hyphenated().to_string();
        let steps_alive = 1;
        let steps_positive= 1;
        let staleness = 0.0;

        let max_staleness = kwargs["max_staleness"];
        let score = kwargs.get("score0").map(|v| *v);
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
            update_score_fn: exponential_moving_average_fn(kwargs.get("smooth_score_gamma").map(|v| *v).unwrap_or(0.8)),
            update_feature_fn: exponential_moving_average_fn(kwargs.get("smooth_score_gamma").map(|v| *v).unwrap_or(0.9)),
        };

        tracker.class_id = tracker.update_class_id(kwargs.get("class_id0").map(|v| *v as i64));

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

impl Tracker for SingleObjectTracker {
    fn _box(self) -> TrackBox {
        todo!()
    }

    fn is_invalid(self) -> bool {
        todo!();
    }

    fn _predict(&mut self) {
        todo!();
    }
}

struct KalmanTracker {
    model_kwargs: HashMap<String, f64>,
    model: Model,

    _tracker: KalmanFilter<f64>,
    _base: SingleObjectTracker,
}

impl KalmanTracker {
    pub fn new(
        x0: Option<DMatrix<f64>>,
        box0: Option<TrackBox>,
        model_kwargs: HashMap<String, f64>,
        kwargs: HashMap<String, f64>,
    ) -> Self {
        let model = Model::new(model_kwargs["dt"], model_kwargs.clone());
        let x0 = x0.unwrap_or(model.box_to_x(box0.unwrap()));
        let tracker = get_kalman_object_tracker::<f64>(&model, Some(x0));

        Self {
            model_kwargs,
            model,
            _tracker: tracker,
            _base: SingleObjectTracker::new(kwargs)
        }
    }

    fn update_class_id(&mut self, class_id: Option<i64>) -> Option<i64> {
        self._base.update_class_id(class_id)
    }
}

impl Tracker for KalmanTracker {
    fn _box(self) -> TrackBox {
        todo!()
    }

    fn is_invalid(self) -> bool {
        todo!();
    }

    fn _predict(&mut self) {
        self._tracker.predict(None, None, None, None)
    }
}

trait BaseMatchingFunction {
    fn call(&self, trackers: &Vec<Box<dyn Tracker>>, detections: Vec<Detection>);
}

struct IOUAndFeatureMatchingFunction {
    pub min_iou: f64,
    pub multi_match_min_iou: f64,
    pub feature_similarity_fn: Box<dyn FnOnce()>
}

struct Detection {
    pub score: f64,
    pub class_id: f64,
    pub _box: Option<DMatrix<f64>>
}

struct MultiObjectTracker {
    trackers: Vec<Box<dyn Tracker>>,
    tracker_kwargs: HashMap<String, f64>,
    tracker_clss: Option<Box<dyn FnOnce(Option<DMatrix<f64>>, Option<DMatrix<f64>>, Detection) -> KalmanTracker>>,
    matching_fn: Option<Box<dyn BaseMatchingFunction>>,
    matching_fn_kwargs: HashMap<String, f64>,
    active_tracks_kwargs: HashMap<String, f64>,
    detections_matched_ids: Vec<String>,
}

impl MultiObjectTracker {
    pub fn new(
        dt: f64,
        model_spec: HashMap<String, f64>,
        matching_fn: Option<Box<dyn BaseMatchingFunction>>,
        tracker_kwargs: Option<HashMap<String, f64>>,
        matching_fn_kwargs: Option<HashMap<String, f64>>,
        active_tracks_kwargs: Option<HashMap<String, f64>>
    ) -> Self {
        let mut model_kwards = model_spec;
        model_kwards.insert(String::from("dt"), dt);

        let tracker_kwargs_ = tracker_kwargs.clone();
        let tracker_clss = move |x0, box0, det: Detection| {
            let mut kwargs = tracker_kwargs_.unwrap_or_default();
            kwargs.insert(String::from("score0"), det.score);
            kwargs.insert(String::from("class_id0"), det.class_id);

            KalmanTracker::new(
                x0,
                box0,
                model_kwards,
                kwargs,
            )
        };

        Self {
            trackers: Vec::default(),
            tracker_kwargs: tracker_kwargs.unwrap_or_default(),
            tracker_clss: Some(Box::new(tracker_clss)),
            matching_fn,
            matching_fn_kwargs: matching_fn_kwargs.unwrap_or_default(),
            active_tracks_kwargs: active_tracks_kwargs.unwrap_or_default(),
            detections_matched_ids: Vec::default(),
        }
    }

    pub fn step(&mut self, detections: Vec<Detection>) {
        let detections = detections
            .into_iter()
            .filter(|det| det._box.is_some())
            .collect::<Vec<_>>();

        for t in self.trackers.iter_mut() {
            t._predict();
        }

        self.matching_fn.as_ref().map(|v| v.call(&self.trackers, detections));


    }
}

#[cfg(test)]
mod test {
    use std::ops::Mul;

    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::dmatrix;


    #[test]
    fn test_tracker_diverges() {
        let spec = ModelPreset::constant_velocity_and_static_box_size_2d().values();
        let _box = dmatrix![0., 0., 10., 10.];
        let mot = MultiObjectTracker::new(
            0.1,
            spec,
            None,
            None,
            None,
            None
        );
    }
}
