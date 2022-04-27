use std::collections::HashMap;
use std::collections::HashSet;
use std::vec;
use nalgebra::{ DMatrix, dmatrix };
use std::iter::FromIterator;

use crate::filter::KalmanFilter;
use crate::model::{ Model, ModelPreset };
use crate::metrics::*;
use crate::matrix::*;

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

fn exponential_moving_average_fn(gamma: f64) -> Box<dyn Fn(f64, f64) -> f64> {
    Box::new(move |old, new| -> f64 {
        gamma * old + (1.0 - gamma) * new
    })
}

fn exponential_moving_average_matrix_fn(gamma: f64) -> Box<dyn Fn(DMatrix<f64>, DMatrix<f64>) -> DMatrix<f64>> {
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

    update_score_fn: Box<dyn Fn(f64, f64) -> f64>,
    update_feature_fn: Box<dyn Fn(DMatrix<f64>, DMatrix<f64>) -> DMatrix<f64>>,

    score: Option<f64>,
    pub feature: Option<DMatrix<f64>>,

    class_id_counts: HashMap<i64, i64>,
    class_id: Option<i64>,
}

type TrackBox = DMatrix<f64>;

trait Tracker {
    fn _box(&self) -> TrackBox;
    fn is_invalid(&self) -> bool;
    fn _predict(&mut self);
    fn _feature(&self) -> Option<DMatrix<f64>>;
    fn update(&mut self, detection: &Detection);
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
            update_feature_fn: exponential_moving_average_matrix_fn(kwargs.get("smooth_score_gamma").map(|v| *v).unwrap_or(0.9)),
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
    fn _box(&self) -> TrackBox {
        todo!()
    }

    fn is_invalid(&self) -> bool {
        todo!();
    }

    fn _predict(&mut self) {
        todo!();
    }

    fn update(&mut self, detection: &Detection) {
        todo!();
    }

    fn _feature(&self) -> Option<DMatrix<f64>> {
        self.feature.clone()
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

    fn _update_box(&mut self, detection: &Detection) {
        let z = self.model.box_to_z(detection._box.unwrap());
        self._tracker.update(&z, None, None);
    }

    fn unstale(&mut self, rate: Option<f64>) -> f64 {
        self._base.staleness = 0_f64.max(self._base.staleness - rate.unwrap_or(2.));
        self._base.staleness
    }
}

impl Tracker for KalmanTracker {
    fn _box(&self) -> TrackBox {
        todo!()
    }

    fn is_invalid(&self) -> bool {
        todo!();
    }

    fn _predict(&mut self) {
        self._tracker.predict(None, None, None, None)
    }

    fn update(&mut self, detection: &Detection) {
        self._update_box(detection);
        self._base.steps_positive += 1;
        self._base.class_id = self.update_class_id(Some(detection.class_id));
        self._base.score = Some((*self._base.update_score_fn)(self._base.score.unwrap(), detection.score));
        self._base.feature = Some((*self._base.update_feature_fn)(self._base.feature.unwrap(), detection.feature.unwrap()));
        self.unstale(Some(3.));
    }

    fn _feature(&self) -> Option<DMatrix<f64>> {
        self._base.feature
    }
}

trait BaseMatchingFunction {
    fn call(&self, trackers: &Vec<Box<dyn Tracker>>, detections: &Vec<Detection>) -> DMatrix<f64>;
}

struct IOUAndFeatureMatchingFunction {
    pub min_iou: f64,
    pub multi_match_min_iou: f64,
    pub feature_similarity_fn: Option<Box<dyn FnOnce(Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) -> f64>>,
    pub feature_similarity_beta: Option<f64>,
}

impl IOUAndFeatureMatchingFunction {
    pub fn new(
        min_iou: f64,
        multi_match_min_iou: f64,
        feature_similarity_fn: Option<Box<dyn FnOnce(Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) -> f64>>,
        feature_similarity_beta: Option<f64>
    ) -> Self {
        Self {
            min_iou,
            multi_match_min_iou,
            feature_similarity_fn,
            feature_similarity_beta
       }
    }
}

impl Default for IOUAndFeatureMatchingFunction {
    fn default() -> Self {
        Self {
            min_iou: 0.1,
            multi_match_min_iou: 1.0,
            feature_similarity_beta: None,
            feature_similarity_fn: None,
        }
    }
}

impl BaseMatchingFunction for IOUAndFeatureMatchingFunction {
    fn call(&self, trackers: &Vec<Box<dyn Tracker>>, detections: &Vec<Detection>) -> DMatrix<f64> {
        match_by_cost_matrix(
            trackers,
            detections,
            self.min_iou,
            self.multi_match_min_iou,
            None,
            self.feature_similarity_beta
        )
    }
}

struct Detection {
    pub score: f64,
    pub class_id: i64,
    pub _box: Option<DMatrix<f64>>,
    pub feature: Option<DMatrix<f64>>,
}

struct MultiObjectTracker {
    trackers: Vec<Box<dyn Tracker>>,
    tracker_kwargs: HashMap<String, f64>,
    tracker_clss: Option<Box<dyn FnOnce(Option<DMatrix<f64>>, Option<DMatrix<f64>>, Detection) -> KalmanTracker>>,
    matching_fn: Option<IOUAndFeatureMatchingFunction>,
    matching_fn_kwargs: HashMap<String, f64>,
    active_tracks_kwargs: HashMap<String, f64>,
    detections_matched_ids: Vec<String>,
}

impl MultiObjectTracker {
    pub fn new(
        dt: f64,
        model_spec: HashMap<String, f64>,
        matching_fn: Option<IOUAndFeatureMatchingFunction>,
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
            kwargs.insert(String::from("class_id0"), det.class_id as f64);

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

        let matches = self.matching_fn.as_ref().map(|v| v.call(&self.trackers, &detections));
        self.detections_matched_ids = Vec::with_capacity(detections.len());

        let matches = matches.unwrap();
        for c in 0..matches.nrows() {
            let track_idx = matches[(c, 0)];
            let det_idx = matches[(c, 1)];

            self.trackers[track_idx as usize].update(&detections[det_idx as usize]);
        }

        let assigned_det_idxs = if matches.len() > 0 {
            matches.index((.., 1)).data.into_slice().to_vec()
        } else {
            vec![]
        };

        let assigned_det_idxs = assigned_det_idxs.iter().map(|v| *v as u64);
        let idxs: HashSet<u64> = HashSet::from_iter(assigned_det_idxs);
        let diff: HashSet<u64> = HashSet::from_iter((0..detections.len()).into_iter().map(|v| v as u64));
        for det_idx in diff.difference(&idxs) {
            let det: Detection = detections[*det_idx as usize];

            let tracker = (*self.tracker_clss.unwrap())(
                None,
                det._box,
                det.clone()
            );

            self.trackers.push(Box::new(tracker));
        }
    }
}

fn cost_matrix_iou_feature(
    trackers: &Vec<Box<dyn Tracker>>,
    detections: &Vec<Detection>,
    feature_similarity_fn: Option<Box<dyn FnOnce(Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) -> f64>>,
    feature_similarity_beta: Option<f64>
) -> (DMatrix<f64>, DMatrix<f64>) {
    let r = trackers.len();
    let c = (trackers.first()).unwrap()._box().shape().1;
    let mut data = Vec::new();
    for tracker in trackers.iter() {
        let mut vs = (*tracker)._box().iter().map(|m| *m).collect::<Vec<_>>();
        data.append(&mut vs);
    }
    let b1 = DMatrix::from_vec(r, c, data);

    let r = detections.len();
    let c = (detections.first()).unwrap()._box.clone().map(|b| b.shape().1.clone()).unwrap_or(0);
    let mut data = Vec::new();
    for detection in detections.iter() {
        let mut vs = (*detection)._box
            .iter()
            .map(|m| matrix_to_vec(&m))
            .flatten()
            .collect::<Vec<_>>();
        data.append(&mut vs);
    }
    let b2 = DMatrix::from_vec(r, c, data);
    let inferred_dim = b1.shape().1 / 2;
    let iou_mat = calculate_iou(b1, b2, inferred_dim);
    let mut apt_mat = iou_mat.clone();

    if feature_similarity_beta.is_some() {
        let f1 = trackers
            .iter()
            .map(|t| (*t)._feature())
            .collect::<Vec<_>>();
        let f2 =detections
            .iter()
            .map(|t| t.feature.clone())
            .collect::<Vec<_>>();

        if _sequence_has_none(&f1) || _sequence_has_none(&f2) {
            apt_mat = iou_mat.clone();
        } else {
            let f1 = f1
                .into_iter()
                .map(|v| v.unwrap())
                .collect();
            let f2 = f2
                .into_iter()
                .map(|v| v.unwrap())
                .collect();

            let sim_mat = feature_similarity_fn.unwrap()(f1, f2);
            let feature_similarity_beta = feature_similarity_beta.unwrap_or_default();
            let sim_mat = feature_similarity_beta + (1. - feature_similarity_beta) * sim_mat;

            apt_mat = iou_mat.clone() * sim_mat;
        }
    } else {
        apt_mat = iou_mat.clone();
    }

    let cost_mat = -1. * apt_mat;
    (cost_mat, iou_mat.clone())
}

fn match_by_cost_matrix(
    trackers: &Vec<Box<dyn Tracker>>,
    detections: &Vec<Detection>,
    min_iou: f64,
    multi_match_min_iou: f64,
    feature_similarity_fn: Option<Box<dyn FnOnce(Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) -> f64>>,
    feature_similarity_beta: Option<f64>
) -> DMatrix<f64> {
    if trackers.len() == 0 || detections.len() == 0 {
        return dmatrix![];
    }

    let (cost_mat, iou_mat)
        = cost_matrix_iou_feature(trackers, detections, feature_similarity_fn, feature_similarity_beta);
    let (row_ind, col_ind) = linear_sum_assignment(&cost_mat);
    let mut matches = vec![];

    for (r, c) in row_ind.iter().zip(col_ind.iter()) {
        if iou_mat[(*r, *c)] >= min_iou {
            matches.push((*r, *c))
        }

        if multi_match_min_iou < 1. {
            for c2 in 0..iou_mat.shape().1 {
                if c2 != *c && iou_mat[(*r, c2)] > multi_match_min_iou {
                    matches.push((*r, c2))
                }
            }
        }
    }

    DMatrix::from_fn(1, matches.len(), |r, c| if c == 0 { matches[r].0 as f64 } else { matches[r].1 as f64 })
}

#[cfg(test)]
mod test {
    use std::ops::Mul;

    use super::*;

    use nalgebra::{dmatrix};
    use approx::assert_relative_eq;

    #[test]
    fn test_tracker_diverges() {
        let spec = ModelPreset::constant_velocity_and_static_box_size_2d();
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
