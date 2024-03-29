use nalgebra as na;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::iter::FromIterator;
use std::sync::{Arc, Mutex};

use crate::filter::KalmanFilter;
use crate::matrix::*;
use crate::metrics::*;
use crate::model::{Model, ModelKwargs, ModelPreset};
use std::vec;

macro_rules! array {
    ($r:expr, $c:expr, $i:expr) => {
        na::DMatrix::from_row_slice($r, $c, $i)
    };
}

#[derive(Debug, Clone)]
pub struct Track {
    pub id: String,
    pub _box: na::DMatrix<f32>,
    pub score: Option<f32>,
    pub class_id: Option<i64>,
}

fn get_kalman_object_tracker(
    model: &Model,
    x0: &mut Option<na::DMatrix<f32>>,
) -> KalmanFilter<f32> {
    let mut tracker = KalmanFilter::new(model.state_length, model.measurement_lengths, 0);

    tracker.F = model.build_F();
    tracker.Q = model.build_Q();
    tracker.H = model.build_H();
    tracker.R = model.build_R();
    tracker.P = model.build_P();

    if let Some(x) = x0.as_mut() {
        std::mem::swap(&mut tracker.x, x);
    }

    tracker
}

fn exponential_moving_average_fn(gamma: f32) -> Box<dyn Fn(f32, f32) -> f32 + Send + Sync> {
    Box::new(move |old, new| -> f32 { gamma * old + (1.0 - gamma) * new })
}

fn exponential_moving_average_matrix_fn(
    gamma: f32,
) -> Box<dyn Fn(na::DMatrix<f32>, na::DMatrix<f32>) -> na::DMatrix<f32> + Send + Sync> {
    Box::new(move |old, new| -> na::DMatrix<f32> { gamma * old + (1.0 - gamma) * new })
}

pub struct SingleObjectTracker {
    id: String,
    steps_alive: i64,
    steps_positive: i64,
    staleness: f32,
    max_staleness: f32,

    update_score_fn: Box<dyn Fn(f32, f32) -> f32 + Send + Sync>,
    update_feature_fn:
        Box<dyn Fn(na::DMatrix<f32>, na::DMatrix<f32>) -> na::DMatrix<f32> + Send + Sync>,

    score: Option<f32>,
    pub feature: Option<na::DMatrix<f32>>,

    class_id_counts: HashMap<i64, i64>,
    class_id: Option<i64>,
}

type TrackBox = na::DMatrix<f32>;

pub trait Tracker {
    fn is_invalid(&self) -> bool;
    fn is_stale(&self) -> bool;
    fn _predict(&mut self);
    fn _box(&self) -> TrackBox;
    fn _feature(&self) -> Option<na::DMatrix<f32>>;
    fn update(&mut self, detection: &Detection);

    fn staleness(&self) -> f32 {
        todo!()
    }
    fn steps_positive(&self) -> i64 {
        todo!()
    }
    fn steps_alive(&self) -> i64 {
        todo!()
    }
    fn set_steps_alive(&mut self, new_value: i64) {
        todo!()
    }
    fn id(&self) -> Cow<'static, str> {
        todo!()
    }
    fn score(&self) -> Option<f32> {
        todo!()
    }
    fn class_id(&self) -> Option<i64> {
        todo!()
    }
    fn model(&self) -> &Model {
        todo!()
    }
    fn stale(&mut self, _rate: Option<f32>) -> f32 {
        todo!()
    }
}

impl SingleObjectTracker {
    pub fn new_with_kwargs(kwargs: Option<SingleObjectTrackerKwargs>) -> Self {
        let kwargs = kwargs.unwrap_or_default();

        let id = uuid::Uuid::new_v4().hyphenated().to_string();
        let steps_alive = 1;
        let steps_positive = 1;
        let staleness = 0.0;

        let max_staleness = kwargs.max_staleness;
        let score = kwargs.score0;
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
            update_score_fn: exponential_moving_average_fn(kwargs.smooth_score_gamma),
            update_feature_fn: exponential_moving_average_matrix_fn(kwargs.smooth_feature_gamma),
        };

        tracker.class_id = tracker.update_class_id(kwargs.class_id0);

        tracker
    }

    fn update_class_id(&mut self, class_id: Option<i64>) -> Option<i64> {
        let Some(class_id) = class_id else { return None };

        let entry = self.class_id_counts.entry(class_id).or_insert(1);
        *entry += 1;

        self.class_id_counts
            .iter()
            .max_by_key(|entry| entry.1)
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

    fn is_stale(&self) -> bool {
        todo!();
    }

    fn _predict(&mut self) {
        todo!();
    }

    fn update(&mut self, detection: &Detection) {
        todo!();
    }

    fn _feature(&self) -> Option<na::DMatrix<f32>> {
        self.feature.clone()
    }
    fn staleness(&self) -> f32 {
        self.staleness
    }
    fn steps_positive(&self) -> i64 {
        self.steps_positive
    }
    fn set_steps_alive(&mut self, new_value: i64) {
        self.steps_alive = new_value;
    }
    fn steps_alive(&self) -> i64 {
        self.steps_alive
    }
    fn id(&self) -> Cow<'static, str> {
        self.id.clone().into()
    }
    fn score(&self) -> Option<f32> {
        self.score
    }
    fn class_id(&self) -> Option<i64> {
        self.class_id
    }
    fn stale(&mut self, rate: Option<f32>) -> f32 {
        let rate = rate.unwrap_or(1.0);
        self.staleness += rate;
        self.staleness
    }
}

#[derive(Clone)]
pub struct SingleObjectTrackerKwargs {
    pub max_staleness: f32,
    pub smooth_score_gamma: f32,
    pub smooth_feature_gamma: f32,
    pub score0: Option<f32>,
    pub class_id0: Option<i64>,
}

impl Default for SingleObjectTrackerKwargs {
    fn default() -> Self {
        SingleObjectTrackerKwargs {
            max_staleness: 12.,
            smooth_score_gamma: 0.8,
            smooth_feature_gamma: 0.9,
            score0: None,
            class_id0: None,
        }
    }
}

impl Default for SingleObjectTracker {
    fn default() -> Self {
        Self::new_with_kwargs(None)
    }
}

pub struct KalmanTracker {
    model_kwargs: (f32, Option<ModelKwargs>),
    model: Model,

    _tracker: KalmanFilter<f32>,
    _base: SingleObjectTracker,
}

impl KalmanTracker {
    pub fn new(
        x0: Option<na::DMatrix<f32>>,
        box0: Option<TrackBox>,
        model_kwargs: (f32, Option<ModelKwargs>),
        kwargs: Option<SingleObjectTrackerKwargs>,
    ) -> Self {
        let model = Model::new(model_kwargs.0, model_kwargs.1.clone());
        let x0 = x0.unwrap_or(model.box_to_x(box0.unwrap()));
        let tracker = get_kalman_object_tracker(&model, &mut Some(x0));

        Self {
            model_kwargs: (model_kwargs.0, model_kwargs.1),
            model,
            _tracker: tracker,
            _base: SingleObjectTracker::new_with_kwargs(kwargs),
        }
    }

    fn update_class_id(&mut self, class_id: Option<i64>) -> Option<i64> {
        self._base.update_class_id(class_id)
    }

    fn _update_box(&mut self, detection: &Detection) {
        let z = self.model.box_to_z(detection._box.clone().unwrap());
        self._tracker.update(&z, None, None);
    }

    fn unstale(&mut self, rate: Option<f32>) -> f32 {
        self._base.staleness = 0_f32.max(self._base.staleness - rate.unwrap_or(2.));
        self._base.staleness
    }
}

impl Tracker for KalmanTracker {
    fn _box(&self) -> TrackBox {
        self.model.x_to_box(&self._tracker.x)
    }

    fn is_invalid(&self) -> bool {
        let x = matrix_to_vec(&self._tracker.x);
        x.iter().any(|v| v.is_nan())
    }

    fn is_stale(&self) -> bool {
        self._base.staleness >= self._base.max_staleness
    }

    fn _predict(&mut self) {
        self._tracker.predict();
        self._base.steps_alive += 1;
    }

    fn update(&mut self, detection: &Detection) {
        self._update_box(detection);
        self._base.steps_positive += 1;
        self._base.class_id = self.update_class_id(Some(detection.class_id));
        self._base.score = Some((*self._base.update_score_fn)(
            self._base.score.unwrap(),
            detection.score,
        ));
        if self._base.feature.is_some() && detection.feature.is_some() {
            self._base.feature = Some((*self._base.update_feature_fn)(
                self._base.feature.clone().unwrap(),
                detection.feature.clone().unwrap(),
            ));
        }
        self.unstale(Some(3.));
    }

    fn _feature(&self) -> Option<na::DMatrix<f32>> {
        self._base.feature.clone()
    }
    fn staleness(&self) -> f32 {
        self._base.staleness()
    }
    fn steps_positive(&self) -> i64 {
        self._base.steps_positive()
    }
    fn steps_alive(&self) -> i64 {
        self._base.steps_alive()
    }
    fn id(&self) -> Cow<'static, str> {
        self._base.id()
    }
    fn score(&self) -> Option<f32> {
        self._base.score()
    }
    fn class_id(&self) -> Option<i64> {
        self._base.class_id()
    }
    fn model(&self) -> &Model {
        &self.model
    }
    fn stale(&mut self, rate: Option<f32>) -> f32 {
        self._base.stale(rate)
    }
}

trait BaseMatchingFunction {
    fn call(
        &self,
        trackers: &[Box<dyn Tracker + Send + Sync>],
        detections: &[Detection],
    ) -> na::DMatrix<usize>;
}

pub struct IOUAndFeatureMatchingFunction {
    pub min_iou: f32,
    pub multi_match_min_iou: f32,
    pub feature_similarity_fn:
        Option<Box<dyn FnOnce(Vec<na::DMatrix<f32>>, Vec<na::DMatrix<f32>>) -> f32 + Send + Sync>>,
    pub feature_similarity_beta: Option<f32>,
}

impl IOUAndFeatureMatchingFunction {
    pub fn new(
        min_iou: f32,
        multi_match_min_iou: f32,
        feature_similarity_fn: Option<
            Box<dyn FnOnce(Vec<na::DMatrix<f32>>, Vec<na::DMatrix<f32>>) -> f32 + Send + Sync>,
        >,
        feature_similarity_beta: Option<f32>,
    ) -> Self {
        Self {
            min_iou,
            multi_match_min_iou,
            feature_similarity_fn,
            feature_similarity_beta,
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
    fn call(
        &self,
        trackers: &[Box<dyn Tracker + Send + Sync>],
        detections: &[Detection],
    ) -> na::DMatrix<usize> {
        match_by_cost_matrix(
            trackers,
            detections,
            self.min_iou,
            self.multi_match_min_iou,
            None,
            self.feature_similarity_beta,
        )
    }
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub score: f32,
    pub class_id: i64,
    pub _box: Option<na::DMatrix<f32>>,
    pub feature: Option<na::DMatrix<f32>>,
}

impl Default for Detection {
    fn default() -> Self {
        Self {
            _box: Some(na::DMatrix::identity(1, 1)),
            score: 0.,
            class_id: 0,
            feature: None,
        }
    }
}

#[derive(Clone)]
pub struct ActiveTracksKwargs {
    pub max_staleness_to_positive_ratio: f32,
    pub max_staleness: f32,
    pub min_steps_alive: i64,
}

impl Default for ActiveTracksKwargs {
    fn default() -> Self {
        Self {
            max_staleness_to_positive_ratio: 3.,
            max_staleness: 999.,
            min_steps_alive: -1,
        }
    }
}

pub struct MultiObjectTracker {
    pub trackers: Vec<Box<dyn Tracker + Send + Sync>>,
    tracker_kwargs: Option<SingleObjectTrackerKwargs>,
    matching_fn: Option<IOUAndFeatureMatchingFunction>,
    matching_fn_kwargs: HashMap<String, f32>,
    active_tracks_kwargs: ActiveTracksKwargs,
    detections_matched_ids: Vec<String>,
    model_kwargs: (f32, Option<ModelKwargs>),
}

impl MultiObjectTracker {
    pub fn new(
        dt: f32,
        model_spec: ModelPreset,
        matching_fn: Option<IOUAndFeatureMatchingFunction>,
        tracker_kwargs: Option<SingleObjectTrackerKwargs>,
        matching_fn_kwargs: Option<HashMap<String, f32>>,
        active_tracks_kwargs: Option<ActiveTracksKwargs>,
    ) -> Self {
        let model_kwards = ModelKwargs {
            order_pos: model_spec.order_pos,
            dim_pos: model_spec.dim_pos,
            order_size: model_spec.order_size,
            dim_size: model_spec.dim_size,
            ..Default::default()
        };

        Self {
            trackers: Vec::default(),
            tracker_kwargs,
            model_kwargs: (dt, Some(model_kwards)),
            matching_fn,
            matching_fn_kwargs: matching_fn_kwargs.unwrap_or_default(),
            active_tracks_kwargs: active_tracks_kwargs.unwrap_or_default(),
            detections_matched_ids: Vec::default(),
        }
    }

    fn tracker_clss(
        &self,
        x0: Option<na::DMatrix<f32>>,
        box0: Option<na::DMatrix<f32>>,
        det: Detection,
    ) -> Box<impl Tracker + Send + Sync> {
        let mut kwargs = self.tracker_kwargs.clone().unwrap_or_default();

        kwargs.score0 = Some(det.score);
        kwargs.class_id0 = Some(det.class_id);

        Box::new(KalmanTracker::new(
            x0,
            box0,
            self.model_kwargs.clone(),
            Some(kwargs),
        ))
    }

    pub fn step(&mut self, detections: Vec<Detection>) -> Vec<Track> {
        let detections = detections
            .into_iter()
            .filter(|det| det._box.is_some())
            .collect::<Vec<_>>();

        for t in self.trackers.iter_mut() {
            t._predict();
            // Arc::get_mut(t).map(|t| t._predict());
        }

        let matches = self
            .matching_fn
            .as_ref()
            .map(|v| v.call(&self.trackers, &detections));
        self.detections_matched_ids = vec![String::default(); detections.len()];
        let matches = matches.unwrap();

        for c in 0..matches.nrows() {
            let track_idx = matches[(c, 0)];
            let det_idx = matches[(c, 1)];
            let det = &detections[det_idx];

            let tracker = &mut self.trackers[track_idx as usize];
            tracker.update(det);
            self.detections_matched_ids[det_idx as usize] = tracker.id().to_string();
        }

        let assigned_det_idxs: HashSet<usize> = if matches.len() > 0 {
            let assigned_det_idxs = matches.index((.., 1)).data.into_slice();
            HashSet::from_iter(assigned_det_idxs.to_vec().into_iter())
        } else {
            HashSet::from_iter(Vec::default())
        };

        let detection_ranges: HashSet<usize> = HashSet::from_iter(0..detections.len());
        let mut diff = detection_ranges
            .difference(&assigned_det_idxs)
            .into_iter()
            .collect::<Vec<_>>();
        diff.sort();

        for det_idx in diff {
            let det = &detections[*det_idx as usize];
            let tracker = self.tracker_clss(None, det._box.clone(), det.clone());
            let det_id: usize = det_idx.clone() as usize;

            self.detections_matched_ids[det_id] = tracker.id().to_string();
            self.trackers.push(tracker);
        }

        let assigned_track_idxs: HashSet<usize> = if matches.len() > 0 {
            let assigned_track_idxs = matches.index((.., 0)).data.into_slice().to_vec();
            let assigned_track_idxs = assigned_track_idxs.iter().map(|v| *v as usize);
            HashSet::from_iter(assigned_track_idxs)
        } else {
            HashSet::from_iter(Vec::default())
        };

        let track_ranges = HashSet::from_iter(0..self.trackers.len());
        let mut diff = track_ranges
            .difference(&assigned_track_idxs)
            .into_iter()
            .collect::<Vec<_>>();
        diff.sort();

        for track_idx in diff {
            let tracker = &mut self.trackers[*track_idx as usize];
            tracker.stale(None);
        }

        self.cleanup_trackers();

        self.active_tracks()
    }

    pub fn cleanup_trackers(&mut self) {
        let count_before = self.trackers.len();

        self.trackers.retain(|t| {
            let tr = t;
            !(tr.is_stale() || tr.is_invalid())
        });

        let count_after = self.trackers.len();

        println!(
            "deleted {} / {} trackers",
            count_before - count_after,
            count_before
        )
    }

    pub fn active_tracks(&self) -> Vec<Track> {
        let mut tracks: Vec<Track> = Vec::default();
        let kwargs = &self.active_tracks_kwargs;

        for tracker in self.trackers.iter() {
            let tr = tracker;
            let cond1 = tr.staleness() / (tr.steps_positive() as f32)
                < kwargs.max_staleness_to_positive_ratio;
            let cond2 = tr.staleness() < kwargs.max_staleness;
            let cond3 = tr.steps_alive() >= kwargs.min_steps_alive;

            if cond1 && cond2 && cond3 {
                let t = Track {
                    id: tr.id().to_string(),
                    _box: tr._box(),
                    score: tr.score(),
                    class_id: tr.class_id(),
                };
                tracks.push(t);
            }
        }

        tracks
    }
}

impl Default for MultiObjectTracker {
    fn default() -> Self {
        MultiObjectTracker {
            trackers: Vec::default(),
            tracker_kwargs: None,
            matching_fn: None,
            matching_fn_kwargs: HashMap::default(),
            active_tracks_kwargs: ActiveTracksKwargs::default(),
            detections_matched_ids: Vec::default(),
            model_kwargs: (0.0, None),
        }
    }
}

fn cost_matrix_iou_feature(
    trackers: &[Box<dyn Tracker + Send + Sync>],
    detections: &[Detection],
    feature_similarity_fn: Option<
        Box<dyn FnOnce(Vec<na::DMatrix<f32>>, Vec<na::DMatrix<f32>>) -> f32>,
    >,
    feature_similarity_beta: Option<f32>,
) -> (na::DMatrix<f32>, na::DMatrix<f32>) {
    let r1 = trackers.len();
    let c1 = (trackers.first()).unwrap()._box().shape().1;
    let mut data = Vec::new();

    for tracker in trackers.iter() {
        let mut vs = tracker._box().iter().map(|m| *m).collect::<Vec<_>>();
        data.append(&mut vs);
    }

    let b1 = na::DMatrix::from_row_slice(r1, c1, data.as_slice());

    let r2 = detections.len();
    let c2 = (detections.first())
        .unwrap()
        ._box
        .clone()
        .map(|b| b.shape().1.clone())
        .unwrap_or(0);
    let mut data = Vec::new();
    for detection in detections.iter() {
        let mut vs = (*detection)
            ._box
            .iter()
            .map(|m| matrix_to_vec(&m))
            .flatten()
            .collect::<Vec<_>>();

        data.append(&mut vs);
    }

    let b2 = na::DMatrix::from_row_slice(r2, c2, data.as_slice());
    let inferred_dim = b1.shape().1 / 2;
    let iou_mat = calculate_iou(b1, b2, inferred_dim);

    let apt_mat = if feature_similarity_beta.is_some() {
        let f1 = trackers.iter().map(|t| (*t)._feature()).collect::<Vec<_>>();
        let f2 = detections
            .iter()
            .map(|t| t.feature.clone())
            .collect::<Vec<_>>();

        if _sequence_has_none(&f1) || _sequence_has_none(&f2) {
            iou_mat
        } else {
            let f1 = f1.into_iter().map(|v| v.unwrap()).collect();
            let f2 = f2.into_iter().map(|v| v.unwrap()).collect();

            let sim_mat = feature_similarity_fn.unwrap()(f1, f2);
            let feature_similarity_beta = feature_similarity_beta.unwrap_or_default();
            let sim_mat = feature_similarity_beta + (1. - feature_similarity_beta) * sim_mat;

            iou_mat * sim_mat
        }
    } else {
        iou_mat
    };

    let cost_mat = -1. * apt_mat.clone();
    (cost_mat, apt_mat)
}

pub fn match_by_cost_matrix(
    trackers: &[Box<dyn Tracker + Send + Sync>],
    detections: &[Detection],
    min_iou: f32,
    multi_match_min_iou: f32,
    feature_similarity_fn: Option<
        Box<dyn FnOnce(Vec<na::DMatrix<f32>>, Vec<na::DMatrix<f32>>) -> f32>,
    >,
    feature_similarity_beta: Option<f32>,
) -> na::DMatrix<usize> {
    if trackers.len() == 0 || detections.len() == 0 {
        return array!(0, 0, &[]);
    }

    let (cost_mat, iou_mat) = cost_matrix_iou_feature(
        trackers,
        detections,
        feature_similarity_fn,
        feature_similarity_beta,
    );
    let (row_ind, col_ind) = linear_sum_assignment(&cost_mat);
    let mut matches = vec![];

    for (r, c) in row_ind.iter().zip(col_ind.iter()) {
        if iou_mat[(*r, *c)] >= min_iou {
            matches.push(vec![*r, *c]);
        }

        if multi_match_min_iou < 1. {
            for c2 in 0..iou_mat.shape().1 {
                if c2 != *c && iou_mat[(*r, c2)] > multi_match_min_iou {
                    matches.push(vec![*r, c2]);
                }
            }
        }
    }

    na::DMatrix::from_fn(matches.len(), matches[0].len(), |r, c| matches[r][c])
}
