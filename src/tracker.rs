use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Div;
use nalgebra::{DMatrix, dmatrix, DMatrixSlice};
use nalgebra::base::dimension::{ Const, Dynamic };
use std::cmp::max;

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
    fn _box(&self) -> TrackBox;
    fn is_invalid(&self) -> bool;
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
    fn _box(&self) -> TrackBox {
        todo!()
    }

    fn is_invalid(&self) -> bool {
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
    fn _box(&self) -> TrackBox {
        todo!()
    }

    fn is_invalid(&self) -> bool {
        todo!();
    }

    fn _predict(&mut self) {
        self._tracker.predict(None, None, None, None)
    }
}

fn matrix_to_vec(mat: &DMatrix<f64>) -> Vec<f64> {
    let (row, col) = mat.shape();
    let mut result = Vec::default();

    for r in 0..row {
        for  c in 0..col {
            result.push(mat[(r, c)]);
        }
    }

    result
}

fn matrix_split(mat: &DMatrix<f64>, indecies_num: usize) -> Vec<DMatrix<f64>> {
    let c = mat.ncols() / indecies_num;
    let r = mat.nrows() / c;

    let mut splitted = Vec::default();
    for i in 0..indecies_num {
        let sp = DMatrix::from(mat.index((.., (i*c)..((i+1)*c))));
        splitted.push(sp);
    }

    splitted
}

fn matrix_maximum(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let cols = a.ncols();
    let rows = a.nrows();

    let mut result = DMatrix::zeros(rows, cols);

    for r in 0..rows {
        for c in 0..cols {
            result[(r, c)] = b[(0, c)].max(a[(r, c)]);
        }
    }

    result
}

fn matrix_minimum(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let cols = a.ncols();
    let rows = a.nrows();

    let mut result = DMatrix::zeros(rows, cols);

    for r in 0..rows {
        for c in 0..cols {
            result[(r, c)] = b[(0, c)].min(a[(r, c)]);
        }
    }

    result
}

fn matrix_clip(mat: &DMatrix<f64>, min_value: f64, max_value: f64) -> DMatrix<f64> {
    let cols = mat.ncols();
    let rows = mat.nrows();

    let mut result = DMatrix::zeros(rows, cols);

    for r in 0..rows {
        for c in 0..cols {
            let v = mat[(r, c)].min(min_value);
            let v = v.max(min_value);

            result[(r, c)] = v;
        }
    }

    result
}

fn matrix_div(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let cols = a.ncols();
    let rows = a.nrows();

    let mut result = DMatrix::zeros(rows, cols);

    for r in 0..rows {
        for c in 0..cols {
            let v = a[(r, c)] / b[(r, c)];
            result[(r, c)] = v;
        }
    }

    result
}


fn calculate_iou(bboxes1: DMatrix<f64>, bboxes2: DMatrix<f64>, dim: usize) -> DMatrix<f64> {
    let r = bboxes1.nrows();
    let bboxes1 = bboxes1.reshape_generic(Dynamic::new(r / dim * 2), Dynamic::new(dim * 2));

    let r = bboxes2.nrows();
    let bboxes2 = bboxes2.reshape_generic(Dynamic::new(r / dim * 2), Dynamic::new(dim * 2));

    let coords_b1 = matrix_split(&bboxes1, 2 * dim);
    let coords_b2 = matrix_split(&bboxes2, 2 * dim);

    let mut coords1: Vec<DMatrix<f64>> = Vec::default();
    for _ in 0..dim {
        coords1.push(DMatrix::zeros(bboxes1.nrows(), bboxes2.nrows()));
    }

    let mut coords2: Vec<DMatrix<f64>> = Vec::default();
    for _ in 0..dim {
        coords2.push(DMatrix::zeros(bboxes1.nrows(), bboxes2.nrows()));
    }

    let zero = DMatrix::zeros(bboxes1.nrows(), bboxes1.ncols());
    let mut val_inter: DMatrix<f64> = DMatrix::repeat(bboxes1.nrows(), bboxes1.ncols(), 1.);
    let mut val_b1: DMatrix<f64> = DMatrix::repeat(bboxes1.nrows(), bboxes1.ncols(), 1.);
    let mut val_b2: DMatrix<f64> = DMatrix::repeat(bboxes1.nrows(), bboxes1.ncols(), 1.);

    for d in 0..dim {
        coords1[d] = matrix_maximum(&coords_b1[d], &coords_b2[d].transpose());
        coords2[d] = matrix_minimum(&coords_b1[d + dim], &coords_b2[d + dim].transpose());

        let sub = coords2[d].clone() - coords1[d].clone();
        val_inter *= matrix_maximum(&sub, &zero);
        val_b1 *= coords_b1[d + dim].clone() - coords_b1[d].clone();
        val_b2 *= coords_b2[d + dim].clone() - coords_b2[d].clone();
    }

    let tmp = val_b1 + val_b2.transpose() - val_inter.clone();
    let iou = matrix_div(&val_inter, &matrix_clip(&tmp, 0., 0.));

    iou
}

fn cost_matrix_iou_feature(
    trackers: &Vec<Box<dyn Tracker>>,
    detections: &Vec<Detection>,
    feature_similarity_fn: Box<dyn FnOnce() -> f64>,
    feature_similarity_beta: Option<f64>
) {
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
        let mut vs = (*detection)._box.iter().map(|m| m.clone()).collect::<Vec<_>>();
        data.append(&mut vs);
    }
    let b2 = DMatrix::from_vec(r, c, data);
    let inferred_dim = b1.shape().1 / 2;
    //let iou_mat = calculate_iou(b1, b2, inferred_dim);
}

fn match_by_cost_matrix(
    trackers: &Vec<Box<dyn Tracker>>,
    detections: &Vec<Detection>,
    min_iou: f64,
    multi_match_min_iou: f64,
    kwargs: HashMap<String, f64>
) -> DMatrix<f64> {
    if trackers.len() == 0 || detections.len() == 0 {
        return dmatrix![];
    }



    dmatrix![]
}

trait BaseMatchingFunction {
    fn call(&self, trackers: &Vec<Box<dyn Tracker>>, detections: &Vec<Detection>) -> DMatrix<f64>;
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

        let matches = self.matching_fn.as_ref().map(|v| v.call(&self.trackers, &detections));
        self.detections_matched_ids = Vec::with_capacity(detections.len());

        for _match in matches {

        }
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

    #[test]
    fn test_matrix_to_vec() {
        let _box = DMatrix::from_row_slice(2, 4, &[
            1., 2., 3., 4.,
            5., 6., 7., 8.,
        ]);
        let actual = matrix_to_vec(&_box);

        assert!(actual == vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn test_matrix_split() {
        let _box = DMatrix::from_row_slice(2, 4, &[
            1., 2., 3., 4.,
            5., 6., 7., 8.,
        ]);

        let actual = matrix_split(&_box, 4);
        let expect = vec![
            DMatrix::from_row_slice(2, 1, &[1., 5.]),
            DMatrix::from_row_slice(2, 1, &[2., 6.]),
            DMatrix::from_row_slice(2, 1, &[3., 7.]),
            DMatrix::from_row_slice(2, 1, &[4., 8.]),
        ];

        assert!(actual == expect);
    }

    #[test]
    fn test_matrix_maximum() {
        let a = DMatrix::from_row_slice(2, 4, &[
            1., 2., 3., 4.,
            5., 6., 7., 8.,
        ]);
        let b = DMatrix::from_row_slice(1, 4, &[0., 1., 9., 10.]);

        let actual = matrix_maximum(&a, &b);
        let expect = DMatrix::from_row_slice(2, 4, &[
            1., 2., 9., 10.,
            5., 6., 9., 10.,
        ]);

        assert!(actual == expect);
    }
}
