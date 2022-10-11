use motrs::model::*;
use motrs::tracker::*;
use wasm_bindgen::prelude::*;
use serde_derive::{Serialize, Deserialize};
use gloo_utils::format::JsValueSerdeExt;
use web_sys::console;
use nalgebra as na;

#[wasm_bindgen]
pub struct MOT {
    tracker: MultiObjectTracker,
}

#[derive(Serialize, Deserialize)]
pub struct Box {
    pub _box: Vec<f32>,
}

#[wasm_bindgen]
impl MOT {
    pub fn new() -> Self {
        let model_spec = ModelPreset::constant_acceleration_and_static_box_size_2d();
        let min_iou = 0.25;
        let multi_match_min_iou = 1. + 1e-7;
        let feature_similarity_fn = None;
        let feature_similarity_beta = None;
        let matching_fn = IOUAndFeatureMatchingFunction::new(
            min_iou,
            multi_match_min_iou,
            feature_similarity_fn,
            feature_similarity_beta,
        );

        let tracker = MultiObjectTracker::new(
            1. / 30., // fps
            model_spec,
            Some(matching_fn),
            Some(SingleObjectTrackerKwargs {
                max_staleness: 15.,
                ..Default::default()
            }),
            None,
            None,
        );

        Self { tracker }
    }

    pub fn step(&mut self, val: &JsValue) {
        let _box: Vec<Box> = val
            .into_serde()
            .unwrap();

        console::log_1(&format!("len: {}", _box.len()).into());

        let _box = na::DMatrix::from_row_slice(1, 4, &_box[0]._box);

        let dets = Detection {
            _box: Some(_box.clone()),
            score: 1.,
            class_id: 1,
            feature: None,
        };

        self.tracker.step(vec![dets]);
    }

    pub fn active_tracks(&mut self) {
        let tracks = self.tracker.active_tracks();

        console::log_1(&format!("MOT tracker tracks {} objects", tracks.len()).into());
        console::log_1(&format!("first track box: {}", tracks[0]._box).into());
    }
}
