use gloo_utils::format::JsValueSerdeExt;
use motrs::matrix::matrix_to_vec;
use motrs::model::*;
use motrs::tracker::*;
use nalgebra as na;
use serde_derive::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_sys::console;

#[wasm_bindgen]
pub struct MOT {
    tracker: MultiObjectTracker,
}

#[derive(Serialize, Deserialize)]
pub struct TrackBox {
    pub id: String,
    pub _box: Vec<f32>,
    pub score: Option<f32>,
    pub class_id: Option<i64>,
}

#[derive(Serialize, Deserialize)]
pub struct Box {
    pub _box: Vec<f32>,
}

#[wasm_bindgen]
impl MOT {
    pub fn new() -> Self {
        let model_spec = ModelPreset::constant_acceleration_and_static_box_size_2d();
        let min_iou = 0.1;
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
            0.1, // fps
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
        let _box: Vec<Box> = val.into_serde().unwrap();

        let _box = na::DMatrix::from_row_slice(1, 4, &_box[0]._box);
        let dets = Detection {
            _box: Some(_box.clone()),
            score: 1.,
            class_id: 1,
            feature: None,
        };
        //console::log_1(&format!("{:?}", self.tracker.detections_matched_ids).into())
        self.tracker.step(vec![dets]);
    }

    pub fn active_tracks(&mut self) -> JsValue {
        let tracks = self.tracker.active_tracks();

        let track_objects = tracks
            .iter()
            .map(|v| TrackBox {
                id: v.id.clone(),
                _box: matrix_to_vec(&v._box),
                score: v.score,
                class_id: v.class_id,
            })
            .collect::<Vec<TrackBox>>();

        JsValue::from_serde(&track_objects).unwrap()
    }
}
