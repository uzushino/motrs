use motrs::model::*;
use motrs::tracker::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct MOT {
    tracker: MultiObjectTracker,
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
}
