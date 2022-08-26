use nalgebra as na;

use motrs::model::ModelPreset;
use motrs::tracker::{Detection, IOUAndFeatureMatchingFunction, MultiObjectTracker};

fn main() {
    let mut object_box = na::DMatrix::from_row_slice(1, 4, &[1., 1., 10., 10.]);
    let model_spec = ModelPreset::constant_velocity_and_static_box_size_2d();
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
    let mut tracker = MultiObjectTracker::new(0.1, model_spec, Some(matching_fn), None, None, None);

    for _step in 0..10 {
        object_box += na::DMatrix::from_row_slice(1, 4, &[1., 1., 1., 1.]);

        let det = Detection {
            _box: Some(object_box.clone()),
            score: 1.,
            class_id: 1,
            feature: None,
        };

        tracker.step(vec![det]);

        let tracks = tracker.active_tracks(None);

        println!("MOT tracker tracks {} objects", tracks.len());
        println!("first track box: {}", tracks[0]._box);
    }
}
