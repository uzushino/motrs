use std::collections::HashMap;

use motrs::model::*;
use motrs::tracker::*;

use approx::assert_relative_eq;
use nalgebra as na;
use nalgebra::dmatrix;

mod testing;
use testing::data_generator;

#[test]
fn test_simple_tracking_objects_1() {
    let fps = 24.;
    let dt = 1. / fps;
    let num_steps = 240;
    let num_steps_warmup = 1. * fps;

    let mut model_spec = ModelPreset::constant_velocity_and_static_box_size_2d();
    model_spec.order_pos = 1;

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
    let mut mot = MultiObjectTracker::new(
        dt,
        model_spec,
        Some(matching_fn),
        None,
        None,
        Some(ActiveTracksKwargs::default()),
    );
    let mut history: HashMap<i64, Vec<String>> = HashMap::from([(0, vec![]), (1, vec![])]);
    let mut gen = data_generator(num_steps, 2, 0.01, 0.2, 0.0, 1.0).into_iter();

    for i in 0..num_steps {
        if let Some((dets_gt, dets_pred)) = gen.next() {
            let detections = dets_pred
                .into_iter()
                .filter(|d| d._box.is_some())
                .collect::<Vec<_>>();
            let _ = mot.step(detections);

            if (i as f32) <= num_steps_warmup {
                continue;
            }

            let matches = match_by_cost_matrix(
                &mot.trackers,
                &dets_gt,
                min_iou,
                multi_match_min_iou,
                None,
                feature_similarity_beta,
            );

            for m in 0..matches.shape().0 {
                let (gidx, tidx) = (matches[(m, 0)], matches[(m, 1)]);
                let track_id = mot.trackers[tidx as usize].id();

                history.get_mut(&(gidx as i64)).map(|f| f.push(track_id));
            }
        }

        assert!(mot.trackers.len() == 2);
    }
}

#[test]
fn test_tracker_diverges() {
    let spec = ModelPreset::constant_velocity_and_static_box_size_2d();
    let _box = dmatrix![0., 0., 10., 10.];
    let mut mot = MultiObjectTracker::new(
        0.1,
        spec,
        Some(IOUAndFeatureMatchingFunction::default()),
        None,
        None,
        None,
    );
    mot.step(vec![Detection {
        _box: Some(_box),
        ..Default::default()
    }]);

    assert!(mot.trackers.len() == 1);

    let _ = mot.active_tracks()[0].id.clone();

    assert_relative_eq!(mot.trackers[0].model().dt, 0.1, epsilon = 1e-3f32)
}

#[test]
fn main() {
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

    let mut mot = MultiObjectTracker::new(0.1, model_spec, Some(matching_fn), None, None, None);
    let mut object_box = na::DMatrix::from_row_slice(1, 4, &[1., 1., 10., 10.]);
    let det = Detection {
        _box: Some(object_box.clone()),
        score: 1.,
        class_id: 1,
        feature: None,
    };
    mot.step(vec![det]);

    let tracks = mot.active_tracks();
    let object_id = tracks[0].id.clone();

    for _step in 0..10 {
        object_box += na::DMatrix::from_row_slice(1, 4, &[1., 1., 1., 1.]);

        let det = Detection {
            _box: Some(object_box.clone()),
            score: 1.,
            class_id: 1,
            feature: None,
        };
        mot.step(vec![det]);

        let track = &tracks[0];
        assert!(object_id == track.id);
    }
}
