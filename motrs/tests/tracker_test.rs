use approx::assert_relative_eq;
use motrs::{BoundingBox, Detection, ModelConfig, MotionModel, MultiObjectTracker, TrackerConfig};

fn detection(offset: f32, dimensions: usize) -> Detection {
    Detection::new(
        BoundingBox::new(vec![offset; dimensions], vec![offset + 10.0; dimensions]).unwrap(),
    )
}

#[test]
fn follows_motion_with_stable_ids_in_one_two_and_three_dimensions() {
    for dimensions in [1, 2, 3] {
        for position in [
            MotionModel::ConstantVelocity,
            MotionModel::ConstantAcceleration,
        ] {
            let mut tracker = MultiObjectTracker::new(TrackerConfig {
                model: ModelConfig {
                    dimensions,
                    position,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
            let first = tracker
                .step([detection(0.0, dimensions), detection(100.0, dimensions)])
                .unwrap();
            for frame in 1..100 {
                let offset = frame as f32 * 0.2;
                // Observation order changes, but the identities must stay stable.
                let tracks = tracker
                    .step([
                        detection(offset + 100.0, dimensions),
                        detection(offset, dimensions),
                    ])
                    .unwrap();
                assert_eq!(tracks.len(), 2);
                assert_eq!(
                    tracker.matched_ids(),
                    &[first[1].id.clone(), first[0].id.clone()]
                );
                for axis in 0..dimensions {
                    assert_relative_eq!(tracks[0].bounds.min()[axis], offset, epsilon = 0.1);
                    assert_relative_eq!(
                        tracks[1].bounds.min()[axis],
                        offset + 100.0,
                        epsilon = 0.1
                    );
                }
            }
        }
    }
}

#[test]
fn missed_observations_are_predicted_and_recovery_resets_the_counter() {
    let mut tracker = MultiObjectTracker::new(TrackerConfig {
        max_missed_frames: 2,
        ..Default::default()
    })
    .unwrap();
    let first = tracker.step([detection(0.0, 2)]).unwrap().remove(0);
    tracker.step([detection(0.2, 2)]).unwrap();
    let predicted = tracker.step([]).unwrap().remove(0);
    assert_eq!(predicted.id, first.id);
    assert_eq!(predicted.missed_frames, 1);
    assert!(predicted.bounds.min()[0] > 0.2);
    let recovered = tracker.step([detection(0.6, 2)]).unwrap().remove(0);
    assert_eq!(recovered.id, first.id);
    assert_eq!(recovered.missed_frames, 0);
    assert_eq!(recovered.hits, 3);
    assert_eq!(recovered.age, 4);
    assert_eq!(tracker.step([]).unwrap().len(), 1);
    assert_eq!(tracker.step([]).unwrap().len(), 1);
    assert!(tracker.step([]).unwrap().is_empty());
    assert!(tracker.is_empty());
}

#[test]
fn new_tracks_are_current_and_confirmation_requires_observations() {
    let mut tracker = MultiObjectTracker::new(TrackerConfig {
        min_hits: 2,
        ..Default::default()
    })
    .unwrap();
    assert!(tracker.step([detection(0.0, 2)]).unwrap().is_empty());
    assert_eq!(tracker.len(), 1);
    assert!(tracker.step([]).unwrap().is_empty());
    let tracks = tracker.step([detection(0.0, 2)]).unwrap();
    assert_eq!(tracks.len(), 1);
    assert_eq!(tracks[0].missed_frames, 0);
    assert_eq!(tracks[0].score, 1.0);
}

#[test]
fn unrelated_observations_create_new_tracks() {
    let mut tracker = MultiObjectTracker::default();
    let first = tracker.step([detection(0.0, 2)]).unwrap().remove(0);
    let tracks = tracker.step([detection(100.0, 2)]).unwrap();
    assert_eq!(tracks.len(), 2);
    assert_ne!(tracker.matched_ids()[0], first.id);
    assert_eq!(tracks[0].missed_frames, 1);
    assert_eq!(tracks[1].missed_frames, 0);
}

#[test]
fn appearance_is_available_from_the_first_detection() {
    let mut tracker = MultiObjectTracker::new(TrackerConfig {
        appearance_weight: 0.5,
        ..Default::default()
    })
    .unwrap();
    let mut red = detection(0.0, 2);
    red.feature = Some(vec![1.0, 0.0]);
    let mut blue = detection(0.0, 2);
    blue.feature = Some(vec![0.0, 1.0]);
    let first = tracker.step([red.clone(), blue.clone()]).unwrap();
    tracker.step([blue, red]).unwrap();
    assert_eq!(
        tracker.matched_ids(),
        &[first[1].id.clone(), first[0].id.clone()]
    );
}

#[test]
fn score_is_smoothed_and_class_uses_all_observations() {
    let mut tracker = MultiObjectTracker::new(TrackerConfig {
        smoothing: 0.5,
        ..Default::default()
    })
    .unwrap();
    let mut observed = detection(0.0, 2);
    observed.score = 0.8;
    observed.class_id = Some(2);
    tracker.step([observed.clone()]).unwrap();
    observed.score = 0.4;
    observed.class_id = Some(1);
    let second = tracker.step([observed.clone()]).unwrap().remove(0);
    assert_relative_eq!(second.score, 0.6);
    assert_eq!(second.class_id, Some(2)); // deterministic tie
    let third = tracker.step([observed]).unwrap().remove(0);
    assert_relative_eq!(third.score, 0.5);
    assert_eq!(third.class_id, Some(1));
}

#[test]
fn size_can_be_tracked_with_velocity_or_acceleration() {
    for size in [
        MotionModel::ConstantVelocity,
        MotionModel::ConstantAcceleration,
    ] {
        let mut tracker = MultiObjectTracker::new(TrackerConfig {
            model: ModelConfig {
                size,
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap();
        let first = tracker.step([detection(0.0, 2)]).unwrap().remove(0);
        for frame in 1..50 {
            let width = 10.0 + frame as f32 * 0.1;
            let observed = Detection::new(BoundingBox::new([0.0, 0.0], [width, 10.0]).unwrap());
            let track = tracker.step([observed]).unwrap().remove(0);
            assert_eq!(track.id, first.id);
            assert_relative_eq!(track.bounds.extent(0), width, epsilon = 0.1);
        }
    }
}
