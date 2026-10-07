use motrs::{BoundingBox, Detection, Error, ModelConfig, MultiObjectTracker, TrackerConfig};

fn detection() -> Detection {
    Detection::new(BoundingBox::new([0.0; 2], [10.0; 2]).unwrap())
}

#[test]
fn default_tracker_accepts_empty_and_multiple_detection_frames() {
    let mut tracker = MultiObjectTracker::default();
    assert!(tracker.step([]).unwrap().is_empty());
    let mut second = detection();
    second.bounds = BoundingBox::new([100.0; 2], [110.0; 2]).unwrap();
    assert_eq!(tracker.step([detection(), second]).unwrap().len(), 2);
    assert_eq!(tracker.matched_ids().len(), 2);
}

#[test]
fn invalid_inputs_do_not_advance_existing_tracks() {
    let mut invalid = vec![];
    for score in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        let mut value = detection();
        value.score = score;
        invalid.push(value);
    }
    for feature in [vec![], vec![f32::NAN], vec![f32::INFINITY]] {
        let mut value = detection();
        value.feature = Some(feature);
        invalid.push(value);
    }
    invalid.push(Detection::new(BoundingBox::new([0.0], [10.0]).unwrap()));
    for value in invalid {
        let mut tracker = MultiObjectTracker::default();
        let before = tracker.step([detection()]).unwrap();
        let ids = tracker.matched_ids().to_vec();
        assert!(matches!(
            tracker.step([detection(), value]),
            Err(Error::InvalidDetection { index: 1, .. })
        ));
        let after = tracker.active_tracks();
        assert_eq!(before[0].bounds, after[0].bounds);
        assert_eq!(before[0].age, after[0].age);
        assert_eq!(before[0].hits, after[0].hits);
        assert_eq!(tracker.matched_ids(), ids);
    }
}

#[test]
fn invalid_configurations_are_rejected_at_construction() {
    let mut invalid = vec![];
    for dt in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX] {
        invalid.push(TrackerConfig {
            dt,
            ..Default::default()
        });
    }
    invalid.push(TrackerConfig {
        min_hits: 0,
        ..Default::default()
    });
    invalid.push(TrackerConfig {
        min_iou: 1.1,
        ..Default::default()
    });
    invalid.push(TrackerConfig {
        appearance_weight: -0.1,
        ..Default::default()
    });
    invalid.push(TrackerConfig {
        smoothing: f32::NAN,
        ..Default::default()
    });
    for model in [
        ModelConfig {
            dimensions: 0,
            ..Default::default()
        },
        ModelConfig {
            position_process_variance: -1.0,
            ..Default::default()
        },
        ModelConfig {
            size_process_variance: f32::NAN,
            ..Default::default()
        },
        ModelConfig {
            position_measurement_variance: 0.0,
            ..Default::default()
        },
        ModelConfig {
            size_measurement_variance: f32::INFINITY,
            ..Default::default()
        },
        ModelConfig {
            initial_state_variance: -1.0,
            ..Default::default()
        },
    ] {
        invalid.push(TrackerConfig {
            model,
            ..Default::default()
        });
    }
    for config in invalid {
        assert!(matches!(
            MultiObjectTracker::new(config),
            Err(Error::InvalidConfig(_))
        ));
    }
}

#[test]
fn zero_norm_or_different_length_features_do_not_break_tracking() {
    let mut tracker = MultiObjectTracker::new(TrackerConfig {
        appearance_weight: 1.0,
        ..Default::default()
    })
    .unwrap();
    let mut observed = detection();
    observed.feature = Some(vec![0.0, 0.0]);
    let first = tracker.step([observed.clone()]).unwrap().remove(0);
    observed.feature = Some(vec![1.0]);
    let second = tracker.step([observed]).unwrap().remove(0);
    assert_eq!(first.id, second.id);
}

#[test]
fn zero_overlap_never_matches_even_when_the_threshold_is_zero() {
    let mut tracker = MultiObjectTracker::new(TrackerConfig {
        min_iou: 0.0,
        ..Default::default()
    })
    .unwrap();
    let first = tracker.step([detection()]).unwrap().remove(0);
    let distant = Detection::new(BoundingBox::new([100.0; 2], [110.0; 2]).unwrap());
    tracker.step([distant]).unwrap();
    assert_ne!(first.id, tracker.matched_ids()[0]);
}
