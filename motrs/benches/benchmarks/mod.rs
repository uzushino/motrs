use criterion::{BatchSize, Criterion, criterion_group};
use motrs::{BoundingBox, Detection, ModelConfig, MotionModel, MultiObjectTracker, TrackerConfig};
use std::hint::black_box;

pub fn step(criterion: &mut Criterion) {
    criterion.bench_function("step 2", |bench| {
        bench.iter_batched(
            || {
                MultiObjectTracker::new(TrackerConfig {
                    model: ModelConfig {
                        position: MotionModel::ConstantAcceleration,
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .unwrap()
            },
            |mut tracker| {
                for frame in 0..2 {
                    let offset = frame as f32;
                    let mut detection = Detection::new(
                        BoundingBox::new([1.0 + offset; 2], [10.0 + offset; 2]).unwrap(),
                    );
                    detection.class_id = Some(1);
                    black_box(tracker.step([detection]).unwrap());
                }
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, step);
