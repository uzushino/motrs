use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group};
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

fn tracking_frame(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tracking frame");
    for count in [1, 10, 50, 200] {
        let frame = |offset: f32| -> Vec<_> {
            (0..count)
                .map(|object| {
                    let x = (object % 20) as f32 * 30.0 + offset;
                    let y = (object / 20) as f32 * 30.0;
                    Detection::new(BoundingBox::new([x, y], [x + 12.0, y + 12.0]).unwrap())
                })
                .collect()
        };
        let frames = [frame(0.0), frame(0.5)];
        let mut tracker = MultiObjectTracker::default();
        tracker.step(frames[0].clone()).unwrap();
        let mut next_frame = 1;
        group.bench_function(BenchmarkId::from_parameter(count), |bench| {
            bench.iter_batched(
                || {
                    let detections = frames[next_frame].clone();
                    next_frame ^= 1;
                    detections
                },
                |detections| black_box(tracker.step(detections).unwrap()),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, step, tracking_frame);
