use criterion::{criterion_group, Criterion};

use nalgebra as na;
use motrs::model::*;
use motrs::tracker::*;

pub fn step(c: &mut Criterion) {
    let model_spec = ModelPreset::constant_acceleration_and_static_box_size_2d();
    let mut tracker = MultiObjectTracker::new(
        0.1, // fps
        model_spec,
        None,
        Some(SingleObjectTrackerKwargs {
            ..Default::default()
        }),
        None,
        None,
    );

    c.bench_function("step 2", |b| {
        b.iter(|| {
            let mut _box = na::DMatrix::from_row_slice(1, 4, &[1., 1., 1., 1.]);
            for _ in 0..2 {
                _box = _box.clone() + na::DMatrix::from_row_slice(1, 4, &[1., 1., 1., 1.]);
                let det = Detection {
                    _box: Some(_box.clone()),
                    score: 1.,
                    class_id: 1,
                    feature: None,
                };
                tracker.step(vec![det]);

            }
        })
    });
}

criterion_group!(benches, step);
