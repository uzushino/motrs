use motrs::{BoundingBox, Detection};
use rand::{RngExt, SeedableRng, rngs::StdRng};

pub struct Frame {
    pub expected: Vec<BoundingBox>,
    pub detections: Vec<Detection>,
}

struct Object {
    phase: f32,
    speed: f32,
    color: [f32; 3],
}

/// A repeatable scene with noisy observations and occasional missed detections.
pub struct SyntheticScene {
    objects: Vec<Object>,
    rng: StdRng,
    frame: usize,
    total_frames: usize,
}

impl SyntheticScene {
    pub fn new(total_frames: usize, object_count: usize) -> Self {
        let mut rng = StdRng::from_seed([13; 32]);
        let objects = (0..object_count)
            .map(|_| Object {
                phase: rng.random_range(0.0..std::f32::consts::TAU),
                speed: rng.random_range(0.3..0.8),
                color: [
                    rng.random_range(0.2..1.0),
                    rng.random_range(0.2..1.0),
                    rng.random_range(0.2..1.0),
                ],
            })
            .collect();
        Self {
            objects,
            rng,
            frame: 0,
            total_frames,
        }
    }
}

impl Iterator for SyntheticScene {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        if self.frame >= self.total_frames {
            return None;
        }
        let time = self.frame as f32 / 30.0;
        self.frame += 1;
        let mut expected = Vec::with_capacity(self.objects.len());
        let mut detections = Vec::with_capacity(self.objects.len());
        for object in &self.objects {
            let angle = object.phase + object.speed * time;
            let left = 300.0 + 220.0 * angle.cos();
            let top = 300.0 + 200.0 * (angle * 1.3).sin();
            expected.push(
                BoundingBox::new([left, top], [left + 35.0, top + 45.0])
                    .expect("scene boxes have positive size"),
            );
            if self.rng.random::<f32>() < 0.15 {
                continue;
            }
            let noisy_left = left + self.rng.random_range(-3.0..3.0);
            let noisy_top = top + self.rng.random_range(-3.0..3.0);
            let mut detection = Detection::new(
                BoundingBox::new(
                    [noisy_left, noisy_top],
                    [noisy_left + 35.0, noisy_top + 45.0],
                )
                .expect("noise preserves box size"),
            );
            detection.feature = Some(object.color.to_vec());
            detections.push(detection);
        }
        Some(Frame {
            expected,
            detections,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_is_repeatable_and_stops_at_the_requested_frame() {
        let left = SyntheticScene::new(5, 20).collect::<Vec<_>>();
        let right = SyntheticScene::new(5, 20).collect::<Vec<_>>();
        assert_eq!(left.len(), 5);
        for (left, right) in left.iter().zip(&right) {
            assert_eq!(left.expected, right.expected);
            assert_eq!(left.detections.len(), right.detections.len());
            for (left, right) in left.detections.iter().zip(&right.detections) {
                assert_eq!(left.bounds, right.bounds);
                assert_eq!(left.feature, right.feature);
            }
        }
        assert!(left.iter().any(|frame| frame.detections.len() < 20));
    }
}
