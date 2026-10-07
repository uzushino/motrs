use crate::{BoundingBox, Error, ModelConfig, MotionModel};
use faer::{Mat, Scale};

/// Immutable matrices shared by all tracks. State entries are grouped by axis:
/// [position, velocity, acceleration, ...size and its derivatives...].
pub(crate) struct MotionMatrices {
    pub transition: Mat<f32>,
    pub process_noise: Mat<f32>,
    pub observation: Mat<f32>,
    pub measurement_noise: Mat<f32>,
    pub initial_covariance: Mat<f32>,
    dimensions: usize,
    position_stride: usize,
    size_stride: usize,
}

impl MotionMatrices {
    pub fn new(config: &ModelConfig, dt: f32) -> Self {
        let position_stride = config.position.state_size();
        let size_stride = config.size.state_size();
        let state_size = config.dimensions * (position_stride + size_stride);
        let measurement_size = config.dimensions * 2;
        let mut transition = Mat::zeros(state_size, state_size);
        let mut process_noise = Mat::zeros(state_size, state_size);
        let mut observation = Mat::zeros(measurement_size, state_size);
        let mut measurement_noise = Mat::zeros(measurement_size, measurement_size);
        let mut state_offset = 0;

        for (measurement_group, motion, process_variance, measurement_variance) in [
            (
                0,
                config.position,
                config.position_process_variance,
                config.position_measurement_variance,
            ),
            (
                config.dimensions,
                config.size,
                config.size_process_variance,
                config.size_measurement_variance,
            ),
        ] {
            let stride = motion.state_size();
            let axis_transition = transition_block(motion, dt);
            let axis_noise = process_noise_block(motion, dt, process_variance);
            for axis in 0..config.dimensions {
                transition
                    .submatrix_mut(state_offset, state_offset, stride, stride)
                    .copy_from(&axis_transition);
                process_noise
                    .submatrix_mut(state_offset, state_offset, stride, stride)
                    .copy_from(&axis_noise);
                observation[(measurement_group + axis, state_offset)] = 1.0;
                measurement_noise[(measurement_group + axis, measurement_group + axis)] =
                    measurement_variance;
                state_offset += stride;
            }
        }
        Self {
            transition,
            process_noise,
            observation,
            measurement_noise,
            initial_covariance: Mat::<f32>::identity(state_size, state_size)
                * Scale(config.initial_state_variance),
            dimensions: config.dimensions,
            position_stride,
            size_stride,
        }
    }

    pub fn initial_state(&self, bounds: &BoundingBox) -> Mat<f32> {
        let mut state = Mat::zeros(self.transition.nrows(), 1);
        for axis in 0..self.dimensions {
            state[(axis * self.position_stride, 0)] = bounds.center(axis);
            let size_index = self.dimensions * self.position_stride + axis * self.size_stride;
            state[(size_index, 0)] = bounds.extent(axis);
        }
        state
    }

    pub fn measurement(&self, bounds: &BoundingBox) -> Mat<f32> {
        Mat::from_fn(self.dimensions * 2, 1, |row, _| {
            if row < self.dimensions {
                bounds.center(row)
            } else {
                bounds.extent(row - self.dimensions)
            }
        })
    }

    pub fn bounds(&self, state: &Mat<f32>) -> Result<BoundingBox, Error> {
        let mut min = Vec::with_capacity(self.dimensions);
        let mut max = Vec::with_capacity(self.dimensions);
        for axis in 0..self.dimensions {
            let center = state[(axis * self.position_stride, 0)];
            let size_index = self.dimensions * self.position_stride + axis * self.size_stride;
            let half_extent = state[(size_index, 0)] * 0.5;
            min.push(center - half_extent);
            max.push(center + half_extent);
        }
        BoundingBox::new(min, max)
    }
}

fn transition_block(motion: MotionModel, dt: f32) -> Mat<f32> {
    match motion {
        MotionModel::Static => faer::mat![[1.0]],
        MotionModel::ConstantVelocity => faer::mat![[1.0, dt], [0.0, 1.0]],
        MotionModel::ConstantAcceleration => {
            faer::mat![[1.0, dt, dt * dt * 0.5], [0.0, 1.0, dt], [0.0, 0.0, 1.0]]
        }
    }
}

fn process_noise_block(motion: MotionModel, dt: f32, variance: f32) -> Mat<f32> {
    // Discrete white noise is the outer product of its effect on each state
    // derivative, multiplied by the driving-noise variance.
    let influence = match motion {
        MotionModel::Static => faer::mat![[1.0]],
        MotionModel::ConstantVelocity => faer::mat![[0.5 * dt * dt], [dt]],
        MotionModel::ConstantAcceleration => faer::mat![[0.5 * dt * dt], [dt], [1.0]],
    };
    &influence * influence.transpose() * Scale(variance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn noise_coefficients_follow_the_outer_product() {
        let noise = process_noise_block(MotionModel::ConstantVelocity, 2.0, 3.0);
        assert_eq!(noise, faer::mat![[12.0, 12.0], [12.0, 12.0]]);
        let noise = process_noise_block(MotionModel::ConstantAcceleration, 2.0, 3.0);
        assert_eq!(
            noise,
            faer::mat![[12.0, 12.0, 6.0], [12.0, 12.0, 6.0], [6.0, 6.0, 3.0]]
        );
    }

    #[test]
    fn state_round_trip_preserves_two_and_three_dimensional_boxes() {
        for dimensions in [2, 3] {
            let config = ModelConfig {
                dimensions,
                position: MotionModel::ConstantAcceleration,
                ..Default::default()
            };
            let model = MotionMatrices::new(&config, 0.1);
            let bounds = BoundingBox::new(vec![10.0; dimensions], vec![20.0; dimensions]).unwrap();
            let state = model.initial_state(&bounds);
            assert_eq!(model.bounds(&state).unwrap(), bounds);
            let observation = &model.observation * state;
            for axis in 0..dimensions {
                assert_relative_eq!(observation[(axis, 0)], 15.0);
                assert_relative_eq!(observation[(dimensions + axis, 0)], 10.0);
            }
        }
    }
}
