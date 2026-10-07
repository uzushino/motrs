use crate::model::MotionMatrices;
use crate::{BoundingBox, Error};
use faer::{Mat, linalg::solvers::Solve};

pub(crate) struct KalmanFilter {
    pub state: Mat<f32>,
    pub covariance: Mat<f32>,
}

impl KalmanFilter {
    pub fn new(model: &MotionMatrices, bounds: &BoundingBox) -> Self {
        Self {
            state: model.initial_state(bounds),
            covariance: model.initial_covariance.clone(),
        }
    }

    /// Advance the estimated center, velocity and size by one frame.
    /// P grows by Q because motion between observations is uncertain.
    pub fn predict(&mut self, model: &MotionMatrices) {
        self.state = &model.transition * &self.state;
        self.covariance = &model.transition * &self.covariance * model.transition.transpose()
            + &model.process_noise;
    }

    /// Blend the observation with the prediction according to their uncertainty.
    pub fn update(&mut self, model: &MotionMatrices, bounds: &BoundingBox) -> Result<(), Error> {
        // innovation = observed center/size - predicted center/size (z - Hx).
        let residual = model.measurement(bounds) - &model.observation * &self.state;
        let state_measurement_covariance = &self.covariance * model.observation.transpose();
        let innovation_covariance =
            &model.observation * &state_measurement_covariance + &model.measurement_noise;

        // S K^T = (P H^T)^T. Solving this system avoids constructing an inverse.
        let decomposition = innovation_covariance
            .llt(faer::Side::Lower)
            .map_err(|_| Error::NumericalFailure)?;
        let gain = decomposition
            .solve(state_measurement_covariance.transpose())
            .transpose()
            .to_owned();
        self.state = &self.state + &gain * residual;

        // Joseph form protects symmetry and positive semidefiniteness.
        let correction = Mat::<f32>::identity(self.state.nrows(), self.state.nrows())
            - &gain * &model.observation;
        self.covariance = &correction * &self.covariance * correction.transpose()
            + &gain * &model.measurement_noise * gain.transpose();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelConfig, MotionModel};
    use approx::assert_relative_eq;

    #[test]
    fn measurement_moves_state_and_reduces_uncertainty() {
        let model = MotionMatrices::new(
            &ModelConfig {
                dimensions: 1,
                position: MotionModel::Static,
                initial_state_variance: 1.0,
                ..Default::default()
            },
            0.1,
        );
        let mut filter = KalmanFilter::new(&model, &BoundingBox::new([0.0], [2.0]).unwrap());
        filter
            .update(&model, &BoundingBox::new([2.0], [4.0]).unwrap())
            .unwrap();
        assert_relative_eq!(filter.state[(0, 0)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(filter.covariance[(0, 0)], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn repeated_prediction_and_update_keep_covariance_symmetric() {
        let model = MotionMatrices::new(&ModelConfig::default(), 0.1);
        let mut filter = KalmanFilter::new(&model, &BoundingBox::new([0.0; 2], [10.0; 2]).unwrap());
        for frame in 1..100 {
            filter.predict(&model);
            let offset = frame as f32 * 0.1;
            filter
                .update(
                    &model,
                    &BoundingBox::new([offset; 2], [offset + 10.0; 2]).unwrap(),
                )
                .unwrap();
            for row in 0..filter.covariance.nrows() {
                assert!(filter.covariance[(row, row)] >= 0.0);
                for col in 0..filter.covariance.ncols() {
                    assert_relative_eq!(
                        filter.covariance[(row, col)],
                        filter.covariance[(col, row)],
                        epsilon = 1e-4
                    );
                }
            }
        }
    }

    #[test]
    fn prediction_integrates_velocity_and_acceleration_over_dt() {
        let model = MotionMatrices::new(
            &ModelConfig {
                dimensions: 1,
                position: MotionModel::ConstantAcceleration,
                ..Default::default()
            },
            2.0,
        );
        let mut filter = KalmanFilter::new(&model, &BoundingBox::new([0.0], [2.0]).unwrap());
        filter.state[(1, 0)] = 3.0;
        filter.state[(2, 0)] = 4.0;
        filter.predict(&model);
        assert_eq!(filter.state[(0, 0)], 15.0); // 1 + 3*2 + 4*2^2/2
        assert_eq!(filter.state[(1, 0)], 11.0);
        assert_eq!(filter.state[(2, 0)], 4.0);
    }

    #[test]
    fn less_reliable_observations_have_less_influence() {
        for (variance, expected_center) in [(1.0, 2.0), (3.0, 1.5)] {
            let model = MotionMatrices::new(
                &ModelConfig {
                    dimensions: 1,
                    position: MotionModel::Static,
                    initial_state_variance: 1.0,
                    position_measurement_variance: variance,
                    ..Default::default()
                },
                0.1,
            );
            let mut filter = KalmanFilter::new(&model, &BoundingBox::new([0.0], [2.0]).unwrap());
            filter
                .update(&model, &BoundingBox::new([2.0], [4.0]).unwrap())
                .unwrap();
            assert_relative_eq!(filter.state[(0, 0)], expected_center, epsilon = 1e-6);
        }
    }
}
