/**
 * Refer: https://github.com/MichaelMauderer/filter-rs
 */
use nalgebra::{DVector, DMatrix, RealField};

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct KalmanFilter<F: RealField>
{
    pub dim_x: usize,
    pub dim_z: usize,
    pub dim_u: usize,

    pub x: DVector<F>,
    pub P: DMatrix<F>,
    pub x_prior: DVector<F>,
    pub P_prior: DMatrix<F>,
    pub x_post: DVector<F>,
    pub P_post: DMatrix<F>,
    pub z: Option<DVector<F>>,
    pub R: DMatrix<F>,
    pub Q: DMatrix<F>,
    pub B: Option<DMatrix<F>>,
    pub F: DMatrix<F>,
    pub H: DMatrix<F>,
    pub y: DVector<F>,
    pub K: DMatrix<F>,
    pub S: DMatrix<F>,
    pub SI: DMatrix<F>,
    pub alpha_sq: F,
}

#[allow(non_snake_case)]
impl<F> KalmanFilter<F> where F: RealField + Copy {
    pub fn new(dim_x: usize, dim_z: usize, dim_u: usize) -> Self {
        let x = DVector::<F>::from_element(dim_x, F::zero());
        let P = DMatrix::<F>::identity(dim_x, dim_x);
        let Q = DMatrix::<F>::identity(dim_x, dim_x);
        let F = DMatrix::<F>::identity(dim_x, dim_x);
        let H = DMatrix::<F>::from_element(dim_z, dim_x, F::zero());
        let R = DMatrix::<F>::identity(dim_z, dim_z);
        let alpha_sq = F::one();

        let z = None;

        let K = DMatrix::<F>::from_element(dim_x, dim_z, F::zero());
        let y = DVector::<F>::from_element(dim_z, F::one());
        let S = DMatrix::<F>::from_element(dim_z, dim_z, F::zero());
        let SI = DMatrix::<F>::from_element(dim_z, dim_z, F::zero());

        let x_prior = x.clone();
        let P_prior = P.clone();

        let x_post = x.clone();
        let P_post = P.clone();

        KalmanFilter {
            dim_x,
            dim_u,
            dim_z,

            x,
            P,
            x_prior,
            P_prior,
            x_post,
            P_post,
            z,
            R,
            Q,
            B: None,
            F,
            H,
            y,
            K,
            S,
            SI,
            alpha_sq,
        }
    }

    pub fn predict(&mut self,
        u: Option<&DVector<F>>,
        B: Option<&DMatrix<F>>, F: Option<&DMatrix<F>>,
        Q: Option<&DMatrix<F>>,
    ) {
        let B = if B.is_some() { B } else { self.B.as_ref() };
        let F = F.unwrap_or(&self.F);
        let Q = Q.unwrap_or(&self.Q);

        if B.is_some() && u.is_some() {
            self.x = F * self.x.clone() + B.unwrap() * u.unwrap();
        } else {
            self.x = F * self.x.clone();
        }

        self.P = ((F * self.P.clone()) * F.transpose()) * self.alpha_sq + Q;

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    /// Add a new measurement (z) to the Kalman filter.
    pub fn update(&mut self, z: &DVector<F>, R: Option<&DMatrix<F>>, H: Option<&DMatrix<F>>) {
        let R = R.unwrap_or(&self.R);
        let H = H.unwrap_or(&self.H);

        self.y = z - H * &self.x;

        let PHT = self.P.clone() * H.transpose();
        self.S = H * &PHT + R;

        self.SI = self.S.clone().try_inverse().unwrap();

        self.K = PHT * &self.SI;

        self.x = &self.x + &self.K * &self.y;

        let I_KH = DMatrix::identity(self.dim_x, self.dim_x) - &self.K * H;
        self.P =
            ((I_KH.clone() * &self.P) * I_KH.transpose()) + ((&self.K * R) * &self.K.transpose());

        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();
    }

    /// Predict state (prior) using the Kalman filter state propagation equations.
    /// Only x is updated, P is left unchanged.
    pub fn predict_steadystate(&mut self, u: Option<&DVector<F>>, B: Option<&DMatrix<F>>) {
        let B = if B.is_some() { B } else { self.B.as_ref() };

        if B.is_some() && u.is_some() {
            self.x = &self.F * &self.x + B.unwrap() * u.unwrap();
        } else {
            self.x = &self.F * &self.x;
        }

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    /// Add a new measurement (z) to the Kalman filter without recomputing the Kalman gain K,
    /// the state covariance P, or the system uncertainty S.
    pub fn update_steadystate(&mut self, z: &DVector<F>) {
        self.y = z - &self.H * &self.x;
        self.x = &self.x + &self.K * &self.y;

        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();
    }

    pub fn get_prediction(&self, u: Option<&DVector<F>>,) -> (DVector<F>, DMatrix<F>) {
        let Q = &self.Q;
        let F = &self.F;
        let P = &self.P;
        let FT = F.transpose();

        let B = self.B.as_ref();
        let x = {
            if B.is_some() && u.is_some() {
                F * &self.x + B.unwrap() * u.unwrap()
            } else {
                F * &self.x
            }
        };

        let P = ((F * P) * FT) * self.alpha_sq + Q;

        (x, P)
    }

    ///  Computes the new estimate based on measurement `z` and returns it without altering the state of the filter.
    pub fn get_update(&self, z: &DVector<F>) -> (DVector<F>, DMatrix<F>) {
        let R = &self.R;
        let H = &self.H;
        let P = &self.P;
        let x = &self.x;

        let y = z - H * &self.x;

        let PHT = &(P * H.transpose());

        let S = H * PHT + R;
        let SI = S.try_inverse().unwrap();

        let K = &(PHT * SI);

        let x = x + K * y;

        let I_KH = &(DMatrix::<F>::identity(self.dim_x, self.dim_x) - (K * H));

        let P = ((I_KH * P) * I_KH.transpose()) + ((K * R) * &K.transpose());

        (x, P)
    }

    /// Returns the residual for the given measurement (z). Does not alter the state of the filter.
    pub fn residual_of(&self, z: &DVector<F>) -> DVector<F> {
        z - (&self.H * &self.x_prior)
    }

    /// Helper function that converts a state into a measurement.
    pub fn measurement_of_state(&self, x: &DVector<F>) -> DVector<F> {
        &self.H * x
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::base::Vector1;
    use nalgebra::{U1, U2, Vector2, Matrix2, Matrix1, dvector};

    use super::*;

    #[test]
    fn test_univariate_kf_setup() {
        let mut kf: KalmanFilter<f32> = KalmanFilter::<f32>::new(1, 1, 1);

        for i in 0..1000 {
            let zf = i as f32;
            let z = dvector!(zf);

            kf.predict(None, None, None, None);
            kf.update(&z, None, None);

            assert_approx_eq!(zf, kf.z.clone().unwrap()[0]);
        }
    }

    #[test]
    fn test_1d_reference() {
        let mut kf: KalmanFilter<f64> = KalmanFilter::new(2, 1, 1);

        kf.x = dvector!(2.0, 0.0);

        kf.F = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0 ]);
        kf.H = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]).transpose();
        kf.P *= 1000.0;
        kf.R = DMatrix::from_row_slice(1, 1, &[5.0]).transpose();
        kf.Q = DMatrix::from_row_slice(2, 2, &[0.0001, 0.001, 0.0001, 0.001]).transpose();

        for t in 0..100 {
            let z = dvector!(t as f64);

            kf.update(&z, None, None);
            kf.predict(None, None, None, None);

            assert_approx_eq!(kf.x[0],
                              if t == 0 { 0.0099502487 } else { t as f64 + 1.0 },
                              0.05);
        }
    }
}
