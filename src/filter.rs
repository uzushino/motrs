use crate::matrix::*;
/**
 * Refer: https://github.com/MichaelMauderer/filter-rs
 */
use nalgebra::{DMatrix, RealField};
use std::ops::Mul;

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct KalmanFilter {
    pub dim_x: usize,
    pub dim_z: usize,
    pub dim_u: usize,

    pub x: DMatrix<f64>,
    pub P: DMatrix<f64>,
    pub x_prior: DMatrix<f64>,
    pub P_prior: DMatrix<f64>,
    pub x_post: DMatrix<f64>,
    pub P_post: DMatrix<f64>,
    pub z: Option<DMatrix<f64>>,
    pub R: DMatrix<f64>,
    pub Q: DMatrix<f64>,
    pub B: Option<DMatrix<f64>>,
    pub F: DMatrix<f64>,
    pub H: DMatrix<f64>,
    pub y: DMatrix<f64>,
    pub K: DMatrix<f64>,
    pub S: DMatrix<f64>,
    pub SI: DMatrix<f64>,
    pub alpha_sq: f64,
}

#[allow(non_snake_case)]
impl KalmanFilter {
    pub fn new(dim_x: usize, dim_z: usize, dim_u: usize) -> Self {
        let x = DMatrix::<f64>::from_element(1, dim_x, 0.0);
        let P = DMatrix::<f64>::identity(dim_x, dim_x);
        let Q = DMatrix::<f64>::identity(dim_x, dim_x);
        let F = DMatrix::<f64>::identity(dim_x, dim_x);
        let H = DMatrix::<f64>::from_element(dim_z, dim_x, 0.0);
        let R = DMatrix::<f64>::identity(dim_z, dim_z);
        let alpha_sq = 1.0;

        let z = None;

        let K = DMatrix::<f64>::from_element(dim_x, dim_z, 0.0);
        let y = DMatrix::<f64>::from_element(1, dim_z, 1.0);
        let S = DMatrix::<f64>::from_element(dim_z, dim_z, 0.0);
        let SI = DMatrix::<f64>::from_element(dim_z, dim_z, 0.0);

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

    pub fn predict(
        &mut self,
        u: Option<&DMatrix<f64>>,
        B: Option<&DMatrix<f64>>,
        F: Option<&DMatrix<f64>>,
        Q: Option<&DMatrix<f64>>,
    ) {
        let B = if B.is_some() { B } else { self.B.as_ref() };
        let F = F.unwrap_or(&self.F);
        let Q = Q.unwrap_or(&self.Q);

        if B.is_some() && u.is_some() {
            self.x = matrix_dot(&F, &self.x.clone()) + matrix_dot(&B.unwrap(), &u.unwrap());
        } else {
            self.x = matrix_dot(&F, &self.x.clone());
        }

        self.P = matrix_dot(&matrix_dot(&F, &self.P.clone()), &F.transpose()) * self.alpha_sq + Q;

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    /// Add a new measurement (z) to the Kalman filter.
    pub fn update(&mut self, z: &DMatrix<f64>, R: Option<&DMatrix<f64>>, H: Option<&DMatrix<f64>>) {
        let R = R.unwrap_or(&self.R);
        let H = H.unwrap_or(&self.H);

        self.y = z - matrix_dot(&H, &self.x.clone());

        let PHT = matrix_dot(&self.P.clone(), &H.transpose());

        self.S = matrix_dot(&H, &PHT) + R;
        self.SI = self.S.clone().pseudo_inverse(0.0001).unwrap();
        self.K = matrix_dot(&PHT, &self.SI);
        self.x = matrix_add(&self.x, &matrix_dot(&self.K, &self.y.clone()));

        let I_KH = matrix_sub(
            &DMatrix::identity(self.dim_x, self.dim_x),
            &matrix_dot(&self.K, &H),
        );

        let p1 = matrix_dot(&matrix_dot(&I_KH.clone(), &self.P), &I_KH.transpose());
        let p2 = matrix_dot(&matrix_dot(&self.K.clone(), &R), &self.K.transpose());

        self.P = p1 + p2;

        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();
    }

    pub fn predict_steadystate(&mut self, u: Option<&DMatrix<f64>>, B: Option<&DMatrix<f64>>) {
        let B = if B.is_some() { B } else { self.B.as_ref() };

        if B.is_some() && u.is_some() {
            self.x = &self.F * &self.x + B.unwrap() * u.unwrap();
        } else {
            self.x = &self.F * &self.x;
        }

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    pub fn update_steadystate(&mut self, z: &DMatrix<f64>) {
        self.y = z - &self.H * &self.x;
        self.x = &self.x + &self.K * &self.y;

        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();
    }

    pub fn get_prediction(&self, u: Option<&DMatrix<f64>>) -> (DMatrix<f64>, DMatrix<f64>) {
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

    pub fn get_update(&self, z: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
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

        let I_KH = &(DMatrix::<f64>::identity(self.dim_x, self.dim_x) - (K * H));

        let P = ((I_KH * P) * I_KH.transpose()) + ((K * R) * &K.transpose());

        (x, P)
    }

    pub fn residual_of(&self, z: &DMatrix<f64>) -> DMatrix<f64> {
        z - (&self.H * &self.x_prior)
    }

    pub fn measurement_of_state(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        &self.H * x
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::base::Vector1;
    use nalgebra::{dmatrix, Matrix1, Matrix2, Vector2, U1, U2};

    use super::*;

    #[test]
    fn test_univariate_kf_setup() {
        let mut kf: KalmanFilter = KalmanFilter::new(1, 1, 1);

        for i in 0..1000 {
            let zf = i as f64;
            let z = dmatrix!(zf);

            kf.predict(None, None, None, None);
            kf.update(&z, None, None);

            assert_approx_eq!(zf, kf.z.clone().unwrap()[0]);
        }
    }
}
