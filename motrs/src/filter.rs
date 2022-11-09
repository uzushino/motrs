use crate::matrix::*;
/**
 * Refer: https://github.com/MichaelMauderer/filter-rs
 */
use nalgebra::DMatrix;
use nalgebra::Matrix5xX;
use nalgebra::Scalar;
use nalgebra::{ComplexField, RealField};
use num_traits::{Float, Num, One, Zero};
use std::ops::AddAssign;
use std::{fmt::Debug, ops::Add};

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct KalmanFilter<T> {
    pub dim_x: usize,
    pub dim_z: usize,
    pub dim_u: usize,

    pub x: DMatrix<T>,
    pub P: DMatrix<T>,
    pub x_prior: DMatrix<T>,
    pub P_prior: DMatrix<T>,
    pub x_post: DMatrix<T>,
    pub P_post: DMatrix<T>,
    pub z: Option<DMatrix<T>>,
    pub R: DMatrix<T>,
    pub Q: DMatrix<T>,
    pub B: Option<DMatrix<T>>,
    pub F: DMatrix<T>,
    pub H: DMatrix<T>,
    pub y: DMatrix<T>,
    pub K: DMatrix<T>,
    pub S: DMatrix<T>,
    pub SI: DMatrix<T>,
    pub alpha_sq: T,
}

#[allow(non_snake_case)]
impl<
        T: Zero
            + Debug
            + Clone
            + Scalar
            + Copy
            + One
            + Float
            + Num
            + AddAssign
            + ComplexField<RealField = T>,
    > KalmanFilter<T>
{
    pub fn new(dim_x: usize, dim_z: usize, dim_u: usize) -> Self {
        let x = DMatrix::<T>::from_element(1, dim_x, T::zero());
        let P = DMatrix::<T>::identity(dim_x, dim_x);
        let Q = DMatrix::<T>::identity(dim_x, dim_x);
        let F = DMatrix::<T>::identity(dim_x, dim_x);
        let H = DMatrix::<T>::from_element(dim_z, dim_x, T::zero());
        let R = DMatrix::<T>::identity(dim_z, dim_z);
        let alpha_sq = T::one();

        let z = None;

        let K = DMatrix::<T>::from_element(dim_x, dim_z, T::zero());
        let y = DMatrix::<T>::from_element(1, dim_z, T::one());
        let S = DMatrix::<T>::from_element(dim_z, dim_z, T::zero());
        let SI = DMatrix::<T>::from_element(dim_z, dim_z, T::zero());

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

    pub fn predict(&mut self) {
        let b = &self.B;
        let f = &self.F;
        let q = &self.Q;

        self.x = matrix_dot(&f, &self.x);
        self.P = matrix_dot(&matrix_dot(&f, &self.P), &f.transpose()) * self.alpha_sq + q;

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    /// Add a new measurement (z) to the Kalman filter.
    pub fn update(&mut self, z: &DMatrix<T>, R: Option<&DMatrix<T>>, H: Option<&DMatrix<T>>) {
        let R = R.unwrap_or(&self.R);
        let H = H.unwrap_or(&self.H);

        self.y = z - matrix_dot(&H, &self.x);
        let PHT = matrix_dot(&self.P, &H.transpose());

        self.S = matrix_dot(&H, &PHT) + R;
        self.SI = self
            .S
            .clone()
            .pseudo_inverse(T::from_f32(0.00001).unwrap())
            .unwrap();
        self.K = matrix_dot(&PHT, &self.SI);
        self.x = matrix_add(&self.x, &matrix_dot(&self.K, &self.y));

        let I_KH = matrix_sub(
            &DMatrix::identity(self.dim_x, self.dim_x),
            &matrix_dot(&self.K, &H),
        );

        let p1 = matrix_dot(&matrix_dot(&I_KH, &self.P), &I_KH.transpose());
        let p2 = matrix_dot(&matrix_dot(&self.K, &R), &self.K.transpose());

        self.P = p1 + p2;

        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();
    }

    pub fn predict_steadystate(&mut self, u: Option<&DMatrix<T>>, B: Option<&DMatrix<T>>) {
        let B = if B.is_some() { B } else { self.B.as_ref() };

        if B.is_some() && u.is_some() {
            self.x = &self.F * &self.x + B.unwrap() * u.unwrap();
        } else {
            self.x = &self.F * &self.x;
        }

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    pub fn update_steadystate(&mut self, z: &DMatrix<T>) {
        self.y = z - &self.H * &self.x;
        self.x = &self.x + &self.K * &self.y;

        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();
    }

    pub fn get_prediction(&self, u: Option<&DMatrix<T>>) -> (DMatrix<T>, DMatrix<T>) {
        let Q = &self.Q;
        let F = &self.F;
        let P = &self.P;
        let FT = F.transpose();

        let B = self.B.as_ref();
        let x = {
            match (B, u) {
                (Some(b), Some(u)) => F * &self.x + b * u,
                _ => F * &self.x
            }
        };

        let P = ((F * P) * FT) * self.alpha_sq + Q;
        (x, P)
    }

    pub fn get_update(&self, z: &DMatrix<T>) -> (DMatrix<T>, DMatrix<T>) {
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
        let I_KH = &(DMatrix::<T>::identity(self.dim_x, self.dim_x) - (K * H));
        let P = ((I_KH * P) * I_KH.transpose()) + ((K * R) * &K.transpose());

        (x, P)
    }

    pub fn residual_of(&self, z: &DMatrix<T>) -> DMatrix<T> {
        z - (&self.H * &self.x_prior)
    }

    pub fn measurement_of_state(&self, x: &DMatrix<T>) -> DMatrix<T> {
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
        let mut kf: KalmanFilter<f32> = KalmanFilter::new(1, 1, 1);

        for i in 0..1000 {
            let zf = i as f32;
            let z = dmatrix!(zf);

            kf.predict();
            kf.update(&z, None, None);

            assert_approx_eq!(zf, kf.z.clone().unwrap()[0]);
        }
    }
}
