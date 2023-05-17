use crate::matrix::*;
/**
 * Refer: https://github.com/MichaelMauderer/filter-rs
 */
use ndarray::Array2;

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct KalmanFilter<T> {
    pub dim_x: usize,
    pub dim_z: usize,
    pub dim_u: usize,

    pub x: Array2<T>,
    pub P: Array2<T>,
    pub x_prior: Array2<T>,
    pub P_prior: Array2<T>,
    pub x_post: Array2<T>,
    pub P_post: Array2<T>,
    pub z: Option<Array2<T>>,
    pub R: Array2<T>,
    pub Q: Array2<T>,
    pub B: Option<Array2<T>>,
    pub F: Array2<T>,
    pub H: Array2<T>,
    pub y: Array2<T>,
    pub K: Array2<T>,
    pub S: Array2<T>,
    pub SI: Array2<T>,
    pub alpha_sq: T,
}

#[allow(non_snake_case)]
impl<T> KalmanFilter<T> {
    pub fn new(dim_x: usize, dim_z: usize, dim_u: usize) -> Self {
        let x = Array2::<T>::from_element(1, dim_x, T::zero());
        let P = Array2::<T>::identity(dim_x, dim_x);
        let Q = Array2::<T>::identity(dim_x, dim_x);
        let F = Array2::<T>::identity(dim_x, dim_x);
        let H = Array2::<T>::from_element(dim_z, dim_x, T::zero());
        let R = Array2::<T>::identity(dim_z, dim_z);
        let alpha_sq = T::one();

        let z = None;

        let K = Array2::<T>::from_element(dim_x, dim_z, T::zero());
        let y = Array2::<T>::from_element(1, dim_z, T::one());
        let S = Array2::<T>::from_element(dim_z, dim_z, T::zero());
        let SI = Array2::<T>::from_element(dim_z, dim_z, T::zero());

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

        self.x = f.dot(&self.x);
        self.P = f.dot(&self.P).dot(f.transpose()) * self.alpha_sq + q;

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    /// Add a new measurement (z) to the Kalman filter.
    pub fn update(&mut self, z: &Array2<T>, R: Option<&Array2<T>>, H: Option<&Array2<T>>) {
        let R = R.unwrap_or(&self.R);
        let H = H.unwrap_or(&self.H);

        self.y = z - H.dot(&self.x);
        let PHT = self.P.dot(&H.transpose());

        self.S = H.dot(&PHT) + R;
        self.SI = self
            .S
            .clone()
            .pseudo_inverse(T::from_f32(0.00001).unwrap())
            .unwrap();
        self.K = PHT.dot(&self.SI);
        self.x = &self.x + self.K.dot(&self.y);

        let I_KH = &Array2::identity(self.dim_x, self.dim_x) - self.K.dot(&H);
        
        let p1 = I_KH.dot(&self.P).dot(I_KH.transpose());
        let p2 = self.K.dot(&R).dot(&self.K.transpose());

        self.P = p1 + p2;

        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();
    }

    pub fn predict_steadystate(&mut self, u: Option<&Array2<T>>, B: Option<&Array2<T>>) {
        let B = if B.is_some() { B } else { self.B.as_ref() };

        if B.is_some() && u.is_some() {
            self.x = &self.F * &self.x + B.unwrap() * u.unwrap();
        } else {
            self.x = &self.F * &self.x;
        }

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    pub fn update_steadystate(&mut self, z: &Array2<T>) {
        self.y = z - &self.H * &self.x;
        self.x = &self.x + &self.K * &self.y;

        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();
    }

    pub fn get_prediction(&self, u: Option<&Array2<T>>) -> (Array2<T>, Array2<T>) {
        let Q = &self.Q;
        let F = &self.F;
        let P = &self.P;
        let FT = F.transpose();

        let B = self.B.as_ref();
        let x = {
            match (B, u) {
                (Some(b), Some(u)) => F * &self.x + b * u,
                _ => F * &self.x,
            }
        };

        let P = ((F * P) * FT) * self.alpha_sq + Q;
        (x, P)
    }

    pub fn get_update(&self, z: &Array2<T>) -> (Array2<T>, Array2<T>) {
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
        let I_KH = &(Array2::<T>::identity(self.dim_x, self.dim_x) - (K * H));
        let P = ((I_KH * P) * I_KH.transpose()) + ((K * R) * &K.transpose());

        (x, P)
    }

    pub fn residual_of(&self, z: &Array2<T>) -> Array2<T> {
        z - (&self.H * &self.x_prior)
    }

    pub fn measurement_of_state(&self, x: &Array2<T>) -> Array2<T> {
        &self.H * x
    }
}

/*
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::base::Vector1;
    use nalgebra::{Array2, Matrix1, Matrix2, Vector2, U1, U2};

    use super::*;

    #[test]
    fn test_univariate_kf_setup() {
        let mut kf: KalmanFilter<f32> = KalmanFilter::new(1, 1, 1);

        for i in 0..1000 {
            let zf = i as f32;
            let z = Array2!(zf);

            kf.predict();
            kf.update(&z, None, None);

            assert_approx_eq!(zf, kf.z.clone().unwrap()[0]);
        }
    }
}
 */