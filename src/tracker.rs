use filter::kalman::kalman_filter::KalmanFilter;
use nalgebra::{U1, U2, Vector2, Matrix2, Matrix1};

fn get_kalman_object_tracker() -> KalmanFilter<f64, U1, U1, U1> {
    let mut tracker = KalmanFilter::<f64, U1, U1, U1>::default();
    tracker
}
