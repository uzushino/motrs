use filter::kalman::kalman_filter::KalmanFilter;
use nalgebra::{DefaultAllocator, DVector, MatrixMN, RealField, VectorN};

use crate::model::Model;

fn get_kalman_object_tracker<F, DimX, DimZ, DimU>(model: &Model, x0: Option<DVector<f64>>) -> KalmanFilter<f64, DimX, DimZ, DimU>
{
    let mut tracker = KalmanFilter::<f64, DimX, DimZ, DimU>::default();

    tracker.F = model.build_F();
    tracker.Q = model.build_Q();
    tracker.H = model.build_H();
    tracker.R = model.build_R();
    tracker.P = model.build_P();

    tracker
}
