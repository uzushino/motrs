use filter::kalman::kalman_filter::KalmanFilter;
use nalgebra::base::Vector1;
use nalgebra::{dmatrix, DVector, DMatrix, Matrix2x1, dvector, VectorN, MatrixMN, RealField, Dim};
use nalgebra::base::dimension::DimName;

use crate::model::Model;

fn get_kalman_object_tracker<F, DimX, DimZ, DimU>(model: &Model, x0: Option<DVector<f64>>) -> KalmanFilter<f32, U1, U1, U1>
    where
        F: RealField,
        DimX: DimName,
        DimZ: DimName,
        DimU: DimName,
{
    let mut tracker = KalmanFilter::<f64, DimX, DimZ, DimU>::default();

    tracker.F = model.build_F();
    tracker.Q = model.build_Q();
    tracker.H = model.build_H();
    tracker.R = model.build_R();
    tracker.P = model.build_P();

    tracker
}
