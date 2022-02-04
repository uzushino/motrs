use std::collections::HashMap;
use nalgebra::{dmatrix, DVector, DMatrix};

struct ModelPreset {
    pub constant_velocity_and_static_box_size_2d: HashMap<String, usize>,
    pub constant_acceleration_and_static_box_size_2d: HashMap<String, usize>,
}

impl ModelPreset {
    pub fn _base_dim_block<'a>(dt: f64, order: usize) -> DMatrix<f64> {
        let block = dmatrix![
            1., dt, (dt.powf(2.)) / 2.,
            0., 1., dt,
            0., 0., 1.
        ];

        let cutoff = order + 1;

        DMatrix::from(block.index((0..cutoff, 0..cutoff)))
    }

    pub fn _zero_pad(arr: DMatrix<f64>, length: usize) -> DVector<f64> {
        let mut ret = DVector::zeros(length);
        ret.index_mut((0.., 0..arr.shape().0)).copy_from(&arr);
        ret
    }
}
