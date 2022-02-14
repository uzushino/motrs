use std::collections::HashMap;
use nalgebra::{dmatrix, DVector, DMatrix};

struct ModelPreset {
    pub constant_velocity_and_static_box_size_2d: HashMap<String, usize>,
    pub constant_acceleration_and_static_box_size_2d: HashMap<String, usize>,
}

fn base_dim_block<'a>(dt: f64, order: usize) -> DMatrix<f64> {
    let block = dmatrix![
        1., dt, (dt.powf(2.)) / 2.,
        0., 1., dt,
        0., 0., 1.
    ];

    let cutoff = order + 1;

    DMatrix::from(block.index((0..cutoff, 0..cutoff)))
}

fn zero_pad(arr: DVector<f64>, length: usize) -> DVector<f64> {
    let mut ret = DVector::zeros(length);
    ret.index_mut((..arr.shape().0, ..)).copy_from(&arr);
    ret
}

pub struct Model {
    dt: f64,
    order_pos: i64,
    dim_pos: i64,
    order_size: usize,
    dim_size: usize,
    q_var_pos: f64,
    q_var_size: f64,
    r_var_pos: f64,
    r_var_size: f64,
    p_cov_p0: f64,
    dim_box: i64,
    pos_idxs: Vec<usize>,
    size_idxs: Vec<usize>,
    z_in_x_ids: Vec<usize>,
    offset_idx: usize,
    state_length: usize,
    measurement_length: usize
}

impl Model {
    pub fn new(
        dt: f64,
        order_pos: i64,
        dim_pos: i64,
        order_size: usize,
        dim_size: usize,
        q_var_pos: f64,
        q_var_size: f64,
        r_var_pos: f64,
        r_var_size: f64,
        p_cov_p0: f64
    ) -> Self {
        let mut model = Model {
            dt,
            order_pos,
            dim_pos,
            order_size,
            dim_size,
            q_var_pos,
            q_var_size,
            r_var_pos,
            r_var_size,
            p_cov_p0
        };
    }
}
#[cfg(test)]
mod test {
    use nalgebra::dvector;
    use super::*;

    #[test]
    fn test_zero_pad() {
        let arr = dvector![1., 2., 3.];
        let pad_arr = zero_pad(arr, 5);

        assert!(pad_arr == dvector![1., 2., 3., 0., 0.])
    }
}
