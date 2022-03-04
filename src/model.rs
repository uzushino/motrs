use std::collections::HashMap;
use std::cmp::max;
use nalgebra::{dmatrix, DVector, DMatrix, Matrix2x1, dvector};
use crate::Q_discrete_white_noise;


struct ModelPreset {
    pub constant_velocity_and_static_box_size_2d: HashMap<String, usize>,
    pub constant_acceleration_and_static_box_size_2d: HashMap<String, usize>,
}

fn base_dim_block<'a>(dt: f64, order: usize) -> DMatrix<f64> {
    let block = DMatrix::from_row_slice(3, 3, &[
        1., dt, (dt.powf(2.)) / 2.,
        0., 1., dt,
        0., 0., 1.
    ]);

    let cutoff = order + 1;

    DMatrix::from(block.index((0..cutoff, 0..cutoff)))
}

fn zero_pad(arr: DVector<f64>, length: usize) -> DVector<f64> {
    let mut ret = DVector::zeros(length);
    ret.index_mut((..arr.shape().0, ..)).copy_from(&arr);
    ret
}

fn repeat_vec<T: Clone>(x: Vec<T>, size: usize) -> Vec<T> {
    x.iter().cycle().take(x.len() * size).map(|v| v.clone()).collect::<Vec<_>>()
}

fn block_diag(arrs: Vec<DMatrix<f64>>) -> DMatrix<f64> {
    let shapes = arrs
        .iter()
        .map(|m| {
            let (a, b) = m.shape();
            vec![a, b]
        })
        .collect::<Vec<_>>();

    let sum_shapes = DMatrix::from_row_slice(3, 2, shapes.clone().into_iter().flatten().collect::<Vec<_>>().as_slice());
    let sum_shape = sum_shapes.row_sum();

    let mut out = DMatrix::zeros(sum_shape[(0, 0)], sum_shape[(0, 1)]);

    let mut r = 0;
    let mut c = 0;

    for (i, sh) in shapes.iter().enumerate() {
        let rr = sh[0];
        let cc = sh[1];
        out.index_mut((r..(r+rr), c..(c+cc))).copy_from(&arrs[i]);
        r += rr;
        c += cc;
    }

    out
}

pub struct Model {
    dt: f64,
    order_pos: usize,
    dim_pos: usize,
    order_size: usize,
    dim_size: usize,
    q_var_pos: f64,
    q_var_size: f64,
    r_var_pos: f64,
    r_var_size: f64,
    p_cov_p0: f64,
    dim_box: usize,
    pos_idxs: Vec<usize>,
    size_idxs: Vec<usize>,
    z_in_x_ids: Vec<usize>,
    offset_idx: usize,
    state_length: usize,
    measurement_lengths: usize
}

impl Model {
    pub fn new(
        dt: f64,
        order_pos: usize,
        dim_pos: usize,
        order_size: usize,
        dim_size: usize,
        q_var_pos: f64,
        q_var_size: f64,
        r_var_pos: f64,
        r_var_size: f64,
        p_cov_p0: f64
    ) -> Self {
        let dim_box = 2 * max(dim_pos, dim_size);
        let (pos_idxs, size_idxs, z_in_x_ids, offset_idx) =
            Self::_calc_idxs(dim_pos, dim_size, order_pos, order_size);
        let state_length = dim_pos * (order_pos + 1) + dim_size * (order_size + 1);
        let measurement_lengths = dim_pos + dim_size;

        Self {
            dt,
            dim_box,
            dim_pos,
            dim_size,
            pos_idxs,
            size_idxs,
            order_pos,
            order_size,
            q_var_pos,
            q_var_size,
            r_var_pos,
            r_var_size,
            p_cov_p0,
            z_in_x_ids,
            offset_idx,
            state_length,
            measurement_lengths
        }
    }

    fn _calc_idxs(dim_pos: usize, dim_size: usize, order_pos: usize, order_size: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>, usize) {
        let offset_idx = max(dim_pos, dim_size);
        let pos_idxs: Vec<usize> = (0..dim_pos).map(|pidx| pidx * (order_pos + 1)).collect();
        let mut size_idxs: Vec<usize> = (0..dim_size).map(|sidx| dim_pos * (order_pos + 1) + sidx * (order_size + 1)).collect();
        let mut z_in_idxs = pos_idxs.clone();
        z_in_idxs.append(&mut size_idxs);

        (pos_idxs, size_idxs, z_in_idxs, offset_idx)
    }


    pub fn build_F(&self) -> DMatrix<f64> {
        let block_pos = base_dim_block(self.dt, self.order_pos);
        let block_size = base_dim_block(self.dt, self.order_size);

        let diag_components = {
            let _block_pos = repeat_vec(vec![block_pos], self.dim_pos);
            let _block_size = repeat_vec(vec![block_size], self.dim_size);
            let mut diag_components = Vec::new();

            diag_components.extend(_block_pos);
            diag_components.extend(_block_size);

            diag_components
        };

        block_diag(diag_components)
    }

    pub fn build_Q(&self) -> DMatrix<f64> {
        let var_pos = self.q_var_pos;
        let var_size = self.q_var_size;

        let q_pos = if self.order_pos == 0 {
            dmatrix![var_pos]
        } else {
            // dim=self.order_pos + 1, dt=self.dt, var=var_pos
            Q_discrete_white_noise(self.order_pos + 1, self.dt, var_pos, 1, true)
        };

        let q_size= if self.order_size == 0 {
            dmatrix![var_size]
        } else {
            // dim=self.order_pos + 1, dt=self.dt, var=var_pos
            Q_discrete_white_noise(self.order_size + 1, self.dt, var_size, 1, true)
        };

        let diag_components = {
            let _block_pos = repeat_vec(vec![q_pos], self.dim_pos);
            let _block_size = repeat_vec(vec![q_size], self.dim_size);
            let mut diag_components = Vec::new();

            diag_components.extend(_block_pos);
            diag_components.extend(_block_size);

            diag_components
        };

        block_diag(diag_components)
    }

    pub fn build_H(&self) {
        fn _base_block(order: usize) -> DVector<usize> {
            return dvector![1] + dvector![0] * order;
        }
    }

}

#[cfg(test)]
mod test {
    use nalgebra::{ dvector, Matrix3x4, Matrix };
    use super::*;

    #[test]
    fn test_vec() {
        let mut a=  dvector![1];
        a.append(&mut repeat_vec(vec![0], 3));
        println!("{:?}", a);
    }

    #[test]
    fn test_zero_pad() {
        let arr = dvector![1., 2., 3.];
        let pad_arr = zero_pad(arr, 5);

        assert!(pad_arr == dvector![1., 2., 3., 0., 0.])
    }

    #[test]
    fn test_repeat_vec() {
        let arr = vec![1., 2., 3.];
        assert!(repeat_vec(arr, 3) == vec![1., 2., 3., 1., 2., 3., 1., 2., 3.])
    }

    #[test]
    fn test_block_diag() {
        let a = DMatrix::from_row_slice(2, 2, &[1., 0., 0., 1.]);
        let b = DMatrix::from_row_slice(2, 3, &[3., 4., 5., 6., 7., 8.]);
        let c = DMatrix::from_row_slice(1, 1, &[7.]);

        let expect = DMatrix::from_row_slice(5, 6, &[
            1., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0.,
            0., 0., 3., 4., 5., 0.,
            0., 0., 6., 7., 8., 0.,
            0., 0., 0., 0., 0., 7.
        ]);
        let out = block_diag(vec![a, b, c]);

        assert!(out == expect)
    }
}
