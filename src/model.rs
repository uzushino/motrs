use nalgebra as na;
use std::cmp::max;
use std::collections::HashMap;

use crate::Q_discrete_white_noise;

pub struct ModelPreset {}

impl ModelPreset {
    pub fn new() -> Self {
        ModelPreset {}
    }

    pub fn constant_velocity_and_static_box_size_2d() -> HashMap<String, f64> {
        let key = vec![
            String::from("order_pos"),
            String::from("dim_pos"),
            String::from("order_size"),
            String::from("dim_size"),
        ];
        let value = vec![1., 2., 0., 2.];

        key.into_iter()
            .zip(value.into_iter())
            .collect::<HashMap<_, _>>()
    }

    pub fn constant_acceleration_and_static_box_size_2d() -> HashMap<String, f64> {
        let key = vec![
            String::from("order_pos"),
            String::from("dim_pos"),
            String::from("order_size"),
            String::from("dim_size"),
        ];

        let value = vec![2., 2., 0., 2.];

        key.into_iter()
            .zip(value.into_iter())
            .collect::<HashMap<_, _>>()
    }
}

fn base_dim_block<'a>(dt: f64, order: usize) -> na::DMatrix<f64> {
    let block =
        na::DMatrix::from_row_slice(3, 3, &[1., dt, (dt.powf(2.)) / 2., 0., 1., dt, 0., 0., 1.]);

    let cutoff = order + 1;

    na::DMatrix::from(block.index((0..cutoff, 0..cutoff)))
}

fn zero_pad(arr: na::DMatrix<f64>, length: usize) -> na::DMatrix<f64> {
    let mut ret = na::DMatrix::zeros(1, length);
    ret.index_mut((.., ..arr.shape().1)).copy_from(&arr);
    ret
}

fn repeat_vec<T: Clone>(x: Vec<T>, size: usize) -> Vec<T> {
    x.iter()
        .cycle()
        .take(x.len() * size)
        .map(|v| v.clone())
        .collect::<Vec<_>>()
}

fn block_diag(arrs: Vec<na::DMatrix<f64>>) -> na::DMatrix<f64> {
    let shapes = arrs
        .iter()
        .map(|m| {
            let (a, b) = m.shape();
            vec![a, b]
        })
        .collect::<Vec<_>>();

    let sum_shapes = na::DMatrix::from_row_slice(
        arrs.len(),
        2,
        shapes
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let sum_shape = sum_shapes.row_sum();

    let mut out = na::DMatrix::zeros(sum_shape[(0, 0)], sum_shape[(0, 1)]);

    let mut r = 0;
    let mut c = 0;

    for (i, sh) in shapes.iter().enumerate() {
        let rr = sh[0];
        let cc = sh[1];
        for row in r..(r + rr) {
            for column in c..(c + cc) {
                out[(row, column)] = arrs[i][(row - r, column - c)];
            }
        }
        // out.index_mut((r..(r+rr), c..(c+cc))).copy_from(&arrs[i]);

        r += rr;
        c += cc;
    }

    out
}

fn eye(block: usize) -> na::DMatrix<f64> {
    na::DMatrix::identity(block, block)
}

pub struct Model {
    pub dt: f64,
    pub order_pos: usize,
    pub dim_pos: usize,
    pub order_size: usize,
    pub dim_size: usize,
    pub q_var_pos: f64,
    pub q_var_size: f64,
    pub r_var_pos: f64,
    pub r_var_size: f64,
    pub p_cov_p0: f64,
    pub dim_box: usize,
    pub pos_idxs: Vec<usize>,
    pub size_idxs: Vec<usize>,
    pub z_in_x_ids: Vec<usize>,
    pub offset_idx: usize,
    pub state_length: usize,
    pub measurement_lengths: usize,
}

#[derive(Clone, Debug)]
pub struct ModelKwargs {
    pub order_pos: i64,
    pub dim_pos: i64,
    pub order_size: i64,
    pub dim_size: i64,
    pub q_var_pos: f64,
    pub q_var_size: f64,
    pub r_var_pos: f64,
    pub r_var_size: f64,
    pub p_cov_p0: f64,
}

impl Default for ModelKwargs {
    fn default() -> Self {
        Self {
            order_pos: 1,
            dim_pos: 2,
            order_size: 0,
            dim_size: 2,
            q_var_pos: 70.,
            q_var_size: 10.,
            r_var_pos: 1.,
            r_var_size: 1.,
            p_cov_p0: 1000.,
        }
    }
}

impl Model {
    pub fn new(dt: f64, kwargs: Option<ModelKwargs>) -> Self {
        let kwargs = kwargs.unwrap_or_default();

        let order_pos = kwargs.order_pos as usize;
        let dim_pos = kwargs.dim_pos as usize;
        let order_size = kwargs.order_size as usize;
        let dim_size = kwargs.dim_size as usize;
        let q_var_pos = kwargs.q_var_pos;
        let q_var_size = kwargs.q_var_size;
        let r_var_pos = kwargs.r_var_pos;
        let r_var_size = kwargs.r_var_size;
        let p_cov_p0 = kwargs.p_cov_p0;

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
            measurement_lengths,
        }
    }

    fn _calc_idxs(
        dim_pos: usize,
        dim_size: usize,
        order_pos: usize,
        order_size: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>, usize) {
        let offset_idx = max(dim_pos, dim_size);
        let pos_idxs: Vec<usize> = (0..dim_pos).map(|pidx| pidx * (order_pos + 1)).collect();
        let size_idxs: Vec<usize> = (0..dim_size)
            .map(|sidx| dim_pos * (order_pos + 1) + sidx * (order_size + 1))
            .collect();
        let mut z_in_idxs = pos_idxs.clone();
        z_in_idxs.append(&mut size_idxs.clone());

        (pos_idxs, size_idxs, z_in_idxs, offset_idx)
    }

    pub fn build_F(&self) -> na::DMatrix<f64> {
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

    pub fn build_Q(&self) -> na::DMatrix<f64> {
        let var_pos = self.q_var_pos;
        let var_size = self.q_var_size;

        let q_pos = if self.order_pos == 0 {
            na::dmatrix![var_pos]
        } else {
            // dim=self.order_pos + 1, dt=self.dt, var=var_pos
            Q_discrete_white_noise(self.order_pos + 1, self.dt, var_pos, 1, true)
        };

        let q_size = if self.order_size == 0 {
            na::dmatrix![var_size]
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

    pub fn build_H(&self) -> na::DMatrix<f64> {
        fn _base_block(order: usize) -> na::DMatrix<f64> {
            let mut diag_components = Vec::new();

            let a = vec![1.];
            let b = repeat_vec(vec![0.], order);

            diag_components.extend(a);
            diag_components.extend(b);

            na::DMatrix::from_vec(1, order + 1, diag_components)
        }

        let _block_pos = repeat_vec(vec![_base_block(self.order_pos)], self.dim_pos);
        let _block_size = repeat_vec(vec![_base_block(self.order_size)], self.dim_size);
        let mut diag_components = Vec::new();

        diag_components.extend(_block_pos);
        diag_components.extend(_block_size);

        block_diag(diag_components)
    }

    pub fn build_P(&self) -> na::DMatrix<f64> {
        let n = eye(self.state_length);
        n * self.p_cov_p0
    }

    pub fn build_R(&self) -> na::DMatrix<f64> {
        let block_pos = eye(self.dim_pos) * self.r_var_pos;
        let block_size = eye(self.dim_size) * self.r_var_size;

        block_diag(vec![block_pos, block_size])
    }

    pub fn box_to_z(&self, _box: na::DMatrix<f64>) -> na::DMatrix<f64> {
        let rep = _box.iter().map(|v| *v).collect::<Vec<f64>>();
        let _box = na::DMatrix::from_row_slice(2, self.dim_box / 2, rep.as_slice());
        let a = _box.row_sum() / 2.0;
        let center = a.columns(0, self.dim_pos);
        let b = _box.index((1, ..)) - _box.index((0, ..));
        let length = b.columns(0, self.dim_size);

        let mut result = center.iter().map(|v| *v).collect::<Vec<f64>>();
        result.append(&mut length.iter().map(|v| *v).collect::<Vec<f64>>());

        na::DMatrix::from_row_slice(1, result.len(), result.as_slice())
    }

    pub fn box_to_x(&self, _box: na::DMatrix<f64>) -> na::DMatrix<f64> {
        let mut x: na::DMatrix<f64> = na::DMatrix::zeros(1, self.state_length);
        let z = self.box_to_z(_box);
        for (idx, i) in self.z_in_x_ids.iter().enumerate() {
            x[*i] = z[idx];
        }
        x
    }

    pub fn x_to_box(&self, x: na::DMatrix<f64>) -> na::DMatrix<f64> {
        let size = max(self.dim_pos, self.dim_size);

        let mut xs = Vec::default();
        for i in &self.pos_idxs {
            xs.push(x[*i]);
        }
        let center = zero_pad(
            na::DMatrix::from_row_slice(1, xs.len(), xs.as_slice()),
            size,
        );

        let mut ys = Vec::default();
        for i in &self.size_idxs {
            ys.push(x[*i]);
        }
        let length = zero_pad(
            na::DMatrix::from_row_slice(1, ys.len(), ys.as_slice()),
            size,
        );

        let mut result = (center.clone() - length.clone() / 2.)
            .iter()
            .map(|v| *v)
            .collect::<Vec<f64>>();
        result.append(
            &mut (center + length / 2.)
                .iter()
                .map(|v| *v)
                .collect::<Vec<f64>>(),
        );

        na::DMatrix::from_row_slice(1, result.len(), result.as_slice())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;

    #[test]
    fn test_zero_pad() {
        let arr = na::dmatrix![1., 2., 3.];
        let pad_arr = zero_pad(arr, 5);

        assert!(pad_arr == na::dmatrix![1., 2., 3., 0., 0.])
    }

    #[test]
    fn test_repeat_vec() {
        let arr = vec![1., 2., 3.];
        assert!(repeat_vec(arr, 3) == vec![1., 2., 3., 1., 2., 3., 1., 2., 3.])
    }

    #[test]
    fn test_block_diag() {
        let a = na::DMatrix::from_row_slice(2, 2, &[1., 0., 0., 1.]);
        let b = na::DMatrix::from_row_slice(2, 3, &[3., 4., 5., 6., 7., 8.]);
        let c = na::DMatrix::from_row_slice(1, 1, &[7.]);

        let expect = na::DMatrix::from_row_slice(
            5,
            6,
            &[
                1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 3., 4., 5., 0., 0., 0., 6.,
                7., 8., 0., 0., 0., 0., 0., 0., 7.,
            ],
        );

        let out = block_diag(vec![a, b, c]);

        assert!(out == expect)
    }

    #[test]
    fn test_eye() {
        let a = eye(3);
        let actual = a * 3.0;
        let expect = na::DMatrix::from_row_slice(3, 3, &[3., 0., 0., 0., 3., 0., 0., 0., 3.]);

        assert!(actual == expect)
    }

    #[test]
    fn test_builder() {
        let kwargs = ModelKwargs {
            r_var_pos: 0.1,
            r_var_size: 0.3,
            p_cov_p0: 100.,
            ..Default::default()
        };
        let m1 = Model::new(0.1, Some(kwargs));

        assert!(m1.state_length == 6);
        assert!(m1.measurement_lengths == 4);

        let F1 = m1.build_F();
        let F1_exp = na::DMatrix::from_row_slice(
            6,
            6,
            &[
                1., 0.1, 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.1, 0., 0., 0., 0.,
                0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
            ],
        );

        assert!(F1_exp == F1);

        let H1 = m1.build_H();
        let H1_exp = na::DMatrix::from_row_slice(
            4,
            6,
            &[
                1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                0., 0., 1.,
            ],
        );
        assert!(H1_exp == H1);

        _ = m1.build_Q();

        let R1 = m1.build_R();
        let R1_exp = na::DMatrix::from_row_slice(
            4,
            4,
            &[
                0.1, 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0.3, 0., 0., 0., 0., 0.3,
            ],
        );
        assert!(R1 == R1_exp);

        _ = m1.build_P();

        let mut kwargs = ModelKwargs {
            order_pos: 2,
            dim_pos: 1,
            order_size: 1,
            dim_size: 1,
            ..Default::default()
        };
        let m2 = Model::new(0.1, Some(kwargs));
        let F2 = m2.build_F();
        let F2_exp = na::DMatrix::from_row_slice(
            5,
            5,
            &[
                1., 0.1, 0.005, 0., 0., 0., 1., 0.1, 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
                0.1, 0., 0., 0., 0., 1.,
            ],
        );

        assert_relative_eq!(F2_exp, F2, epsilon = 1e-3f64);
    }

    #[test]
    fn test_state_to_observation_converters() {
        let kwargs = ModelKwargs {
            order_pos: 1,
            dim_pos: 2,
            order_size: 0,
            dim_size: 2,
            ..Default::default()
        };

        let model = Model::new(0.1, Some(kwargs));
        let _box = na::dmatrix![10., 10., 20., 30.];

        let x = model.box_to_x(_box.clone());
        assert!(na::dmatrix![15., 0., 20., 0., 10., 20.] == x);

        let box_ret = model.x_to_box(x);

        assert!(box_ret == _box);

        let mut kwargs = ModelKwargs {
            order_pos: 1,
            dim_pos: 3,
            order_size: 0,
            dim_size: 3,
            ..Default::default()
        };
        let model = Model::new(0.1, Some(kwargs));
        let _box = na::dmatrix![10., 10., 10., 20., 30., 40.];
        let x = model.box_to_x(_box.clone());

        assert!(na::dmatrix![15., 0., 20., 0., 25., 0., 10., 20., 30.] == x);

        let box_ret = model.x_to_box(x);
        assert!(box_ret == _box);
    }

    #[test]
    fn test_box_to_z() {
        let mut kwargs = ModelKwargs {
            order_pos: 1,
            dim_pos: 2,
            order_size: 0,
            dim_size: 2,
            ..Default::default()
        };

        let model = Model::new(0.1, Some(kwargs));
        let _box = na::dmatrix![10f64, 10., 20., 20.];
        let result = model.box_to_z(_box);

        assert!(result == na::dmatrix![15., 15., 10., 10.]);

        let mut kwargs = ModelKwargs {
            order_pos: 1,
            dim_pos: 3,
            order_size: 1,
            dim_size: 2,
            ..Default::default()
        };

        let model = Model::new(0.1, Some(kwargs));
        let _box = na::dmatrix![10f64, 10., 0., 20., 20., 50.];
        let result = model.box_to_z(_box);

        assert!(result == na::dmatrix![15., 15., 25., 10., 10.]);
    }
}
