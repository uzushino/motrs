use std::cmp::max;
use std::fmt::Debug;
use ndarray as nd;

#[derive(Default)]
pub struct ModelPreset {
    pub order_pos: i32,
    pub dim_pos: i32,
    pub order_size: usize,
    pub dim_size: usize,
}

impl ModelPreset {
    pub fn new() -> Self {
        ModelPreset {
            ..Default::default()
        }
    }

    pub fn constant_velocity_and_static_box_size_2d() -> Self {
        Self {
            order_pos: 1,
            dim_pos: 2,
            order_size: 0,
            dim_size: 2,
        }
    }

    pub fn constant_acceleration_and_static_box_size_2d() -> Self {
        Self {
            order_pos: 2,
            dim_pos: 2,
            order_size: 0,
            dim_size: 2,
        }
    }
}

fn base_dim_block<T>(dt: T, order: usize) -> nd::Array2<T> {
    let dt: f32 = dt.to_f32().unwrap_or_default();
    let block = nd::Array::from_shape_vec((3, 3), vec![1., dt, (dt.powf(2.)) / 2., 0., 1., dt, 0., 0., 1.]).unwrap();
    let cutoff = order + 1;

    nd::Array::from_shape_fn((cutoff, cutoff), |(r, c)| T::from_f32(block[[r, c]]).unwrap())
}

fn zero_pad<T>(arr: &nd::Array2<T>, length: usize) -> nd::Array2<T> {
    let n_cols = arr.shape()[1];
    let mut ret = nd::Array::zeros((1, length));
    ret.slice_mut(nd::s![.., ..n_cols]).assign(arr);
    ret
}

fn repeat_vec<T: Clone>(x: Vec<T>, size: usize) -> Vec<T> {
    x.iter()
        .cycle()
        .take(x.len() * size)
        .cloned()
        .collect::<Vec<_>>()
}

fn block_diag(arrs: Vec<nd::Array2<f32>>) -> nd::Array2<f32> {
    let shapes = arrs.iter().map(|m| m.shape()).collect::<Vec<_>>();
    let sum_shapes = shapes.iter().fold((0, 0), |(ra, ca), (rb, cb)| (ra + rb, ca + cb));
    let mut out = nd::Array2::zeros(sum_shapes);

    let mut r = 0;
    let mut c = 0;

    for arr in arrs.iter() {
        let (rr, cc) = arr.shape();
        let block = arr.to_owned();
        out.slice_mut(nd::s![r..(r+rr), c..(c+cc)]).assign(&block);

        r += rr;
        c += cc;
    }

    out
}

fn eye(block: usize) -> nd::Array2<f32> {
    let mut arr = nd::Array::zeros((block, block));
    for i in 0..block {
        arr[[i, i]] = 1.0;
    }
    arr
}

pub struct Model {
    pub dt: f32,
    pub order_pos: usize,
    pub dim_pos: usize,
    pub order_size: usize,
    pub dim_size: usize,
    pub q_var_pos: f32,
    pub q_var_size: f32,
    pub r_var_pos: f32,
    pub r_var_size: f32,
    pub p_cov_p0: f32,
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
    pub order_pos: i32,
    pub dim_pos: i32,
    pub order_size: usize,
    pub dim_size: usize,
    pub q_var_pos: f32,
    pub q_var_size: f32,
    pub r_var_pos: f32,
    pub r_var_size: f32,
    pub p_cov_p0: f32,
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
    pub fn new(dt: f32, kwargs: Option<ModelKwargs>) -> Self {
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
        let (pos_idxs, size_idxs, offset_idx) =
            Self::_calc_idxs(dim_pos, dim_size, order_pos, order_size);

        let z_in_x_ids = pos_idxs
            .iter()
            .chain(&size_idxs)
            .cloned()
            .collect::<Vec<_>>();

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
    ) -> (Vec<usize>, Vec<usize>, usize) {
        let offset_idx = max(dim_pos, dim_size);
        let pos_idxs: Vec<usize> = (0..dim_pos).map(|pidx| pidx * (order_pos + 1)).collect();
        let size_idxs: Vec<usize> = (0..dim_size)
            .map(|sidx| dim_pos * (order_pos + 1) + sidx * (order_size + 1))
            .collect();

        (pos_idxs, size_idxs, offset_idx)
    }

    pub fn build_F(&self) -> nd::Array2<f32> {
        let block_pos = base_dim_block(self.dt, self.order_pos);
        let block_size = base_dim_block(self.dt, self.order_size);
        let dim_pos = self.dim_pos;
        let dim_size = self.dim_size;
    
        let diag_components = {
            let _block_pos = nd::Array::from(vec![block_pos]).repeat(dim_pos, 1);
            let _block_size = nd::Array::from(vec![block_size]).repeat(dim_size, 1);
    
            let mut diag_components = Vec::new();
            diag_components.extend(_block_pos.iter());
            diag_components.extend(_block_size.iter());
            diag_components
        };

        block_diag(diag_components) 
    }

    pub fn build_Q(&self) -> nd::Array2<f32> {
        let var_pos = self.q_var_pos;
        let var_size = self.q_var_size as f32;
    
        let q_pos = if self.order_pos == 0 {
            nd::Array::from_elem((1, 1), var_pos)
        } else {
            crate::Q_discrete_white_noise(self.order_pos + 1, self.dt, var_pos, 1, true).into()
        };
    
        let q_size = if self.order_size == 0 {
            nd::Array::from_elem((1, 1), var_size)
        } else {
            crate::Q_discrete_white_noise(self.order_size + 1, self.dt, var_size, 1, true).into()
        };
    
        let block_pos = q_pos.repeat(self.dim_pos, self.dim_pos);
        let block_size = q_size.repeat(self.dim_size, self.dim_size);
    
        let diag_components = block_pos.outer_iter().chain(block_size.outer_iter())
            .map(|a| a.to_owned()).collect();
    
        block_diag(diag_components)
    }

    pub fn build_H(&self) -> nd::Array2<f32> {
        fn base_block(order: usize) -> nd::Array2<f32> {
            let a = nd::Array::from_elem((1, 1), 1.0);
            let b = nd::Array::from_elem((1, order), 0.0);
            nd::stack![1, [a, b]]
        }
    
        let block_pos = repeat_vec(vec![base_block(self.order_pos)], self.dim_pos);
        let block_size = repeat_vec(vec![base_block(self.order_size)], self.dim_size);
    
        let diag_components = block_pos.iter().chain(block_size.iter())
            .cloned().map(|a| a.to_owned()).collect();

        block_diag(diag_components)
    }

    pub fn build_P(&self) -> nd::Array2<f32> {
        let n = eye(self.state_length);
        n * self.p_cov_p0
    }

    pub fn build_R(&self) -> nd::Array2<f32> {
        let block_pos = eye(self.dim_pos) * self.r_var_pos;
        let block_size = eye(self.dim_size) * (self.r_var_size as f32);

        block_diag(vec![block_pos, block_size])
    }

    pub fn box_to_z(&self, _box: nd::Array2<f32>) -> nd::Array2<f32> {
        let rep = _box.iter().cloned().collect::<Vec<f32>>();
        let _box = nd::Array2::from_shape_vec((2, self.dim_box / 2), rep).unwrap();
        let a = _box.sum_axis(ndarray::Axis(0)) / 2.0;
        let center = a.slice(nd::s![..self.dim_pos]).to_owned();
        let b = _box.row(1) - _box.row(0);
        let length = b.slice(nd::s![..self.dim_size]).to_owned();
    
        let mut result = center.iter().copied().collect::<Vec<f32>>();
        result.append(&mut length.iter().copied().collect::<Vec<f32>>());
    
        nd::Array2::from_shape_vec((1, result.len()), result).unwrap()
    }

    pub fn box_to_x(&self, _box: nd::Array2<f32>) -> nd::Array2<f32> {
        let mut x: nd::Array2<f32> = nd::Array::zeros((1, self.state_length));
        let z = self.box_to_z(_box);
        for (idx, i) in self.z_in_x_ids.iter().enumerate() {
            x[[0, *i]] = z[[0, idx]];
        }
        x
    }

    pub fn x_to_box(&self, x: &nd::Array1<f32>) -> nd::Array2<f32> {
        let size = self.dim_pos.max(self.dim_size);
    
        let center = zero_pad(
            &x.slice(self.pos_idxs.clone()).to_owned(),
            size,
        );
    
        let length = zero_pad(
            &x.slice(self.size_idxs.clone()).to_owned(),
            size,
        );
    
        let mut result = nd::concatenate(
            nd::Axis(1),
            &[&(center.clone() - length.clone() / 2.), &(center + length / 2.)],
        )
        .unwrap()
        .into_iter()
        .map(|v| *v)
        .collect::<Vec<f32>>();
    
        nd::Array2::from_shape_vec((1, result.len()), result).unwrap()
    }

}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;

    #[test]
    fn test_zero_pad() {
        let arr = nd::Array::from_shape_vec((1, 3), vec![1., 2., 3.]).unwrap();
        let pad_arr: nd::Array2<f32> = zero_pad(arr, 5);
    
        assert!(pad_arr == nd::Array::from_shape_vec((1, 5), vec![1., 2., 3., 0., 0.]).unwrap())
    }

    #[test]
    fn test_repeat_vec() {
        let arr = vec![1., 2., 3.];
        assert!(repeat_vec(arr, 3) == vec![1., 2., 3., 1., 2., 3., 1., 2., 3.])
    }

    #[test]
    fn test_block_diag() {
        let a = nd::Array::from_shape_vec((2, 2), vec![1., 0., 0., 1.]).unwrap();
        let b = nd::Array::from_shape_vec((2, 3), vec![3., 4., 5., 6., 7., 8.]).unwrap();
        let c = nd::Array::from_shape_vec((1, 1), vec![7.]).unwrap();

        let expect = nd::Array::from_shape_vec(
            (5, 6),
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
        let expect = nd::Array::from_shape_vec((3, 3), &[3., 0., 0., 0., 3., 0., 0., 0., 3.]);

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
        let F1_exp = nd::Array::from_shape_vec(
            (6, 6),
            &[
                1., 0.1, 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.1, 0., 0., 0., 0.,
                0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
            ],
        );

        assert!(F1_exp.unwrap() == F1);

        let H1 = m1.build_H();
        let H1_exp = nd::Array::from_shape_slice(
            (4, 6),
            &[
                1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                0., 0., 1.,
            ],
        );
        assert!(H1_exp.unwrap() == H1);

        _ = m1.build_Q();

        let R1 = m1.build_R();
        let R1_exp = nd::Array::from_shape_vec(
            (4, 4),
            &[
                0.1, 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0.3, 0., 0., 0., 0., 0.3,
            ],
        );
        assert!(R1_exp.unwrap() == R1);

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
        let F2_exp = nd::Array::from_row_slice(
            (5, 5),
            &[
                1., 0.1, 0.005, 0., 0., 0., 1., 0.1, 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
                0.1, 0., 0., 0., 0., 1.,
            ],
        );

        assert_relative_eq!(F2_exp, F2, epsilon = 1e-3f32);
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

        let box_ret = model.x_to_box(&x);
        assert!(box_ret == _box);

        let kwargs = ModelKwargs {
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

        let box_ret = model.x_to_box(&x);
        assert!(box_ret == _box);
    }

    #[test]
    fn test_box_to_z() {
        let kwargs = ModelKwargs {
            order_pos: 1,
            dim_pos: 2,
            order_size: 0,
            dim_size: 2,
            ..Default::default()
        };

        let model = Model::new(0.1, Some(kwargs));
        let _box = na::dmatrix![10f32, 10., 20., 20.];
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
        let _box = na::dmatrix![10f32, 10., 0., 20., 20., 50.];
        let result = model.box_to_z(_box);

        assert!(result == na::dmatrix![15., 15., 25., 10., 10.]);
    }
}
