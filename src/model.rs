use std::{collections::HashMap, hash::Hash};
use std::cmp::max;
use nalgebra::{dmatrix, DVector, DMatrix, Matrix2x1, dvector};
use crate::Q_discrete_white_noise;
use nalgebra::Dynamic;

struct ModelPreset {
}

impl ModelPreset {
    pub fn new() -> Self {
        ModelPreset {}
    }

    pub fn constant_velocity_and_static_box_size_2d() -> HashMap<String, usize> {
        let key = vec![
            String::from("order_pos"),
            String::from("dim_pos"),
            String::from("order_size"),
            String::from("dim_size"),
        ];
        let value = vec![1, 2, 0, 2];

        key.into_iter().zip(value.into_iter()).collect::<HashMap<_, _> >()
    }

    pub fn constant_acceleration_and_static_box_size_2d() -> HashMap<String, usize> {
        let key = vec![
            String::from("order_pos"),
            String::from("dim_pos"),
            String::from("order_size"),
            String::from("dim_size"),
        ];

        let value = vec![2, 2, 0, 2];

        key.into_iter().zip(value.into_iter()).collect::<HashMap<_, _> >()
    }
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

fn eye(block: usize) -> DMatrix<f64> {
    DMatrix::identity(block, block)
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
        kwargs: HashMap<String, f64>,
    ) -> Self {
        let default_kwargs = HashMap::from([
            (String::from("order_pos"), 1.),
            (String::from("dim_pos"), 2.),
            (String::from("order_size"), 0.),
            (String::from("dim_size"), 2.),
            (String::from("q_var_pos"), 70.),
            (String::from("q_var_size"), 10.),
            (String::from("r_var_pos"), 1.),
            (String::from("r_var_size"), 1.),
            (String::from("p_cov_p0"), 1000.),
        ]);

        let kwargs = kwargs.into_iter().chain(default_kwargs).collect::<HashMap<_, _>>();
        let order_pos: usize = kwargs["order_pos"] as usize;
        let dim_pos: usize = kwargs["dim_pos"] as usize;
        let order_size: usize = kwargs["order_size"] as usize;
        let dim_size: usize = kwargs["dim_size"] as usize;
        let q_var_pos: f64 = kwargs["q_var_pos"];
        let q_var_size: f64 = kwargs["q_var_size"];
        let r_var_pos: f64 = kwargs["r_var_pos"];
        let r_var_size: f64 = kwargs["r_var_size"];
        let p_cov_p0: f64 = kwargs["p_cov_p0"] ;

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

    pub fn build_H(&self) -> DMatrix<f64> {
        fn _base_block(order: usize) -> DMatrix<f64> {
            let mut diag_components = Vec::new();

            let a = vec![1.];
            let b = repeat_vec(vec![0.], order);

            diag_components.extend(a);
            diag_components.extend(b);

            DMatrix::from_vec(1, order, diag_components)
        }

        let _block_pos = repeat_vec(vec![_base_block(self.order_pos)], self.dim_pos);
        let _block_size = repeat_vec(vec![_base_block(self.order_size)], self.dim_size);
        let mut diag_components = Vec::new();

        diag_components.extend(_block_pos);
        diag_components.extend(_block_size);

        block_diag(diag_components)
    }

    pub fn build_P(&self) -> DMatrix<f64> {
        let n = eye(self.state_length);
        n * self.p_cov_p0
    }

    pub fn build_R(&self) -> DMatrix<f64> {
        let block_pos = eye(self.dim_pos) * self.r_var_pos;
        let block_size = eye(self.dim_size) * self.r_var_size;

        block_diag(vec![block_pos, block_size])
    }

    /*
       def box_to_z(self, box: Box) -> Vector:
        assert self.dim_box == len(box)
        box = np.array(box).reshape(2, (int(self.dim_box / 2)))
        center = (np.sum(box, axis=0) / 2.0)[:self.dim_pos]
        length = (box[1, :] - box[0, :])[:self.dim_size]
        return np.concatenate((center, length))

    def box_to_x(self, box: Box) -> Vector:
        x = np.zeros((self.state_length,))
        x[self.z_in_x_idxs] = self.box_to_z(box)
        return x
    */
    pub fn box_to_z(&self, _box: DMatrix<f64>) -> DVector<f64> {
        let _box = DMatrix::from_data(_box).reshape_generic(
            Dynamic::new(self.dim_box / 2),
            Dynamic::new(2),
        );

        let tmp: Vec<f64> = _box.row(0).iter().map(|x| x.clone()).collect::<Vec<_>>().clone();
        dbg!(&tmp);
        dbg!(&self.dim_pos);

        let a = _box.row_sum() / 2.0;
        dbg!(&a);

        let center = a.index((..self.dim_pos, ..));

        let b = _box.index((1, ..)) - _box.index((0, ..));
        let length = b.index((..self.dim_size, ..));

        let mut result = center.row(0).as_slice().to_vec();
        result.append(&mut length.row(0).as_slice().to_vec());

        DVector::from(result)
    }

    /*
    pub fn box_to_x(&self, _box: DVector<f64>) -> DVector<f64> {
        let mut x = DVector::zeros(self.state_length);
    } */
}

#[cfg(test)]
mod test {
    use nalgebra::{ dvector, Matrix3x4, Matrix };
    use super::*;

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

    #[test]
    fn test_eye() {
        let a = eye(3);
        let actual = a * 3.0;
        let expect = DMatrix::from_row_slice(3, 3, &[
            3., 0., 0.,
            0., 3., 0.,
            0., 0., 3.,
        ]);

        assert!(actual == expect)
    }

    #[test]
    fn test_box_to_z() {
        let model = Model::new(1.0, HashMap::default());
        let _box = DMatrix::from_column_slice(1, 4, &[1f64, 2., 3., 4.]);

        let result = model.box_to_z(_box);

        dbg!(&result);
    }
}
