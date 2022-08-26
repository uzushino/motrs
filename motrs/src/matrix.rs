use core::num;
use std::fmt::Debug;

use na::Scalar;
use nalgebra as na;
use num_traits::Zero;

pub fn matrix_to_vec<T: Copy>(mat: &na::DMatrix<T>) -> Vec<T> {
    let (row, col) = mat.shape();
    let mut result = Vec::default();

    for r in 0..row {
        for c in 0..col {
            result.push(mat[(r, c)]);
        }
    }

    result
}

pub fn matrix_split<T>(mat: &na::DMatrix<T>, indecies_num: usize) -> Vec<na::DMatrix<T>>
where
    T: Zero + Debug + Clone + Scalar + Copy,
{
    let c = mat.ncols() / indecies_num;
    let r = mat.nrows() / c;

    let mut splitted = Vec::default();
    for i in 0..indecies_num {
        let sp = na::DMatrix::from(mat.index((.., (i * c)..((i + 1) * c))));
        splitted.push(sp);
    }

    splitted
}

pub fn create_matrix_broadcasting<T>(rows: usize, cols: usize, a: &na::DMatrix<T>) -> na::DMatrix<T>
where
    T: Zero + Debug + Clone + Scalar + Copy,
{
    if a.ncols() == 1 && a.nrows() == 1 {
        na::DMatrix::from_fn(rows, cols, |_r, _c| a[(0, 0)])
    } else if a.ncols() == 1 {
        na::DMatrix::from_fn(rows, cols, |r, _c| a[(r, 0)])
    } else {
        na::DMatrix::from_fn(rows, cols, |_r, c| a[(0, c)])
    }
}

pub fn matrix_broadcasting<F, T>(a: &na::DMatrix<T>, b: &na::DMatrix<T>, f: F) -> na::DMatrix<T>
where
    F: Fn(usize, usize, &na::DMatrix<T>, &na::DMatrix<T>) -> T,
    T: num_traits::Num + Zero + Debug + Copy + Scalar,
{
    if a.ncols() == b.ncols() && a.nrows() == b.nrows() {
        let cols = a.ncols();
        let rows = a.nrows();
        let mut result = na::DMatrix::zeros(rows, cols);

        for r in 0..rows {
            for c in 0..cols {
                result[(r, c)] = f(r, c, a, b);
            }
        }

        result
    } else {
        let a = if a.nrows() == 1 && a.ncols() == 1 {
            na::DMatrix::repeat(b.nrows(), b.ncols(), a[(0, 0)])
        } else if a.nrows() == 1 {
            create_matrix_broadcasting(b.nrows(), a.ncols(), a)
        } else if a.ncols() == 1 {
            create_matrix_broadcasting(a.nrows(), b.ncols(), a)
        } else {
            a.clone()
        };
        let b = if b.nrows() == 1 && b.ncols() == 1 {
            na::DMatrix::repeat(a.nrows(), a.ncols(), b[(0, 0)])
        } else if b.nrows() == 1 {
            create_matrix_broadcasting(a.nrows(), b.ncols(), b)
        } else if b.ncols() == 1 {
            create_matrix_broadcasting(b.nrows(), a.ncols(), b)
        } else {
            b.clone()
        };

        if !(a.nrows() == b.nrows() && a.ncols() == b.ncols()) {
            panic!(
                "Can not broadcasting . a: {:?}, b: {:?}",
                a.shape(),
                b.shape()
            );
        }

        matrix_broadcasting(&a, &b, f)
    }
}

pub fn matrix_maximum(a: &na::DMatrix<f32>, b: &na::DMatrix<f32>) -> na::DMatrix<f32> {
    matrix_broadcasting(a, b, |r, c, a, b| b[(r, c)].max(a[(r, c)]))
}

pub fn matrix_minimum(a: &na::DMatrix<f32>, b: &na::DMatrix<f32>) -> na::DMatrix<f32> {
    matrix_broadcasting(a, b, |r, c, a, b| b[(r, c)].min(a[(r, c)]))
}

pub fn matrix_add<T>(a: &na::DMatrix<T>, b: &na::DMatrix<T>) -> na::DMatrix<T>
where
    T: num_traits::Num + Zero + Debug + Copy + Scalar,
{
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] + b[(r, c)])
}

pub fn matrix_sub<T>(a: &na::DMatrix<T>, b: &na::DMatrix<T>) -> na::DMatrix<T>
where
    T: num_traits::Num + Zero + Debug + Copy + Scalar,
{
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] - b[(r, c)])
}

pub fn matrix_mul<T>(a: &na::DMatrix<T>, b: &na::DMatrix<T>) -> na::DMatrix<T>
where
    T: num_traits::Num + Zero + Debug + Copy + Scalar,
{
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] * b[(r, c)])
}

pub fn matrix_div<T>(a: &na::DMatrix<T>, b: &na::DMatrix<T>) -> na::DMatrix<T>
where
    T: num_traits::Num + Zero + Debug + Copy + Scalar,
{
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] / b[(r, c)])
}

pub fn matrix_dot<T: num_traits::Num + num_traits::Float + Debug + Scalar>(
    a: &na::DMatrix<T>,
    b: &na::DMatrix<T>,
) -> na::DMatrix<T> {
    if b.nrows() == 1 {
        let mut mat: na::DMatrix<T> = na::DMatrix::repeat(1, a.nrows(), T::zero());
        for r in 0..a.nrows() {
            let mut total = 0.0;
            for c in 0..b.ncols() {
                total += (a[(r, c)] * b[(0, c)]).to_f32().unwrap_or_default();
            }
            mat[(0, r)] = num_traits::NumCast::from(total).unwrap();
        }
        mat
    } else if b.ncols() == 1 {
        let b = create_matrix_broadcasting(b.nrows(), a.ncols(), &b);
        let dot = matrix_dot(a, &b);
        na::DMatrix::from_fn(a.nrows(), 1, |r, c| dot[(r, 0)])
    } else {
        let mat = na::DMatrix::from_fn(a.nrows(), b.ncols(), |r, c| {
            let col = a.row(r);
            let row = b.column(c);

            let col = na::DMatrix::from_fn(col.nrows(), col.ncols(), |r, c| {
                col[(r, c)].to_f32().unwrap_or_default()
            });
            let row = na::DMatrix::from_fn(row.nrows(), row.ncols(), |r, c| {
                row[(r, c)].to_f32().unwrap_or_default()
            });

            (col * row).sum()
        });

        na::DMatrix::from_fn(mat.nrows(), mat.ncols(), |r, c| {
            num_traits::NumCast::from(mat[(r, c)]).unwrap()
        })
    }
}

pub fn matrix_clip(
    mat: &na::DMatrix<f32>,
    min_value: Option<f32>,
    max_value: Option<f32>,
) -> na::DMatrix<f32> {
    let cols = mat.ncols();
    let rows = mat.nrows();

    let mut result = na::DMatrix::zeros(rows, cols);

    for r in 0..rows {
        for c in 0..cols {
            let mut v = mat[(r, c)];
            if let Some(mv) = min_value {
                v = v.max(mv);
            }
            if let Some(mv) = max_value {
                v = v.min(mv);
            }

            result[(r, c)] = v;
        }
    }

    result
}

#[cfg(test)]
mod test {
    use na::dmatrix;

    use super::*;

    #[test]
    fn test_broadcasting() {
        let a = na::DMatrix::from_row_slice(2, 3, &[2, 4, 6, 8, 10, 12]);
        let b = na::DMatrix::from_row_slice(2, 1, &[10, 100]);

        assert!(
            matrix_add(&a, &b) == na::DMatrix::from_row_slice(2, 3, &[12, 14, 16, 108, 110, 112])
        );
    }

    #[test]
    fn test_matrix_to_vec() {
        let _box = na::DMatrix::from_row_slice(2, 4, &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let actual = matrix_to_vec(&_box);

        assert!(actual == vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn test_matrix_split() {
        let _box = na::DMatrix::from_row_slice(2, 4, &[1., 2., 3., 4., 5., 6., 7., 8.]);

        let actual = matrix_split(&_box, 4);
        let expect = vec![
            na::DMatrix::from_row_slice(2, 1, &[1., 5.]),
            na::DMatrix::from_row_slice(2, 1, &[2., 6.]),
            na::DMatrix::from_row_slice(2, 1, &[3., 7.]),
            na::DMatrix::from_row_slice(2, 1, &[4., 8.]),
        ];

        assert!(actual == expect);
    }

    #[test]
    fn test_matrix_maximum() {
        let a = na::DMatrix::from_row_slice(2, 4, &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = na::DMatrix::from_row_slice(1, 4, &[0., 1., 9., 10.]);

        let actual = matrix_maximum(&a, &b);
        let expect = na::DMatrix::from_row_slice(2, 4, &[1., 2., 9., 10., 5., 6., 9., 10.]);

        assert!(actual == expect);

        let a = na::DMatrix::from_row_slice(1, 1, &[20.]);
        let b = na::DMatrix::from_row_slice(1, 2, &[10., 30.]);
        assert!(matrix_maximum(&a, &b) == na::dmatrix![20., 30.]);

        let a = na::DMatrix::from_row_slice(2, 1, &[20., 15.]);
        let b = na::DMatrix::from_row_slice(1, 1, &[10.]);
        assert!(matrix_maximum(&a, &b) == na::DMatrix::from_row_slice(2, 1, &[20., 15.]));
    }

    #[test]
    fn test_matrix_dot() {
        let a = na::DMatrix::from_row_slice(2, 2, &[1., 2., 3., 4.]);
        let b = na::DMatrix::from_row_slice(2, 2, &[5., 6., 7., 8.]);
        assert!(matrix_dot(&a, &b) == na::DMatrix::from_row_slice(2, 2, &[19., 22., 43., 50.]));

        let a = na::DMatrix::from_row_slice(2, 2, &[1., 2., 3., 4.]);
        let b = na::DMatrix::from_row_slice(2, 1, &[10., 20.]);
        assert!(matrix_dot(&a, &b) == na::DMatrix::from_row_slice(2, 1, &[50., 110.]));

        let a = na::DMatrix::from_row_slice(3, 3, &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let b = na::DMatrix::from_row_slice(1, 3, &[10., 20., 30.]);
        assert!(matrix_dot(&a, &b) == na::DMatrix::from_row_slice(1, 3, &[140., 320., 500.]));
    }
}
