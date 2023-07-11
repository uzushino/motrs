use num_traits::Zero;
use ndarray as nd;

pub fn matrix_to_vec<T: Copy>(mat: &nd::Array2<T>) -> Vec<T> {
    let (row, col) = mat.shape();
    let mut result = Vec::default();

    for r in 0..row {
        for c in 0..col {
            result.push(mat[[r, c]]);
        }
    }

    result
}

pub fn matrix_split<T: Clone>(mat: &nd::Array2<T>, indices_num: usize) -> Vec<nd::Array2<T>> {
    let c = mat.ncols() / indices_num;
    let r = mat.nrows() / c;

    let mut splitted = Vec::default();
    for i in 0..indices_num {
        let sp = mat.index_axis(nd::Axis(1), i * c..(i + 1) * c).to_owned();
        splitted.push(sp);
    }

    splitted
}

pub fn create_matrix_broadcasting<T>(rows: usize, cols: usize, a: &nd::Array2<T>) -> nd::Array2<T>
where
    T: Copy,
{
    if a.ncols() == 1 && a.nrows() == 1 {
        nd::Array::from_elem((rows, cols), a[[0, 0]])
    } else if a.ncols() == 1 {
        nd::Array::from_shape_fn((rows, cols), |(r, _c)| a[[r, 0]])
    } else {
        nd::Array::from_shape_fn((rows, cols), |(_r, c)| a[[0, c]])
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

/*
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_broadcasting() {
        let a = na::DMatrix::from_row_slice(2, 3, &[2., 4., 6., 8., 10., 12.]);
        let b = na::DMatrix::from_row_slice(2, 1, &[10., 100.]);

        assert!(
            matrix_add(&a, &b)
                == na::DMatrix::from_row_slice(2, 3, &[12., 14., 16., 108., 110., 112.])
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
 */