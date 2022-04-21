use nalgebra::{ dmatrix, DMatrix };

pub fn matrix_to_vec(mat: &DMatrix<f64>) -> Vec<f64> {
    let (row, col) = mat.shape();
    let mut result = Vec::default();

    for r in 0..row {
        for  c in 0..col {
            result.push(mat[(r, c)]);
        }
    }

    result
}

pub fn matrix_split(mat: &DMatrix<f64>, indecies_num: usize) -> Vec<DMatrix<f64>> {
    let c = mat.ncols() / indecies_num;
    let r = mat.nrows() / c;

    let mut splitted = Vec::default();
    for i in 0..indecies_num {
        let sp = DMatrix::from(mat.index((.., (i*c)..((i+1)*c))));
        splitted.push(sp);
    }

    splitted
}

pub fn create_matrix_broadcasting(rows: usize, cols: usize, a: &DMatrix<f64>) -> DMatrix<f64> {
    if a.ncols() == 1 && a.nrows() == 1 {
        DMatrix::from_fn(rows, cols, |_r, _c| a[(0, 0)])
    } else if a.ncols() == 1 {
        DMatrix::from_fn(rows, cols, |r, _c| a[(r, 0)])
    } else {
        DMatrix::from_fn(rows, cols, |_r, c| a[(0, c)])
    }
}

pub fn matrix_broadcasting<F>(a: &DMatrix<f64>, b: &DMatrix<f64>, f: F) -> DMatrix<f64> where F: Fn(usize, usize, &DMatrix<f64>, &DMatrix<f64>) -> f64 {
    if a.ncols() == b.ncols() && a.nrows() == b.nrows() {
        let cols = a.ncols();
        let rows = a.nrows();
        let mut result = DMatrix::zeros(rows, cols);

        for r in 0..rows {
            for c in 0..cols {
                result[(r, c)] = f(r, c, a, b);
            }
        }

        result
    } else {
        if a.nrows() == b.nrows() {
            if a.ncols() > b.ncols() {
                let b = create_matrix_broadcasting(a.nrows(), a.ncols(), b);
                matrix_broadcasting(&a, &b, f)
            } else {
                let a = create_matrix_broadcasting(b.nrows(), b.ncols(), a);
                matrix_broadcasting(&a, &b, f)
            }
        } else if a.nrows() >= b.nrows() {
            let b = if b.ncols() == 1 || b.nrows() == 1 {
                create_matrix_broadcasting(a.nrows(), a.ncols(), b)
            } else {
                b.clone()
            };
            matrix_broadcasting(&a, &b, f)
        } else {
            let a = if a.ncols() == 1 || a.nrows() == 1 {
                create_matrix_broadcasting(b.nrows(), b.ncols(), a)
            } else {
                a.clone()
            };
            matrix_broadcasting(&a, &b, f)
        }
    }
}

pub fn matrix_maximum(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| b[(r, c)].max(a[(r, c)]))
}

pub fn matrix_minimum(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| b[(r, c)].min(a[(r, c)]))
}

pub fn matrix_add(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] + b[(r, c)])
}

pub fn matrix_sub(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] - b[(r, c)])
}

pub fn matrix_mul(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] * b[(r, c)])
}

pub fn matrix_div(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] / b[(r, c)])
}

pub fn matrix_clip(mat: &DMatrix<f64>, min_value: Option<f64>, max_value: Option<f64>) -> DMatrix<f64> {
    let cols = mat.ncols();
    let rows = mat.nrows();

    let mut result = DMatrix::zeros(rows, cols);

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
    use super::*;

    #[test]
    fn test_matrix_to_vec() {
        let _box = DMatrix::from_row_slice(2, 4, &[
            1., 2., 3., 4.,
            5., 6., 7., 8.,
        ]);
        let actual = matrix_to_vec(&_box);

        assert!(actual == vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn test_matrix_split() {
        let _box = DMatrix::from_row_slice(2, 4, &[
            1., 2., 3., 4.,
            5., 6., 7., 8.,
        ]);

        let actual = matrix_split(&_box, 4);
        let expect = vec![
            DMatrix::from_row_slice(2, 1, &[1., 5.]),
            DMatrix::from_row_slice(2, 1, &[2., 6.]),
            DMatrix::from_row_slice(2, 1, &[3., 7.]),
            DMatrix::from_row_slice(2, 1, &[4., 8.]),
        ];

        assert!(actual == expect);
    }

    #[test]
    fn test_matrix_maximum() {
        let a = DMatrix::from_row_slice(2, 4, &[
            1., 2., 3., 4.,
            5., 6., 7., 8.,
        ]);
        let b = DMatrix::from_row_slice(1, 4, &[0., 1., 9., 10.]);

        let actual = matrix_maximum(&a, &b);
        let expect = DMatrix::from_row_slice(2, 4, &[
            1., 2., 9., 10.,
            5., 6., 9., 10.,
        ]);

        assert!(actual == expect);

        let a = DMatrix::from_row_slice(1, 1, &[20.]);
        let b = DMatrix::from_row_slice(1, 2, &[10., 30.]);
        assert!(matrix_maximum(&a, &b) == dmatrix![20., 30.]);

        let a = DMatrix::from_row_slice(2, 1, &[20., 15.]);
        let b = DMatrix::from_row_slice(1, 1, &[10.]);
        assert!(matrix_maximum(&a, &b) == DMatrix::from_row_slice(2, 1, &[20., 15.]));
    }
}
