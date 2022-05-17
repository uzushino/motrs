use nalgebra as na;

pub fn matrix_to_vec(mat: &na::DMatrix<f64>) -> Vec<f64> {
    let (row, col) = mat.shape();
    let mut result = Vec::default();

    for r in 0..row {
        for c in 0..col {
            result.push(mat[(r, c)]);
        }
    }

    result
}

pub fn matrix_split(mat: &na::DMatrix<f64>, indecies_num: usize) -> Vec<na::DMatrix<f64>> {
    let c = mat.ncols() / indecies_num;
    let r = mat.nrows() / c;

    let mut splitted = Vec::default();
    for i in 0..indecies_num {
        let sp = na::DMatrix::from(mat.index((.., (i * c)..((i + 1) * c))));
        splitted.push(sp);
    }

    splitted
}

pub fn create_matrix_broadcasting(
    rows: usize,
    cols: usize,
    a: &na::DMatrix<f64>,
) -> na::DMatrix<f64> {
    if a.ncols() == 1 && a.nrows() == 1 {
        na::DMatrix::from_fn(rows, cols, |_r, _c| a[(0, 0)])
    } else if a.ncols() == 1 {
        na::DMatrix::from_fn(rows, cols, |r, _c| a[(r, 0)])
    } else {
        na::DMatrix::from_fn(rows, cols, |_r, c| a[(0, c)])
    }
}

pub fn matrix_broadcasting<F>(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>, f: F) -> na::DMatrix<f64>
where
    F: Fn(usize, usize, &na::DMatrix<f64>, &na::DMatrix<f64>) -> f64,
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

pub fn matrix_maximum(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| b[(r, c)].max(a[(r, c)]))
}

pub fn matrix_minimum(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| b[(r, c)].min(a[(r, c)]))
}

pub fn matrix_add(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] + b[(r, c)])
}

pub fn matrix_sub(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] - b[(r, c)])
}

pub fn matrix_mul(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] * b[(r, c)])
}

pub fn matrix_div(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    matrix_broadcasting(a, b, |r, c, a, b| a[(r, c)] / b[(r, c)])
}

pub fn matrix_dot(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    //if a.ncols() == 1 && b.ncols() == 1 && b.nrows() == 1 {
    //    return matrix_mul(a, b);
    //}
    //if a.nrows() == 1 && b.nrows() == 1 {
    //    return na::dmatrix![a.dot(b)];
    //}

    if b.nrows() == 1 {
        let mut mat = na::DMatrix::repeat(1, a.nrows(), 0.0);
        for r in 0..a.nrows() {
            let mut total = 0.0;
            for c in 0..b.ncols() {
                total += a[(r, c)] * b[(0, c)];
            }
            mat[(0, r)] = total;
        }

        mat
    } else if b.ncols() == 1 {
        let b = create_matrix_broadcasting(b.nrows(), a.ncols(), &b);
        let dot = matrix_dot(a, &b);
        na::DMatrix::from_fn(a.nrows(), 1, |r, c| dot[(r, 0)])
    } else {
        na::DMatrix::from_fn(a.nrows(), b.ncols(), |r, c| {
            let col = a.row(r);
            let row = b.column(c);

            (col * row).sum()
        })
    }
}

pub fn matrix_clip(
    mat: &na::DMatrix<f64>,
    min_value: Option<f64>,
    max_value: Option<f64>,
) -> na::DMatrix<f64> {
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
    use super::*;

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
