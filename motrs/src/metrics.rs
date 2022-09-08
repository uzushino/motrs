use nalgebra as na;
use nalgebra::base::dimension::Dynamic;

use ordered_float::OrderedFloat;

use crate::assignment::minimize;
use crate::matrix::*;

pub fn calculate_iou(
    bboxes1: na::DMatrix<f32>,
    bboxes2: na::DMatrix<f32>,
    dim: usize,
) -> na::DMatrix<f32> {
    let r = bboxes1.nrows();
    let bboxes1 = bboxes1.reshape_generic(Dynamic::new(r), Dynamic::new(dim * 2));
    let r = bboxes2.nrows();
    let bboxes2 = bboxes2.reshape_generic(Dynamic::new(r), Dynamic::new(dim * 2));
    let coords_b1 = matrix_split(&bboxes1, 2 * dim);
    let coords_b2 = matrix_split(&bboxes2, 2 * dim);
    let mut coords1: Vec<na::DMatrix<f32>> = Vec::default();

    for _ in 0..dim {
        coords1.push(na::DMatrix::zeros(bboxes1.nrows(), bboxes2.nrows()));
    }

    let mut coords2: Vec<na::DMatrix<f32>> = Vec::default();
    for _ in 0..dim {
        coords2.push(na::DMatrix::zeros(bboxes1.nrows(), bboxes2.nrows()));
    }

    let zero = na::DMatrix::zeros(1, 1);
    let mut val_inter: na::DMatrix<f32> = na::DMatrix::repeat(1, 1, 1.);
    let mut val_b1: na::DMatrix<f32> = na::DMatrix::repeat(1, 1, 1.);
    let mut val_b2: na::DMatrix<f32> = na::DMatrix::repeat(1, 1, 1.);

    for d in 0..dim {
        coords1[d] = matrix_maximum(&coords_b1[d], &coords_b2[d].transpose());
        coords2[d] = matrix_minimum(&coords_b1[d + dim], &coords_b2[d + dim].transpose());

        let sub = coords2[d].clone() - coords1[d].clone();
        let tmp = matrix_maximum(&sub, &zero);

        val_inter = matrix_mul(&tmp, &val_inter);

        val_b1 = matrix_mul(
            &(coords_b1[d + dim].clone() - coords_b1[d].clone()),
            &val_b1,
        );
        val_b2 = matrix_mul(
            &val_b2,
            &(coords_b2[d + dim].clone() - coords_b2[d].clone()),
        );
    }

    let tmp = matrix_sub(
        &matrix_add(&val_b1, &val_b2.transpose()),
        &val_inter.clone(),
    );
    let iou = matrix_div(&val_inter, &matrix_clip(&tmp, Some(0.), None));

    iou
}

pub fn _sequence_has_none(seq: &Vec<Option<na::DMatrix<f32>>>) -> bool {
    seq.iter().any(|v| v.is_none())
}

pub fn linear_sum_assignment(mat: &na::DMatrix<f32>) -> (Vec<usize>, Vec<usize>) {
    let height = mat.nrows();
    let width = mat.ncols();
    let mut matrix = vec![];

    for row in 0..mat.nrows() {
        for col in 0..mat.ncols() {
            matrix.push(OrderedFloat(mat[(row, col)]));
        }
    }

    let assignment: Vec<Option<usize>> = minimize(&matrix, height, width);

    let mut row_idxs = vec![];
    let mut col_idxs = vec![];

    for (i, a) in assignment.iter().enumerate() {
        if let Some(n) = a {
            row_idxs.push(i);
            col_idxs.push(*n);
        }
    }

    (row_idxs, col_idxs)
}

mod test {
    use super::*;

    use approx::assert_relative_eq;
    use nalgebra::dmatrix;

    #[test]
    fn test_iou() {
        let b1 = na::DMatrix::from_row_slice(1, 2, &[10., 20.]);
        let b2 = na::DMatrix::from_row_slice(2, 2, &[10., 21., 30., 40.]);
        let iou_1d = calculate_iou(b1, b2, 1);

        assert_relative_eq!(iou_1d, dmatrix![0.9091, 0.], epsilon = 1e-3f32);

        let b1 = na::DMatrix::from_row_slice(2, 4, &[20.1, 20.1, 30.1, 30.1, 15., 15., 25., 25.]);
        let b2 = na::DMatrix::from_row_slice(1, 4, &[10., 10., 20., 20.]);
        let iou_2d = calculate_iou(b1, b2, 2);

        assert_relative_eq!(
            iou_2d,
            na::DMatrix::from_row_slice(2, 1, &[0., 0.1429]),
            epsilon = 1e-3f32
        );

        let b1 = na::DMatrix::from_row_slice(1, 6, &[10., 10., 10., 20., 20., 20.]);
        let b2 = na::DMatrix::from_row_slice(
            2,
            6,
            &[
                10., 11., 10.2, 21., 19.9, 20.3, 30., 30., 30., 90., 90., 90.,
            ],
        );
        let iou_3d = calculate_iou(b1, b2, 3);

        assert_relative_eq!(
            iou_3d,
            na::DMatrix::from_row_slice(1, 2, &[0.7811, 0.]),
            epsilon = 1e-3f32
        );
    }

    #[test]
    fn test_linear_sum_assignment() {
        let a = na::DMatrix::from_row_slice(3, 3, &[4., 1., 3., 2., 0., 5., 3., 2., 2.]);

        assert!(linear_sum_assignment(&a) == (vec![0, 1, 2], vec![1, 0, 2]));

        let a = na::DMatrix::from_row_slice(2, 1, &[2., 1.]);

        assert!(linear_sum_assignment(&a) == (vec![1], vec![0]));

        let a = na::DMatrix::from_row_slice(3, 2, &[2., 3., 4., 5., 1., 0.]);

        assert!(linear_sum_assignment(&a) == (vec![0, 2], vec![0, 1]));

        let a = na::DMatrix::from_row_slice(2, 1, &[-0., -0.806]);

        assert!(linear_sum_assignment(&a) == (vec![1], vec![0]));
    }
}
