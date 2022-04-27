use nalgebra::{DMatrix, dmatrix};
use nalgebra::base::dimension::Dynamic;

use pathfinding::kuhn_munkres::kuhn_munkres_min;
use pathfinding::matrix::Matrix;
use ordered_float::OrderedFloat;

use crate::matrix::*;

pub fn calculate_iou(bboxes1: DMatrix<f64>, bboxes2: DMatrix<f64>, dim: usize) -> DMatrix<f64> {
    let r = bboxes1.nrows();
    let bboxes1 = bboxes1.reshape_generic(
        Dynamic::new(r),
        Dynamic::new(dim * 2)
    );

    let r = bboxes2.nrows();
    let bboxes2 = bboxes2.reshape_generic(
        Dynamic::new(r),
        Dynamic::new(dim * 2)
    );

    let coords_b1 = matrix_split(&bboxes1, 2 * dim);
    let coords_b2 = matrix_split(&bboxes2, 2 * dim);

    let mut coords1: Vec<DMatrix<f64>> = Vec::default();
    for _ in 0..dim {
        coords1.push(DMatrix::zeros(bboxes1.nrows(), bboxes2.nrows()));
    }

    let mut coords2: Vec<DMatrix<f64>> = Vec::default();
    for _ in 0..dim {
        coords2.push(DMatrix::zeros(bboxes1.nrows(), bboxes2.nrows()));
    }

    let zero = DMatrix::zeros(1, 1);
    let mut val_inter: DMatrix<f64> = DMatrix::repeat(1, 1, 1.);
    let mut val_b1: DMatrix<f64> = DMatrix::repeat(1, 1, 1.);
    let mut val_b2: DMatrix<f64> = DMatrix::repeat(1, 1, 1.);

    for d in 0..dim {
        coords1[d] = matrix_maximum(&coords_b1[d], &coords_b2[d].transpose());
        coords2[d] = matrix_minimum(&coords_b1[d + dim], &coords_b2[d + dim].transpose());

        let sub = coords2[d].clone() - coords1[d].clone();

        val_inter = matrix_mul(&matrix_maximum(&sub, &zero), &val_inter);
        val_b1 = matrix_mul(&(coords_b1[d + dim].clone() - coords_b1[d].clone()), &val_b1);
        val_b2 = matrix_mul(&val_b2, &(coords_b2[d + dim].clone() - coords_b2[d].clone()));
    }

    let b1 = create_matrix_broadcasting(val_inter.nrows(), val_inter.ncols(), &val_b1);
    let tmp = matrix_sub(&matrix_add(&b1, &val_b2.transpose()), &val_inter.clone());
    let iou = matrix_div(&val_inter, &matrix_clip(&tmp, Some(0.), None));

    iou
}

pub fn _sequence_has_none(seq: &Vec<Option<DMatrix<f64>>>) -> bool {
    seq.iter().any(|v| v.is_none())
}

pub fn cosine_distance(vector1: &DMatrix<f64>, vector2: &DMatrix<f64>) -> f64 {
    let norm = vector1.dot(vector1) * vector2.dot(vector2);

    if norm > 0.0 {
        return 1. - vector1.dot(vector2) / norm.sqrt()
    }

    0.0
}

pub fn angular_similarity(vector1: DMatrix<f64>, vector2: DMatrix<f64>) -> DMatrix<f64> {
    let mut result = DMatrix::zeros(vector1.nrows(), 1);

    for row in 0..vector1.nrows() {
        let mat = DMatrix::from(vector1.rows(row, 1));
        result[(row, 0)] = 1. - (cosine_distance(&mat, &vector2) / 2.);
    }

    result
}

pub fn linear_sum_assignment(mat: &DMatrix<f64>) -> (Vec<usize>, Vec<usize>) {
    let mut mat_vec = vec![vec![OrderedFloat(0.); mat.ncols()]; mat.nrows()];

    for row in 0..mat.nrows() {
        for col in 0..mat.nrows() {
            mat_vec[row][col] = OrderedFloat(mat[(row, col)]);
        }
    }

    let weights: Matrix<OrderedFloat<f64>> = Matrix::from_rows(mat_vec).unwrap();
    let (cash_flow, assignments) = kuhn_munkres_min(&weights);

    ((0..mat.nrows()).into_iter().collect::<Vec<_>>(), assignments)
}

mod test {
    use super::*;

    use nalgebra::{dmatrix};
    use approx::*;

    #[test]
    fn test_iou() {
        let b1 = DMatrix::from_row_slice(1, 2, &[10., 20.]);
        let b2 = DMatrix::from_row_slice(2, 2, &[10., 21., 30., 40.]);
        let iou_1d = calculate_iou(b1, b2, 1);

        assert_relative_eq!(iou_1d , dmatrix![0.9091, 0.], epsilon = 1e-3f64);

        let b1 = DMatrix::from_row_slice(2, 4, &[20.1, 20.1, 30.1, 30.1, 15., 15., 25., 25.]);
        let b2 = DMatrix::from_row_slice(1, 4, &[10., 10., 20., 20.]);
        let iou_2d = calculate_iou(b1, b2, 2);

        assert_relative_eq!(iou_2d , DMatrix::from_row_slice(2, 1, &[0., 0.1429]), epsilon = 1e-3f64);

        let b1 = DMatrix::from_row_slice(1, 6, &[10., 10., 10., 20., 20., 20.]);
        let b2 = DMatrix::from_row_slice(2, 6, &[10., 11., 10.2, 21., 19.9, 20.3, 30., 30., 30., 90., 90., 90.]);
        let iou_3d = calculate_iou(b1, b2, 3);

        assert_relative_eq!(iou_3d, DMatrix::from_row_slice(1, 2, &[0.7811, 0.]), epsilon = 1e-3f64);
    }

    #[test]
    fn test_match_by_cost_matrix() {
        /*
        let matching_fn = IOUAndFeatureMatchingFunction::default();
        let b1 = MultiObjectTracker::new(
            0.041666666666666664,
            HashMap::default(),
            Some(Box::new(matching_fn)),
            tracker_kwargs,
            matching_fn_kwargs,
            active_tracks_kwargs
        );
        let b2 = DMatrix::from_row_slice(2, 2, &[10., 21., 30., 40.]);
        let iou_1d = match_by_cost_matrix(
            b1,
            b2,
            1.,
            0.,
            None,
            None
        );

        assert_relative_eq!(iou_1d , dmatrix![0.9091, 0.], epsilon = 1e-3f64);
*/
    }

    #[test]
    fn test_anguler_similarity() {
        let a = DMatrix::from_row_slice(2, 3, &[
            11., 136., 234.,
            44., 159., 201.,
        ]);

        let b = DMatrix::from_row_slice(1, 3, &[
            30., 160., 208.
        ]);

        let actual = angular_similarity(a, b);
        let expect = DMatrix::from_row_slice(2, 1, &[
                0.99452275, 0.99916559
        ]);

        assert_relative_eq!(expect, actual, epsilon = 1e-3f64);
    }
}
