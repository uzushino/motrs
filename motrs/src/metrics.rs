use ordered_float::OrderedFloat;
use ndarray as nd;
use crate::assignment::minimize;
use crate::matrix::*;

pub fn calculate_iou(
    bboxes1: &nd::Array2<f32>,
    bboxes2: &nd::Array2<f32>,
    dim: usize,
) -> nd::Array2<f32> {
    let r1 = bboxes1.nrows();
    let r2 = bboxes2.nrows();
    let coords_b1 = matrix_split(&bboxes1, dim);
    let coords_b2 = matrix_split(&bboxes2, dim);
    let mut coords1: Vec<nd::Array2<f32>> = vec![nd::Array2::zeros((r1, r2)); dim];
    let mut coords2: Vec<nd::Array2<f32>> = vec![nd::Array2::zeros((r1, r2)); dim];

    for d in 0..dim {
        let coords_b1_d = &coords_b1[d];
        let coords_b2_d_t = &coords_b2[d].t();
        coords1[d].assign(&coords_b1_d.max(coords_b2_d_t));
        coords2[d].assign(&coords_b1[d+dim].min(coords_b2[d+dim].t()));
    }

    let zero = nd::Array2::zeros((1, 1));
    let mut val_inter: nd::Array2<f32> = nd::Array2::ones((r1, r2));
    let mut val_b1: nd::Array2<f32> = nd::Array2::ones((r1, r2));
    let mut val_b2: nd::Array2<f32> = nd::Array2::ones((r1, r2));

    for d in 0..dim {
        let sub = &coords2[d] - &coords1[d];
        let tmp = sub.mapv_into(|x| x.max(0.));

        val_inter = &val_inter * &tmp;
        val_b1 = &val_b1 * &(&coords_b1[d + dim] - &coords_b1[d]);
        val_b2 = &(&coords_b2[d + dim] - &coords_b2[d]) * &val_b2;
    }

    let tmp = &val_b1 + &val_b2.t() - &val_inter;

    &val_inter / &tmp.mapv_into(|x| x.max(0.))
}

pub fn _sequence_has_none<T>(seq: &Vec<Option<T>>) -> bool {
    seq.iter().any(|v| v.is_none())
}

pub fn linear_sum_assignment(mat: &nd::Array2<f32>) -> (Vec<usize>, Vec<usize>) {
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

    #[test]
    fn test_iou() {
        let b1 = nd::Array2::from_shape_vec((1, 2), vec![10., 20.]);
        let b2 = nd::Array2::from_shape_vec((2, 2), vec![10., 21., 30., 40.]);
        let iou_1d = calculate_iou(b1, b2, 1);

        assert_relative_eq!(iou_1d, nd::array![0.9091, 0.], epsilon = 1e-3f32);
/*
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
        ); */
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
