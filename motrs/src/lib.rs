use std::ops::Mul;

use ndarray as nd;

pub mod matrix;
pub mod model;
pub mod tracker;

mod assignment;
mod filter;
mod metrics;

use crate::tracker::Track;

pub fn Q_discrete_white_noise<T>(
    dim: usize,
    dt: f32,
    var: f32,
    block_size: usize,
    order_by_dim: bool,
) -> nd::Array2<T> {
    if !vec![2, 3, 4].contains(&dim) {
        panic!();
    }

    let Q = match dim {
        2 => vec![
            vec![(0.25 * dt).powf(4.), (0.5 * dt).powf(3.)],
            vec![(0.5 * dt).powf(3.), dt.powf(2.)],
        ],
        3 => vec![
            vec![
                (0.25 * dt).powf(4.),
                (0.5 * dt).powf(3.),
                (0.5 * dt).powf(2.0),
            ],
            vec![(0.5 * dt).powf(3.), dt.powf(2.), dt],
            vec![(0.5 * dt).powf(2.), dt, 1.],
        ],
        _ => vec![
            vec![
                dt.powf(6.) / 36.,
                dt.powf(5.) / 12.,
                dt.powf(4.) / 6.,
                dt.powf(3.) / 6.,
            ],
            vec![
                dt.powf(5.) / 12.,
                dt.powf(4.) / 4.,
                dt.powf(3.) / 2.,
                dt.powf(2.) / 2.,
            ],
            vec![dt.powf(4.) / 6., dt.powf(3.) / 2., dt.powf(2.), dt],
            vec![dt.powf(3.) / 6., dt.powf(2.) / 2., dt, 1.],
        ],
    };

    order_by_derivative(Q, dim, block_size).map(|v| T::from_f32(v).unwrap_or(T::zero()))
        * T::from_f32(var).unwrap_or(T::zero())
}

fn order_by_derivative<T>(
    q: Vec<Vec<T>>,
    dim: usize,
    block_size: usize,
) -> nd::Array2<T> {
    let n = dim * block_size;
    let mut d = nd::Array2::<T>::zeros(n);

    for (i, &x) in q.iter().flatten().enumerate() {
        let f = nd::Array2(block_size, block_size) * x;
        let ix = (i / dim) * block_size;
        let iy = (i % dim) * block_size;

        d.index_mut((ix..(ix + block_size), iy..(iy + block_size)))
            .copy_from(&f);
    }

    d
}

pub fn tracker_to_string(track: Track) -> String {
    let score = if let Some(score) = track.score {
        score
    } else {
        -1.
    };
    format!(
        "ID: {} | S: {} | C: {}",
        track.id,
        score,
        track.class_id.unwrap_or_default()
    )
}
