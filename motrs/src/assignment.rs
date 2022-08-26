// Refer: https://github.com/nwtnni/hungarian

use fixedbitset::FixedBitSet;
use nalgebra as na;
use num_traits::NumAssign;
use num_traits::{Bounded, Signed, Zero};
use std::fmt::Debug;
use std::iter::Sum;

macro_rules! on {
    ($s:expr, $i:expr) => {
        $s.contains($i)
    };
}

macro_rules! off {
    ($s:expr, $i:expr) => {
        !$s.contains($i)
    };
}

pub fn minimize<N: 'static + NumAssign + Bounded + Sum<N> + Zero + Signed + Ord + Copy + Debug>(
    matrix: &[N],
    height: usize,
    width: usize,
) -> Vec<Option<usize>> {
    if height == 0 || width == 0 {
        return Vec::new();
    }
    let rotated = width < height;
    let (w, h) = if rotated {
        (height, width)
    } else {
        (width, height)
    };
    let mut m = na::DMatrix::zeros(h, w);

    for i in 0..height {
        for j in 0..width {
            let cost = matrix[width * i + j];
            if rotated {
                m[(width - 1 - j, i)] = cost;
            } else {
                m[(i, j)] = cost;
            }
        }
    }

    let mut stars = na::DMatrix::repeat(h, w, false);
    let mut primes = na::DMatrix::repeat(h, w, false);
    let mut row_cover = FixedBitSet::with_capacity(h);
    let mut col_cover = FixedBitSet::with_capacity(w);

    for r in 0..m.nrows() {
        let row = m.row(r);
        let min = row.iter().min().unwrap().clone();

        for c in 0..m.ncols() {
            m[(r, c)] -= min;
        }
    }

    //********************************************//
    //                                            //
    //                   Step 2                   //
    //                                            //
    //********************************************//

    for i in 0..h {
        for j in 0..w {
            if on!(col_cover, j) {
                continue;
            }

            if m[(i, j)].is_zero() {
                stars[(i, j)] = true;
                col_cover.insert(j);
                break;
            }
        }
    }

    // Reset cover
    col_cover.clear();
    let mut verify = true;

    loop {
        if verify {
            //********************************************//
            //                                            //
            //                   Step 3                   //
            //                                            //
            //********************************************//

            for c in 0..stars.ncols() {
                let col = stars.column(c);
                if col.iter().any(|&s| s) {
                    col_cover.insert(c)
                }
            }

            if col_cover.count_ones(..) == h {
                let assign = (0..stars.nrows()).map(|_r| {
                    let r = stars.row(_r);

                    r.iter()
                        .enumerate()
                        .find(|&(_, &v)| v)
                        .map(|(i, _)| i)
                        .unwrap()
                });

                if rotated {
                    let mut result = vec![None; w];
                    assign
                        .enumerate()
                        .for_each(|(i, j)| result[j] = Some(h - i - 1));
                    return result;
                } else {
                    return assign.map(|j| Some(j)).collect();
                }
            }
        }

        //********************************************//
        //                                            //
        //                   Step 4                   //
        //                                            //
        //********************************************//

        let mut uncovered = None;

        'outer: for i in 0..h {
            if on!(row_cover, i) {
                continue;
            }
            for j in 0..w {
                if on!(col_cover, j) {
                    continue;
                }
                if m[(i, j)].is_zero() {
                    uncovered = Some((i, j));
                    primes[(i, j)] = true;
                    break 'outer;
                }
            }
        }

        if let None = uncovered {
            //********************************************//
            //                                            //
            //                   Step 6                   //
            //                                            //
            //********************************************//
            let mut min = N::max_value();

            for i in 0..h {
                if on!(row_cover, i) {
                    continue;
                }
                for j in 0..w {
                    if on!(col_cover, j) {
                        continue;
                    }
                    let value = m[(i, j)];
                    min = if value < min { value } else { min };
                }
            }

            for i in (0..h).filter(|&i| on!(row_cover, i)) {
                m.row_mut(i).iter_mut().for_each(|c| *c += min);
            }

            for j in (0..w).filter(|&j| off!(col_cover, j)) {
                m.column_mut(j).iter_mut().for_each(|c| *c -= min);
            }

            verify = false;
            continue;
        }

        let (i, j) = uncovered.unwrap();
        if let Some(j) = (0..w).find(|&j| stars[(i, j)]) {
            row_cover.insert(i);
            col_cover.set(j, false);
            verify = false;
            continue;
        }

        //********************************************//
        //                                            //
        //                   Step 5                   //
        //                                            //
        //********************************************//

        let mut path = vec![(i, j)];
        loop {
            let (_, j) = path[path.len() - 1];
            let next_star = (0..h).find(|&i| stars[(i, j)]);

            if let None = next_star {
                break;
            }
            let i = next_star.unwrap();
            path.push((i, j));

            let j = (0..w).find(|&j| primes[(i, j)]).unwrap();
            path.push((i, j));
        }

        for (i, j) in path {
            stars[(i, j)] = primes[(i, j)];
        }

        row_cover.clear();
        col_cover.clear();

        primes.iter_mut().for_each(|p| *p = false);

        verify = true;
    }
}
