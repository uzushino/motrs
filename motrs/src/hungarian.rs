use faer::Mat;

/// Minimum-cost rectangular assignment using shortest augmenting paths.
/// Each row is assigned to at most one column and vice versa.
pub(crate) fn minimize(cost: &Mat<f32>) -> Vec<(usize, usize)> {
    if cost.nrows() == 0 || cost.ncols() == 0 {
        return Vec::new();
    }
    if cost.nrows() > cost.ncols() {
        let mut pairs = minimize(&cost.transpose().to_owned())
            .into_iter()
            .map(|(column, row)| (row, column))
            .collect::<Vec<_>>();
        pairs.sort_unstable();
        return pairs;
    }

    // Index zero is a temporary unmatched column used to start each path.
    let (row_count, column_count) = cost.shape();
    let mut row_potential = vec![0.0_f64; row_count + 1];
    let mut column_potential = vec![0.0_f64; column_count + 1];
    let mut column_to_row = vec![0; column_count + 1];
    let mut predecessor_column = vec![0; column_count + 1];

    for row in 1..=row_count {
        column_to_row[0] = row;
        let mut current_column = 0;
        let mut distance = vec![f64::INFINITY; column_count + 1];
        let mut visited = vec![false; column_count + 1];
        loop {
            visited[current_column] = true;
            let current_row = column_to_row[current_column];
            let (mut next_column, mut cost_adjustment) = (0, f64::INFINITY);
            for column in 1..=column_count {
                if visited[column] {
                    continue;
                }
                let reduced_cost = cost[(current_row - 1, column - 1)] as f64
                    - row_potential[current_row]
                    - column_potential[column];
                if reduced_cost < distance[column] {
                    distance[column] = reduced_cost;
                    predecessor_column[column] = current_column;
                }
                if distance[column] < cost_adjustment {
                    cost_adjustment = distance[column];
                    next_column = column;
                }
            }
            for column in 0..=column_count {
                if visited[column] {
                    row_potential[column_to_row[column]] += cost_adjustment;
                    column_potential[column] -= cost_adjustment;
                } else {
                    distance[column] -= cost_adjustment;
                }
            }
            current_column = next_column;
            if column_to_row[current_column] == 0 {
                break;
            }
        }
        // Reverse the path to make room for the new row.
        while current_column != 0 {
            let previous_column = predecessor_column[current_column];
            column_to_row[current_column] = column_to_row[previous_column];
            current_column = previous_column;
        }
    }

    let mut pairs = (1..=column_count)
        .filter(|&column| column_to_row[column] != 0)
        .map(|column| (column_to_row[column] - 1, column - 1))
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimizes_square_and_rectangular_costs() {
        assert_eq!(
            minimize(&faer::mat![[4., 1., 3.], [2., 0., 5.], [3., 2., 2.]]),
            vec![(0, 1), (1, 0), (2, 2)]
        );
        assert_eq!(minimize(&faer::mat![[2.], [1.]]), vec![(1, 0)]);
        assert_eq!(
            minimize(&faer::mat![[2., 3.], [4., 5.], [1., 0.]]),
            vec![(0, 0), (2, 1)]
        );
        assert!(minimize(&Mat::zeros(0, 3)).is_empty());
    }

    #[test]
    fn ties_produce_unique_assignments() {
        let pairs = minimize(&Mat::zeros(3, 4));
        assert_eq!(pairs, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn agrees_with_exhaustive_search_for_small_rectangular_matrices() {
        fn exhaustive(cost: &Mat<f32>, row: usize, used: &mut [bool]) -> f32 {
            if row == cost.nrows() {
                return 0.0;
            }
            let mut best = f32::INFINITY;
            for column in 0..cost.ncols() {
                if used[column] {
                    continue;
                }
                used[column] = true;
                best = best.min(cost[(row, column)] + exhaustive(cost, row + 1, used));
                used[column] = false;
            }
            best
        }
        for seed in 0..128 {
            let cost = Mat::from_fn(3, 4, |row, column| {
                ((seed * 17 + row * 29 + column * 13 + row * column * 11) % 31) as f32 - 15.0
            });
            let actual: f32 = minimize(&cost)
                .iter()
                .map(|&(row, column)| cost[(row, column)])
                .sum();
            let expected = exhaustive(&cost, 0, &mut [false; 4]);
            assert_eq!(actual, expected, "seed={seed}");
        }
    }
}
