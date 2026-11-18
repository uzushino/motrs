//! Decide which observation belongs to each predicted track.
//! Rows represent tracks; columns represent detections plus an unmatched option.
use faer::Mat;

use crate::hungarian::minimize;
use crate::{BoundingBox, Detection};

pub(crate) struct Candidate<'a> {
    pub track_index: usize,
    pub bounds: &'a BoundingBox,
    pub feature: Option<&'a [f32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Match {
    pub track_index: usize,
    pub detection_index: usize,
}

const UNMATCHED_COST: f32 = 1.0;
const FORBIDDEN_COST: f32 = 2.0;

pub(crate) fn associate(
    tracks: &[Candidate<'_>],
    detections: &[Detection],
    min_iou: f32,
    appearance_weight: f32,
) -> Vec<Match> {
    if tracks.is_empty() || detections.is_empty() {
        return Vec::new();
    }
    if let ([track], [detection]) = (tracks, detections) {
        return (pair_cost(track, detection, min_iou, appearance_weight) < UNMATCHED_COST)
            .then_some(Match {
                track_index: track.track_index,
                detection_index: 0,
            })
            .into_iter()
            .collect();
    }

    // Give every track an unmatched column. Without these columns the optimizer
    // would have to assign unrelated objects and could displace valid matches.
    let cost = Mat::from_fn(
        tracks.len(),
        detections.len() + tracks.len(),
        |track_index, column| {
            if column >= detections.len() {
                return UNMATCHED_COST;
            }
            pair_cost(
                &tracks[track_index],
                &detections[column],
                min_iou,
                appearance_weight,
            )
        },
    );
    // A set of eligible pairs with no shared tracks or detections is already optimal.
    if let Some(matches) = independent_matches(tracks, detections.len(), &cost) {
        return matches;
    }
    minimize(&cost)
        .into_iter()
        .filter_map(|(row, detection_index)| {
            (detection_index < detections.len() && cost[(row, detection_index)] < UNMATCHED_COST)
                .then_some(Match {
                    track_index: tracks[row].track_index,
                    detection_index,
                })
        })
        .collect()
}

fn independent_matches(
    tracks: &[Candidate<'_>],
    detection_count: usize,
    cost: &Mat<f32>,
) -> Option<Vec<Match>> {
    let mut assigned = vec![false; detection_count];
    let mut matches = Vec::with_capacity(tracks.len().min(detection_count));
    for (row, track) in tracks.iter().enumerate() {
        let mut eligible =
            (0..detection_count).filter(|&column| cost[(row, column)] < UNMATCHED_COST);
        let Some(detection_index) = eligible.next() else {
            continue;
        };
        if assigned[detection_index] || eligible.next().is_some() {
            return None;
        }
        assigned[detection_index] = true;
        matches.push(Match {
            track_index: track.track_index,
            detection_index,
        });
    }
    Some(matches)
}

fn pair_cost(
    track: &Candidate<'_>,
    detection: &Detection,
    min_iou: f32,
    appearance_weight: f32,
) -> f32 {
    let overlap = track.bounds.iou(&detection.bounds);
    if overlap < min_iou {
        return FORBIDDEN_COST;
    }
    if appearance_weight == 0.0 {
        return 1.0 - overlap;
    }
    let appearance = track
        .feature
        .zip(detection.feature.as_deref())
        .and_then(|(previous, observed)| cosine_similarity(previous, observed));
    // Missing or incompatible appearance falls back to geometry alone.
    let similarity = appearance.map_or(1.0, |similarity| {
        1.0 - appearance_weight + appearance_weight * similarity
    });
    1.0 - overlap * similarity
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() {
        return None;
    }
    let (mut dot, mut left_norm, mut right_norm) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&left, &right) in left.iter().zip(right) {
        let (left, right) = (f64::from(left), f64::from(right));
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    (denominator > 0.0).then(|| (dot / denominator).clamp(0.0, 1.0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_pairs_do_not_displace_a_valid_match() {
        let left = BoundingBox::new([0.0], [10.0]).unwrap();
        let right = BoundingBox::new([100.0], [110.0]).unwrap();
        let candidates = [
            Candidate {
                track_index: 0,
                bounds: &left,
                feature: None,
            },
            Candidate {
                track_index: 1,
                bounds: &right,
                feature: None,
            },
        ];
        let detections = [
            Detection::new(left.clone()),
            Detection::new(BoundingBox::new([200.0], [210.0]).unwrap()),
        ];
        assert_eq!(
            associate(&candidates, &detections, 0.1, 0.0),
            vec![Match {
                track_index: 0,
                detection_index: 0
            }]
        );
    }

    #[test]
    fn appearance_resolves_identical_boxes() {
        let bounds = BoundingBox::new([0.0], [10.0]).unwrap();
        let candidates = [
            Candidate {
                track_index: 0,
                bounds: &bounds,
                feature: Some(&[1.0, 0.0]),
            },
            Candidate {
                track_index: 1,
                bounds: &bounds,
                feature: Some(&[0.0, 1.0]),
            },
        ];
        let mut detections = [
            Detection::new(bounds.clone()),
            Detection::new(bounds.clone()),
        ];
        detections[0].feature = Some(vec![0.0, 1.0]);
        detections[1].feature = Some(vec![1.0, 0.0]);
        assert_eq!(
            associate(&candidates, &detections, 0.1, 0.5),
            vec![
                Match {
                    track_index: 0,
                    detection_index: 1
                },
                Match {
                    track_index: 1,
                    detection_index: 0
                }
            ]
        );
    }

    #[test]
    fn conflicting_pairs_use_the_global_optimum() {
        let first = BoundingBox::new([0.0], [10.0]).unwrap();
        let second = BoundingBox::new([1.0], [11.0]).unwrap();
        let tracks = [
            Candidate {
                track_index: 2,
                bounds: &first,
                feature: None,
            },
            Candidate {
                track_index: 4,
                bounds: &second,
                feature: None,
            },
        ];
        let detections = [
            Detection::new(second.clone()),
            Detection::new(BoundingBox::new([-2.0], [8.0]).unwrap()),
        ];
        assert_eq!(
            associate(&tracks, &detections, 0.1, 0.0),
            vec![
                Match {
                    track_index: 2,
                    detection_index: 1,
                },
                Match {
                    track_index: 4,
                    detection_index: 0,
                },
            ]
        );
    }

    #[test]
    fn unusable_features_fall_back_to_overlap() {
        assert_eq!(cosine_similarity(&[0.0], &[1.0]), None);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]), Some(1.0));
    }
}
