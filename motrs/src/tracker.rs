use crate::association::{Candidate, Match, associate};
use crate::kalman::KalmanFilter;
use crate::model::MotionMatrices;
use crate::{BoundingBox, Error, TrackerConfig};
use std::collections::BTreeMap;

/// One observed object. Appearance is an optional embedding or color vector.
#[derive(Debug, Clone)]
pub struct Detection {
    pub bounds: BoundingBox,
    pub score: f32,
    pub class_id: Option<i64>,
    pub feature: Option<Vec<f32>>,
}

impl Detection {
    pub fn new(bounds: BoundingBox) -> Self {
        Self {
            bounds,
            score: 1.0,
            class_id: None,
            feature: None,
        }
    }
}

/// A snapshot of a confirmed track after the current frame.
#[derive(Debug, Clone)]
pub struct Track {
    pub id: String,
    pub bounds: BoundingBox,
    pub score: f32,
    pub class_id: Option<i64>,
    pub age: u64,
    pub hits: u64,
    pub missed_frames: u32,
}

struct TrackState {
    snapshot: Track,
    kalman: KalmanFilter,
    feature: Option<Vec<f32>>,
    class_votes: BTreeMap<i64, u64>,
}

struct PredictedTrack {
    kalman: KalmanFilter,
    bounds: BoundingBox,
    detection_index: Option<usize>,
}

impl TrackState {
    fn new(model: &MotionMatrices, detection: &Detection) -> Self {
        let mut class_votes = BTreeMap::new();
        if let Some(class_id) = detection.class_id {
            class_votes.insert(class_id, 1);
        }
        Self {
            snapshot: Track {
                id: uuid::Uuid::new_v4().to_string(),
                bounds: detection.bounds.clone(),
                score: detection.score,
                class_id: detection.class_id,
                age: 1,
                hits: 1,
                missed_frames: 0,
            },
            kalman: KalmanFilter::new(model, &detection.bounds),
            feature: detection.feature.clone(),
            class_votes,
        }
    }

    fn predict(&self, model: &MotionMatrices) -> Option<PredictedTrack> {
        let mut kalman = self.kalman.predict(model);
        let bounds = model
            .predicted_bounds(
                &mut kalman.state,
                &mut kalman.covariance,
                &self.snapshot.bounds,
            )
            .ok()?;
        Some(PredictedTrack {
            kalman,
            bounds,
            detection_index: None,
        })
    }

    fn record_detection(&mut self, detection: &Detection, smoothing: f32) {
        self.snapshot.hits += 1;
        self.snapshot.missed_frames = 0;
        self.snapshot.score = smoothing * self.snapshot.score + (1.0 - smoothing) * detection.score;
        if let Some(class_id) = detection.class_id {
            *self.class_votes.entry(class_id).or_default() += 1;
            // BTreeMap makes ties deterministic: the larger class ID wins.
            self.snapshot.class_id = self
                .class_votes
                .iter()
                .max_by_key(|(_, votes)| **votes)
                .map(|(&class_id, _)| class_id);
        }
        if let Some(feature) = &detection.feature {
            match &mut self.feature {
                Some(previous) if previous.len() == feature.len() => {
                    for (previous, &observed) in previous.iter_mut().zip(feature) {
                        *previous = smoothing * *previous + (1.0 - smoothing) * observed;
                    }
                }
                _ => self.feature = Some(feature.clone()),
            }
        }
    }
}

/// Predict, associate, and update objects one frame at a time.
pub struct MultiObjectTracker {
    config: TrackerConfig,
    model: MotionMatrices,
    tracks: Vec<TrackState>,
    matched_ids: Vec<String>,
}

impl MultiObjectTracker {
    pub fn new(config: TrackerConfig) -> Result<Self, Error> {
        config.validate()?;
        let model = MotionMatrices::new(&config.model, config.dt);
        for matrix in [
            &model.transition,
            &model.process_noise,
            &model.measurement_noise,
            &model.initial_covariance,
        ] {
            if matrix
                .col_iter()
                .any(|col| col.iter().any(|value| !value.is_finite()))
            {
                return Err(Error::InvalidConfig(
                    "derived matrices overflow; reduce dt or variances",
                ));
            }
        }
        Ok(Self {
            config,
            model,
            tracks: Vec::new(),
            matched_ids: Vec::new(),
        })
    }

    /// Advance one frame. Pass an empty collection when no objects are observed.
    /// On error, the tracker stays at the last successful frame.
    pub fn step(
        &mut self,
        detections: impl IntoIterator<Item = Detection>,
    ) -> Result<Vec<Track>, Error> {
        let detections = detections.into_iter().collect::<Vec<_>>();
        self.validate_detections(&detections)?;
        let mut predictions = self
            .tracks
            .iter()
            .map(|track| track.predict(&self.model))
            .collect::<Vec<_>>();
        for Match {
            track_index,
            detection_index,
        } in self.associate(&predictions, &detections)
        {
            let prediction = predictions[track_index]
                .as_mut()
                .expect("only valid predictions can match");
            prediction
                .kalman
                .update(&self.model, &detections[detection_index].bounds)?;
            prediction.bounds = self
                .model
                .bounds(&prediction.kalman.state)
                .map_err(|_| Error::NumericalFailure)?;
            prediction.detection_index = Some(detection_index);
        }
        self.commit_frame(predictions, &detections);
        Ok(self.active_tracks())
    }

    pub fn config(&self) -> &TrackerConfig {
        &self.config
    }
    pub fn len(&self) -> usize {
        self.tracks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Track IDs aligned with the detections after a successful `step`.
    pub fn matched_ids(&self) -> &[String] {
        &self.matched_ids
    }

    pub fn active_tracks(&self) -> Vec<Track> {
        self.tracks
            .iter()
            .filter(|track| track.snapshot.hits >= u64::from(self.config.min_hits))
            .map(|track| track.snapshot.clone())
            .collect()
    }

    fn validate_detections(&self, detections: &[Detection]) -> Result<(), Error> {
        for (index, detection) in detections.iter().enumerate() {
            let reason = if detection.bounds.dimensions() != self.config.model.dimensions {
                Some("box dimension differs from the model")
            } else if !detection.score.is_finite() || !(0.0..=1.0).contains(&detection.score) {
                Some("score must be finite and in [0, 1]")
            } else if detection.feature.as_ref().is_some_and(|feature| {
                feature.is_empty() || feature.iter().any(|value| !value.is_finite())
            }) {
                Some("appearance must contain finite values and cannot be empty")
            } else {
                None
            };
            if let Some(reason) = reason {
                return Err(Error::InvalidDetection { index, reason });
            }
        }
        Ok(())
    }

    fn associate(
        &self,
        predictions: &[Option<PredictedTrack>],
        detections: &[Detection],
    ) -> Vec<Match> {
        if detections.is_empty() {
            return Vec::new();
        }
        let candidates = self
            .tracks
            .iter()
            .zip(predictions)
            .enumerate()
            .filter_map(|(track_index, (track, prediction))| {
                Some(Candidate {
                    track_index,
                    bounds: &prediction.as_ref()?.bounds,
                    feature: track.feature.as_deref(),
                })
            })
            .collect::<Vec<_>>();
        associate(
            &candidates,
            detections,
            self.config.min_iou,
            self.config.appearance_weight,
        )
    }

    fn commit_frame(&mut self, predictions: Vec<Option<PredictedTrack>>, detections: &[Detection]) {
        let mut predictions = predictions.into_iter();
        let mut matched_ids = vec![String::new(); detections.len()];
        self.tracks.retain_mut(|track| {
            let Some(prediction) = predictions.next().expect("one prediction per track") else {
                return false;
            };
            track.kalman = prediction.kalman;
            track.snapshot.bounds = prediction.bounds;
            track.snapshot.age += 1;
            if let Some(index) = prediction.detection_index {
                track.record_detection(&detections[index], self.config.smoothing);
                matched_ids[index] = track.snapshot.id.clone();
            } else {
                if track.snapshot.missed_frames == self.config.max_missed_frames {
                    return false;
                }
                track.snapshot.missed_frames += 1;
            }
            true
        });
        for (detection, matched_id) in detections.iter().zip(&mut matched_ids) {
            if matched_id.is_empty() {
                let track = TrackState::new(&self.model, detection);
                *matched_id = track.snapshot.id.clone();
                self.tracks.push(track);
            }
        }
        self.matched_ids = matched_ids;
    }
}

impl Default for MultiObjectTracker {
    fn default() -> Self {
        Self::new(TrackerConfig::default()).expect("default configuration is valid")
    }
}
