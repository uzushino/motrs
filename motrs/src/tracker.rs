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

    fn predict(&mut self, model: &MotionMatrices) -> bool {
        self.kalman.predict(model);
        self.snapshot.age += 1;
        match model.bounds(&self.kalman.state) {
            Ok(bounds) => {
                self.snapshot.bounds = bounds;
                true
            }
            Err(_) => false,
        }
    }

    fn update(
        &mut self,
        model: &MotionMatrices,
        detection: &Detection,
        smoothing: f32,
    ) -> Result<(), Error> {
        self.kalman.update(model, &detection.bounds)?;
        self.snapshot.bounds = model
            .bounds(&self.kalman.state)
            .map_err(|_| Error::NumericalFailure)?;
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
        Ok(())
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
    /// All detections are validated before any track is changed.
    pub fn step(
        &mut self,
        detections: impl IntoIterator<Item = Detection>,
    ) -> Result<Vec<Track>, Error> {
        let detections = detections.into_iter().collect::<Vec<_>>();
        self.validate_detections(&detections)?;
        self.predict_tracks();
        let matches = self.associate(&detections);
        let (matched_tracks, matched_detections) = self.update_matches(&detections, &matches)?;
        self.retire_unmatched(&matched_tracks);
        self.create_unmatched(&detections, &matched_detections);
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

    fn predict_tracks(&mut self) {
        self.tracks.retain_mut(|track| track.predict(&self.model));
    }

    fn associate(&self, detections: &[Detection]) -> Vec<Match> {
        let candidates = self
            .tracks
            .iter()
            .map(|track| Candidate {
                bounds: &track.snapshot.bounds,
                feature: track.feature.as_deref(),
            })
            .collect::<Vec<_>>();
        associate(
            &candidates,
            detections,
            self.config.min_iou,
            self.config.appearance_weight,
        )
    }

    fn update_matches(
        &mut self,
        detections: &[Detection],
        matches: &[Match],
    ) -> Result<(Vec<bool>, Vec<bool>), Error> {
        let mut matched_tracks = vec![false; self.tracks.len()];
        let mut matched_detections = vec![false; detections.len()];
        self.matched_ids = vec![String::new(); detections.len()];
        for &Match {
            track_index,
            detection_index,
        } in matches
        {
            let track = &mut self.tracks[track_index];
            track.update(
                &self.model,
                &detections[detection_index],
                self.config.smoothing,
            )?;
            self.matched_ids[detection_index] = track.snapshot.id.clone();
            matched_tracks[track_index] = true;
            matched_detections[detection_index] = true;
        }
        Ok((matched_tracks, matched_detections))
    }

    fn retire_unmatched(&mut self, matched_tracks: &[bool]) {
        for (track, &matched) in self.tracks.iter_mut().zip(matched_tracks) {
            if !matched {
                track.snapshot.missed_frames += 1;
            }
        }
        self.tracks
            .retain(|track| track.snapshot.missed_frames <= self.config.max_missed_frames);
    }

    fn create_unmatched(&mut self, detections: &[Detection], matched_detections: &[bool]) {
        for (index, detection) in detections.iter().enumerate() {
            if !matched_detections[index] {
                let track = TrackState::new(&self.model, detection);
                self.matched_ids[index] = track.snapshot.id.clone();
                self.tracks.push(track);
            }
        }
    }
}

impl Default for MultiObjectTracker {
    fn default() -> Self {
        Self::new(TrackerConfig::default()).expect("default configuration is valid")
    }
}
