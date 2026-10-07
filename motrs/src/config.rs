use crate::Error;

/// Derivatives retained for each position or size coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionModel {
    Static,
    ConstantVelocity,
    ConstantAcceleration,
}

impl MotionModel {
    pub(crate) fn state_size(self) -> usize {
        match self {
            Self::Static => 1,
            Self::ConstantVelocity => 2,
            Self::ConstantAcceleration => 3,
        }
    }
}

/// Parameters for the linear motion model and its uncertainty.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of spatial axes: 1 for an interval, 2 for a rectangle, 3 for a box.
    pub dimensions: usize,
    /// Motion assumptions for the box center on each axis.
    pub position: MotionModel,
    /// Motion assumptions for the box extent on each axis.
    pub size: MotionModel,
    /// Driving motion-noise variance. Larger values allow motion to change faster.
    pub position_process_variance: f32,
    /// Driving size-noise variance; zero assumes exact size dynamics.
    pub size_process_variance: f32,
    /// Center observation variance. Larger values trust detections less.
    pub position_measurement_variance: f32,
    /// Extent observation variance. Larger values trust detected sizes less.
    pub size_measurement_variance: f32,
    /// Initial variance of every state component; velocities start at zero.
    pub initial_state_variance: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dimensions: 2,
            position: MotionModel::ConstantVelocity,
            size: MotionModel::Static,
            position_process_variance: 70.0,
            size_process_variance: 10.0,
            position_measurement_variance: 1.0,
            size_measurement_variance: 1.0,
            initial_state_variance: 1000.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Time between frames, in seconds.
    pub dt: f32,
    pub model: ModelConfig,
    /// Minimum geometric overlap required for a match, in [0, 1].
    pub min_iou: f32,
    /// A track survives this many consecutive frames without a detection.
    pub max_missed_frames: u32,
    /// Number of detections required before a track is returned.
    pub min_hits: u32,
    /// Weight of cosine appearance similarity in [0, 1]; zero uses only IoU.
    pub appearance_weight: f32,
    /// Weight retained from the previous score and appearance vector.
    pub smoothing: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            dt: 0.1,
            model: ModelConfig::default(),
            min_iou: 0.1,
            max_missed_frames: 12,
            min_hits: 1,
            appearance_weight: 0.0,
            smoothing: 0.8,
        }
    }
}

impl TrackerConfig {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(Error::InvalidConfig("dt must be finite and positive"));
        }
        if self.model.dimensions == 0 {
            return Err(Error::InvalidConfig("dimensions must be positive"));
        }
        for value in [self.min_iou, self.appearance_weight, self.smoothing] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(Error::InvalidConfig(
                    "overlap, appearance weight, and smoothing must be in [0, 1]",
                ));
            }
        }
        if self.min_hits == 0 {
            return Err(Error::InvalidConfig("min_hits must be positive"));
        }
        for value in [
            self.model.position_process_variance,
            self.model.size_process_variance,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::InvalidConfig(
                    "process variances must be finite and nonnegative",
                ));
            }
        }
        for value in [
            self.model.position_measurement_variance,
            self.model.size_measurement_variance,
            self.model.initial_state_variance,
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::InvalidConfig(
                    "measurement variances and initial covariance must be finite and positive",
                ));
            }
        }
        Ok(())
    }
}
