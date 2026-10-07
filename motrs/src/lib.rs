//! Multi-object tracking with configurable motion models and IoU association.
//!
//! ```
//! use motrs::{BoundingBox, Detection, MultiObjectTracker};
//!
//! let mut tracker = MultiObjectTracker::default();
//! let bounds = BoundingBox::new([10.0, 20.0], [30.0, 40.0])?;
//! let tracks = tracker.step([Detection::new(bounds)])?;
//! assert_eq!(tracks.len(), 1);
//! # Ok::<(), motrs::Error>(())
//! ```

mod association;
mod bounds;
mod config;
mod error;
mod hungarian;
mod kalman;
mod model;
mod tracker;

pub use bounds::BoundingBox;
pub use config::{ModelConfig, MotionModel, TrackerConfig};
pub use error::Error;
pub use tracker::{Detection, MultiObjectTracker, Track};
