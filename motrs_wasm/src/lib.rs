use gloo_utils::format::JsValueSerdeExt;
use motrs::{BoundingBox, Detection, ModelConfig, MotionModel, MultiObjectTracker, TrackerConfig};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// JavaScript facade for two-dimensional tracking.
#[wasm_bindgen]
pub struct MOT {
    tracker: MultiObjectTracker,
}

#[derive(Deserialize)]
struct InputBox {
    #[serde(rename = "_box")]
    bounds: [f32; 4],
    #[serde(default = "default_score")]
    score: f32,
    #[serde(default)]
    class_id: Option<i64>,
    #[serde(default)]
    feature: Option<Vec<f32>>,
}

fn default_score() -> f32 {
    1.0
}

impl TryFrom<InputBox> for Detection {
    type Error = motrs::Error;

    fn try_from(input: InputBox) -> Result<Self, Self::Error> {
        let [left, top, right, bottom] = input.bounds;
        Ok(Self {
            bounds: BoundingBox::new([left, top], [right, bottom])?,
            score: input.score,
            class_id: input.class_id,
            feature: input.feature,
        })
    }
}

#[derive(Serialize)]
struct OutputBox {
    id: String,
    #[serde(rename = "_box")]
    bounds: [f32; 4],
    score: f32,
    class_id: Option<i64>,
    missed_frames: u32,
}

impl From<motrs::Track> for OutputBox {
    fn from(track: motrs::Track) -> Self {
        Self {
            id: track.id,
            bounds: [
                track.bounds.min()[0],
                track.bounds.min()[1],
                track.bounds.max()[0],
                track.bounds.max()[1],
            ],
            score: track.score,
            class_id: track.class_id,
            missed_frames: track.missed_frames,
        }
    }
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsError::new(&error.to_string()).into()
}

#[wasm_bindgen]
impl MOT {
    pub fn new() -> Self {
        Self::with_max_missed_frames(15)
    }

    /// Create a tracker with a configurable lifetime for unobserved objects.
    pub fn with_max_missed_frames(max_missed_frames: u32) -> Self {
        let tracker = MultiObjectTracker::new(TrackerConfig {
            model: ModelConfig {
                position: MotionModel::ConstantAcceleration,
                ..Default::default()
            },
            max_missed_frames,
            ..Default::default()
        })
        .expect("WASM configuration is valid");
        Self { tracker }
    }

    /// Accept every detection in the array. An empty array advances a missed frame.
    /// Malformed input throws a JavaScript error before advancing the tracker.
    pub fn step(&mut self, value: &JsValue) -> Result<(), JsValue> {
        let inputs: Vec<InputBox> = value.into_serde().map_err(js_error)?;
        let detections = inputs
            .into_iter()
            .map(Detection::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(js_error)?;
        self.tracker.step(detections).map_err(js_error)?;
        Ok(())
    }

    pub fn active_tracks(&self) -> Result<JsValue, JsValue> {
        let tracks = self
            .tracker
            .active_tracks()
            .into_iter()
            .map(OutputBox::from)
            .collect::<Vec<_>>();
        JsValue::from_serde(&tracks).map_err(js_error)
    }
}

impl Default for MOT {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_and_output_preserve_corner_order_and_metadata() {
        let detection = Detection::try_from(InputBox {
            bounds: [1.0, 2.0, 11.0, 22.0],
            score: 0.8,
            class_id: Some(3),
            feature: None,
        })
        .unwrap();
        let mut tracker = MultiObjectTracker::default();
        let output = OutputBox::from(tracker.step([detection]).unwrap().remove(0));
        assert_eq!(output.bounds, [1.0, 2.0, 11.0, 22.0]);
        assert_eq!(output.score, 0.8);
        assert_eq!(output.class_id, Some(3));
    }

    #[test]
    fn invalid_bounds_are_rejected_instead_of_panicking() {
        assert!(
            Detection::try_from(InputBox {
                bounds: [10.0, 2.0, 1.0, 22.0],
                score: 1.0,
                class_id: None,
                feature: None,
            })
            .is_err()
        );
    }
}
