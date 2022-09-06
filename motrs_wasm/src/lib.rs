use motrs::model::*;
use motrs::tracker::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct MOT {
    tracker: MultiObjectTracker,
}
