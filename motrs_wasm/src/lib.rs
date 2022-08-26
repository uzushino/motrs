use wasm_bindgen::prelude::*;
use motrs::model::*;
use motrs::tracker::*;

#[wasm_bindgen]
pub struct MOT {
    tracker: MultiObjectTracker
}
