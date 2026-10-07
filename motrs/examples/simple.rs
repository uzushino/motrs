use motrs::{BoundingBox, Detection, Error, MultiObjectTracker};

fn main() -> Result<(), Error> {
    let mut tracker = MultiObjectTracker::default();
    for frame in 0..10 {
        let offset = frame as f32 * 0.5;
        let bounds = BoundingBox::new([offset, 20.0], [offset + 10.0, 30.0])?;
        for track in tracker.step([Detection::new(bounds)])? {
            println!(
                "frame={frame} id={} min={:?} max={:?}",
                track.id,
                track.bounds.min(),
                track.bounds.max()
            );
        }
    }
    Ok(())
}
