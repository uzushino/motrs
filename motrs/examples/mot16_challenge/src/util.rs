use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use motrs::{BoundingBox, Detection};

type ReadResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone)]
pub struct FrameDetections {
    pub number: u64,
    pub detections: Vec<Detection>,
}

pub fn video_frame_path(directory: &Path, number: u64) -> PathBuf {
    directory.join(format!("{number:06}.jpg"))
}

pub fn read_detections(path: &Path) -> ReadResult<Vec<FrameDetections>> {
    parse_detections(std::fs::File::open(path)?)
}

fn parse_detections(reader: impl Read) -> ReadResult<Vec<FrameDetections>> {
    let mut csv = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(reader);
    let mut frames = BTreeMap::<u64, Vec<Detection>>::new();
    for record in csv.records() {
        let record = record?;
        if record.len() < 6 {
            return Err("MOT16 rows must contain at least six columns".into());
        }
        let number: u64 = record[0].parse()?;
        if number == 0 {
            return Err("MOT16 frame numbers must start at one".into());
        }
        let left: f32 = record[2].parse()?;
        let top: f32 = record[3].parse()?;
        let width: f32 = record[4].parse()?;
        let height: f32 = record[5].parse()?;
        let bounds = BoundingBox::new([left, top], [left + width, top + height])?;
        // This visualization uses annotated boxes as observations, not detector
        // confidence or ground-truth IDs. It does not measure tracking accuracy.
        frames
            .entry(number)
            .or_default()
            .push(Detection::new(bounds));
    }
    let last_frame = frames.last_key_value().map_or(0, |(&number, _)| number);
    Ok((1..=last_frame)
        .map(|number| FrameDetections {
            number,
            detections: frames.remove(&number).unwrap_or_default(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_frame_uses_six_digits() {
        assert_eq!(
            video_frame_path(Path::new("/tmp"), 1),
            Path::new("/tmp/000001.jpg")
        );
    }

    #[test]
    fn reads_fractional_bounds_and_preserves_empty_frames() {
        let csv = b"3,2,10.5,20.25,30.5,40.75,1,-1,-1,-1\n1,1,1,2,3,4,1,1,1\n";
        let frames = parse_detections(&csv[..]).unwrap();
        assert_eq!(
            frames.iter().map(|frame| frame.number).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(frames[1].detections.is_empty());
        assert_eq!(frames[2].detections[0].bounds.min(), &[10.5, 20.25]);
        assert_eq!(frames[2].detections[0].bounds.max(), &[41.0, 61.0]);
    }

    #[test]
    fn rejects_malformed_or_invalid_rows() {
        for csv in [
            "1,2,3\n",
            "1,2,no,4,5,6\n",
            "0,2,3,4,5,6\n",
            "1,2,3,4,-5,6\n",
            "1,2,NaN,4,5,6\n",
        ] {
            assert!(parse_detections(csv.as_bytes()).is_err());
        }
    }
}
