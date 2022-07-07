use std::env;
use std::path::PathBuf;
use polars::{prelude::*, frame};
use nalgebra as na;
use motrs::tracker::Detection;
use genawaiter::{sync::gen, yield_};

use rand::rngs::StdRng;
use rand::distributions::Uniform;
use rand::Rng;
use rand_distr::{ Normal, Distribution };

use iced::{ canvas::Path, Color, Point, Size, canvas };

const CANVAS_SIZE: i64 = 1000;

pub fn rand_int<R: Rng>(rng: &mut R, min_val: i64, max_val: i64) -> i64 {
    rng.sample(Uniform::new(min_val, max_val))
}

pub fn rand_uniform<R: Rng>(rng: &mut R, min_val: f64, max_val: f64) -> f64 {
    rng.sample::<f64, _>(Uniform::new(min_val, max_val))
}

pub fn random<R: Rng>(rng: &mut R) -> f64 {
    rng.gen()
}

pub fn rand_color<R: Rng>(rng: &mut R) -> [i64; 3] {
    let r = rand_int(rng, 0, 255);
    let g = rand_int(rng, 0, 255);
    let b = rand_int(rng, 0, 255);

    [r, g, b]
}

pub fn rand_guass<R: Rng>(rng: &mut R, mu: f64, sigma2: f64) -> f64 {
    let normal = Normal::new(mu, sigma2.sqrt()).unwrap();
    normal.sample(rng)
}

pub fn read_video_frame(dir: &std::path::Path, frame_idx: u64) -> PathBuf {
    let frame = format!("{0:>06}.jpg", frame_idx);
    let fpath = dir.join(frame);
    fpath
}

fn read_bounds_csv(path: &std::path::Path) -> DataFrame {
    let mut df = LazyCsvReader::new(path.to_string_lossy().to_string())
        .with_ignore_parser_errors(true)
        .finish()
        .collect()
        .unwrap();

    let result = df.set_column_names(&[
        "frame_idx",
        "id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        // "conf",
        "x",
        "y",
        "z"
    ]);

    df
}

fn read_max_frame(df: &DataFrame) -> i64 {
    let max_frame = df
        .clone()
        .lazy()
        .collect()
        .unwrap()
        .max();

    let frame_idx = max_frame.find_idx_by_name("frame_idx").unwrap();
    let max_frame = max_frame.get(frame_idx).unwrap();

    match max_frame[0] {
        AnyValue::Int64(v) => v,
        _ => 0,
    }
}

fn read_bounds(df: &DataFrame, frame_idx: i64, drop_detection_prob: f64, add_detection_noise: f64) -> Vec<Detection> {
    let seed: [u8; 32] = [13; 32];
    let mut rng: StdRng = rand::SeedableRng::from_seed(seed);

    let bb_left = df.find_idx_by_name("bb_left").unwrap();
    let bb_top = df.find_idx_by_name("bb_top").unwrap();
    let bb_width = df.find_idx_by_name("bb_width").unwrap();
    let bb_height = df.find_idx_by_name("bb_height").unwrap();

    let mask = df.column("frame_idx").unwrap().eq(frame_idx);
    let filter_df = df.filter(&mask).unwrap();
    let mut detections = vec![];

    for row_idx in 0..filter_df.height() {
        if random(&mut rng) < drop_detection_prob {
            continue
        }

        if let Some(row) = filter_df.get(row_idx){
            let mut _box = vec![
                to_num(&row[bb_left]),
                to_num(&row[bb_top]),
                to_num(&row[bb_left]) + to_num(&row[bb_width]),
                to_num(&row[bb_top]) + to_num(&row[bb_height]),
            ];

            if add_detection_noise > 0. {
                for i in 0..4 {
                    _box[i] += rand_uniform(&mut rng, -add_detection_noise, add_detection_noise);
                }
            }

            let det = Detection {
                _box: Some(na::DMatrix::from_row_slice(1, 4, _box.as_slice())),
                score: rand_uniform(&mut rng, 0.5, 1.),
                class_id: std::cmp::max(0, rand_int(&mut rng, -1, 1)),
                feature: None,
            };

            detections.push(det);
        }
    }

    detections
}

fn to_num(v: &AnyValue) -> f64 {
    match v {
        AnyValue::Int16(v) => v.clone() as f64,
        AnyValue::Int32(v) => v.clone() as f64,
        AnyValue::Int64(v) => v.clone() as f64,
        _ => 0.
    }
}

pub fn read_detections(path: &std::path::Path, drop_detection_prob: f64, add_detection_noise: f64) -> impl Iterator<Item=(i64, Vec<Detection>)> {
    let path = env::current_dir().unwrap().join(path);
    if !path.is_file() {
        panic!()
    }

    let df = read_bounds_csv(&path);
    gen!({
        let max_frame = read_max_frame(&df);
        dbg!(&max_frame);
        for frame_idx in 0..max_frame {
            let mut detections = read_bounds(&df, frame_idx, drop_detection_prob, add_detection_noise);
            yield_!((frame_idx, detections));
        }
    }).into_iter()
}

pub fn draw_rectangle(frame: &mut canvas::Frame, _box: (usize, usize, usize, usize), color: (u8, u8, u8), fill: bool) {
    let top_left = Point {
        x: _box.0 as f32,
        y: _box.1 as f32,
    };

    let size = Size {
        width: (_box.3 as f32 - _box.1 as f32).abs(),
        height: (_box.3 as f32 - _box.1 as f32).abs()
    };

    let color = Color {
        r: (color.0 / 255u8) as f32,
        g: (color.1 / 255u8) as f32,
        b: (color.2 / 255u8) as f32,
        a: 0.5,
    };

    frame.with_save(|frame| {
        let path = Path::rectangle(top_left, size);

        if fill {
            frame.fill(&path, canvas::Fill {
                color,
                ..Default::default()
            });
        }

        frame.stroke(&path, canvas::Stroke::default().with_color(color));
    })
}

mod test {
    use super::*;

    #[test]
    fn test_read_video_frame() {
        let path = read_video_frame(std::path::Path::new("/tmp"), 1);
        assert_eq!(path.as_os_str(), "/tmp/00000001.jpg");
    }

    #[test]
    fn test_read_bounds_csv() {
        let df = read_bounds_csv(std::path::Path::new("MOT16/train/MOT16-02/gt/gt.txt"));
        println!("{}", df);
    }
}
