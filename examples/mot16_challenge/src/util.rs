use std::env;
use std::path::PathBuf;
use polars::prelude::*;
use nalgebra as na;
use motrs::tracker::Detection;
use genawaiter::rc::{ Gen, Co };

use crate::testing::{rand_uniform, rand_int, random};


fn read_video_frame(dir: &std::path::Path, frame_idx: u64) -> PathBuf {
    let frame = format!("{0:>08}.jpg", frame_idx);
    let fpath = dir.join(frame);
    fpath
}

async fn detections(co: Co<(i32, Vec<Detection>)>, df: &DataFrame, add_detection_noise: f64, drop_detection_prob: f64) {
    let _df = df.clone();

    let max_frame = _df
        .lazy()
        .groupby(vec![col("frame_idx")])
        .agg(vec![col("frame_idx").max()])
        .collect()
        .unwrap();

    let max_frame = max_frame[0]
        .i32()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap_or(0))
        .collect::<Vec<_>>();
    let max_frame = max_frame[0];

    let bb_left = df.find_idx_by_name("bb_left").unwrap();
    let bb_top = df.find_idx_by_name("bb_top").unwrap();
    let bb_width = df.find_idx_by_name("bb_width").unwrap();
    let bb_height = df.find_idx_by_name("bb_height").unwrap();

    fn to_num(v: &AnyValue) -> f64 {
        match v {
            AnyValue::Int16(v) => v.clone() as f64,
            AnyValue::Int32(v) => v.clone() as f64,
            AnyValue::Int64(v) => v.clone() as f64,
        }
    }

    let mut rng = rand::thread_rng();

    for frame_idx in 0..max_frame {
        let mut detections = vec![];
        let mask = df.column("frame_idx").unwrap().eq(frame_idx);
        let filter_df = df.filter(&mask).unwrap();

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

        co.yield_((frame_idx, detections)).await;
    }
}

fn read_detections(path: &std::path::Path, drop_detection_prob: f64, add_detection_noise: f64) -> Gen<(i32, Vec<Detection>)> {
    let path = env::current_dir().unwrap().join(path);
    if ! path.is_file() {
        panic!()
    }

    let df = LazyCsvReader::new("../datasets/foods1.csv".into())
        .finish()
        .collect()
        .unwrap();

    Gen::new(|co| async {
        let df = df.clone();

        detections(co, &df, add_detection_noise, drop_detection_prob).await;
    })
}

/*
def read_detections(path, drop_detection_prob: float = 0.0, add_detection_noise: float = 0.0):
    """ parses and converts MOT16 benchmark annotations to known [xmin, ymin, xmax, ymax] format """
    path = os.path.expanduser(path)
    logger.debug(f'reading detections from {path}')
    if not os.path.isfile(path):
        raise ValueError('file does not exist')

    df = pd.read_csv(path, names=COL_NAMES)

    max_frame = df.frame_idx.max()
    for frame_idx in range(max_frame):
        detections = []
        for _, row in df[df.frame_idx == frame_idx].iterrows():
            if random.random() < drop_detection_prob:
                continue

            box = [row.bb_left, row.bb_top,
                   row.bb_left + row.bb_width,
                   row.bb_top + row.bb_height]

            if add_detection_noise > 0:
                for i in range(4):
                    box[i] += random.uniform(-add_detection_noise, add_detection_noise)

            detections.append(Detection(box=box))

        yield frame_idx, detections
 */
