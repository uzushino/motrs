use std::env;
use std::path::{PathBuf, Path};
use polars::prelude::*;

fn read_video_frame(dir: &std::path::Path, frame_idx: u64) -> PathBuf {
    let frame = format!("{0:>08}.jpg", frame_idx);
    let fpath = dir.join(frame);
    fpath
}

fn read_detections(path: &std::path::Path, drop_detetion_prob: f64, add_detection_noise: f64) {
    let path = env::current_dir().unwrap().join(path);
    if ! path.is_file() {
        panic!()
    }

    let mut df = LazyCsvReader::new("../datasets/foods1.csv".into())
        .finish()
        .collect()
        .unwrap();

    let max_frame = df
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

    for frame_idx in 0..max_frame {
        let detections = vec![];
        let mask = df.column("frame_idx").unwrap().eq(frame_idx);
        let filter_df = df.filter(&mask).unwrap();

        for row_idx in 0..filter_df.height() {
            let row = filter_df.get(row_idx);
            if let Some(row) = row {
                let bb_left = df.find_idx_by_name("bb_left");
                let bb_top = df.find_idx_by_name("bb_top");
                let bb_width = df.find_idx_by_name("bb_width");
                let bb_height = df.find_idx_by_name("bb_height");
            }
        }
    }
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
