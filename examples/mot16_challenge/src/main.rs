use std::borrow::Borrow;
use std::env;
use std::sync::{Arc, Mutex};
use std::path::{Path, PathBuf};

use motrs::tracker::*;
use motrs::model::*;

use iced::{
    Application, Command, executor,
    Container, Element, Length, Settings,
    Rectangle, Subscription, canvas,
    Image, Alignment, Column
};

use iced_native::subscription;
use iced_native::image::Handle;
use polars::frame;
use std::hash::Hash;
use image::{Luma, GenericImageView};
use image::{Rgba, RgbImage};

use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

mod util;

use crate::util::{ read_detections, read_video_frame, draw_rectangle };

pub fn main() -> iced::Result {
    Mot16Challenge::run(Settings {
        antialiasing: true,
        ..Settings::default()
    })
}

#[derive(Default)]
struct Mot16Challenge {
    pub viewer: canvas::Cache,
    pub active_tracks: Vec<Track>,
    pub detections: Vec<Detection>,
    pub frame_path: PathBuf,
}

#[derive(Debug, Clone)]
pub enum Message {
    Void,
    Tracking(Vec<Track>, Vec<Detection>, PathBuf),
}

impl Application for Mot16Challenge {
    type Message = Message;
    type Executor = executor::Default;

    type Flags = ();

    fn new(_: Self::Flags) -> (Self, Command<Message>) {
        (
            Self {
                viewer: Default::default(),
                active_tracks: Vec::default(),
                detections: Vec::default(),
                frame_path: PathBuf::default()
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("mot16_challenge")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::Tracking(active_tracks, detections, frame) => {
                self.active_tracks = active_tracks;
                self.detections = detections;
                self.frame_path = frame;
            },
            _ => {}
        };
        Command::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        worker(0).map(|v| {
            match v.1 {
                Progress::Advanced(c, active_tracks, detections, frame) => {
                    Message::Tracking(active_tracks, detections, frame)
                },
                _ => Message::Void
            }
        })
    }

    fn view(&mut self) -> Element<Message> {
        self.viewer.clear();

        let frame_path = self.frame_path.clone();
        if frame_path.is_file() {
            let mut img = image::open(frame_path.clone()).unwrap();
            let white = Rgba([255u8, 255u8, 255u8, 255u8]);

            self.active_tracks.iter().for_each(|track| {
                let x1 = track._box[0] as i32;
                let y1 = track._box[1] as i32;
                let x2 = track._box[2] as i32;
                let y2 = track._box[3] as i32;

                draw_hollow_rect_mut(
                    &mut img,
                    Rect::at(x1, y1).of_size((x2 - x1) as u32, (y2 - y1) as u32),
                    white
                );
            });


            let w = img.width();
            let h = img.height();
            let mut bytes = vec![0u8; (w*h*4) as usize];

            for y in 0..img.height() {
                for x in 0..img.width() {
                    let pixel = img.get_pixel(x, y);
                    let idx: usize = y as usize * w as usize * 4 + (x as usize * 4);

                    bytes[idx + 0] = pixel[2];
                    bytes[idx + 1] = pixel[1];
                    bytes[idx + 2] = pixel[0];
                    bytes[idx + 3] = pixel[3];
                }
            }

            let image = Container::new(
                Image::new(Handle::from_pixels(img.width(), img.height(), bytes))
                            .width(Length::Fill)
                            .height(Length::Fill)
                )
                .height(Length::Fill)
                .width(Length::Fill);

            let content  = Column::new()
                .width(Length::Fill)
                .height(Length::Fill)
                // .push(canvas)
                .push(image);

            Container::new(content)
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(20)
                .into()
        } else {
            let content = Column::new()
                .width(Length::Fill)
                .height(Length::Fill);
            Container::new(content)
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(20)
                .into()
        }
    }
}

impl<Message> canvas::Program<Message> for Mot16Challenge {
    fn draw(&self, bounds: Rectangle, _cursor: canvas::Cursor) -> Vec<canvas::Geometry> {
        let viewer = self.viewer.draw(bounds.size(), |frame| {
            self.active_tracks.iter().for_each(|track| {
                let rect = (track._box[0] as usize, track._box[1] as usize, track._box[2] as usize, track._box[3] as usize);
                draw_rectangle(frame, rect, (0, 255, 0), false);
            });
       });

        vec![viewer]
    }
}
pub struct MyTracker {}

impl MyTracker {
    pub fn new() -> Self {
        Self {}
    }

    pub fn create() -> MultiObjectTracker {
        let model_spec = ModelPreset::constant_acceleration_and_static_box_size_2d();
        let min_iou = 0.25;
        let multi_match_min_iou = 1. + 1e-7;
        let feature_similarity_fn = None;
        let feature_similarity_beta = None;
        let matching_fn = IOUAndFeatureMatchingFunction::new(
            min_iou,
            multi_match_min_iou,
            feature_similarity_fn,
            feature_similarity_beta,
        );

        let tracker = MultiObjectTracker::new(
            1. / 30.,
            model_spec,
            Some(matching_fn),
            Some(SingleObjectTrackerKwargs {
                max_staleness: 15.,
                ..Default::default()
            }),
            None,
            None
        );

        tracker
    }
}

fn worker<I: 'static + Hash + Copy + Send + Sync>(id: I) -> iced::Subscription<(I, Progress)> {
    let fps = 30.0;
    let split = "train";
    let seq_id = "04";
    let sel = "gt";
    let drop_detection_prob = 0.1;
    let add_detection_noise = 5.0;

    let dataset_root = "examples/mot16_challenge/MOT16";
    let dataset_root = env::current_dir().unwrap().join(dataset_root);
    let dataset_root2 = format!("{}/{}/MOT16-{}", dataset_root.as_path().display().to_string(), split, seq_id);

    let frames_dir = format!("{}/img1", dataset_root2);
    let _dets_path = format!("{}/{}/{}.txt", dataset_root2, sel, sel);

    let init_state = MyState::Ready(MyTracker::create(), 1000);
    let dets_path = std::path::Path::new(_dets_path.as_str());
    let dets_gen = read_detections(
            dets_path,
            drop_detection_prob,
            add_detection_noise
    );

    let gen = Arc::new(Mutex::new(dets_gen));

    subscription::unfold(id, init_state, move |state| {
        let gen = gen.clone();
        tracking(id, gen, state, frames_dir.clone())
    })
}

async fn tracking<T, I: Copy>(id: I, gen: Arc<Mutex<T>>, state: MyState, frame_dir: String) -> (Option<(I, Progress)>, MyState) where T: Iterator<Item=(i64, Vec<Detection>)> {
    match state {
        MyState::Ready(tracker, num_steps) => {
            (Some((id, Progress::Started)), MyState::Tracking { total: num_steps, count: 0, tracker: tracker })
        }
        MyState::Tracking { total, count, mut tracker,} => {
            if count <= total {
                if let Some((frame_idx, det_gt)) = gen.lock().unwrap().next() {
                    let target = det_gt
                        .to_vec()
                        .into_iter()
                        .filter(|v| v._box.is_some())
                        .collect::<Vec<_>>();

                    let frame_dir_path = Path::new(&frame_dir);
                    let frame = read_video_frame(frame_dir_path, frame_idx as u64);
                    let active_tracks = tracker.step(target.clone());

                    (Some((id, Progress::Advanced(count, active_tracks, target, frame))), MyState::Tracking{ total, count: count + 1, tracker })
                } else {
                    (Some((id, Progress::Finished)), MyState::Finished)
                }
            } else {
                (Some((id, Progress::Finished)), MyState::Finished)
            }
        },
        MyState::Finished => {
            (None, MyState::Finished)
        }
    }
}

#[derive(Debug, Clone)]
pub enum Progress {
    Started,
    Advanced(usize, Vec<Track>, Vec<Detection>, PathBuf),
    Finished,
    Errored,
}

pub enum MyState {
    Ready(MultiObjectTracker, usize),
    Tracking {
        total: usize,
        count: usize,
        tracker: MultiObjectTracker,
    },
    Finished,
}
