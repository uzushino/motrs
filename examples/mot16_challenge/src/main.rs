use std::borrow::Borrow;
use std::env;
use std::sync::{Arc, Mutex};

use motrs::tracker::*;
use motrs::model::*;

use iced::{
    Application, Command, executor,
    Container, Element, Length, Settings,
    Rectangle, Subscription, canvas
};
use iced_native::subscription;
use std::hash::Hash;

mod util;

use crate::util::{ read_detections };

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
    pub detections: Vec<Detection>
}

#[derive(Debug, Clone)]
pub enum Message {
    Void,
    Tracking(Vec<Track>, Vec<Detection>),
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
                detections: Vec::default()
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("mot16_challenge")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::Tracking(active_tracks, detections) => {
                self.active_tracks = active_tracks;
                self.detections = detections;
            },
            _ => {}
        };
        Command::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        worker(0).map(|v| {
            match v.1 {
                Progress::Advanced(c, active_tracks, detections) => {
                    Message::Tracking(active_tracks, detections)
                },
                _ => Message::Void
            }
        })
    }

    fn view(&mut self) -> Element<Message> {
        self.viewer.clear();

        let canvas = canvas::Canvas::new(self)
            .width(Length::Fill)
            .height(Length::Fill);

        Container::new(canvas)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(20)
            .into()
    }
}

impl<Message> canvas::Program<Message> for Mot16Challenge {
    fn draw(&self, bounds: Rectangle, _cursor: canvas::Cursor) -> Vec<canvas::Geometry> {
        let viewer = self.viewer.draw(bounds.size(), |frame| {

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

    let init_state = MyState::Ready(MyTracker::create(), 0);
    let dets_path = std::path::Path::new(_dets_path.as_str());
    let dets_gen = read_detections(
            dets_path,
            drop_detection_prob,
            add_detection_noise
    );

    let gen = Arc::new(Mutex::new(dets_gen));

    subscription::unfold(id, init_state, move |state| {
        let gen = gen.clone();
        tracking(id, gen, state)
    })
}

async fn tracking<T, I: Copy>(id: I, gen: Arc<Mutex<T>>, state: MyState) -> (Option<(I, Progress)>, MyState) where T: Iterator<Item=(i64, Vec<Detection>)> {
    match state {
        MyState::Ready(tracker, num_steps) => {
            (Some((id, Progress::Started)), MyState::Tracking { total: num_steps, count: 0, tracker: tracker })
        }
        MyState::Tracking { total, count, mut tracker} => {
            if count <= total {
                if let Some((_det_pred, det_gt)) = gen.lock().unwrap().next() {
                    let target = det_gt
                        .to_vec()
                        .into_iter()
                        .filter(|v| v._box.is_some())
                        .collect::<Vec<_>>();

                    let active_tracks = tracker.step(target.clone());

                    (Some((id, Progress::Advanced(count, active_tracks, target))), MyState::Tracking{ total, count: count + 1, tracker })
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
    Advanced(usize, Vec<Track>, Vec<Detection>),
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
