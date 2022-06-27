use motrs::tracker::*;
use motrs::model::*;
use genawaiter::rc::Gen;

use iced::{
    futures, Clipboard, Application, Command, executor,
    Container, Element, Length, Settings,
    canvas, Rectangle, Subscription
};
use std::env;

mod testing;
mod util;

use crate::util::read_detections;

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

    fn update(&mut self, message: Message, _: &mut Clipboard) -> Command<Message> {
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
        iced::Subscription::from_recipe(MyTracker::new()).map(|v| {
            match v {
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

use iced_futures::subscription::Recipe;

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
            1 / 30.,
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



impl<H, E> Recipe<H, E> for MyTracker where H: std::hash::Hasher {
    type Output = Progress;

    fn hash(&self, state: &mut H) {
        use std::hash::Hash;
        std::any::TypeId::of::<Self>().hash(state);
    }

    fn stream(self: Box<Self>, _input: futures::stream::BoxStream<'static, E>) -> futures::stream::BoxStream<'static, Self::Output> {
        let num_steps = self.num_steps;

        let fps = 30.0;
        let split = "train";
        let seq_id = "04";
        let sel = "gt";
        let drop_detection_prob = 0.1;
        let add_detection_noise = 5.0;

        let dataset_root = "./";
        let dataset_root = env::current_dir().unwrap().join(dataset_root);
        let dataset_root2 = format!("{}/{}/MOT16-{}", dataset_root.as_os_str().as_ref(), split, seq_id);

        let frames_dir = format!("{}/img1", dataset_root2);
        let dets_path = format!("{}/{}/{}.txt", dataset_root2, sel, sel);
        let dets_path = std::path::Path::new(dets_path.as_str());
        let dets_gen = read_detections(dets_path, drop_detection_prob, add_detection_noise);

        Box::pin(futures::stream::unfold(MyState::Ready(Self::create(), dets_gen, num_steps), |state| async move {
                match state {
                    MyState::Ready(tracker, gen, num_steps) => {
                        Some((Progress::Started, MyState::Tracking { total: num_steps, count: 0, tracker: tracker, gen: gen }))
                    }
                    MyState::Tracking { total, count, mut tracker, mut gen} => {
                        if count <= total {
                            if let genawaiter::GeneratorState::Yielded((_det_pred, det_gt)) = gen.resume() {
                                let target = det_gt
                                    .to_vec()
                                    .into_iter()
                                    .filter(|v| v._box.is_some())
                                    .collect::<Vec<_>>();
                                let active_tracks = tracker.step(target.clone());

                                Some((Progress::Advanced(count, active_tracks, target), MyState::Tracking{ total, count: count + 1, tracker, gen }))
                            } else {
                                Some((Progress::Finished, MyState::Finished))
                            }
                        } else {
                            Some((Progress::Finished, MyState::Finished))
                        }
                    },
                    MyState::Finished => {
                        let _: () = iced::futures::future::pending().await;
                        None
                    }
                }
            },
        ))
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
    Ready(MultiObjectTracker, Gen<(i32, Vec<Detection>)>, usize),
    Tracking {
        total: usize,
        count: usize,
        tracker: MultiObjectTracker,
        gen: Gen<(i32, Vec<Detection>)>,
    },
    Finished,
}
