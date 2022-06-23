use motrs::tracker::*;
use motrs::model::*;
use genawaiter::sync::{Gen, GenBoxed};

use iced::{
    futures, Clipboard, Application, Command, executor,
    Container, Element, Length, Settings, canvas::Path, Point, Size,
    canvas, Rectangle, Color, Subscription
};

mod testing;
mod util;

use crate::testing::data_generator;

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
        let model_spec = ModelPreset::constant_acceleration_and_static_box_size_2d();
        let min_iou = 1. / 24.;
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
            0.1,
            model_spec,
            Some(matching_fn),
            Some(SingleObjectTrackerKwargs {
                max_staleness: 12.,
                ..Default::default()
            }),
            None,
            Some(ActiveTracksKwargs {
                min_steps_alive: 2,
                max_staleness: 6.,
                ..Default::default()
            })
        );

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

impl<Message> canvas::Program<Message> for ImageViewer {
    fn draw(&self, bounds: Rectangle, _cursor: canvas::Cursor) -> Vec<canvas::Geometry> {
        let viewer = self.viewer.draw(bounds.size(), |frame| {
            self.active_tracks.iter().for_each(|track| {
                let rect = (track._box[0] as usize, track._box[1] as usize, track._box[2] as usize, track._box[3] as usize);
                draw_rectangle(frame, rect, (0, 255, 0), false);
            });

            self.detections.iter().for_each(|det|  {
                let b = det._box.clone().unwrap();
                let f = det.feature.clone().unwrap();
                let rect = (b[0] as usize, b[1] as usize, b[2] as usize, b[3] as usize);
                draw_rectangle(frame, rect, ((f[0] * 256. % 256.) as u8, (f[1] * 256. % 256.) as u8, (f[2] * 256. % 256.) as u8), true);
            });
        });

        vec![viewer]
    }
}

use iced_futures::subscription::Recipe;

pub struct MyTracker {
    num_steps: usize,
}

impl MyTracker {
    pub fn new() -> Self {
        Self {
            num_steps: 1000,
        }
    }

    pub fn create() -> MultiObjectTracker {
        let model_spec = ModelPreset::constant_acceleration_and_static_box_size_2d();
        let min_iou = 1. / 24.;
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
            0.1,
            model_spec,
            Some(matching_fn),
            Some(SingleObjectTrackerKwargs {
                max_staleness: 12.,
                ..Default::default()
            }),
            None,
            Some(ActiveTracksKwargs {
                min_steps_alive: 2,
                max_staleness: 6.,
                ..Default::default()
            })
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

        let gen = data_generator(
            num_steps as i64,
            20,
            0.03,
            0.33,
            0.0,
            3.33
        );

        Box::pin(futures::stream::unfold(MyState::Ready(Self::create(), gen, num_steps), |state| async move {
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
    Ready(MultiObjectTracker, GenBoxed<(Vec<Detection>, Vec<Detection>)>, usize),
    Tracking {
        total: usize,
        count: usize,
        tracker: MultiObjectTracker,
        gen: GenBoxed<(Vec<Detection>, Vec<Detection>)>,
    },
    Finished,
}
