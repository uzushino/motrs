use motrs::tracker::*;
use motrs::model::*;

use nalgebra as na;

use iced::{
    button, futures, image, Clipboard, Application, Button, Column, Command, executor,
    Container, Element, Length, Row, Settings, Text, canvas::Path, Point, Size, Sandbox,
    canvas, Rectangle, Color, time, Subscription
};
use std::time::{Duration, Instant};

mod testing;
mod tracker;

use crate::testing::data_generator;

fn draw_rectangle(frame: &mut canvas::Frame, _box: (usize, usize, usize, usize), color: (u8, u8, u8)) {
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
        a: 1.,
    };

    frame.with_save(|frame| {
        let path = Path::rectangle(top_left, size);
        frame.stroke(&path, canvas::Stroke::default().with_color(color));
    })
}

pub fn main() -> iced::Result {
    ImageViewer::run(Settings {
        antialiasing: true,
        ..Settings::default()
    })
}

#[derive(Default)]
struct ImageViewer {
    pub speed: usize,
    pub num_steps: i64,
    pub tracker: Option<MultiObjectTracker>,
    pub viewer: canvas::Cache,
}

#[derive(Debug, Clone)]
pub enum Message {
    Void,
    Tick(Instant),
}

impl Application for ImageViewer {
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
                speed: 5,
                num_steps: 0,
                tracker: Some(tracker),
                viewer: Default::default(),
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("2d_multi_object_tracking")
    }

    fn update(&mut self, _message: Message, _: &mut Clipboard) -> Command<Message> {
        Command::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::Subscription::from_recipe(MyTracker::new()).map(|v| Message::Void)
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
            draw_rectangle(frame, (10, 10, 100, 100), (255, 0, 0));
            draw_rectangle(frame, (300, 300, 500, 500), (0, 255, 0));
        });

        vec![viewer]
    }
}

use iced_futures::subscription::Recipe;
use motrs::tracker::*;
use motrs::model::*;


pub struct MyTracker {
    uuid: String,
    dets: Vec<(Vec<Detection>, Vec<Detection>)>,
}

impl MyTracker {
    pub fn new() -> Self {
        let mut dets: Vec<(Vec<Detection>, Vec<Detection>)> = data_generator(
            200,
            20,
            0.03,
            0.33,
            0.0,
            3.33
        )
        .into_iter()
        .take(200)
        .collect::<Vec<_>>();

        Self {
            dets,
            uuid: String::default(),
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
        let dets = self.dets.clone();

        Box::pin(futures::stream::unfold(MyState::Ready(Self::create(), dets), |state| async move {
                match state {
                    MyState::Ready(tracker, dets) => {
                        Some((Progress::Started, MyState::Tracking { total: 1, count: 0, tracker: tracker, dets: dets }))
                    }
                    MyState::Tracking { total, count, mut tracker, dets} => {
                        if count <= total {
                            let detections =
                                dets
                                .get(0)
                                .unwrap()
                                .0
                                .clone()
                                .into_iter()
                                .filter(|d| d._box.is_some())
                                .collect::<Vec<_>>();

                            tracker.step(detections);

                            Some((Progress::Advanced(count), MyState::Tracking{ total, count: count + 1, tracker, dets }))
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
    Advanced(u64),
    Finished,
    Errored,
}


pub enum MyState {
    Ready(MultiObjectTracker, Vec<(Vec<Detection>, Vec<Detection>)>),
    Tracking {
        total: u64,
        count: u64,
        tracker: MultiObjectTracker,
        dets: Vec<(Vec<Detection>, Vec<Detection>)>
    },
    Finished,
}
