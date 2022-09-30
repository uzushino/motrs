use motrs::model::*;
use motrs::tracker::*;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use iced::{
    canvas, canvas::Path, executor, Application, Color, Command, Container, Element, Length, Point,
    Rectangle, Settings, Size, Subscription,
};
use iced_native::subscription;

mod testing;

use crate::testing::data_generator;

pub fn main() -> iced::Result {
    MultiObject2dTracking::run(Settings {
        antialiasing: true,
        ..Settings::default()
    })
}

fn draw_rectangle(
    frame: &mut canvas::Frame,
    _box: (usize, usize, usize, usize),
    color: (u8, u8, u8),
    fill: bool,
) {
    let top_left = Point {
        x: _box.0 as f32,
        y: _box.1 as f32,
    };

    let size = Size {
        width: (_box.3 as f32 - _box.1 as f32).abs(),
        height: (_box.3 as f32 - _box.1 as f32).abs(),
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
            frame.fill(
                &path,
                canvas::Fill {
                    color,
                    ..Default::default()
                },
            );
        }

        frame.stroke(&path, canvas::Stroke::default().with_color(color));
    })
}

#[derive(Default)]
struct MultiObject2dTracking {
    pub num_steps: usize,
    pub viewer: canvas::Cache,
    pub active_tracks: Vec<Track>,
    pub detections: Vec<Detection>,
}

#[derive(Debug, Clone)]
pub enum Message {
    Void,
    Tracking(Vec<Track>, Vec<Detection>),
}

impl Application for MultiObject2dTracking {
    type Message = Message;
    type Executor = executor::Default;
    type Flags = ();

    fn new(_: Self::Flags) -> (Self, Command<Message>) {
        (
            Self {
                num_steps: 1000,
                viewer: Default::default(),
                active_tracks: Vec::default(),
                detections: Vec::default(),
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("2d_multi_object_tracking")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::Tracking(active_tracks, detections) => {
                self.active_tracks = active_tracks;
                self.detections = detections;
            }
            _ => {}
        };
        Command::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        worker(0, self.num_steps).map(|v| match v.1 {
            Progress::Advanced(_, active_tracks, detections) => {
                Message::Tracking(active_tracks, detections)
            }
            _ => Message::Void,
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

fn worker<I: 'static + Hash + Copy + Send + Sync>(
    id: I,
    num_steps: usize,
) -> iced::Subscription<(I, Progress)> {
    let gen = data_generator(num_steps as i64, 20, 0.03, 0.33, 0.0, 3.33);

    let gen = Arc::new(Mutex::new(gen));
    let init_state = MyState::Ready(MyTracker::create(), num_steps);

    subscription::unfold(id, init_state, move |state| {
        let gen = gen.clone();
        tracking(id, gen, state)
    })
}

async fn tracking<T, I: Copy>(
    id: I,
    gen: Arc<Mutex<T>>,
    state: MyState,
) -> (Option<(I, Progress)>, MyState)
where
    T: Iterator<Item = (Vec<Detection>, Vec<Detection>)>,
{
    let gen = gen.clone();

    match state {
        MyState::Ready(tracker, num_steps) => (
            Some((id, Progress::Started)),
            MyState::Tracking {
                total: num_steps,
                count: 0,
                tracker: tracker,
            },
        ),
        MyState::Tracking {
            total,
            count,
            mut tracker,
        } => {
            if count <= total {
                if let Some((_, det_gt)) = gen.lock().unwrap().next() {
                    let target = det_gt
                        .to_vec()
                        .into_iter()
                        .filter(|v| v._box.is_some())
                        .collect::<Vec<_>>();
                    let active_tracks = tracker.step(target.clone());

                    (
                        Some((id, Progress::Advanced(count, active_tracks, target))),
                        MyState::Tracking {
                            total,
                            count: count + 1,
                            tracker,
                        },
                    )
                } else {
                    (Some((id, Progress::Finished)), MyState::Finished)
                }
            } else {
                (Some((id, Progress::Finished)), MyState::Finished)
            }
        }
        MyState::Finished => {
            // let _: () = iced::futures::future::pending().await;
            (Some((id, Progress::Finished)), MyState::Finished)
        }
    }
}

impl<Message> canvas::Program<Message> for MultiObject2dTracking {
    fn draw(&self, bounds: Rectangle, _cursor: canvas::Cursor) -> Vec<canvas::Geometry> {
        let viewer = self.viewer.draw(bounds.size(), |frame| {
            self.active_tracks.iter().for_each(|track| {
                let rect = (
                    track._box[0] as usize,
                    track._box[1] as usize,
                    track._box[2] as usize,
                    track._box[3] as usize,
                );
                draw_rectangle(frame, rect, (0, 255, 0), false);
            });

            self.detections.iter().for_each(|det| {
                let b = det._box.clone().unwrap();
                let f = det.feature.clone().unwrap();
                let rect = (b[0] as usize, b[1] as usize, b[2] as usize, b[3] as usize);
                draw_rectangle(
                    frame,
                    rect,
                    (
                        (f[0] * 256. % 256.) as u8,
                        (f[1] * 256. % 256.) as u8,
                        (f[2] * 256. % 256.) as u8,
                    ),
                    true,
                );
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
            }),
        );

        tracker
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
