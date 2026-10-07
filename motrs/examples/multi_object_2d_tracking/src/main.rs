use std::time::Duration;

use iced::widget::{canvas, column, container, text};
use iced::{Color, Element, Length, Point, Rectangle, Renderer, Size, Subscription, Theme, mouse};
use motrs::{BoundingBox, Detection, MultiObjectTracker, Track, TrackerConfig};

mod scene;
use scene::SyntheticScene;

fn main() -> iced::Result {
    iced::application(Viewer::new, Viewer::update, Viewer::view)
        .title("Multi-object tracking")
        .subscription(Viewer::subscription)
        .antialiasing(true)
        .run()
}

#[derive(Debug, Clone, Copy)]
enum Message {
    Tick,
}

struct Viewer {
    scene: SyntheticScene,
    tracker: MultiObjectTracker,
    expected: Vec<BoundingBox>,
    detections: Vec<Detection>,
    tracks: Vec<Track>,
    canvas: canvas::Cache,
    finished: bool,
}

impl Viewer {
    fn new() -> Self {
        Self {
            scene: SyntheticScene::new(1000, 20),
            tracker: MultiObjectTracker::new(TrackerConfig {
                dt: 1.0 / 30.0,
                min_iou: 1.0 / 24.0,
                min_hits: 2,
                appearance_weight: 0.3,
                ..Default::default()
            })
            .expect("demo configuration is valid"),
            expected: Vec::new(),
            detections: Vec::new(),
            tracks: Vec::new(),
            canvas: canvas::Cache::new(),
            finished: false,
        }
    }

    fn update(&mut self, _: Message) {
        if let Some(frame) = self.scene.next() {
            self.expected = frame.expected;
            self.detections = frame.detections;
            match self.tracker.step(self.detections.clone()) {
                Ok(tracks) => self.tracks = tracks,
                Err(error) => {
                    eprintln!("tracking failed: {error}");
                    self.finished = true;
                }
            }
            self.canvas.clear();
        } else {
            self.finished = true;
        }
    }

    fn subscription(&self) -> Subscription<Message> {
        if self.finished {
            Subscription::none()
        } else {
            iced::time::every(Duration::from_millis(33)).map(|_| Message::Tick)
        }
    }

    fn view(&self) -> Element<'_, Message> {
        container(column![
            text("Gray: true position / Color: detection / Green: track"),
            canvas::Canvas::new(self)
                .width(Length::Fill)
                .height(Length::Fill),
        ])
        .padding(20)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }
}

fn draw_box(frame: &mut canvas::Frame, bounds: &BoundingBox, color: Color, fill: bool) {
    let path = canvas::Path::rectangle(
        Point::new(bounds.min()[0], bounds.min()[1]),
        Size::new(bounds.extent(0), bounds.extent(1)),
    );
    if fill {
        frame.fill(&path, color);
    }
    frame.stroke(&path, canvas::Stroke::default().with_color(color));
}

impl canvas::Program<Message> for Viewer {
    type State = ();

    fn draw(
        &self,
        _: &Self::State,
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        _: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        vec![self.canvas.draw(renderer, bounds.size(), |frame| {
            frame.scale((bounds.width / 650.0).min(bounds.height / 650.0));
            for bounds in &self.expected {
                draw_box(frame, bounds, Color::from_rgb(0.5, 0.5, 0.5), false);
            }
            for detection in &self.detections {
                let color = detection
                    .feature
                    .as_ref()
                    .map_or(Color::from_rgba(1.0, 0.0, 0.0, 0.3), |rgb| {
                        Color::from_rgba(rgb[0], rgb[1], rgb[2], 0.3)
                    });
                draw_box(frame, &detection.bounds, color, true);
            }
            for track in &self.tracks {
                draw_box(frame, &track.bounds, Color::from_rgb(0.0, 1.0, 0.0), false);
            }
        })]
    }
}
