use motrs::tracker::{ Track, Detection };

use nalgebra as na;
use iced::{
    button, futures, image, Alignment, Application, Button, Column, Command, executor,
    Container, Element, Length, Row, Settings, Text, canvas::Path, Point, Size, Sandbox,
    canvas, Rectangle, Color
};

mod testing;

use crate::testing::data_generator;

const CANVAS_SIZE: usize = 1000;

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
    pub viewer: canvas::Cache,
}

#[derive(Debug, Clone)]
pub enum Message {
    Void,
}

impl Application for ImageViewer {
    type Message = Message;
    type Executor = executor::Default;
    type Flags = ();

    fn new(_: Self::Flags) -> (Self, Command<Message>) {
        (
            Self {
                width: 640,
                height: 360,
                viewer: Default::default(),
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("2d_multi_object_tracking")
    }

    fn update(&mut self, _message: Message) -> Command<Message> {
        Command::none()
    }

    fn view(&mut self) -> Element<Message> {
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
        let clock = self.viewer.draw(bounds.size(), |frame| {
            draw_rectangle(frame, (10, 10, 100, 100), (255, 0, 0));
            draw_rectangle(frame, (300, 300, 500, 500), (0, 255, 0));
        });

        vec![clock]
    }
}