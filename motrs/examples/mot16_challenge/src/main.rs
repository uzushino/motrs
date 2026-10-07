use std::path::{Path, PathBuf};
use std::time::Duration;

use ab_glyph::FontArc;
use iced::widget::{Image, column, container, image::Handle, text};
use iced::{Element, Length, Subscription};
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use motrs::{BoundingBox, ModelConfig, MotionModel, MultiObjectTracker, TrackerConfig};

mod util;
use util::{FrameDetections, read_detections, video_frame_path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let example_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut arguments = std::env::args_os().skip(1);
    let first_argument = arguments.next();
    let check_only = first_argument.as_deref() == Some(std::ffi::OsStr::new("--check"));
    let dataset_argument = if check_only {
        arguments.next()
    } else {
        first_argument
    };
    if arguments.next().is_some() {
        return Err("Usage: mot16_challenge [--check] [sequence-directory]".into());
    }
    let dataset = dataset_argument
        .map(PathBuf::from)
        .unwrap_or_else(|| example_root.join("MOT16/train/MOT16-04"));
    let annotation_path = dataset.join("gt/gt.txt");
    let frames = read_detections(&annotation_path)
        .map_err(|error| format!("Cannot read {}: {error}", annotation_path.display()))?;
    let font = load_font(example_root);
    if check_only {
        return check_sequence(frames, dataset.join("img1"), font);
    }
    iced::application(
        move || Viewer::new(frames.clone(), dataset.join("img1"), font.clone()),
        Viewer::update,
        Viewer::view,
    )
    .title("MOT16 tracking")
    .subscription(Viewer::subscription)
    .run()?;
    Ok(())
}

fn load_font(example_root: &Path) -> Option<FontArc> {
    let bundled = example_root.join("assets/fonts/Dela_Gothic_One/DelaGothicOne-Regular.ttf");
    let system = if cfg!(target_os = "macos") {
        PathBuf::from("/System/Library/Fonts/Supplemental/Arial.ttf")
    } else if cfg!(target_os = "windows") {
        PathBuf::from(r"C:\Windows\Fonts\arial.ttf")
    } else {
        PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    };
    [bundled, system].into_iter().find_map(|path| {
        std::fs::read(path)
            .ok()
            .and_then(|bytes| FontArc::try_from_vec(bytes).ok())
    })
}

fn create_tracker() -> MultiObjectTracker {
    MultiObjectTracker::new(TrackerConfig {
        dt: 1.0 / 30.0,
        min_iou: 0.25,
        max_missed_frames: 15,
        model: ModelConfig {
            position: MotionModel::ConstantAcceleration,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("demo configuration is valid")
}

/// Run the same image loading, tracking and drawing as the GUI, without a window.
fn check_sequence(
    frames: Vec<FrameDetections>,
    image_directory: PathBuf,
    font: Option<FontArc>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut viewer = Viewer::new(frames, image_directory, font);
    let (mut processed_frames, mut peak_tracks) = (0, 0);
    while let Some(frame) = viewer.frames.next() {
        viewer.render_frame(frame)?;
        processed_frames += 1;
        peak_tracks = peak_tracks.max(viewer.tracker.len());
        if processed_frames % 100 == 0 {
            println!("{}", viewer.status);
        }
    }
    if processed_frames == 0 {
        return Err("Sequence has no annotated frames".into());
    }
    println!(
        "Checked {processed_frames} frames: images, tracking and overlays OK; peak tracks={peak_tracks}"
    );
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum Message {
    Tick,
}

struct Viewer {
    frames: std::vec::IntoIter<FrameDetections>,
    image_directory: PathBuf,
    font: Option<FontArc>,
    tracker: MultiObjectTracker,
    image: Option<Handle>,
    status: String,
    finished: bool,
}

impl Viewer {
    fn new(frames: Vec<FrameDetections>, image_directory: PathBuf, font: Option<FontArc>) -> Self {
        Self {
            frames: frames.into_iter(),
            image_directory,
            font,
            tracker: create_tracker(),
            image: None,
            status: "White: track / Red: observation".into(),
            finished: false,
        }
    }

    fn update(&mut self, _: Message) {
        let Some(frame) = self.frames.next() else {
            self.finished = true;
            return;
        };
        match self.render_frame(frame) {
            Ok(image) => self.image = Some(image),
            Err(error) => {
                self.status = error.to_string();
                self.finished = true;
            }
        }
    }

    fn render_frame(
        &mut self,
        frame: FrameDetections,
    ) -> Result<Handle, Box<dyn std::error::Error>> {
        let path = video_frame_path(&self.image_directory, frame.number);
        let mut image = image::open(&path)
            .map_err(|error| format!("Cannot read {}: {error}", path.display()))?
            .into_rgba8();
        let tracks = self.tracker.step(frame.detections.clone())?;
        let white = Rgba([255, 255, 255, 255]);
        for detection in &frame.detections {
            draw_box(&mut image, &detection.bounds, Rgba([255, 0, 0, 255]));
        }
        for track in tracks {
            draw_box(&mut image, &track.bounds, white);
            if let Some(font) = &self.font {
                draw_text_mut(
                    &mut image,
                    white,
                    track.bounds.min()[0] as i32,
                    track.bounds.min()[1] as i32,
                    12.0,
                    font,
                    &track.id[..8],
                );
            }
        }
        self.status = format!("Frame {} — White: track / Red: observation", frame.number);
        Ok(Handle::from_rgba(
            image.width(),
            image.height(),
            image.into_raw(),
        ))
    }

    fn subscription(&self) -> Subscription<Message> {
        if self.finished {
            Subscription::none()
        } else {
            iced::time::every(Duration::from_millis(33)).map(|_| Message::Tick)
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let mut content = column![text(&self.status)];
        if let Some(image) = &self.image {
            content = content.push(
                Image::new(image.clone())
                    .width(Length::Fill)
                    .height(Length::Fill),
            );
        }
        container(content)
            .padding(20)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

fn draw_box(image: &mut RgbaImage, bounds: &BoundingBox, color: Rgba<u8>) {
    let rectangle = Rect::at(bounds.min()[0] as i32, bounds.min()[1] as i32).of_size(
        (bounds.extent(0) as u32).max(1),
        (bounds.extent(1) as u32).max(1),
    );
    draw_hollow_rect_mut(image, rectangle, color);
}
