use iced_futures::futures;

use motrs::tracker::*;
use motrs::model::*;

// Just a little utility function
pub fn file<T: ToString>(url: T) -> iced::Subscription<i64> {
    iced::Subscription::from_recipe(MyTracker { uuid: "aa".to_string()})
}

pub struct MyTracker {
    uuid: String
}

impl MyTracker {
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

impl<H, I> iced_native::subscription::Recipe<H, I> for MyTracker where H: std::hash::Hasher {
    type Output = Progress;

    fn hash(&self, state: &mut H) {
        use std::hash::Hash;
        std::any::TypeId::of::<Self>().hash(state);
        self.uuid.hash(state);
    }

    fn stream(
        self: Box<Self>,
        _input: futures::stream::BoxStream<'static, I>,
    ) -> futures::stream::BoxStream<'static, Self::Output> {
        Box::pin(futures::stream::unfold(State::Ready(Self::create()), |state| async move {
                match state {
                    State::Ready(tracker) => {
                        Some((Progress::Started, State::Tracking { total: 1, count: 0, }))
                    }
                    State::Tracking { total, count } => {
                        if count <= total {
                            Some((Progress::Advanced(count), State::Tracking{ total, count }))
                        } else {
                            Some((Progress::Finished, State::Finished))
                        }
                    },
                    State::Finished => {
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

pub enum State {
    Ready(MultiObjectTracker),
    Tracking {
        total: u64,
        count: u64,
    },
    Finished,
}
