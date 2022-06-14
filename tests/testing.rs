use rand::distributions::Uniform;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use genawaiter::sync::{Gen, GenBoxed};
use nalgebra as na;

use motrs::tracker::Detection;

const CANVAS_SIZE: i64 = 1000;

pub fn rand_int<R: Rng>(rng: &mut R, min_val: i64, max_val: i64) -> i64 {
    rng.sample(Uniform::new(min_val, max_val))
}

pub fn rand_uniform<R: Rng>(rng: &mut R, min_val: f64, max_val: f64) -> f64 {
    rng.sample::<f64, _>(Uniform::new(min_val, max_val))
}

pub fn rand_color<R: Rng>(rng: &mut R) -> [i64; 3] {
    let r = rand_int(rng, 0, 255);
    let g = rand_int(rng, 0, 255);
    let b = rand_int(rng, 0, 255);

    [r, g, b]
}

pub fn rand_guass<R: Rng>(rng: &mut R, mu: f64, sigma2: f64) -> f64 {
    let normal = Normal::new(mu, sigma2.sqrt()).unwrap();
    normal.sample(rng)
}

struct Actor {
    pub max_omega: f64,
    pub miss_prob: f64,
    pub disappear_prob: f64,
    pub det_err_sigma: f64,
    pub canvas_size: i64,
    pub class_id: i64,

    pub width: i64,
    pub height: i64,

    pub omega_x: f64,
    pub omega_y: f64,
    pub fi_x: i64,
    pub fi_y: i64,

    pub color: [i64; 3],
    pub disappear_steps: i64,
}

impl Actor {
    pub fn new(
        max_omega: f64,
        miss_prob: f64,
        disappear_prob: f64,
        det_err_sigma: f64,
        canvas_size: i64,
        color: Option<[i64; 3]>,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let max_omega = max_omega;
        let miss_prob = miss_prob;
        let disappear_prob = disappear_prob;
        let det_err_sigma = det_err_sigma;
        let canvas_size = canvas_size;
        let class_id = rand_int(&mut rng, 1, 10);

        let width = rand_int(&mut rng, 50, 120);
        let height = rand_int(&mut rng, 50, 120);

        let omega_x = rand_uniform(&mut rng, -max_omega, max_omega);
        let omega_y = rand_uniform(&mut rng, -max_omega, max_omega);
        let fi_x = rand_int(&mut rng, -180, 180);
        let fi_y = rand_int(&mut rng, -90, 90);

        let color = if let Some(c) = color {
            c
        } else {
            rand_color(&mut rng)
        };

        let disappear_steps = 0;

        Self {
            max_omega,
            miss_prob,
            disappear_prob,
            det_err_sigma,
            canvas_size,
            class_id,
            width,
            height,
            omega_x,
            omega_y,
            fi_x,
            fi_y,
            color,
            disappear_steps,
        }
    }

    fn position_at(&self, step: i64) -> (f64, f64) {
        let half = (self.canvas_size as f64) / 2. - 50.;
        let x = half * (self.omega_x * (step as f64) + (self.fi_x as f64)).cos() + half;
        let y = half * (self.omega_y * (step as f64) + (self.fi_y as f64)).cos() + half;

        (x, y)
    }

    fn detections(&mut self, step: &i64) -> (Detection, Detection) {
        let (xmin, ymin) = self.position_at(*step);
        let mut rng = rand::thread_rng();
        let box_gt = [
            xmin,
            ymin,
            xmin + (self.width as f64),
            ymin + (self.height as f64),
        ];
        let mut box_pred = if rng.gen::<f64>() < self.miss_prob {
            None
        } else {
            let _pred = box_gt
                .iter()
                .map(|v| rand_guass(&mut rng, 0., self.det_err_sigma) + v)
                .collect::<Vec<_>>();
            Some(_pred)
        };

        if rng.gen::<f64>() < self.disappear_prob {
            self.disappear_steps = rand_int(&mut rng, 1, 24);
        }

        if self.disappear_steps > 0 {
            box_pred = None;
            self.disappear_steps -= 1;
        }

        let det_gt = Detection {
            _box: Some(na::DMatrix::from_row_slice(1, 4, &box_gt)),
            score: 1.,
            class_id: self.class_id,
            feature: Some(na::DMatrix::from_fn(1, 3, |r, c| self.color[c] as f64)),
        };

        let feature_pred = self
            .color
            .iter()
            .map(|v| rand_guass(&mut rng, 0., 5.) + (*v as f64))
            .collect::<Vec<_>>();

        let det_pred = if box_pred.is_some() {
            Detection {
                _box: Some(na::DMatrix::from_row_slice(
                    1,
                    4,
                    box_pred.unwrap().as_slice(),
                )),
                score: rand_uniform(&mut rng, 0.5, 1.),
                class_id: std::cmp::max(0, self.class_id + rand_int(&mut rng, -1, 1)),
                feature: Some(na::DMatrix::from_row_slice(1, 3, feature_pred.as_slice())),
            }
        } else {
            Detection {
                _box: None,
                score: rand_uniform(&mut rng, 0.5, 1.),
                class_id: std::cmp::max(0, self.class_id + rand_int(&mut rng, -1, 1)),
                feature: Some(na::DMatrix::from_row_slice(1, 3, feature_pred.as_slice())),
            }
        };

        (det_gt, det_pred)
    }
}

impl Default for Actor {
    fn default() -> Self {
        let max_omega = 0.05;
        let miss_prob = 0.1;
        let disappear_prob = 0.01;
        let det_err_sigma = 1.0;
        let canvas_size = 400;

        Self::new(
            max_omega,
            miss_prob,
            disappear_prob,
            det_err_sigma,
            canvas_size,
            None,
        )
    }
}

pub fn data_generator(
    num_steps: i64,
    num_objects: i64,
    max_omega: f64,
    miss_prob: f64,
    disappear_prob: f64,
    det_err_sigma: f64,
) -> GenBoxed<(Vec<Detection>, Vec<Detection>)> {
    Gen::new_boxed(|co| async move {
        let mut actors = (0..num_objects)
            .map(|_| {
                Actor::new(
                    max_omega,
                    miss_prob,
                    disappear_prob,
                    det_err_sigma,
                    CANVAS_SIZE,
                    None,
                )
            })
            .collect::<Vec<_>>();

        for step in 0..num_steps {
            let mut dets_gt = vec![];
            let mut dets_pred = vec![];

            for mut actor in actors.iter_mut() {
                let (det_gt, det_pred) = (&mut actor).detections(&step);

                dets_gt.push(det_gt);
                dets_pred.push(det_pred);
            }

            co.yield_((dets_gt, dets_pred)).await;
        }
    })
}

pub fn data_generator_file(
    gt: &std::path::Path,
    pred: &std::path::Path,
) -> GenBoxed<(Vec<Detection>, Vec<Detection>)> {
    let mut rdr_gt = csv::Reader::from_path(gt.as_os_str()).unwrap();
    let mut rdr_pred = csv::Reader::from_path(pred.as_os_str()).unwrap();

    Gen::new_boxed(|co| async move {
        let mut iter1 = rdr_gt.deserialize::<(
            Option<f64>,
            Option<f64>,
            Option<f64>,
            Option<f64>,
            f64,
            f64,
            f64,
            i64,
            f64,
        )>();
        let mut iter2 = rdr_pred.deserialize::<(
            Option<f64>,
            Option<f64>,
            Option<f64>,
            Option<f64>,
            f64,
            f64,
            f64,
            f64,
            f64,
        )>();

        for j in 0..2000 {
            let mut dets_gt = vec![];
            let mut dets_pred = vec![];

            for i in 0..2 {
                let record = iter1.next();
                if let Some(r) = record {
                    let r = r.unwrap();
                    if r.0.is_some() {
                        dets_gt.push(Detection {
                            _box: Some(na::DMatrix::from_row_slice(
                                1,
                                4,
                                &[r.0.unwrap(), r.1.unwrap(), r.2.unwrap(), r.3.unwrap()],
                            )),
                            score: r.8,
                            class_id: r.7,
                            feature: Some(na::DMatrix::from_row_slice(1, 3, &[r.4, r.5, r.6])),
                        });
                    } else {
                        dets_gt.push(Detection {
                            _box: None,
                            score: r.8,
                            class_id: r.7,
                            feature: Some(na::DMatrix::from_row_slice(1, 3, &[r.4, r.5, r.6])),
                        });
                    }
                }

                let record = iter2.next();
                if let Some(r) = record {
                    let r = r.unwrap();
                    if r.0.is_some() {
                        dets_pred.push(Detection {
                            _box: Some(na::DMatrix::from_row_slice(
                                1,
                                4,
                                &[r.0.unwrap(), r.1.unwrap(), r.2.unwrap(), r.3.unwrap()],
                            )),
                            score: r.8,
                            class_id: r.7 as i64,
                            feature: Some(na::DMatrix::from_row_slice(1, 3, &[r.4, r.5, r.6])),
                        });
                    } else {
                        dets_pred.push(Detection {
                            _box: None,
                            score: r.8,
                            class_id: r.7 as i64,
                            feature: Some(na::DMatrix::from_row_slice(1, 3, &[r.4, r.5, r.6])),
                        });
                    }
                }
            }

            co.yield_((dets_gt, dets_pred)).await;
        }
    })
}
