use genawaiter::{sync::gen, yield_};
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;

pub fn rand_int(rng: &mut rand::rngs::ThreadRng, min_val: i64,  max_val: i64) -> i64{
   rng.sample(Uniform::new(min_val, max_val))
}

pub fn rand_uniform(rng: &mut rand::rngs::ThreadRng, min_val: f64,  max_val: f64) -> f64{
   rng.sample::<f64, _>(Uniform::new(min_val, max_val))
}

pub fn rand_color(rng: &mut rand::rngs::ThreadRng) -> [i64; 3] {
   let r = rand_int(rng, 0, 255);
   let g = rand_int(rng, 0, 255);
   let b = rand_int(rng, 0, 255);

   [r, g, b]
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

        let omega_x = rand_uniform(&mut rng,-max_omega, max_omega);
        let omega_y = rand_uniform(&mut rng,-max_omega, max_omega);
        let fi_x = rand_int(&mut rng,-180, 180);
        let fi_y = rand_int(&mut rng,-90, 90);

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

    fn detections(&self, step: i64) {
        let (xmin, ymin)  = self.position_at(step);
        let box_gt = [xmin, ymin, xmin + (self.width as f64), ymin + (self.height + f64)];
    }
}

impl Default for Actor {
    fn default() -> Self {
        let max_omega = 0.05;
        let miss_prob = 0.1;
        let disappear_prob = 0.01;
        let det_err_sigma = 1.0;
        let canvas_size = 400;

        Self::new(max_omega, miss_prob, disappear_prob, det_err_sigma, canvas_size, None)
    }
}



pub fn data_generator(num_steps: i64, num_objects: i64, max_omega: f64, miss_prob: f64, disappear_prob: f64, det_err_sigma: f64) {
}
