use crate::Error;

/// An axis-aligned box with finite coordinates and positive extent on each axis.
///
/// The two corners have the same number of coordinates. Two-dimensional boxes
/// use `[x, y]`; three-dimensional boxes use `[x, y, z]`.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    min: Vec<f32>,
    max: Vec<f32>,
}

impl BoundingBox {
    pub fn new(min: impl Into<Vec<f32>>, max: impl Into<Vec<f32>>) -> Result<Self, Error> {
        let (min, max) = (min.into(), max.into());
        if min.is_empty() || min.len() != max.len() {
            return Err(Error::InvalidBounds(
                "corners must have the same nonzero dimension",
            ));
        }
        for (&low, &high) in min.iter().zip(&max) {
            if !low.is_finite() || !high.is_finite() || !(high - low).is_finite() {
                return Err(Error::InvalidBounds(
                    "coordinates and extents must be finite",
                ));
            }
            if high <= low {
                return Err(Error::InvalidBounds("each maximum must exceed its minimum"));
            }
        }
        Ok(Self { min, max })
    }

    pub fn dimensions(&self) -> usize {
        self.min.len()
    }
    pub fn min(&self) -> &[f32] {
        &self.min
    }
    pub fn max(&self) -> &[f32] {
        &self.max
    }
    pub fn center(&self, axis: usize) -> f32 {
        self.min[axis] + self.extent(axis) * 0.5
    }
    pub fn extent(&self, axis: usize) -> f32 {
        self.max[axis] - self.min[axis]
    }

    /// Intersection over union; boxes of different dimensions have no overlap.
    pub fn iou(&self, other: &Self) -> f32 {
        if self.dimensions() != other.dimensions() {
            return 0.0;
        }
        let (mut intersection, mut left_volume, mut right_volume) = (1.0_f64, 1.0_f64, 1.0_f64);
        for axis in 0..self.dimensions() {
            let low = self.min[axis].max(other.min[axis]) as f64;
            let high = self.max[axis].min(other.max[axis]) as f64;
            intersection *= (high - low).max(0.0);
            left_volume *= f64::from(self.max[axis]) - f64::from(self.min[axis]);
            right_volume *= f64::from(other.max[axis]) - f64::from(other.min[axis]);
        }
        let union = left_volume + right_volume - intersection;
        if union.is_finite() && union > 0.0 {
            (intersection / union).clamp(0.0, 1.0) as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn overlap_works_in_one_two_and_three_dimensions() {
        let a = BoundingBox::new([10.0], [20.0]).unwrap();
        let b = BoundingBox::new([10.0], [21.0]).unwrap();
        assert_relative_eq!(a.iou(&b), 10.0 / 11.0);
        let a = BoundingBox::new([15.0, 15.0], [25.0, 25.0]).unwrap();
        let b = BoundingBox::new([10.0, 10.0], [20.0, 20.0]).unwrap();
        assert_relative_eq!(a.iou(&b), 25.0 / 175.0);
        let a = BoundingBox::new([0.0; 3], [2.0; 3]).unwrap();
        let b = BoundingBox::new([1.0; 3], [3.0; 3]).unwrap();
        assert_relative_eq!(a.iou(&b), 1.0 / 15.0);
    }

    #[test]
    fn rejects_degenerate_nonfinite_and_mismatched_boxes() {
        assert!(BoundingBox::new([1.0, 1.0], [1.0, 2.0]).is_err());
        assert!(BoundingBox::new([f32::NAN], [2.0]).is_err());
        assert!(BoundingBox::new([0.0], [f32::INFINITY]).is_err());
        assert!(BoundingBox::new([0.0], [1.0, 2.0]).is_err());
        assert!(BoundingBox::new([0.0; 0], [1.0; 0]).is_err());
        assert!(BoundingBox::new([-f32::MAX], [f32::MAX]).is_err());
    }

    #[test]
    fn touching_and_disjoint_boxes_have_zero_overlap() {
        let a = BoundingBox::new([0.0; 2], [1.0; 2]).unwrap();
        let b = BoundingBox::new([1.0; 2], [2.0; 2]).unwrap();
        let c = BoundingBox::new([3.0; 2], [4.0; 2]).unwrap();
        assert_eq!(a.iou(&b), 0.0);
        assert_eq!(a.iou(&c), 0.0);
        assert_eq!(a.iou(&a), 1.0);
    }

    #[test]
    fn fractional_boxes_have_exact_self_overlap() {
        let bounds = BoundingBox::new([0.1, 0.2], [1000.2, 1000.3]).unwrap();
        assert_eq!(bounds.iou(&bounds), 1.0);
    }
}
