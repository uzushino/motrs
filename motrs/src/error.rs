use std::fmt;

/// Invalid input or a failed numerical update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidBounds(&'static str),
    InvalidConfig(&'static str),
    InvalidDetection { index: usize, reason: &'static str },
    NumericalFailure,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds(reason) => write!(f, "invalid bounding box: {reason}"),
            Self::InvalidConfig(reason) => write!(f, "invalid tracker configuration: {reason}"),
            Self::InvalidDetection { index, reason } => {
                write!(f, "invalid detection {index}: {reason}")
            }
            Self::NumericalFailure => f.write_str("Kalman filter decomposition failed"),
        }
    }
}

impl std::error::Error for Error {}
