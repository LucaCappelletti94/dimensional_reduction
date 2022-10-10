use crate::{
    basic_decomposition::BasicDecomposition,
    basic_iterative_decomposition::BasicIterativeDecomposition,
};
use pyo3::PyResult;
use pyo3::types::PyDict;
use indicatif::{ProgressBar, ProgressStyle};
use indicatif::{ProgressBarIter, ProgressIterator};
use num_traits::{AsPrimitive, Float, One, Zero};
use std::{
    iter::Sum,
    ops::{Add, Div, Mul, MulAssign, Sub, SubAssign},
};

pub trait GenericFeature:
    Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + Sum<Self>
    + One
    + Zero
    + Copy
    + Sync
    + Send
    + 'static
{
}

impl<T> GenericFeature for T where
    T: Mul<Self, Output = Self>
        + Div<Self, Output = Self>
        + Add<Self, Output = Self>
        + Sub<Self, Output = Self>
        + SubAssign<Self>
        + MulAssign<Self>
        + Sum<Self>
        + One
        + Zero
        + Copy
        + Sync
        + Send
        + 'static
{
}

pub trait DimensionalReduction {
    fn fit_transform<Original, Target>(
        &self,
        target: &mut [Target],
        target_dimension: usize,
        original: &[Original],
        original_dimension: usize,
    ) -> Result<(), String>
    where
        Original: num_traits::AsPrimitive<Target> + GenericFeature,
        Target: Float + GenericFeature,
        f32: AsPrimitive<Target>;
}

pub trait Decomposition {
    fn get_basic_decomposition(&self) -> &BasicDecomposition;

    fn get_model_name(&self) -> &str {
        self.get_basic_decomposition().model_name.as_str()
    }

    fn get_random_state(&self) -> u64 {
        self.get_basic_decomposition().random_state
    }

    fn is_verbose(&self) -> bool {
        self.get_basic_decomposition().verbose
    }
}

impl Decomposition for BasicDecomposition {
    fn get_basic_decomposition(&self) -> &BasicDecomposition {
        &self
    }
}

pub trait IterativeDecomposition: Decomposition {
    fn get_iterative_basic_decomposition(&self) -> &BasicIterativeDecomposition;

    fn get_iterations(&self) -> usize {
        self.get_iterative_basic_decomposition().iterations
    }

    fn get_learning_rate(&self) -> f32 {
        self.get_iterative_basic_decomposition().learning_rate
    }

    fn get_loading_bar(&self) -> ProgressBar {
        if self.is_verbose() {
            let pb = ProgressBar::new(self.get_iterations() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(&format!(
                        concat!(
                            "{}{{msg}} {{spinner:.green}} [{{elapsed_precise}}] ",
                            "[{{bar:40.cyan/blue}}] ({{pos}}/{{len}}, ETA {{eta}})"
                        ),
                        self.get_model_name()
                    ))
                    .unwrap(),
            );
            pb
        } else {
            ProgressBar::hidden()
        }
    }

    fn start_iterations(&self) -> ProgressBarIter<std::ops::Range<usize>> {
        (0..self.get_iterations()).progress_with(self.get_loading_bar())
    }
}

impl<T> Decomposition for T
where
    T: IterativeDecomposition,
{
    fn get_basic_decomposition(&self) -> &BasicDecomposition {
        &self.get_iterative_basic_decomposition().basic_decomposition
    }
}

pub(crate) trait FromPyDict {
    fn from_pydict(py_kwargs: Option<&PyDict>) -> PyResult<Self>
    where
        Self: Sized;
}
