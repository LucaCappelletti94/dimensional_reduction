use crate::{
    basic_decomposition::BasicDecomposition,
    basic_iterative_decomposition::BasicIterativeDecomposition,
};
use indicatif::{ProgressBar, ProgressStyle};
use indicatif::{ProgressBarIter, ProgressIterator};
use num_traits::{AsPrimitive, Bounded, Float, One, Zero};
use rayon::prelude::*;
use std::fmt::Debug;
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};
use vec_rand::{random_f32, splitmix64};

pub trait GenericFeature:
    Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + PartialOrd
    + Bounded
    + AsPrimitive<usize>
    + Debug
    + SubAssign<Self>
    + AddAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
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
        + AddAssign<Self>
        + MulAssign<Self>
        + DivAssign<Self>
        + Sum<Self>
        + PartialOrd
        + Debug
        + Bounded
        + AsPrimitive<usize>
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
        Original: num_traits::AsPrimitive<Target> + GenericFeature + Float,
        Target: Float + GenericFeature,
        usize: AsPrimitive<Original> + AsPrimitive<Target>,
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

pub trait RandomUniformInitialization {
    fn random_init(&mut self, random_state: u64);
}

pub trait MatrixSum<F> {
    fn matrix_sum(&self, dimensionality: usize) -> Result<Vec<F>, String>;
}

impl<'a, F> MatrixSum<F> for &[F]
where
    F: GenericFeature,
{
    fn matrix_sum(&self, dimensionality: usize) -> Result<Vec<F>, String> {
        if self.len() % dimensionality != 0 {
            return Err(format!(
                concat!(
                    "The provided dimensionality {} is not compatible ",
                    "with the length of this object {}."
                ),
                dimensionality,
                self.len(),
            ));
        }

        Ok(self
            .par_chunks(dimensionality)
            .map(|slice| slice.to_vec())
            .reduce(
                || vec![F::zero(); dimensionality],
                |mut left, right| {
                    left.iter_mut().zip(right.into_iter()).for_each(|(l, r)| {
                        *l += r;
                    });
                    left
                },
            ))
    }
}

pub trait MatrixMean<F> {
    fn matrix_mean(&self, dimensionality: usize) -> Result<Vec<F>, String>;
}

impl<'a, F> MatrixMean<F> for &[F]
where
    F: GenericFeature,
    usize: AsPrimitive<F>,
{
    fn matrix_mean(&self, dimensionality: usize) -> Result<Vec<F>, String> {
        if self.is_empty() {
            return Err("The provided object is empty".to_string());
        }

        let mut matrix_sum = self.matrix_sum(dimensionality)?;
        let number_of_samples: F = (self.len() / dimensionality).as_();

        matrix_sum.iter_mut().for_each(|v| {
            *v /= number_of_samples;
        });

        Ok(matrix_sum)
    }
}

pub trait MatrixMinMax<F> {
    fn matrix_min_max(&self, dimensionality: usize) -> Result<(Vec<F>, Vec<F>), String>;
}

impl<'a, F> MatrixMinMax<F> for &[F]
where
    F: GenericFeature,
{
    fn matrix_min_max(&self, dimensionality: usize) -> Result<(Vec<F>, Vec<F>), String> {
        if self.is_empty() {
            return Err("The provided object is empty".to_string());
        }

        if self.len() % dimensionality != 0 {
            return Err(format!(
                concat!(
                    "The provided dimensionality {} is not compatible ",
                    "with the length of this object {}."
                ),
                dimensionality,
                self.len(),
            ));
        }

        Ok(self
            .par_chunks(dimensionality)
            .map(|slice| (slice.to_vec(), slice.to_vec()))
            .reduce(
                || {
                    (
                        vec![F::max_value(); dimensionality],
                        vec![F::min_value(); dimensionality],
                    )
                },
                |(mut left_min, mut left_max), (right_min, right_max)| {
                    left_min
                        .iter_mut()
                        .zip(right_min.into_iter())
                        .for_each(|(l, r)| {
                            if *l > r {
                                *l = r;
                            }
                        });
                    left_max
                        .iter_mut()
                        .zip(right_max.into_iter())
                        .for_each(|(l, r)| {
                            if *l < r {
                                *l = r;
                            }
                        });
                    (left_min, left_max)
                },
            ))
    }
}

pub trait MatrixVariance<F> {
    fn matrix_var(&self, dimensionality: usize) -> Result<Vec<F>, String>;
}

impl<'a, F> MatrixVariance<F> for &[F]
where
    F: GenericFeature,
    usize: AsPrimitive<F>,
{
    fn matrix_var(&self, dimensionality: usize) -> Result<Vec<F>, String> {
        let matrix_mean = self.matrix_mean(dimensionality)?;

        let mut unnormalized_variance = self
            .par_chunks(dimensionality)
            .map(|slice| {
                let mut vec = slice.to_vec();
                vec.iter_mut()
                    .zip(matrix_mean.iter().copied())
                    .for_each(|(v, m)| {
                        let delta = *v - m;
                        *v = delta * delta;
                    });
                vec
            })
            .reduce(
                || vec![F::zero(); dimensionality],
                |mut left, right| {
                    left.iter_mut().zip(right.into_iter()).for_each(|(l, r)| {
                        *l += r;
                    });
                    left
                },
            );

        let number_of_samples: F = (self.len() / dimensionality).as_();

        unnormalized_variance.iter_mut().for_each(|v| {
            *v /= number_of_samples;
        });

        Ok(unnormalized_variance)
    }
}

pub trait MatrixStandardDeviation<F> {
    fn matrix_std(&self, dimensionality: usize) -> Result<Vec<F>, String>;
}

impl<'a, F> MatrixStandardDeviation<F> for &[F]
where
    F: GenericFeature + Float,
    usize: AsPrimitive<F>,
{
    fn matrix_std(&self, dimensionality: usize) -> Result<Vec<F>, String> {
        let mut variance = self.matrix_var(dimensionality)?;

        variance.iter_mut().for_each(|v| {
            *v = (*v).sqrt();
        });

        Ok(variance)
    }
}

impl<'a, F> RandomUniformInitialization for &'a mut [F]
where
    F: Send + Sync + Copy + 'static,
    f32: AsPrimitive<F>,
{
    fn random_init(&mut self, random_state: u64) {
        self.par_iter_mut().enumerate().for_each(|(i, weight)| {
            *weight = (2.0_f32 * random_f32(splitmix64(random_state + random_state * i as u64))
                - 1.0_f32)
                .as_();
        });
    }
}

impl<F> RandomUniformInitialization for Vec<F>
where
    Self: AsMut<[F]>,
    F: Send + Sync + 'static,
    F: AsPrimitive<f32>,
    f32: AsPrimitive<F>,
{
    fn random_init(&mut self, random_state: u64) {
        self.as_mut().random_init(random_state);
    }
}
