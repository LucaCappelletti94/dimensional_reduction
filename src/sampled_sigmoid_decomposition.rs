use crate::traits::*;
use crate::{
    basic_iterative_decomposition::BasicIterativeDecomposition,
    traits::{Decomposition, DimensionalReduction, GenericFeature, IterativeDecomposition},
    utils::{dot, normal_dot, sigmoid, DataRaceAware},
};
use num_traits::{AsPrimitive, Float};
use rayon::prelude::*;
use vec_rand::splitmix64;

#[derive(Clone)]
pub struct SampledSigmoidDecomposition {
    decomposition: BasicIterativeDecomposition,
}

impl From<BasicIterativeDecomposition> for SampledSigmoidDecomposition {
    fn from(decomposition: BasicIterativeDecomposition) -> Self {
        Self { decomposition }
    }
}

impl IterativeDecomposition for SampledSigmoidDecomposition {
    fn get_iterative_basic_decomposition(&self) -> &BasicIterativeDecomposition {
        &self.decomposition
    }
}

impl DimensionalReduction for SampledSigmoidDecomposition {
    fn fit_transform<Original, Target>(
        &self,
        mut target: &mut [Target],
        target_dimension: usize,
        original: &[Original],
        original_dimension: usize,
    ) -> Result<(), String>
    where
        Original: AsPrimitive<Target> + GenericFeature,
        Target: Float + GenericFeature,
        usize: AsPrimitive<Original> + AsPrimitive<Target>,
        f32: AsPrimitive<Target>,
    {
        if target.len() % target_dimension != 0 {
            return Err(format!(
                concat!(
                    "The provided target slice has length {} ",
                    "which is not compatible with the provided ",
                    "target dimension {}."
                ),
                target.len(),
                target_dimension
            ));
        }

        target.random_init(self.get_random_state());

        let number_of_samples = target.len() / target_dimension;

        let mean = original.matrix_mean(original_dimension)?;
        let variance = original.matrix_var(original_dimension)?;

        // We wrap the features object in an unsafe cell so
        // it may be shared among threads.
        let wrapped_target = DataRaceAware::from(target);

        if original.len() % original_dimension != 0 {
            return Err(format!(
                concat!(
                    "The provided original slice has length {} ",
                    "which is not compatible with the provided ",
                    "original dimension {}."
                ),
                original.len(),
                original_dimension
            ));
        }

        let learning_rate: Target = self.get_learning_rate().as_();
        let mut random_state = splitmix64(self.get_random_state());

        self.start_iterations()
            .map(|_| {
                random_state = splitmix64(random_state);
                original
                    .par_chunks(original_dimension)
                    .enumerate()
                    .map(|(sample_number, left_original_sample)| unsafe {
                        (
                            sample_number as u64,
                            (
                                &mut (*wrapped_target.get())[(sample_number * target_dimension)
                                    ..((sample_number + 1) * target_dimension)],
                                left_original_sample,
                            ),
                        )
                    })
                    .map(
                        |(sample_number, (left_target_sample, left_original_sample))| unsafe {
                            let inner_sample_number = (splitmix64(
                                random_state.wrapping_add(sample_number.wrapping_mul(random_state)),
                            ) as usize)
                                % number_of_samples;

                            let right_target_sample = &mut (*wrapped_target.get())
                                [(inner_sample_number * target_dimension)
                                    ..((inner_sample_number + 1) * target_dimension)];

                            let right_original_sample = &original[(inner_sample_number
                                * original_dimension)
                                ..((inner_sample_number + 1) * original_dimension)];

                            let target_dot = dot(
                                left_target_sample.iter().copied(),
                                right_target_sample.iter().copied(),
                            );

                            let original_dot: Target = normal_dot(
                                left_original_sample,
                                right_original_sample,
                                &mean,
                                &variance,
                            )
                            .as_();

                            //let left_sigmoid = sigmoid(target_dot);
                            //let right_sigmoid = sigmoid(original_dot);

                            let mut variation = sigmoid(target_dot) - sigmoid(original_dot.ln());
                            variation *= learning_rate;

                            left_target_sample
                                .iter_mut()
                                .zip(right_target_sample.iter_mut())
                                .for_each(|(left, right)| {
                                    let new_left = *left - *right * variation;
                                    let new_right = *right - *left * variation;
                                    if new_left.is_finite() {
                                        *left = new_left;
                                    }
                                    if new_right.is_finite() {
                                        *right = new_right;
                                    }
                                });
                            Ok(())
                        },
                    )
                    .collect::<Result<(), String>>()
            })
            .collect::<Result<(), String>>()
    }
}
