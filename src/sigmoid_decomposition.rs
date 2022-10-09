use crate::{
    basic_iterative_decomposition::BasicIterativeDecomposition,
    traits::{DimensionalReduction, GenericFeature, IterativeDecomposition},
    utils::{dot, sigmoid, DataRaceAware},
};
use num_traits::AsPrimitive;
use rayon::prelude::*;

pub struct SigmoidDecomposition {
    decomposition: BasicIterativeDecomposition,
}

impl IterativeDecomposition for SigmoidDecomposition {
    fn get_iterative_basic_decomposition(&self) -> &BasicIterativeDecomposition {
        &self.decomposition
    }
}

impl DimensionalReduction for SigmoidDecomposition {
    fn fit_transform<Original, Target>(
        &self,
        target: &mut [Target],
        target_dimension: usize,
        original: &[Original],
        original_dimension: usize,
    ) -> Result<(), String>
    where
        Original: AsPrimitive<Target> + GenericFeature,
        Target: GenericFeature,
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

        self.start_iterations()
            .map(|_| {
                original
                    .par_chunks(original_dimension)
                    .enumerate()
                    .map(|(sample_number, left_original_sample)| unsafe {
                        (
                            sample_number,
                            (
                                &mut (*wrapped_target.get())[(sample_number * target_dimension)
                                    ..((sample_number + 1) * target_dimension)],
                                left_original_sample,
                            ),
                        )
                    })
                    .map(
                        |(sample_number, (left_target_sample, left_original_sample))| {
                            original[sample_number * original_dimension..]
                                .chunks(original_dimension)
                                .enumerate()
                                .map(|(mut inner_sample_number, right_original_sample)| unsafe {
                                    inner_sample_number += sample_number;
                                    (
                                        &mut (*wrapped_target.get())[(inner_sample_number
                                            * target_dimension)
                                            ..((inner_sample_number + 1) * target_dimension)],
                                        right_original_sample,
                                    )
                                })
                                .map(|(right_target_sample, right_original_sample)| {
                                    let target_dot = dot(
                                        left_target_sample.iter().copied(),
                                        right_target_sample.iter().copied(),
                                    );
                                    let original_dot: Target = dot(
                                        left_original_sample.iter().copied(),
                                        right_original_sample.iter().copied(),
                                    )
                                    .as_();
                                    let mut variation = sigmoid(target_dot) - sigmoid(original_dot);
                                    variation *= learning_rate;
                                    left_target_sample
                                        .iter_mut()
                                        .zip(right_target_sample.iter_mut())
                                        .for_each(|(left, right)| {
                                            *left -= *right * variation;
                                            *right -= *left * variation;
                                        });
                                    Ok(())
                                })
                                .collect::<Result<(), String>>()
                        },
                    )
                    .collect::<Result<(), String>>()
            })
            .collect::<Result<(), String>>()
    }
}
