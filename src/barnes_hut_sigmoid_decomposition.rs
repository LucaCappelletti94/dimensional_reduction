use std::sync::atomic::{AtomicUsize, Ordering};

use crate::traits::*;
use crate::{
    basic_iterative_decomposition::BasicIterativeDecomposition,
    traits::{Decomposition, DimensionalReduction, GenericFeature, IterativeDecomposition},
    utils::{dot, normal_dot, sigmoid, DataRaceAware},
};
use num_traits::{AsPrimitive, Float, Zero};
use rayon::prelude::*;

use std::arch::asm;

#[inline(always)]
fn pdep(x: u64, mask: u64) -> u64 {
    let mut res;
    unsafe {
        asm!(
            "pdep {res}, {x}, {mask}",
            x = in(reg) x,
            mask = in(reg) mask,
            res = lateout(reg) res,
        )
    };
    res
}

struct GradientGrid<Target, Original> {
    depth: usize,
    target_dimension: usize,
    original_dimension: usize,
    gradients: DataRaceAware<Vec<Target>>,
    target_averages: DataRaceAware<Vec<Target>>,
    original_averages: DataRaceAware<Vec<Original>>,
    populations: Vec<AtomicUsize>,
    reverse_index: Vec<Vec<usize>>,
    index: Vec<usize>,
    min_values: Vec<Target>,
    max_values: Vec<Target>,
}

impl<Target, Original> GradientGrid<Target, Original>
where
    Target: Float + GenericFeature,
    Original: GenericFeature + Zero,
    usize: AsPrimitive<Target>,
    usize: AsPrimitive<Original>,
    Target: AsPrimitive<usize>,
    Original: AsPrimitive<usize>,
{
    fn new(depth: usize, target_dimension: usize, original_dimension: usize) -> Self {
        if target_dimension != 2 {
            unimplemented!("Only target_dimension == 2!");
        }

        let total_number_of_elements = Self::get_number_of_elements_before_layer(depth + 1);
        let grid_size = 2_u32.pow(2 * depth as u32) as usize;

        Self {
            depth,
            target_dimension,
            original_dimension,
            gradients: DataRaceAware::from(vec![
                Target::zero();
                target_dimension * total_number_of_elements
            ]),
            target_averages: DataRaceAware::from(vec![Target::zero(); 0]),
            original_averages: DataRaceAware::from(vec![Original::zero(); 0]),
            populations: (0..total_number_of_elements)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<AtomicUsize>>(),
            reverse_index: vec![Vec::new(); grid_size],
            index: Vec::new(),
            min_values: Vec::new(),
            max_values: Vec::new(),
        }
    }

    unsafe fn reset(&mut self) {
        // First of all, we reset the gradients to zero.
        (*self.gradients.get()).iter_mut().for_each(|v| {
            *v = Target::zero();
        });
        // Then we reset the populations.
        self.populations.iter().for_each(|v| {
            v.store(0, Ordering::Relaxed);
        });
        // The target averages.
        (*self.target_averages.get()).iter_mut().for_each(|v| {
            *v = Target::zero();
        });
        // The original averages.
        (*self.original_averages.get()).iter_mut().for_each(|v| {
            *v = Original::zero();
        });

        self.reverse_index.iter_mut().for_each(|r| r.clear());
    }

    /// Return the cordinates of the cell containing the provided point.
    ///
    /// # Safety
    /// If the minimum and maximum values bonds are not properly
    /// updated when this function is called, the index returned
    /// may cause out-of-bounds exceptions.
    fn get_cell_coordinates_unchecked(&self, x: Target, y: Target, layer: usize) -> (usize, usize) {
        let grid_side: Target = (2_usize.pow(layer as u32)).as_();
        (
            (grid_side
                * ((x - self.min_values[0])
                    / (Target::epsilon() + self.max_values[0] - self.min_values[0])))
                .floor()
                .min(grid_side - Target::one())
                .as_(),
            (grid_side
                * ((y - self.min_values[1])
                    / (Target::epsilon() + self.max_values[1] - self.min_values[1])))
                .floor()
                .min(grid_side - Target::one())
                .as_(),
        )
    }

    /// Returns number of elements before layer.
    fn get_number_of_elements_before_layer(layer: usize) -> usize {
        pdep((1 << layer as u64) - 1, 6148914691236517205) as usize - 1
    }

    /// Return the ID of the cell containing the provided point.
    ///
    /// # Safety
    /// If the minimum and maximum values bonds are not properly
    /// updated when this function is called, the index returned
    /// may cause out-of-bounds exceptions.
    fn get_relative_cell_id_unchecked(&self, x: Target, y: Target, layer: usize) -> usize {
        let (row_number, column_number) = self.get_cell_coordinates_unchecked(x, y, layer);

        (pdep(row_number as u64, 12297829382473034410)
            | pdep(column_number as u64, 6148914691236517205)) as usize
    }

    /// Return the ID of the cell containing the provided point.
    ///
    /// # Safety
    /// If the minimum and maximum values bonds are not properly
    /// updated when this function is called, the index returned
    /// may cause out-of-bounds exceptions.
    fn get_absolute_cell_id_unchecked(&self, x: Target, y: Target, layer: usize) -> usize {
        Self::get_number_of_elements_before_layer(layer)
            + self.get_relative_cell_id_unchecked(x, y, layer)
    }

    /// Return iterator on the child indices.
    fn iter_child_cells(&self, id: usize) -> impl Iterator<Item = usize> + '_ {
        id * 4..(id + 1) * 4
    }

    /// Return iterator on the child indices.
    fn iter_siblings_cells(&self, id: usize) -> impl Iterator<Item = usize> + '_ {
        self.iter_child_cells(id / 4)
            .filter(move |&sybling_id| sybling_id == id)
    }

    /// Return all far-away leafs IDs from given point.
    fn iter_far_away_leafs(&self, x: Target, y: Target) -> impl Iterator<Item = usize> + '_ {
        (1..self.depth).flat_map(move |layer| {
            self.iter_siblings_cells(self.get_absolute_cell_id_unchecked(x, y, layer))
        })
    }

    /// Return iterator on the child indices.
    fn iter_siblings(&self, x: Target, y: Target) -> impl Iterator<Item = usize> + '_ {
        self.reverse_index[self.get_relative_cell_id_unchecked(x, y, self.depth)]
            .iter()
            .copied()
    }

    /// Return all far-away leafs IDs and their properties.
    fn iter_mut_far_away_leafs_properties(
        &self,
        x: Target,
        y: Target,
    ) -> impl Iterator<Item = (&[Target], &[Original], &mut [Target], usize)> + '_ {
        self.iter_far_away_leafs(x, y).map(|id| unsafe {
            (
                &(*self.target_averages.get())
                    [id * self.target_dimension..(id + 1) * self.target_dimension],
                &(*self.original_averages.get())
                    [id * self.original_dimension..(id + 1) * self.original_dimension],
                &mut (*self.gradients.get())
                    [id * self.target_dimension..(id + 1) * self.target_dimension],
                self.populations[id].load(Ordering::Relaxed),
            )
        })
    }

    /// Propagates down to the leafs of the grid the partial gradients.
    fn downpropagate_gradient(&mut self) {
        (1..self.depth).for_each(|layer| {
            // We iterate on the elements of this layer.
            (Self::get_number_of_elements_before_layer(layer)
                ..Self::get_number_of_elements_before_layer(1 + layer))
                .into_par_iter()
                .map(|cell| unsafe {
                    (
                        cell,
                        &(*self.gradients.get())
                            [cell * self.target_dimension..(cell + 1) * self.target_dimension],
                    )
                })
                .for_each(|(cell, gradient)| unsafe {
                    self.iter_child_cells(cell).for_each(|child_cell| {
                        (*self.gradients.get())[child_cell * self.target_dimension
                            ..(child_cell + 1) * self.target_dimension]
                            .iter_mut()
                            .zip(gradient.iter().copied())
                            .for_each(|(cg, g)| {
                                *cg += g;
                            });
                    });
                });
        });
    }

    fn apply_gradient(&self, target_features: &mut [Target]) {
        target_features
            .par_chunks_mut(self.target_dimension)
            .zip(self.index.par_iter())
            .for_each(|(target_feature, index)| unsafe {
                (*self.gradients.get())
                    [index * self.target_dimension..(index + 1) * self.target_dimension]
                    .iter()
                    .copied()
                    .zip(target_feature.iter_mut())
                    .for_each(|(gradient, t)| {
                        *t += gradient;
                    });
            });
    }

    fn prepare(
        &mut self,
        target_features: &[Target],
        original_features: &[Original],
    ) -> Result<(), String> {
        // First we clean up the grid.
        unsafe { self.reset() };

        // We update the cells minimum and maximum values,
        // which define the borders of the cell.
        let (min_values, max_values) = target_features.matrix_min_max(self.target_dimension)?;
        self.min_values = min_values;
        self.max_values = max_values;

        // Now we start to iterate on the features, and we
        // update the various populations and averages.

        let (mut target_cell_sums, mut original_cell_sums) = target_features
            .par_chunks(self.target_dimension)
            .zip(original_features.par_chunks(self.original_dimension))
            .map(|(target_feature, original_feature)| {
                let cell_index = self.get_absolute_cell_id_unchecked(
                    target_feature[0],
                    target_feature[1],
                    self.depth,
                );
                self.populations[cell_index].fetch_add(1, Ordering::Relaxed);

                debug_assert!(cell_index < self.populations.len());

                (cell_index, target_feature, original_feature)
            })
            .fold(
                || {
                    (
                        vec![Target::zero(); self.populations.len() * self.target_dimension],
                        vec![Original::zero(); self.populations.len() * self.original_dimension],
                    )
                },
                |(mut partial_target_sum, mut partial_original_sum): (
                    Vec<Target>,
                    Vec<Original>,
                ),
                 (cell_index, target_feature, original_feature): (
                    usize,
                    &[Target],
                    &[Original],
                )| {
                    partial_target_sum[cell_index * self.target_dimension
                        ..(cell_index + 1) * self.target_dimension]
                        .iter_mut()
                        .zip(target_feature.iter().copied())
                        .for_each(|(p, v)| {
                            *p += v;
                        });

                    partial_original_sum[cell_index * self.original_dimension
                        ..(cell_index + 1) * self.original_dimension]
                        .iter_mut()
                        .zip(original_feature.iter().copied())
                        .for_each(|(p, v)| {
                            *p += v;
                        });

                    (partial_target_sum, partial_original_sum)
                },
            )
            .reduce(
                || {
                    (
                        vec![Target::zero(); self.populations.len() * self.target_dimension],
                        vec![Original::zero(); self.populations.len() * self.original_dimension],
                    )
                },
                |(mut partial_target_sum, mut partial_original_sum): (
                    Vec<Target>,
                    Vec<Original>,
                ),
                 (target_sum, original_sum): (Vec<Target>, Vec<Original>)| {
                    partial_target_sum
                        .iter_mut()
                        .zip(target_sum.into_iter())
                        .for_each(|(p, v)| {
                            *p += v;
                        });

                    partial_original_sum
                        .iter_mut()
                        .zip(original_sum.into_iter())
                        .for_each(|(p, v)| {
                            *p += v;
                        });

                    (partial_target_sum, partial_original_sum)
                },
            );

        // Now we update the total sums by dividing by the number
        // of elements in each cell, obtaining the cell average feature.
        let number_of_elements_before_last_layer =
            Self::get_number_of_elements_before_layer(self.depth);
        self.populations[number_of_elements_before_last_layer..]
            .par_iter()
            .map(|population| population.load(Ordering::Relaxed))
            .zip(
                target_cell_sums[number_of_elements_before_last_layer * self.target_dimension..]
                    .par_chunks_mut(self.target_dimension)
                    .zip(
                        original_cell_sums
                            [number_of_elements_before_last_layer * self.original_dimension..]
                            .par_chunks_mut(self.original_dimension),
                    ),
            )
            .filter(|(population, _)| *population > 0)
            .for_each(|(population, (target_cell_total, original_cell_total))| {
                let population_target: Target = (population as usize).as_();
                let population_original: Original = (population as usize).as_();
                target_cell_total.iter_mut().for_each(|v| {
                    *v /= population_target;
                });
                original_cell_total.iter_mut().for_each(|v| {
                    *v /= population_original;
                });
            });

        self.target_averages = DataRaceAware::from(target_cell_sums);
        self.original_averages = DataRaceAware::from(original_cell_sums);

        // We have now to up-propagate the population counts and the averages
        // up to the parent cells, as we currently only have them for the leafs.

        // We reverse the iteration as we need to start from the penultimate layer.
        (1..self.depth).rev().for_each(|layer| {
            // We iterate on the elements of this layer.
            (Self::get_number_of_elements_before_layer(layer)
                ..Self::get_number_of_elements_before_layer(1 + layer))
                .into_par_iter()
                .map(|cell| unsafe {
                    (
                        cell,
                        &mut (*self.target_averages.get())
                            [cell * self.target_dimension..(cell + 1) * self.target_dimension],
                        &mut (*self.original_averages.get())
                            [cell * self.original_dimension..(cell + 1) * self.original_dimension],
                    )
                })
                .for_each(|(cell, target_sum, original_sum)| unsafe {
                    self.iter_child_cells(cell)
                        .map(|cell| {
                            (
                                self.populations[cell].load(Ordering::Relaxed) as usize,
                                &(*self.target_averages.get())[cell * self.target_dimension
                                    ..(cell + 1) * self.target_dimension],
                                &(*self.original_averages.get())[cell * self.original_dimension
                                    ..(cell + 1) * self.original_dimension],
                            )
                        })
                        .for_each(|(population, target_average, original_average)| {
                            self.populations[cell].fetch_add(population, Ordering::Relaxed);
                            target_sum
                                .iter_mut()
                                .zip(target_average.iter().copied())
                                .for_each(|(s, v)| {
                                    *s += v * population.as_();
                                });
                            original_sum
                                .iter_mut()
                                .zip(original_average.iter().copied())
                                .for_each(|(s, v)| {
                                    *s += v * population.as_();
                                });
                        });

                    let total = self.populations[cell].load(Ordering::Relaxed) as usize;

                    if total.is_zero() {
                        return;
                    }

                    // Since now the values in these slices are the summations of
                    // the features of the child leafs, we need to divide them by the
                    // total population of the leaf to obtain the averages.
                    target_sum.iter_mut().for_each(|s| {
                        *s /= total.as_();
                    });
                    original_sum.iter_mut().for_each(|s| {
                        *s /= total.as_();
                    });
                });
        });

        target_features
            .chunks(self.target_dimension)
            .enumerate()
            .for_each(|(sample_index, target_feature)| {
                let index = self.get_relative_cell_id_unchecked(
                    target_feature[0],
                    target_feature[1],
                    self.depth,
                );
                self.reverse_index[index].push(sample_index);
            });

        self.index = target_features
            .par_chunks(self.target_dimension)
            .map(|feature| self.get_relative_cell_id_unchecked(feature[0], feature[1], self.depth))
            .collect::<Vec<usize>>();

        Ok(())
    }
}

#[derive(Clone)]
pub struct BarnesHutSigmoidDecomposition {
    decomposition: BasicIterativeDecomposition,
    depth: usize,
}

impl BarnesHutSigmoidDecomposition {
    pub fn new(decomposition: BasicIterativeDecomposition, depth: Option<usize>) -> Self {
        Self {
            decomposition,
            depth: depth.unwrap_or(3),
        }
    }
}

impl IterativeDecomposition for BarnesHutSigmoidDecomposition {
    fn get_iterative_basic_decomposition(&self) -> &BasicIterativeDecomposition {
        &self.decomposition
    }
}

impl DimensionalReduction for BarnesHutSigmoidDecomposition {
    fn fit_transform<Original, Target>(
        &self,
        mut target: &mut [Target],
        target_dimension: usize,
        original: &[Original],
        original_dimension: usize,
    ) -> Result<(), String>
    where
        Original: AsPrimitive<Target> + GenericFeature + Float,
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

        if target_dimension != 2 {
            return Err("Currently we only support 2.".to_string());
        }

        target.random_init(self.get_random_state());

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

        let mut grid: GradientGrid<Target, Original> =
            GradientGrid::new(self.depth, 2, original_dimension);
        let learning_rate: Target = self.get_learning_rate().as_();

        self.start_iterations()
            .map(|_| {
                grid.prepare(unsafe { *wrapped_target.get() }, original)?;
                original
                    .par_chunks(original_dimension)
                    .enumerate()
                    .map(|(sample_number, left_original_sample)| unsafe {
                        (
                            sample_number,
                            &mut (*wrapped_target.get())[(sample_number * target_dimension)
                                ..((sample_number + 1) * target_dimension)],
                            left_original_sample,
                        )
                    })
                    .map(
                        |(sample_number, left_target_sample, left_original_sample)| {
                            // First we iterate on the far away elements averages.
                            grid.iter_mut_far_away_leafs_properties(
                                left_target_sample[0],
                                left_target_sample[1],
                            )
                            .for_each(
                                |(
                                    cell_target_average,
                                    cell_original_average,
                                    gradient,
                                    population,
                                )| {
                                    let target_dot = dot(
                                        left_target_sample.iter().copied(),
                                        cell_target_average.iter().copied(),
                                    );
                                    let original_dot: Target = normal_dot(
                                        left_original_sample,
                                        cell_original_average,
                                        &mean,
                                        &variance,
                                    )
                                    .as_();
                                    let mut variation = sigmoid(target_dot) - sigmoid(original_dot);
                                    variation *= learning_rate;
                                    left_target_sample
                                        .iter_mut()
                                        .zip(cell_target_average.iter().zip(gradient.iter_mut()))
                                        .for_each(|(left, (right, gradient))| {
                                            let left_tmp = *left;
                                            *left -= *right * variation * population.as_();
                                            *gradient -= left_tmp * variation;
                                        });
                                },
                            );

                            // First we iterate on the far away elements averages.
                            grid.iter_siblings(left_target_sample[0], left_target_sample[1])
                                .filter(|&sibling_id| sibling_id != sample_number)
                                .map(|sibling_id| unsafe {
                                    (
                                        &mut (*wrapped_target.get())[(sibling_id * target_dimension)
                                            ..((sibling_id + 1) * target_dimension)],
                                        &original[(sibling_id * original_dimension)
                                            ..((sibling_id + 1) * original_dimension)],
                                    )
                                })
                                .for_each(|(sibling_target, sibling_original)| {
                                    let target_dot = dot(
                                        left_target_sample.iter().copied(),
                                        sibling_target.iter().copied(),
                                    );
                                    let original_dot: Target = normal_dot(
                                        left_original_sample,
                                        sibling_original,
                                        &mean,
                                        &variance,
                                    )
                                    .as_();
                                    let mut variation = sigmoid(target_dot) - sigmoid(original_dot);
                                    variation *= learning_rate;
                                    left_target_sample
                                        .iter_mut()
                                        .zip(sibling_target.iter_mut())
                                        .for_each(|(left, right)| {
                                            let left_tmp = *left;
                                            *left -= *right * variation;
                                            *right -= left_tmp * variation;
                                        });
                                });

                            Ok(())
                        },
                    )
                    .collect::<Result<(), String>>()?;
                grid.downpropagate_gradient();
                grid.apply_gradient(unsafe { &mut *wrapped_target.get() });
                Ok(())
            })
            .collect::<Result<(), String>>()
    }
}
