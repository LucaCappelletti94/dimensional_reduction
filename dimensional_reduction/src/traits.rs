use num_traits::AsPrimitive;

pub trait DimensionalReduction {
    fn get_target_dimension(&self) -> usize;
    
    fn fit_transform<Original, Target>(
        &self,
        target: &mut [Target],
        original: &[Original],
    ) -> Result<(), String>
    where
        Original: Copy + 'static,
        Target: AsPrimitive<Original>;
}
