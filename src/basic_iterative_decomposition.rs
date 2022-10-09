use crate::basic_decomposition::BasicDecomposition;

pub struct BasicIterativeDecomposition {
    pub(crate) basic_decomposition: BasicDecomposition,
    pub(crate) iterations: usize,
    pub(crate) learning_rate: f32
}