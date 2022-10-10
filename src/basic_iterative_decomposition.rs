use crate::basic_decomposition::BasicDecomposition;

#[derive(Clone)]
pub struct BasicIterativeDecomposition {
    pub(crate) basic_decomposition: BasicDecomposition,
    pub(crate) iterations: usize,
    pub(crate) learning_rate: f32,
}

impl BasicIterativeDecomposition {
    pub fn new(
        iterations: Option<usize>,
        learning_rate: Option<f32>,
        model_name: &str,
        random_state: Option<u64>,
        verbose: Option<bool>,
    ) -> Result<Self, String> {
        Ok(Self {
            basic_decomposition: BasicDecomposition::new(model_name, random_state, verbose)?,
            iterations: iterations.unwrap_or(100),
            learning_rate: learning_rate.unwrap_or(0.01),
        })
    }
}
