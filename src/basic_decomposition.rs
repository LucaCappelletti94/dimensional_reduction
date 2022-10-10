#[derive(Clone)]
pub struct BasicDecomposition {
    pub(crate) model_name: String,
    pub(crate) random_state: u64,
    pub(crate) verbose: bool,
}

impl BasicDecomposition {
    pub fn new(
        model_name: &str,
        random_state: Option<u64>,
        verbose: Option<bool>,
    ) -> Result<Self, String> {
        if model_name.is_empty() {
            return Err("The provided model name is empty.".to_string());
        }

        Ok(Self {
            model_name: model_name.to_string(),
            random_state: random_state.unwrap_or(42),
            verbose: verbose.unwrap_or(true),
        })
    }
}
