use dimensional_reduction::SigmoidDecomposition as SigmoidDecompositionRust;

impl FromPyDict for SigmoidDecompositionRust {
    fn from_pydict(py_kwargs: Option<&types::PyDict>) -> PyResult<Self>
    where
        Self: Sized,
    {
        let py = pyo3::Python::acquire_gil();
        let kwargs = normalize_kwargs!(py_kwargs, py.python());

        Ok(Self {
            decomposition: pe!(BasicIterativeDecomposition::new(
                extract_value_rust_result!(kwargs, "iterations", usize),
                extract_value_rust_result!(kwargs, "learning_rate", f32),
                "Sigmoid Decomposition",
                extract_value_rust_result!(kwargs, "random_state", u64),
                extract_value_rust_result!(kwargs, "verbose", bool),
            ))?,
        })
    }
}

///
#[pyclass]
#[derive(Clone)]
#[pyo3(text_signature = "(*, iterations, learning_rate, random_state, verbose)")]
pub struct SigmoidDecomposition {
    inner: SigmoidDecompositionRust,
}

#[pymethods]
impl SigmoidDecomposition {
    #[new]
    #[args(py_kwargs = "**")]
    /// Return a new instance of the Sigmoid Decomposition model.
    ///
    /// Parameters
    /// ------------------------
    /// random_state: int = 42
    ///     The random state to reproduce the model initialization and training. By default, 42.
    pub fn new(py_kwargs: Option<&PyDict>) -> PyResult<Self> {
        Ok(Self {
            inner: Self::from_pydict(py_kwargs)?,
        })
    }

    #[pyo3(text_signature = "($self, matrix, number_of_dimensions, dtype)")]
    pub fn fit_transform(
        &self,
        matrix: Py<PyAny>,
        number_of_dimensions: Option<usize>,
        dtype: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        self.inner.fit_transform(matrix, number_of_dimensions, dtype)
    }
}
