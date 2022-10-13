use crate::*;
use crate::traits::*;
use crate::numpy_decomposition::NumpyDecomposition;
use dimensional_reduction::basic_iterative_decomposition::BasicIterativeDecomposition;
use dimensional_reduction::SampledSigmoidDecomposition as SampledSigmoidDecompositionRust;
use pyo3::types::PyDict;
use pyo3::*;
use pyo3::exceptions::PyValueError;

impl FromPyDict for SampledSigmoidDecompositionRust {
    fn from_pydict(py_kwargs: Option<&types::PyDict>) -> PyResult<Self>
    where
        Self: Sized,
    {
        let py = pyo3::Python::acquire_gil();
        let kwargs = normalize_kwargs!(py_kwargs, py.python());

        Ok(Self::from(pe!(BasicIterativeDecomposition::new(
                extract_value_rust_result!(kwargs, "iterations", usize),
                extract_value_rust_result!(kwargs, "learning_rate", f32),
                "Sampled Sigmoid Decomposition",
                extract_value_rust_result!(kwargs, "random_state", u64),
                extract_value_rust_result!(kwargs, "verbose", bool),
            ))?,
        ))
    }
}

///
#[pyclass]
#[derive(Clone)]
#[pyo3(text_signature = "(*, iterations, learning_rate, random_state, verbose)")]
pub struct SampledSigmoidDecomposition {
    inner: SampledSigmoidDecompositionRust,
}

impl DimensionalReductionBinding<SampledSigmoidDecompositionRust> for SampledSigmoidDecomposition {
    fn get_basic_dimensionality_reduction(&self) -> &SampledSigmoidDecompositionRust {
        &self.inner
    }
}

#[pymethods]
impl SampledSigmoidDecomposition {
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
            inner: SampledSigmoidDecompositionRust::from_pydict(py_kwargs)?,
        })
    }

    #[pyo3(text_signature = "($self, matrix, number_of_dimensions, dtype)")]
    pub fn fit_transform(
        &self,
        matrix: Py<PyAny>,
        number_of_dimensions: Option<usize>,
        dtype: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        self.fit_transform_binding(matrix, number_of_dimensions, dtype)
    }
}
