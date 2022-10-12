use crate::numpy_decomposition::NumpyDecomposition;
use crate::traits::*;
use crate::*;
use dimensional_reduction::basic_iterative_decomposition::BasicIterativeDecomposition;
use dimensional_reduction::BarnesHutSigmoidDecomposition as BarnesHutSigmoidDecompositionRust;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;
use pyo3::*;

impl FromPyDict for BarnesHutSigmoidDecompositionRust {
    fn from_pydict(py_kwargs: Option<&types::PyDict>) -> PyResult<Self>
    where
        Self: Sized,
    {
        let py = pyo3::Python::acquire_gil();
        let kwargs = normalize_kwargs!(py_kwargs, py.python());

        Ok(Self::new(
            pe!(BasicIterativeDecomposition::new(
                extract_value_rust_result!(kwargs, "iterations", usize),
                extract_value_rust_result!(kwargs, "learning_rate", f32),
                "Barnes-Hut Sigmoid Decomposition",
                extract_value_rust_result!(kwargs, "random_state", u64),
                extract_value_rust_result!(kwargs, "verbose", bool),
            ))?,
            extract_value_rust_result!(kwargs, "depth", usize),
        ))
    }
}

///
#[pyclass]
#[derive(Clone)]
#[pyo3(text_signature = "(*, iterations, learning_rate, random_state, verbose)")]
pub struct BarnesHutSigmoidDecomposition {
    inner: BarnesHutSigmoidDecompositionRust,
}

impl DimensionalReductionBinding<BarnesHutSigmoidDecompositionRust>
    for BarnesHutSigmoidDecomposition
{
    fn get_basic_dimensionality_reduction(&self) -> &BarnesHutSigmoidDecompositionRust {
        &self.inner
    }
}

#[pymethods]
impl BarnesHutSigmoidDecomposition {
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
            inner: BarnesHutSigmoidDecompositionRust::from_pydict(py_kwargs)?,
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
