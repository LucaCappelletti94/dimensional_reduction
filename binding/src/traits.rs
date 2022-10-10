use dimensional_reduction::traits::DimensionalReduction;
use pyo3::types::PyDict;
use pyo3::PyResult;


pub trait FromPyDict {
    fn from_pydict(py_kwargs: Option<&PyDict>) -> PyResult<Self>
    where
        Self: Sized;
}


pub trait DimensionalReductionBinding<T> where T: DimensionalReduction {
    fn get_basic_dimensionality_reduction(&self) -> &T;
}
