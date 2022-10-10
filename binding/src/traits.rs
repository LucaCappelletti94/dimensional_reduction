use pyo3::types::PyDict;
use pyo3::PyResult;


pub trait FromPyDict {
    fn from_pydict(py_kwargs: Option<&PyDict>) -> PyResult<Self>
    where
        Self: Sized;
}
