#![feature(return_position_impl_trait_in_trait)]

pub mod macros;
pub mod numpy_decomposition;
pub mod sigmoid_decomposition;
pub mod traits;

use pyo3::{pymodule, types::PyModule, PyResult, Python};
pub use sigmoid_decomposition::SigmoidDecomposition;

#[pymodule]
pub fn dimensional_reduction(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SigmoidDecomposition>()?;
    Ok(())
}
