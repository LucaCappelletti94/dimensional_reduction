#![feature(return_position_impl_trait_in_trait)]

pub mod basic_decomposition;
pub mod basic_iterative_decomposition;
pub mod macros;
pub mod numpy_decomposition;
pub mod sigmoid_decomposition;
pub mod traits;
pub mod utils;

use pyo3::{pymodule, types::PyModule, PyResult, Python};
pub use sigmoid_decomposition::SigmoidDecomposition;

#[pymodule]
pub fn dimensional_reduction(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SigmoidDecomposition>()?;
    Ok(())
}
