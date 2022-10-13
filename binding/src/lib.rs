#![feature(return_position_impl_trait_in_trait)]

pub mod macros;
pub mod numpy_decomposition;
pub mod sigmoid_decomposition;
pub mod sampled_sigmoid_decomposition;
pub mod barnes_hut_sigmoid_decomposition;
pub mod traits;

use pyo3::{pymodule, types::PyModule, PyResult, Python};
pub use sigmoid_decomposition::SigmoidDecomposition;
pub use sampled_sigmoid_decomposition::SampledSigmoidDecomposition;
pub use barnes_hut_sigmoid_decomposition::BarnesHutSigmoidDecomposition;

#[pymodule]
pub fn dimensional_reduction(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SigmoidDecomposition>()?;
    m.add_class::<BarnesHutSigmoidDecomposition>()?;
    m.add_class::<SampledSigmoidDecomposition>()?;
    Ok(())
}
