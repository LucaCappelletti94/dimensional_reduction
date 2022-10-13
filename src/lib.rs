#![feature(return_position_impl_trait_in_trait)]

pub mod barnes_hut_sigmoid_decomposition;
pub mod basic_decomposition;
pub mod basic_iterative_decomposition;
pub mod sampled_sigmoid_decomposition;
pub mod sigmoid_decomposition;
pub mod traits;
pub mod utils;

pub use barnes_hut_sigmoid_decomposition::*;
pub use sigmoid_decomposition::*;
pub use sampled_sigmoid_decomposition::*;