use dimensional_reduction::traits::DimensionalReduction;
use crate::*;
use numpy::PyArray2;
use crate::traits::*;
use pyo3::exceptions::PyValueError;
use pyo3::{FromPyObject, IntoPy, Py, PyAny, PyResult};

pub trait NumpyDecomposition<T> where T: DimensionalReduction {
    fn fit_transform_binding(
        &self,
        matrix: Py<PyAny>,
        number_of_dimensions: Option<usize>,
        dtype: Option<&str>,
    ) -> PyResult<Py<PyAny>>;
}

macro_rules! impl_numpy_decompositions {
    ($($dtype:ty),*) => {
        /// Returns cosine similarity of the provided source and destinations using the provided features.
        ///
        /// Arguments
        /// ------------
        /// matrix: np.ndarray
        ///     2D Matrix containing the feaures.
        ///
        fn fit_transform_binding(&self, matrix: Py<PyAny>, number_of_dimensions: Option<usize>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
            let gil = pyo3::Python::acquire_gil();
            let matrix = matrix.as_ref(gil.python());
            $(
                if let Ok(matrix) = <&PyArray2<$dtype>>::extract(&matrix) {

                    if !matrix.is_c_contiguous(){
                        return pe!(Err(
                            concat!(
                                "The provided vector is not a contiguos vector in ",
                                "C orientation."
                            )
                        ));
                    }

                    let matrix_ref = unsafe { matrix.as_slice().unwrap() };
                    let number_of_samples = matrix.shape()[0];
                    let number_of_features = matrix.shape()[1];
                    let number_of_dimensions = number_of_dimensions.unwrap_or(2);

                    match dtype.unwrap_or("f32") {
                        // "f16" => {
                        //     let target = unsafe { PyArray2::new(gil.python(), [number_of_samples, number_of_dimensions], false) };
                        //     let target_ref: &mut [f16] = unsafe { target.as_slice_mut().unwrap() };

                        //     pe!(self.get_basic_dimensionality_reduction().fit_transform(
                        //         target_ref,
                        //         number_of_dimensions,
                        //         matrix_ref,
                        //         number_of_features,
                        //     ))?;

                        //     return Ok(target.to_owned().into_py(gil.python()));
                        // },
                        "f32" => {
                            let target = unsafe { PyArray2::new(gil.python(), [number_of_samples, number_of_dimensions], false) };
                            let target_ref: &mut [f32] = unsafe { target.as_slice_mut().unwrap() };

                            pe!(self.get_basic_dimensionality_reduction().fit_transform(
                                target_ref,
                                number_of_dimensions,
                                matrix_ref,
                                number_of_features,
                            ))?;

                            return Ok(target.to_owned().into_py(gil.python()));
                        },
                        "f64" => {
                            let target = unsafe { PyArray2::new(gil.python(), [number_of_samples, number_of_dimensions], false) };
                            let target_ref: &mut [f64] = unsafe { target.as_slice_mut().unwrap() };

                            pe!(self.get_basic_dimensionality_reduction().fit_transform(
                                target_ref,
                                number_of_dimensions,
                                matrix_ref,
                                number_of_features,
                            ))?;

                            return Ok(target.to_owned().into_py(gil.python()));
                        },
                        dtype => {
                            pe!(Err(
                                format!(
                                    "The data type {} is not supported.",
                                    dtype
                                )
                            ))?;
                        }
                    }
                }
            )*

            pe!(Err(concat!(
                "The provided features are not supported ",
                "in the cosine similarity computation!"
            )
            .to_string()))
        }
    };
}

impl<M, T> NumpyDecomposition<T> for M
where
    M: DimensionalReductionBinding<T>,
    T: DimensionalReduction
{
    impl_numpy_decompositions! {
        //u8, u16, u32, u64, i8, i16, i32, i64,
        f32, f64
    }
}
