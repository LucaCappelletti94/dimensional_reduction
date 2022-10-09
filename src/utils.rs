use std::{
    cell::UnsafeCell,
    iter::Sum,
    ops::{Add, Mul},
};

use crate::traits::GenericFeature;

pub fn dot<I1, I2, E>(left: I1, right: I2) -> E
where
    I1: Iterator<Item = E>,
    I2: Iterator<Item = E>,
    E: Mul<E, Output = E> + Add<E, Output = E> + Sum<E>,
{
    left.zip(right).map(|(l, r)| l * r).sum::<E>()
}

pub fn sigmoid<F>(x: F) -> F
where
    F: GenericFeature,
{
    F::one() / (F::one() + (-x).exp())
}

pub struct DataRaceAware<T>
where
    T: ?Sized,
{
    pub(crate) value: UnsafeCell<T>,
}

impl<T> From<T> for DataRaceAware<T> {
    fn from(value: T) -> Self {
        Self {
            value: UnsafeCell::from(value),
        }
    }
}

impl<T> DataRaceAware<T> {
    pub fn get(&self) -> *mut T {
        self.value.get()
    }

    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }
}

unsafe impl<T> Sync for DataRaceAware<T> {}
unsafe impl<T> Send for DataRaceAware<T> {}
