use std::{
    cell::UnsafeCell,
    iter::Sum,
    ops::{Add, Div, Mul, Sub},
};

use num_traits::Float;

pub fn dot<I1, I2, E>(left: I1, right: I2) -> E
where
    I1: Iterator<Item = E>,
    I2: Iterator<Item = E>,
    E: Mul<E, Output = E> + Add<E, Output = E> + Sum<E>,
{
    left.zip(right).map(|(l, r)| l * r).sum::<E>()
}

pub fn normal_dot<E>(left: &[E], right: &[E], mean: &[E], std: &[E]) -> E
where
    E: Mul<E, Output = E>
        + Div<E, Output = E>
        + Sub<E, Output = E>
        + Add<E, Output = E>
        + Sum<E>
        + Copy,
{
    dot(
        left.iter()
            .copied()
            .zip(mean.iter().copied().zip(std.iter().copied()))
            .map(|(l, (m, s))| (l - m) / s),
        right
            .iter()
            .copied()
            .zip(mean.iter().copied().zip(std.iter().copied()))
            .map(|(l, (m, s))| (l - m) / s),
    )
}

pub fn sigmoid<F>(x: F) -> F
where
    F: Float,
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
