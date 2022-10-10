
#[macro_export]
macro_rules! pe {
    ($value: expr) => {
        ($value).map_err(|err| PyValueError::new_err(err))
    };
}

#[macro_export]
macro_rules! normalize_kwargs {
    ($kwargs: expr, $py: expr) => {
        match $kwargs {
            Some(v) => v,
            None => PyDict::new($py),
        }
    };
}

#[macro_export]
macro_rules! extract_value_rust_result {
    ($kwargs: ident, $key: literal, $_type: ty) => {
        pe!($kwargs
            .get_item($key)
            .map_or(Ok::<_, PyErr>(None), |value| {
                Ok(if value.get_type().name().unwrap() == "NoneType" {
                    None
                } else {
                    Some(pe!(value.extract::<$_type>().map_err(|_| {
                        format!(
                            "The value passed as parameter {} cannot be casted from {} to {}.",
                            $key,
                            value.get_type().name().unwrap(),
                            stringify!($_type)
                        )
                    }))?)
                })
            }))?
    };
}