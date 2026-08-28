use pyo3::prelude::*;

mod algorithms;

/// A Python module implemented in Rust.
#[pymodule]
fn _k_center(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(algorithms::gonzalez::gonzalez, m)?)?;
    Ok(())
}