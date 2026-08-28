use pyo3::prelude::*;

mod algorithms;

/// Python module implemented in Rust.
/// Exposes selected Rust functions to Python.
#[pymodule]
fn _k_center(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(algorithms::gonzalez::fit, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::gonzalez::predict, m)?)?;
    Ok(())
}