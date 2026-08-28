use pyo3::prelude::*;

#[pyfunction]
pub fn gonzalez() -> PyResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {}