# Toy-CSR: Custom CSR Matrix-Vector Multiplication

A toy implementation of sparse matrix-vector multiplication using the Compressed Sparse Row (CSR) format, with Python bindings via pybind11.

## Overview

This is a learning project to explore building C++ extensions for Python. It implements basic CSR matrix-vector multiplication and demonstrates OpenMP parallelization.

## What it does

- Implements CSR sparse matrix format
- Exposes C++ function to Python via pybind11
- Uses OpenMP for basic parallelization
- Includes compiler optimizations for fun

## Requirements

- CMake (≥ 3.15)
- C++ compiler with C++11 support
- Python 3.x
- pybind11
- OpenMP
- NumPy and SciPy (for testing)

## Building

Create a build directory and compile:
```bash
mkdir -p build
cd build
cmake ..
make
```

This will generate the `my_sparse` Python module.

## Usage

```python
import numpy as np
import my_sparse

# Define CSR matrix components
values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float64)
cols = np.array([0, 3, 1, 3, 2, 3, 0], dtype=np.int32)
row_ptr = np.array([0, 2, 4, 6, 7], dtype=np.int32)

# Vector to multiply
x = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

# Perform CSR matrix-vector multiplication
result = my_sparse.csr_matvec(values, cols, row_ptr, x)
print("Result:", result)
```

## CSR Format

The Compressed Sparse Row format stores a sparse matrix using three arrays:

- **values**: Non-zero elements of the matrix
- **cols**: Column indices for each non-zero element
- **row_ptr**: Pointers to the start of each row in the values array

For example, the matrix:
```
[ 10  0  0 20 ]
[  0 30  0 40 ]
[  0  0 50 60 ]
[ 70  0  0  0 ]
```

Is stored as:
- `values = [10, 20, 30, 40, 50, 60, 70]`
- `cols = [0, 3, 1, 3, 2, 3, 0]`
- `row_ptr = [0, 2, 4, 6, 7]`

## Testing

Run the included test script from the build directory:
```bash
cd build
python3 test.py
```

The test compares the custom implementation against SciPy's CSR matrix operations.

## Notes

- **Stride Bug**: Earlier versions had a stride=0 issue when creating NumPy arrays. Fixed by explicitly specifying strides: `py::array_t<double>({n_rows}, {sizeof(double)})`
- Debug output is still in the code
- This is a toy project for learning purposes - not intended for production use!
