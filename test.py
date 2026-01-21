import numpy as np
import my_sparse
from scipy.sparse import csr_matrix

# 1. Setup the test data
# This is the same 4x4 matrix example we discussed:
# [ 10  0  0 20 ]
# [  0 30  0 40 ]
# [  0  0 50 60 ]
# [ 70  0  0  0 ]

values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float64)
cols   = np.array([0, 3, 1, 3, 2, 3, 0], dtype=np.int32)
rows   = np.array([0, 2, 4, 6, 7],       dtype=np.int32)

# The vector to multiply (x)
x = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

# 2. Run YOUR C++ Implementation
print("--- Running Custom C++ Implementation ---")
try:
    my_result = my_sparse.csr_matvec(values, cols, rows, x)
    print("Result vector y = ", my_result)
except Exception as e:
    print(f"Crashed! Error: {e}")
    exit(1)

# 3. Validation: Compare against Scipy (The Gold Standard)
print("\n--- Validating against Scipy ---")
# Reconstruct the scipy matrix object
scipy_mat = csr_matrix((values, cols, rows), shape=(4, 4))
true_result = scipy_mat.dot(x)

print(f"My Result:    {my_result}")
print(f"Scipy Result: {true_result}")

# 4. Final Check
if np.allclose(my_result, true_result):
    print("\n✅ SUCCESS: logic is correct!")
else:
    print("\n❌ FAILURE: The numbers do not match.")