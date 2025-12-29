#include <pybind11/pybind11.h> 
#include <pybind11/numpy.h> 
#include <iostream>
#include <omp.h>
namespace py = pybind11;

py::array_t<double> csr_matvec(
    py::array_t<double> values,
    py::array_t<int> cols,
    py::array_t<int> row_ptr,
    py::array_t<double> x    
){

    auto v = values.unchecked<1>();
    auto c = cols.unchecked<1>();
    auto r = row_ptr.unchecked<1>();
    auto vec = x.unchecked<1>();

    int n_rows = row_ptr.shape(0) - 1;
    py::array_t<double> result = py::array_t<double>({n_rows}, {8});
    auto y = result.mutable_unchecked<1>();
    
    std::cout << "DEBUG: n_rows = " << n_rows << std::endl;
    std::cout << "DEBUG: result shape = " << result.shape(0) << std::endl;
    std::cout << "DEBUG: result stride = " << result.strides(0) << std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i<n_rows; i++)
    {
        double res = 0;

        //row by row
        int rowstart = r(i);
        int rowend = r(i+1);

        std::cout << "Row " << i << ": start=" << rowstart << ", end=" << rowend << std::endl;
        for(int j = rowstart; j<rowend; j++)
        {
            std::cout << v(j)*vec(c(j))<<std::endl;
            res+=v(j)*vec(c(j));
        }
        std::cout << "  Writing " << res << " to index " << i << " at address " << (void*)&y[i] << std::endl;
        y(i) = res;
    }

    return result;
}



PYBIND11_MODULE(my_sparse, m) {
    m.doc() = "Custome CSR Implementation";
    m.def("csr_matvec", &csr_matvec, "A function to multiply CSR matrix with vector");
}