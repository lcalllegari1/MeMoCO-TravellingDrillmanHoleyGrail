#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <string>

#include "lib/utils.hpp"

namespace utils {
  void printm(const Matrix<f64>& matrix, size_t precision) {
    std::vector<f64> mat = matrix.data;

    if (mat.empty()) return;

    size_t n_elements = mat.size();
    size_t dim = static_cast<size_t>(std::sqrt(n_elements));

    if (dim * dim != n_elements) {
      std::cerr << "Error: Vector size " << n_elements 
                << " is not a perfect square. Cannot infer matrix dimensions." << std::endl;
      return;
    }
    size_t max_idx_width = std::to_string(dim - 1).length();
    
    size_t max_val_width = 0;
    for (const auto& val : mat) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(precision) << val;
      max_val_width = std::max(max_val_width, ss.str().length());
    }
    size_t col_width = std::max(max_val_width, max_idx_width) + 2;
    std::cout << std::string(max_idx_width, ' ') << " |";

    for (size_t c = 0; c < dim; ++c) {
      std::cout << std::setw(col_width) << c;
    }
    std::cout << "\n";
    std::cout << std::string(max_idx_width, '-') << "-+";
    std::cout << std::string(dim * col_width, '-') << "\n";

    for (size_t r = 0; r < dim; ++r) {
      std::cout << std::setw(max_idx_width) << r << " |";

      for (size_t c = 0; c < dim; ++c) {
        size_t idx = r * dim + c;
        std::cout << std::fixed << std::setprecision(precision) 
                  << std::setw(col_width) << mat[idx];
      }
      std::cout << "\n";
    }
  }

  void printv(const std::vector<u16>& vector) {
    std::cout << "[";
    for (size_t i = 0; i < vector.size(); ++i) {
      std::cout << vector[i];
      if (i < vector.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]\n";
  }

  void rev_subseq(Tour &tour, size_t start, size_t end) {
    std::reverse(tour.begin() + start, tour.begin() + end + 1);
  }

  size_t gcd(size_t a, size_t b) {
    size_t t;
    while (b != 0) {
      t = b;
      b = a % b;
      a = t;
    }
    return a;
  }

  size_t get_step(size_t n_nodes) {
    if (n_nodes <= 1) {
      return 1;
    }

    size_t step = static_cast<size_t>(n_nodes * 0.61803398875);
    while (gcd(step, n_nodes) > 1) {
      step++;
    }

    return step;
  }
}
