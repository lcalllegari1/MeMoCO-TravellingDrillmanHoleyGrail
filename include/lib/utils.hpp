#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>

#include "lib/types.hpp"

namespace utils {

  void printm(const Matrix<f64> &matrix, size_t precision = 2);
  void printv(const std::vector<u16> &vector);

  void rev_subseq(Tour &tour, size_t start, size_t end);
  
  size_t gcd(size_t a, size_t b);

  size_t get_step(size_t n_nodes);
}

#endif // UTILS_HPP
