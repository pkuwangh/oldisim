#ifndef TRIAD_STREAM_H
#define TRIAD_STREAM_H

#include <cstdlib>
#include <stdint.h>

#include <vector>

namespace search {

class TriadStream {
 public:
  explicit TriadStream(size_t num_elems);
  void Triad(size_t num_iterations);
  void Copy(size_t num_iterations);


 private:
  std::vector<double> data_a_;
  std::vector<double> data_b_;
  std::vector<double> data_c_;
  double scalar_;
};
}  // namespace search

#endif  // TRIAD_STREAM_H
