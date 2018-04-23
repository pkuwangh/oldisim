
#include "TriadStream.h"

namespace search {

TriadStream::TriadStream(size_t num_elems)
    : data_a_(num_elems), data_b_(num_elems), data_c_(num_elems), scalar_(3.) {

    for (size_t i = 0; i < num_elems; i++) {
        data_a_[i] = 1.;
        data_b_[i] = 2.;
        data_c_[i] = 0.;
    }

}


void TriadStream::Triad(size_t num_iterations) {
    for (size_t i = 0; i < num_iterations; i++) {
        std::vector<double> tmp = data_a_;
        data_a_ = data_b_;
        data_b_ = data_c_;
        data_c_ = tmp;

        for (size_t index = 0; index < data_a_.size(); index++) {
            data_a_[index] = data_b_[index] + scalar_ * data_c_[index];
        }

    }
}

void TriadStream::Copy(size_t num_iterations) {
    for (size_t i = 0; i < num_iterations; i++) {
        std::vector<double> tmp = data_a_;
        data_a_ = data_b_;
        data_b_ = data_c_;
        data_c_ = tmp;

        for (size_t index = 0; index < data_a_.size(); index++) {
           data_c_[index] = data_a_[index];
       }
    }
}
}  // namespace search
