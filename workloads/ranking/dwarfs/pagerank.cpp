// Copyright (c) 2015, The Regents of the University of California (Regents).
// All Rights Reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of the Regents nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL REGENTS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "pagerank.h"
#include <string>

#include <gapbs/src/benchmark.h>
#include <gapbs/src/command_line.h>

namespace ranking {
namespace dwarfs {

PageRank::PageRank(int scale, int degrees) : scale_(scale), degrees_(degrees) {
  char *argv[4] = {(char *)"pr", (char *)"-u", (char *)std::to_string(scale_).c_str()};
  CLPageRank cli{3, argv, "pagerank", 1e-4, 20};
  Builder b{cli};
  graph_ = std::move(b.MakeGraph());
}

/** PageRank implementation taken from
* http://gap.cs.berkeley.edu/benchmark.html
*/
pvector<float> PageRank::rank(int max_iters, double epsilon) {
  const float init_score = 1.0f / graph_.num_nodes();
  const float base_score = (1.0f - kDamp) / graph_.num_nodes();
  pvector<float> scores(graph_.num_nodes(), init_score);
  pvector<float> outgoing_contrib(graph_.num_nodes());
  for (int iter = 0; iter < max_iters; iter++) {
    double error = 0;

#pragma omp parallel for
    for (NodeID n = 0; n < graph_.num_nodes(); n++)
      outgoing_contrib[n] = scores[n] / graph_.out_degree(n);

#pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
    for (NodeID u = 0; u < graph_.num_nodes(); u++) {
      float incoming_total = 0;
      for (NodeID v : graph_.in_neigh(u))
        incoming_total += outgoing_contrib[v];
      float old_score = scores[u];
      scores[u] = base_score + kDamp * incoming_total;
      error += fabs(scores[u] - old_score);
    }
    if (error < epsilon)
      break;
  }
  return scores;
}

} // namespace dwarfs
} // namespace ranking
