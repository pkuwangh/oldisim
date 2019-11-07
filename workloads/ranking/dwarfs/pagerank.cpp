// Copyright (c) 2019-present, Facebook, Inc. and its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
  graph_ = std::move(b.MakeGraph(scale_, degrees_, true));
}

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
