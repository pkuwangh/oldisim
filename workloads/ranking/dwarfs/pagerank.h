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

#ifndef PAGERANK_H
#define PAGERANK_H

#include <cstdint>

#include <gapbs/src/graph.h>
#include <gapbs/src/pvector.h>

namespace ranking {
namespace dwarfs {

class PageRank {
public:
  constexpr static const float kDamp = 0.85;

  explicit PageRank(int scale, int degrees);

  pvector<float> rank(int max_iters, double epsilon);

private:
  CSRGraph<int32_t> graph_;
  int scale_;
  int degrees_;
};

} // namespace dwarfs
} // namespace ranking

#endif // PAGERANK_H
