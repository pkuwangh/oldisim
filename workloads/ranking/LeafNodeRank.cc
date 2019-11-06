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

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include "oldisim/LeafNodeServer.h"
#include "oldisim/NodeThread.h"
#include "oldisim/ParentConnection.h"
#include "oldisim/QueryContext.h"
#include "oldisim/Util.h"

#include "LeafNodeRankCmdline.h"
#include "RequestTypes.h"

#include "gen-cpp/ranking_types.h"

using apache::thrift::protocol::TBinaryProtocol;
using apache::thrift::transport::TMemoryBuffer;

// Shared configuration flags
static gengetopt_args_info args;

static const int kMaxResponseSize = 1 << 12;
const int kNumNops = 6;
const int kNumNopIterations = 60;

struct ThreadData {
  std::default_random_engine rng;
  std::gamma_distribution<double> latency_distribution;
  std::string random_string;
};

void ThreadStartup(oldisim::NodeThread &thread,
                   std::vector<ThreadData> &thread_data) {
  auto &this_thread = thread_data[thread.get_thread_num()];
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  this_thread.rng.seed(seed);

  const double alpha = 0.7;
  const double beta = 20000;
  this_thread.latency_distribution =
      std::gamma_distribution<double>(alpha, beta);

  this_thread.random_string = RandomString(kMaxResponseSize);
}

void PageRankRequestHandler(oldisim::NodeThread &thread,
                            oldisim::QueryContext &context,
                            std::vector<ThreadData> &thread_data) {
  auto &this_thread = thread_data[thread.get_thread_num()];
  int num_iterations = this_thread.latency_distribution(this_thread.rng);
  for (int i = 0; i < num_iterations; i++) {
    for (int j = 0; j < kNumNopIterations; j++) {
      for (int k = 0; k < kNumNops; k++) {
        asm volatile("nop");
      }
    }
  }
  std::shared_ptr<TMemoryBuffer> strBuffer(new TMemoryBuffer());
  std::shared_ptr<TBinaryProtocol> proto(new TBinaryProtocol(strBuffer));

  // Serialize random string as Thrift
  ranking::Payload payload;
  payload.message = this_thread.random_string;
  payload.write(proto.get());

  // Get serialized data
  uint8_t* buf;
  uint32_t sz;
  strBuffer->getBuffer(&buf, &sz);

  context.SendResponse(buf, sz);
}

int main(int argc, char **argv) {
  if (cmdline_parser(argc, argv, &args) != 0) {
    DIE("cmdline_parser failed");
  }

  // Set logging level
  for (unsigned int i = 0; i < args.verbose_given; i++) {
    log_level = (log_level_t)(static_cast<int>(log_level) - 1);
  }
  if (args.quiet_given) {
    log_level = QUIET;
  }

  std::vector<ThreadData> thread_data(args.threads_arg);

  oldisim::LeafNodeServer server(args.port_arg);
  server.SetThreadStartupCallback(
      std::bind(ThreadStartup, std::placeholders::_1, std::ref(thread_data)));
  server.RegisterQueryCallback(
      ranking::kPageRankRequestType,
      std::bind(PageRankRequestHandler, std::placeholders::_1,
                std::placeholders::_2, std::ref(thread_data)));
  server.SetNumThreads(args.threads_arg);
  server.SetThreadPinning(!args.noaffinity_given);
  server.SetThreadLoadBalancing(!args.noloadbalance_given);

  server.EnableMonitoring(args.monitor_port_arg);

  server.Run();

  return 0;
}
