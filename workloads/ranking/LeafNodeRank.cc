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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <folly/Range.h>
#include <folly/compression/Compression.h>
#include <folly/compression/Counters.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/init/Init.h>

#include <thrift/lib/cpp2/protocol/CompactProtocol.h>
#include <thrift/lib/cpp2/protocol/Serializer.h>

#include "oldisim/LeafNodeServer.h"
#include "oldisim/NodeThread.h"
#include "oldisim/ParentConnection.h"
#include "oldisim/QueryContext.h"
#include "oldisim/Util.h"

#include "LeafNodeRankCmdline.h"
#include "RequestTypes.h"

#include "TimekeeperPool.h"
#include "dwarfs/pagerank.h"

#include "if/gen-cpp2/ranking_types.h"

#include "../search/ICacheBuster.h"
#include "../search/PointerChase.h"

#include "generators/RankingGenerators.h"

// Shared configuration flags
static gengetopt_args_info args;

static const int kMaxResponseSize = 1 << 12;
const int kNumNops = 6;
const int kNumNopIterations = 60;
const int kNumCompressIterations = 100;
const int kNumICacheBusterMethods = 100000;
const int kPointerChaseSize = 10000000;

struct ThreadData {
  std::shared_ptr<folly::CPUThreadPoolExecutor> cpuThreadPool;
  std::shared_ptr<folly::IOThreadPoolExecutor> ioThreadPool;
  std::shared_ptr<ranking::TimekeeperPool> timekeeperPool;
  std::unique_ptr<ranking::dwarfs::PageRank> page_ranker;
  std::unique_ptr<search::PointerChase> pointer_chaser;
  std::unique_ptr<ICacheBuster> icache_buster;
  std::default_random_engine rng;
  std::gamma_distribution<double> latency_distribution;
  std::string random_string;
};

void ThreadStartup(
    oldisim::NodeThread &thread, std::vector<ThreadData> &thread_data,
    ranking::dwarfs::PageRankParams &params,
    const std::shared_ptr<folly::CPUThreadPoolExecutor> cpuThreadPool,
    const std::shared_ptr<folly::IOThreadPoolExecutor> ioThreadPool,
    const std::shared_ptr<ranking::TimekeeperPool> timekeeperPool) {
  auto &this_thread = thread_data[thread.get_thread_num()];
  auto graph = params.buildGraph();
  this_thread.cpuThreadPool = cpuThreadPool;
  this_thread.ioThreadPool = ioThreadPool;
  this_thread.timekeeperPool = timekeeperPool;
  this_thread.page_ranker.reset(
      new ranking::dwarfs::PageRank{std::move(graph)});
  this_thread.icache_buster.reset(new ICacheBuster(kNumICacheBusterMethods));
  this_thread.pointer_chaser.reset(new search::PointerChase(kPointerChaseSize));

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  this_thread.rng.seed(seed);

  const double alpha = 0.7;
  const double beta = 20000;
  this_thread.latency_distribution =
      std::gamma_distribution<double>(alpha, beta);

  this_thread.random_string = RandomString(args.random_data_size_arg);
}

std::string compressPayload(const std::string &data, int result) {
  folly::StringPiece output(
      data.data(),
      std::min(args.compression_data_size_arg, args.random_data_size_arg));
  auto codec = folly::io::getCodec(folly::io::CodecType::ZSTD);
  std::string compressed = codec->compress(output);
  return std::move(compressed);
}

folly::IOBufQueue serializePayload(const ranking::RankingResponse &resp) {
  apache::thrift::CompactSerializer ser;
  folly::IOBufQueue bufq;
  ser.serialize(resp, &bufq);
  return std::move(bufq);
}

ranking::RankingResponse deserializePayload(const folly::IOBuf *buf) {
  apache::thrift::CompactSerializer ser;
  ranking::RankingResponse resp;
  ser.deserialize(buf, resp);
  return resp;
}

void PageRankRequestHandler(oldisim::NodeThread &thread,
                            oldisim::QueryContext &context,
                            std::vector<ThreadData> &thread_data) {
  auto &this_thread = thread_data[thread.get_thread_num()];
  int num_iterations = this_thread.latency_distribution(this_thread.rng);
  ICacheBuster &buster = *this_thread.icache_buster;
  search::PointerChase &chaser = *this_thread.pointer_chaser;

  for (int i = 0; i < num_iterations; i++) {
    buster.RunNextMethod();
  }

  auto f = folly::via(this_thread.cpuThreadPool.get(), [&this_thread]() {
    return this_thread.page_ranker->rank(args.graph_max_iters_arg, 1e-4);
  });
  int result = std::move(f).get();

  auto timekeeper = this_thread.timekeeperPool->getTimekeeper();
  auto s = folly::futures::sleep(std::chrono::milliseconds(5), timekeeper.get())
               .via(this_thread.ioThreadPool.get())
               .thenValue([&](auto &&) {
                 chaser.Chase(args.io_chase_iterations_arg);
                 return result + 1;
               });
  result = std::move(s).get();

  chaser.Chase(args.chase_iterations_arg);

  // Serialize random string as Thrift
  auto compressed = compressPayload(this_thread.random_string, result);

  // Generate a response
  ranking::RankingResponse resp =
      ranking::generators::generateRandomRankingResponse(20);

  // Serialize into FBThrift
  auto payloadIOBufQ = serializePayload(resp);
  auto buf = payloadIOBufQ.move();

  folly::futures::sleep(std::chrono::milliseconds(2), timekeeper.get()).get();

  auto resp1 = deserializePayload(buf.get());

  context.SendResponse(buf->data(), buf->length());
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
  int fake_argc = 1;
  char *fake_argv[2] = {(char *)"./LeafNodeRank", nullptr};
  char **sargv = reinterpret_cast<char **>(fake_argv);
  folly::init(&fake_argc, &sargv);
  auto cpuThreadPool =
      std::make_shared<folly::CPUThreadPoolExecutor>(args.cpu_threads_arg);
  auto ioThreadPool =
      std::make_shared<folly::IOThreadPoolExecutor>(args.io_threads_arg);

  auto timekeeperPool =
      std::make_shared<ranking::TimekeeperPool>(args.timekeeper_threads_arg);

  std::vector<ThreadData> thread_data(args.threads_arg);
  ranking::dwarfs::PageRankParams params{args.graph_scale_arg,
                                         args.graph_degree_arg};
  oldisim::LeafNodeServer server(args.port_arg);
  server.SetThreadStartupCallback(
      std::bind(ThreadStartup, std::placeholders::_1, std::ref(thread_data),
                std::ref(params), cpuThreadPool, ioThreadPool, timekeeperPool));
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
