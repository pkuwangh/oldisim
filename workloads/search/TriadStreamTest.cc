#include <iostream>

#include "oldisim/Util.h"
#include "TriadStreamTestCmdline.h"
#include "TriadStream.h"

static gengetopt_args_info args;

int main(int argc, char** argv) {
  if (cmdline_parser(argc, argv, &args) != 0) {
    DIE("cmdline_parser failed");
  }

  search::TriadStream triadstreamer(args.size_arg);

  uint64_t start_time = GetTimeAccurateNano();
  for (int i = 0; i < args.iterations_arg; i++) {
    triadstreamer.Triad(args.length_arg);
  }

  uint64_t end_time = GetTimeAccurateNano();

  std::cout << "Time per triad: "
            << static_cast<double>(end_time - start_time) /
               (static_cast<double>(args.iterations_arg) * args.length_arg)
            << " ns" << std::endl;
  return 0;
}

