// Entry point to the program.

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "AbsProgramGenerator.h"
#include "CGOptions.h"
#include "HIPSmith/HIPOptions.h"
#include "HIPSmith/HIPOutputMgr.h"
#include "HIPSmith/HIPProgramGenerator.h"
#include "platform.h"

// Generator seed.
static unsigned long g_Seed = 0;

int main(int argc, char **argv) {
  g_Seed = platform_gen_seed();
  CGOptions::set_default_settings();
  HIPSmith::HIPOptions::ResolveCGOptions();

  // AbsProgramGenerator does other initialisation stuff, besides itself. So
  // we call it, disregarding the returned object. Still need to delete it.
  AbsProgramGenerator *generator =
      AbsProgramGenerator::CreateInstance(argc, argv, g_Seed);
  if (!generator) {
    cout << "error: can't create AbsProgramGenerator. csmith init failed!"
         << std::endl;
    return -1;
  }

  // Create program generator
  HIPSmith::HIPProgramGenerator hip_generator(g_Seed);
  hip_generator.goGenerator();

  // Calls Finalization::doFinalization(), which deletes everything, so must be
  // called after program generation.
  delete generator;

  return 0;
}
