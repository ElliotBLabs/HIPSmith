#pragma once

#include <memory>
#include <string>

#include "AbsProgramGenerator.h"
#include "CommonMacros.h"
#include "DefaultOutputMgr.h"
#include "HIPSmith/HIPOutputMgr.h"
#include "OutputMgr.h"

namespace HIPSmith {

class HIPProgramGenerator : public AbsProgramGenerator {
 public:
  HIPProgramGenerator(unsigned long seed)
      : output_mgr_(new HIPOutputMgr()), seed_(seed) {}

  // Inherited from AbsProgramGenerator
  // Creates the random program.
  void goGenerator();

  // Inherited from AbsProgramGenerator.
  OutputMgr* getOutputMgr();

  // Inherited from AbsProgramGenerator.
  // not needed
  std::string get_count_prefix(const std::string& name);

  // Inherited from AbsProgramGenerator.
  // not needed
  void initialize();

  // Generate the constant memory structures and values early
  void GenerateConstantMemory();

 private:
  std::unique_ptr<OutputMgr> output_mgr_;

  unsigned long seed_;

  // generate __constant__ and __managed__ globals
  void GenerateHIPGlobals();
};

}  // namespace HIPSmith
