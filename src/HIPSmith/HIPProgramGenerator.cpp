#include "HIPSmith/HIPProgramGenerator.h"

#include <VariableSelector.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "Function.h"
#include "Type.h"

class OutputMgr;

namespace HIPSmith {

void HIPProgramGenerator::goGenerator() {
  output_mgr_->OutputHeader(0, NULL, seed_);

  // This creates the random program, the rest handles post-processing and
  // outputting the program.
  GenerateAllTypes();
  GenerateFunctions();
  output_mgr_->Output();
}

OutputMgr* HIPProgramGenerator::getOutputMgr() { return output_mgr_.get(); }

std::string HIPProgramGenerator::get_count_prefix(const std::string& name) {
  assert(false);
  return "";
}

void HIPProgramGenerator::initialize() {}

}  // namespace HIPSmith
