#include "HIPSmith/HIPProgramGenerator.h"

#include <VariableSelector.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "Function.h"
#include "HIPSmith/HIPExpression.h"
#include "HIPSmith/HIPOptions.h"
#include "HIPSmith/Vector.h"
#include "Type.h"

class OutputMgr;

namespace HIPSmith {

void HIPProgramGenerator::goGenerator() {
  output_mgr_->OutputHeader(0, NULL, seed_);

  // This creates the random program, the rest handles post-processing and
  // outputting the program.
  HIPExpression::InitProbabilityTable();
  Vector::GenerateVectorTypes();

  GenerateAllTypes();

  GenerateHIPGlobals();
  GenerateHIPBuiltins();

  GenerateFunctions();
  output_mgr_->Output();
}

OutputMgr* HIPProgramGenerator::getOutputMgr() { return output_mgr_.get(); }

std::string HIPProgramGenerator::get_count_prefix(const std::string& name) {
  assert(false);
  return "";
}

void HIPProgramGenerator::initialize() {}

void HIPProgramGenerator::GenerateHIPGlobals() {
  int max_gen = 20;
  if (HIPSmith::HIPOptions::hip_consts()) {
    int num_consts = rnd_upto(max_gen);
    for (int i = 0; i < num_consts; i++) {
      VariableSelector::GenerateHIPConstant(CGContext::get_empty_context());
    }
    reset_gensym();
  }

  if (HIPSmith::HIPOptions::hip_managed()) {
    int num_managed = rnd_upto(max_gen);
    for (int i = 0; i < num_managed; i++) {
      VariableSelector::GenerateHIPManaged(CGContext::get_empty_context());
    }
    reset_gensym();
  }

  if (HIPSmith::HIPOptions::hip_device()) {
    int num_device = rnd_upto(max_gen);
    for (int i = 0; i < num_device; i++) {
      VariableSelector::GenerateHIPDeviceVar(CGContext::get_empty_context());
    }
    reset_gensym();
  }
}

void HIPProgramGenerator::GenerateHIPBuiltins() {
  if (HIPOptions::hip_builtins()) {
    VariableSelector::GenerateHIPBuiltins(CGContext::get_empty_context());
  }
}

}  // namespace HIPSmith
