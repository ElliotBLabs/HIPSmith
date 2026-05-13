#pragma once

#include <fstream>
#include <string>

#include "CommonMacros.h"
#include "HIPSmith/Globals.h"
#include "OutputMgr.h"

namespace HIPSmith {

class Globals;

class HIPOutputMgr : public OutputMgr {
 public:
  HIPOutputMgr();
  ~HIPOutputMgr() { out_.close(); }

  // Inherited from OutputMgr.
  // Outputs #defines and forward declarations.
  void OutputHeader(int argc, char *argv[], unsigned long seed);

  // Inherited from OutputMgr.
  // Outputs all other code functions and kernel entry
  void Output();

  // Inherited from OutputMgr.
  // Gets the stream used for printing the output.
  std::ostream &get_main_out();

 private:
  std::ofstream out_;

  // Outputs the main function that invokes the random program and reports back
  // a hash of the output
  void OutputEntryFunction(Globals &globals);

  // output HIP constant and managed memory
  void OutputHipGlobals();
};

}  // namespace HIPSmith
