#include "HIPSmith/HIPOutputMgr.h"

#include <cstdio>
#include <fstream>
#include <sstream>

#include "Function.h"
#include "HIPSmith/HIPOptions.h"
#include "OutputMgr.h"
#include "Type.h"
#include "VariableSelector.h"

namespace HIPSmith {

HIPOutputMgr::HIPOutputMgr() : out_(HIPOptions::output()) {}

void HIPOutputMgr::OutputHeader(int argc, char *argv[], unsigned long seed) {
  // Redefine platform independent scalar C types to platform independent scalar
  // OpenCL types.
  std::ostream &out = get_main_out();
  out << "#include <cstdint>" << std::endl;
  out << "#include <cstddef>" << std::endl;
  out << "#include <climits>" << std::endl;
  out << "#include <cstring>" << std::endl;
  out << "#include <iostream>" << std::endl;
  out << "#include \"HIPSmith.h\"" << std::endl;
  out << "#include \"safe_math_macros.h\"" << std::endl;
  out << "#define transparent_crc(X, Y, Z) "
         "transparent_crc_(&crc64_context, X, Y, Z)\n"
         "\n"
      << std::endl;
  out << "// Seed: " << seed << std::endl;
}

void HIPOutputMgr::Output() {
  std::ostream &out = get_main_out();
  OutputStructUnionDeclarations(out);

  Globals *globals = Globals::GetGlobals();
  globals->OutputStructDefinition(out);
  globals->ModifyGlobalVariableReferences();
  globals->AddGlobalStructToAllFunctions();

  OutputForwardDeclarations(out);
  OutputFunctions(out);
  OutputEntryFunction(*globals);
}

std::ostream &HIPOutputMgr::get_main_out() { return out_; }

void HIPOutputMgr::OutputEntryFunction(Globals &globals) {
  std::ostream &out = get_main_out();

  out << "int main(int argc, char** argv) {" << std::endl;
  globals.OutputStructInit(out);
  output_tab(out, 1);
  out << "func_1(";
  globals.GetGlobalStructVar().Output(out);
  out << ");" << std::endl;
  output_tab(out, 1);
  out << "uint64_t crc64_context = 0xFFFFFFFFFFFFFFFFUL;" << std::endl;
  output_tab(out, 1);
  out << "int print_hash_value = 0;" << std::endl;
  HashGlobalVariables(out);
  output_tab(out, 1);

  out << "std::cout <<  (crc64_context ^ 0xFFFFFFFFFFFFFFFFUL) << std::endl;"
      << std::endl;
  out << "}" << endl;
}

}  // namespace HIPSmith
