#include "HIPSmith/HIPOutputMgr.h"

#include <cstdio>
#include <fstream>
#include <sstream>

#include "ArrayVariable.h"
#include "Function.h"
#include "HIPSmith/HIPOptions.h"
#include "OutputMgr.h"
#include "Type.h"
#include "VariableSelector.h"

namespace HIPSmith {

HIPOutputMgr::HIPOutputMgr()
    : out_(HIPOptions::output()), orig_out_("HIP-CCProg.cc") {}

void HIPOutputMgr::OutputHeader(int argc, char *argv[], unsigned long seed) {
  std::ostream &out = get_main_out();

  out << "// "
         "------------------------------------------------------------------\n"
      << "// Standard Headers\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "#include <cstdint>\n"
      << "#include <cstddef>\n"
      << "#include <climits>\n"
      << "#include <cstring>\n"
      << "#include <iostream>\n\n"

      << "// "
         "------------------------------------------------------------------\n"
      << "// HIP Runtime\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "#include <hip/hip_runtime.h>\n\n"

      << "// "
         "------------------------------------------------------------------\n"
      << "// Project Headers\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "#include \"HIPSmith.h\"\n"
      << "#include \"safe_math_macros.h\"\n\n"

      << "// "
         "------------------------------------------------------------------\n"
      << "// Macros & Metadata\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "#define transparent_crc(X, Y, Z) transparent_crc_(&crc64_context, X, "
         "Y, Z)\n"
      << "// Seed: " << seed << "\n\n";
}

void HIPOutputMgr::Output() {
  OutputOriginal();
  // now will emit device coded functions
  HIPOptions::is_emitting_device_code(true);

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

  // ---------------------------------------------------------
  // Generated Kernel
  // ---------------------------------------------------------
  out << "// ------------------------------------------------------------------"
      << std::endl;
  out << "// Kernel" << std::endl;
  out << "// ------------------------------------------------------------------"
      << std::endl;
  out << "__global__ void hipsmith_kernel(uint64_t *results) {" << std::endl;

  output_tab(out, 1);
  out << "// Init globals" << std::endl;
  globals.OutputStructInit(out);
  out << std::endl;

  output_tab(out, 1);
  out << "// Execute core logic" << std::endl;
  output_tab(out, 1);
  out << "func_1(";
  globals.GetGlobalStructVar().Output(out);
  out << ");" << std::endl;
  out << std::endl;

  output_tab(out, 1);
  out << "// CRC Context" << std::endl;
  output_tab(out, 1);
  out << "uint64_t crc64_context = 0xFFFFFFFFFFFFFFFFUL;" << std::endl;
  output_tab(out, 1);
  out << "int print_hash_value = 0;" << std::endl;

  output_tab(out, 1);
  out << "// Hash variables" << std::endl;
  HashGlobalVariables(out);
  out << std::endl;

  output_tab(out, 1);
  out << "// Store result" << std::endl;
  output_tab(out, 1);
  out << "int tid = threadIdx.x + blockIdx.x * blockDim.x;" << std::endl;
  output_tab(out, 1);
  out << "results[tid] = (crc64_context ^ 0xFFFFFFFFFFFFFFFFUL);" << std::endl;
  out << "}" << std::endl << std::endl;

  // ---------------------------------------------------------
  // Generated Main
  // ---------------------------------------------------------
  out << "// ------------------------------------------------------------------"
      << std::endl;
  out << "// Host Main" << std::endl;
  out << "// ------------------------------------------------------------------"
      << std::endl;
  out << "int main(int argc, const char* argv[]) {" << std::endl;

  output_tab(out, 1);
  out << "// Config" << std::endl;
  output_tab(out, 1);
  out << "const unsigned int num_threads = 4;" << std::endl;
  output_tab(out, 1);
  out << "const unsigned int block_size = 4;" << std::endl;
  out << std::endl;

  output_tab(out, 1);
  out << "// Host Alloc" << std::endl;
  output_tab(out, 1);
  out << "std::vector<uint64_t> h_results(num_threads);" << std::endl;
  output_tab(out, 1);
  out << "const size_t results_bytes = sizeof(uint64_t) * h_results.size();"
      << std::endl;
  out << std::endl;

  output_tab(out, 1);
  out << "// Device Alloc" << std::endl;
  output_tab(out, 1);
  out << "uint64_t *d_results;" << std::endl;
  output_tab(out, 1);
  out << "HIP_CHECK(hipMalloc((void**)&d_results, results_bytes));"
      << std::endl;
  output_tab(out, 1);
  out << "HIP_CHECK(hipMemset(d_results, 0, results_bytes));" << std::endl;
  out << std::endl;

  output_tab(out, 1);
  out << "// Dimensions" << std::endl;
  output_tab(out, 1);
  out << "const dim3 block_dim(block_size);" << std::endl;
  output_tab(out, 1);
  out << "const dim3 grid_dim((num_threads + block_size - 1) / block_size);"
      << std::endl;
  out << std::endl;

  output_tab(out, 1);
  out << "// Launch" << std::endl;
  output_tab(out, 1);
  out << "hipLaunchKernelGGL(hipsmith_kernel, grid_dim, block_dim, 0, 0, "
         "d_results);"
      << std::endl;
  output_tab(out, 1);
  out << "HIP_CHECK(hipGetLastError());" << std::endl;
  output_tab(out, 1);
  out << "HIP_CHECK(hipDeviceSynchronize());" << std::endl;
  out << std::endl;

  output_tab(out, 1);
  out << "// Copy Back" << std::endl;
  output_tab(out, 1);
  out << "HIP_CHECK(hipMemcpy(h_results.data(), d_results, results_bytes, "
         "hipMemcpyDeviceToHost));"
      << std::endl;
  out << std::endl;

  output_tab(out, 1);
  out << "// Free" << std::endl;
  output_tab(out, 1);
  out << "HIP_CHECK(hipFree(d_results));" << std::endl;
  out << std::endl;

  output_tab(out, 1);
  out << "// Output" << std::endl;
  output_tab(out, 1);
  out << "for (size_t i = 0; i < h_results.size(); ++i) {" << std::endl;
  output_tab(out, 2);
  out << "printf(\"Thread %zu CRC: %lu\\n\", i, h_results[i]);" << std::endl;
  output_tab(out, 1);
  out << "}" << std::endl;

  output_tab(out, 1);
  out << "return 0;" << std::endl;
  out << "}" << std::endl;
}

void HIPOutputMgr::OutputOriginal() {
  std::ostream &out = orig_out_;

  out << "// "
         "------------------------------------------------------------------\n"
      << "// Standard Headers\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "#include <cstdint>\n"
      << "#include <cstddef>\n"
      << "#include <climits>\n"
      << "#include <cstring>\n"
      << "#include <iostream>\n\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "// Project Headers\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "#include \"csmith.h\"\n"
      << "#include \"safe_math_macros.h\"\n\n"

      << "// "
         "------------------------------------------------------------------\n"
      << "// Macros & Metadata\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "#define transparent_crc(X, Y, Z) transparent_crc_(&crc64_context, X, "
         "Y, Z)\n";

  OutputStructUnionDeclarations(out);
  OutputGlobalVariables(out);
  OutputForwardDeclarations(out);
  OutputFunctions(out);

  // 5. Standard Main
  out << "int main(void) {" << std::endl;
  output_tab(out, 1);
  out << "func_1();" << std::endl;  // Standard entry
  output_tab(out, 1);
  out << "// CRC Context" << std::endl;
  output_tab(out, 1);
  out << "uint64_t crc64_context = 0xFFFFFFFFFFFFFFFFUL;" << std::endl;
  output_tab(out, 1);
  out << "int print_hash_value = 0;" << std::endl;
  output_tab(out, 1);
  out << "// Hash variables" << std::endl;
  HashGlobalVariables(out);
  out << std::endl;
  output_tab(out, 1);

  out << "std::cout <<  (crc64_context ^ 0xFFFFFFFFFFFFFFFFUL) << std::endl;"
      << std::endl;
  output_tab(out, 1);
  out << "return 0;" << std::endl;
  out << "}" << std::endl;
}
}  // namespace HIPSmith