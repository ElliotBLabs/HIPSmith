#include "HIPSmith/FunctionInvocationBuiltIn.h"

#include <map>
#include <ostream>
#include <vector>

#include "CGContext.h"
#include "Expression.h"
#include "HIPSmith/HIPOptions.h"
#include "ProbabilityTable.h"
#include "Type.h"
#include "VectorFilter.h"

namespace HIPSmith {
namespace {

DistributionTable *hip_sync_func_table = NULL;

const char *const kSyncNames[4] = {"",  // Sentinel
                                   "__syncthreads_count", "__syncthreads_and",
                                   "__syncthreads_or"};

}  // namespace

// base class

FunctionInvocationHIPBuiltIn *FunctionInvocationHIPBuiltIn::make_random(
    CGContext &cg_context, const Type &type) {
  if (type.eType == eSimple &&
      (type.simple_type == eInt || type.simple_type == eUInt)) {
    // these function invocations are int returning

    if (HIPOptions::hip_sync()) {
      return FunctionInvocationHIPSyncBuiltIn::make_random(cg_context, type);
    }
  }

  // we do not have something that can return that type
  return NULL;
}

void FunctionInvocationHIPBuiltIn::InitTables() {
  // init the tables associated with each special built in type
  FunctionInvocationHIPSyncBuiltIn::InitTables();
}

void FunctionInvocationHIPBuiltIn::Output(std::ostream &out) const {
  OutputFuncName(out);
  out << '(';
  for (size_t idx = 0; idx < param_value.size(); ++idx) {
    param_value[idx]->Output(out);
    if (idx < param_value.size() - 1) out << ", ";
  }
  out << ')';
}

void FunctionInvocationHIPBuiltIn::indented_output(std::ostream &out,
                                                   int indent) const {
  output_tab(out, indent);
  Output(out);
}

// Hip predicated sync function calls

FunctionInvocationHIPSyncBuiltIn *FunctionInvocationHIPSyncBuiltIn::make_random(
    CGContext &cg_context, const Type &type) {
  std::vector<const Type *> param_types;
  enum BuiltIn func = FunctionSelector(type, &param_types);

  FunctionInvocationHIPSyncBuiltIn *fi =
      new FunctionInvocationHIPSyncBuiltIn(func, type);

  for (const Type *param_type : param_types) {
    // use a random csmith expression as a parameter inside these
    fi->param_value.push_back(Expression::make_random(cg_context, param_type));
  }
  return fi;
}

enum FunctionInvocationHIPSyncBuiltIn::BuiltIn
FunctionInvocationHIPSyncBuiltIn::FunctionSelector(
    const Type &type, std::vector<const Type *> *params) {
  assert(params != NULL);
  params->clear();

  assert(hip_sync_func_table != NULL);
  VectorFilter filter(hip_sync_func_table);

  int rnd = rnd_upto(filter.get_max_prob(), &filter);
  enum BuiltIn func = (enum BuiltIn)filter.lookup(rnd);

  // require only one interger param for predicated hip sync builtins
  params->push_back(&Type::get_simple_type(eInt));

  return func;
}

void FunctionInvocationHIPSyncBuiltIn::InitTables() {
  hip_sync_func_table = new DistributionTable();
  hip_sync_func_table->add_entry(kSyncThreadsCount, 10);
  hip_sync_func_table->add_entry(kSyncThreadsAnd, 10);
  hip_sync_func_table->add_entry(kSyncThreadsOr, 10);
}

FunctionInvocationHIPSyncBuiltIn *FunctionInvocationHIPSyncBuiltIn::clone()
    const {
  FunctionInvocationHIPSyncBuiltIn *fi =
      new FunctionInvocationHIPSyncBuiltIn(built_in_, type_);
  for (const Expression *expr : param_value) {
    fi->param_value.push_back(expr->clone());
  }
  return fi;
}

void FunctionInvocationHIPSyncBuiltIn::OutputFuncName(std::ostream &out) const {
  out << kSyncNames[built_in_];
}

const Type &FunctionInvocationHIPSyncBuiltIn::GetParameterType(
    size_t idx) const {
  // Always an int for these specific functions
  return Type::get_simple_type(eInt);
}

}  // namespace HIPSmith