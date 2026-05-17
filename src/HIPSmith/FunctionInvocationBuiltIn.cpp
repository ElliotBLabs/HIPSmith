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
DistributionTable *hip_warp_vote_func_table = NULL;

const char *const kSyncNames[4] = {"",  // Sentinel
                                   "__syncthreads_count", "__syncthreads_and",
                                   "__syncthreads_or"};

const char *const kWarpVoteNames[] = {
    "",  // Sentinel
    "__all",      "__any",      "__ballot",     "__activemask",
    "__all_sync", "__any_sync", "__ballot_sync"};

}  // namespace

// base class

FunctionInvocationHIPBuiltIn *FunctionInvocationHIPBuiltIn::make_random(
    CGContext &cg_context, const Type &type) {
  // we are a bit cheeky and allow any int type thing to get one of these and
  // rely on implicit casting
  if (type.eType == eSimple) {
    int choice = rnd_upto(100);
    if (choice < 50 && HIPOptions::hip_sync()) {
      return FunctionInvocationHIPSyncBuiltIn::make_random(cg_context, type);
    } else if (HIPOptions::hip_warp()) {
      return FunctionInvocationHIPWarpVoteBuiltIn::make_random(cg_context,
                                                               type);
    }
  }

  // we do not have something that can return that type
  return NULL;
}

void FunctionInvocationHIPBuiltIn::InitTables() {
  // init the tables associated with each special built in type
  FunctionInvocationHIPSyncBuiltIn::InitTables();
  FunctionInvocationHIPWarpVoteBuiltIn::InitTables();
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

// HIP SYNC STUFF

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

// END OF HIP SYNC ONES

// Hip Warp Vote function calls

FunctionInvocationHIPWarpVoteBuiltIn *
FunctionInvocationHIPWarpVoteBuiltIn::make_random(CGContext &cg_context,
                                                  const Type &type) {
  std::vector<const Type *> param_types;
  enum BuiltIn func = FunctionSelector(type, &param_types);

  if (func == kIdentity) {
    return NULL;  // Could not find a valid function for this return type
  }

  FunctionInvocationHIPWarpVoteBuiltIn *fi =
      new FunctionInvocationHIPWarpVoteBuiltIn(func, type);

  for (const Type *param_type : param_types) {
    // use a random csmith expression as a parameter inside these
    fi->param_value.push_back(Expression::make_random(cg_context, param_type));
  }
  return fi;
}

enum FunctionInvocationHIPWarpVoteBuiltIn::BuiltIn
FunctionInvocationHIPWarpVoteBuiltIn::FunctionSelector(
    const Type &type, std::vector<const Type *> *params) {
  assert(params != NULL);
  params->clear();

  assert(hip_warp_vote_func_table != NULL);
  VectorFilter filter(hip_warp_vote_func_table);

  int rnd = rnd_upto(filter.get_max_prob(), &filter);
  enum BuiltIn func = (enum BuiltIn)filter.lookup(rnd);

  // Populate expected parameters based on the selected built-in
  switch (func) {
    case kActiveMask:
      // 0 parameters
      break;
    case kAll:
    case kAny:
    case kBallot:
      // 1 parameter: int predicate
      params->push_back(&Type::get_simple_type(eInt));
      break;
    case kAllSync:
    case kAnySync:
    case kBallotSync:
      // 2 parameters: unsigned long long mask, int predicate
      params->push_back(&Type::get_simple_type(eULongLong));
      params->push_back(&Type::get_simple_type(eInt));
      break;
    default:
      assert(false);
  }

  return func;
}

void FunctionInvocationHIPWarpVoteBuiltIn::InitTables() {
  hip_warp_vote_func_table = new DistributionTable();
  hip_warp_vote_func_table->add_entry(kAll, 10);
  hip_warp_vote_func_table->add_entry(kAny, 10);
  hip_warp_vote_func_table->add_entry(kBallot, 10);
  hip_warp_vote_func_table->add_entry(kActiveMask, 10);
  hip_warp_vote_func_table->add_entry(kAllSync, 10);
  hip_warp_vote_func_table->add_entry(kAnySync, 10);
  hip_warp_vote_func_table->add_entry(kBallotSync, 10);
}

FunctionInvocationHIPWarpVoteBuiltIn *
FunctionInvocationHIPWarpVoteBuiltIn::clone() const {
  FunctionInvocationHIPWarpVoteBuiltIn *fi =
      new FunctionInvocationHIPWarpVoteBuiltIn(built_in_, type_);
  for (const Expression *expr : param_value) {
    fi->param_value.push_back(expr->clone());
  }
  return fi;
}

void FunctionInvocationHIPWarpVoteBuiltIn::OutputFuncName(
    std::ostream &out) const {
  out << kWarpVoteNames[built_in_];
}

void FunctionInvocationHIPWarpVoteBuiltIn::Output(std::ostream &out) const {
  OutputFuncName(out);
  out << '(';

  for (size_t idx = 0; idx < param_value.size(); ++idx) {
    // we force a cast to ulonglong for the exec masks
    if (idx == 0 && (built_in_ == kAllSync || built_in_ == kAnySync ||
                     built_in_ == kBallotSync)) {
      out << "(unsigned long long)(";
      param_value[idx]->Output(out);
      out << ")";
    } else {
      // do not change the output
      param_value[idx]->Output(out);
    }

    // commas between args
    if (idx < param_value.size() - 1) {
      out << ", ";
    }
  }

  out << ')';
}

const Type &FunctionInvocationHIPWarpVoteBuiltIn::GetParameterType(
    size_t idx) const {
  if (built_in_ == kAllSync || built_in_ == kAnySync ||
      built_in_ == kBallotSync) {
    if (idx == 0) return Type::get_simple_type(eULongLong);
    return Type::get_simple_type(eInt);
  }
  return Type::get_simple_type(eInt);
}
// WARP Vote end stuff

}  // namespace HIPSmith