#include "HIPSmith/FunctionInvocationBuiltIn.h"

#include <map>
#include <ostream>
#include <vector>

#include "Bookkeeper.h"
#include "CGContext.h"
#include "Constant.h"
#include "Expression.h"
#include "ExpressionVariable.h"
#include "HIPSmith/HIPOptions.h"
#include "Lhs.h"
#include "ProbabilityTable.h"
#include "Type.h"
#include "Variable.h"
#include "VariableSelector.h"
#include "VectorFilter.h"

namespace HIPSmith {
namespace {

DistributionTable *hip_sync_func_table = NULL;
DistributionTable *hip_warp_vote_func_table = NULL;
DistributionTable *hip_warp_match_func_table = NULL;
DistributionTable *hip_warp_shuffle_func_table = NULL;
DistributionTable *hip_warp_reduce_func_table = NULL;
DistributionTable *hip_atomic_func_table = NULL;

const char *const kSyncNames[4] = {"",  // Sentinel
                                   "__syncthreads_count", "__syncthreads_and",
                                   "__syncthreads_or"};

const char *const kWarpVoteNames[] = {"",  // Sentinel
                                      "__all",
                                      "__any",
                                      "__ballot",
                                      "__activemask",
                                      "safe_all_sync",
                                      "safe_any_sync",
                                      "safe_ballot_sync"};

const char *const kWarpMatchNames[] = {"", "safe_match_any", "safe_match_all",
                                       "safe_match_any_sync",
                                       "safe_match_all_sync"};

const char *const kWarpShuffleNames[] = {"",  // Sentinel
                                         "safe_shfl",
                                         "safe_shfl_up",
                                         "safe_shfl_down",
                                         "safe_shfl_xor",
                                         "safe_shfl_sync",
                                         "safe_shfl_up_sync",
                                         "safe_shfl_down_sync",
                                         "safe_shfl_xor_sync"};

const char *const kWarpReduceNames[] = {"",  // Sentinel
                                        "safe_reduce_add_sync",
                                        "safe_reduce_min_sync",
                                        "safe_reduce_max_sync",
                                        "safe_reduce_and_sync",
                                        "safe_reduce_or_sync",
                                        "safe_reduce_xor_sync"};

const char *const kAtomicNames[] = {
    "",  // Sentinel
    "safe_atomicAdd", "safe_atomicAdd_system",
    "safe_atomicSub", "safe_atomicSub_system",
    "atomicMin", "atomicMin_system",      // Native
    "atomicMax", "atomicMax_system",      // Native
    "atomicExch", "atomicExch_system",    // Native
    "atomicCAS", "atomicCAS_system",      // Native
    "atomicAnd", "atomicAnd_system",      // Native
    "atomicOr", "atomicOr_system",        // Native
    "atomicXor", "atomicXor_system",      // Native
    "atomicInc", "atomicInc_system",      // Native
    "atomicDec", "atomicDec_system"       // Native
};
}  // namespace

// base class

FunctionInvocationHIPBuiltIn *FunctionInvocationHIPBuiltIn::make_random(
    CGContext &cg_context, const Type &type) {
  
  // Built-ins generally operate on simple scalar types (ints)
  if (type.eType != eSimple) {
    return NULL;
  }

  // 1. Dynamically gather only the enabled options into a list using the class enum
  std::vector<BuiltInType> active_categories;

  if (HIPOptions::hip_sync())         active_categories.push_back(kSyncPredicates);
  if (HIPOptions::hip_warp())         active_categories.push_back(kWarpVote);
  if (HIPOptions::hip_warp_match())   active_categories.push_back(kWarpMatch);
  if (HIPOptions::hip_warp_shuffle()) active_categories.push_back(kWarpShuffle);
  if (HIPOptions::hip_warp_reduce())  active_categories.push_back(kWarpReduce);
  if (HIPOptions::hip_atomic())       active_categories.push_back(kAtomic);

  // 2. If no builtin flags were passed, exit safely
  if (active_categories.empty()) {
    return NULL;
  }

  // 3. Select a random index from 0 to (size - 1)
  int random_index = rnd_upto(active_categories.size());

  // 4. Fetch the category at that random index and dispatch
  switch (active_categories[random_index]) {
    case kSyncPredicates:
      return FunctionInvocationHIPSyncBuiltIn::make_random(cg_context, type);
    case kWarpVote:
      return FunctionInvocationHIPWarpVoteBuiltIn::make_random(cg_context, type);
    case kWarpMatch:
      return FunctionInvocationHIPWarpMatchBuiltIn::make_random(cg_context, type);
    case kWarpShuffle:
      return FunctionInvocationHIPWarpShuffleBuiltIn::make_random(cg_context, type);
    case kWarpReduce:
      return FunctionInvocationHIPWarpReduceBuiltIn::make_random(cg_context, type);
    case kAtomic:
      return FunctionInvocationHIPAtomicBuiltIn::make_random(cg_context, type);
    default:
      return NULL;
  }
}

void FunctionInvocationHIPBuiltIn::InitTables() {
  FunctionInvocationHIPSyncBuiltIn::InitTables();
  FunctionInvocationHIPWarpVoteBuiltIn::InitTables();
  FunctionInvocationHIPWarpMatchBuiltIn::InitTables();
  FunctionInvocationHIPWarpShuffleBuiltIn::InitTables();
  FunctionInvocationHIPWarpReduceBuiltIn::InitTables();
  FunctionInvocationHIPAtomicBuiltIn::InitTables();
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

  for (size_t i = 0; i < param_types.size(); ++i) {
    fi->param_value.push_back(
        Expression::make_random(cg_context, param_types[i]));
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

  // Everything except __activemask() takes exactly 1 integer predicate now
  if (func != kActiveMask) {
    params->push_back(&Type::get_simple_type(eInt));
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

// WARP Match start

FunctionInvocationHIPWarpMatchBuiltIn *
FunctionInvocationHIPWarpMatchBuiltIn::make_random(CGContext &cg_context,
                                                   const Type &type) {
  // T can be 32-bit int or 64-bit int only for fuzzing
  // no flots or doubles
  const Type *possible_t_types[] = {&Type::get_simple_type(eInt),
                                    &Type::get_simple_type(eLongLong)};
  const Type *t_type = possible_t_types[rnd_upto(2)];

  std::vector<const Type *> param_types;
  enum BuiltIn func = FunctionSelector(type, &param_types, *t_type);

  if (func == kIdentity) {
    return NULL;
  }

  FunctionInvocationHIPWarpMatchBuiltIn *fi =
      new FunctionInvocationHIPWarpMatchBuiltIn(func, type, *t_type);

  for (size_t i = 0; i < param_types.size(); ++i) {
    // pred always at index 1
    bool is_pred_param =
        (i == 1 && (func == kMatchAll || func == kMatchAllSync));

    if (is_pred_param) {
      const Type *base_type = &Type::get_simple_type(eInt);
      CVQualifiers qfer;
      qfer.add_qualifiers(false, false);
      std::vector<const Variable *> invalid_vars;

      Variable *pred_var =
          VariableSelector::select(Effect::WRITE, cg_context, base_type, &qfer,
                                   invalid_vars, eExact, eParentLocal);

      if (!pred_var || pred_var->isBitfield_ ||
          pred_var->type->simple_type != eInt) {
        if (func == kMatchAll)
          fi->built_in_ = kMatchAny;
        else if (func == kMatchAllSync)
          fi->built_in_ = kMatchAnySync;
        continue;
      }

      ExpressionVariable *addr_expr =
          new ExpressionVariable(*pred_var, param_types[i]);
      Bookkeeper::record_address_taken(pred_var);
      Bookkeeper::record_volatile_access(pred_var, -1, false);

      fi->param_value.push_back(addr_expr);
    } else {
      fi->param_value.push_back(
          Expression::make_random(cg_context, param_types[i]));
    }
  }
  return fi;
}

enum FunctionInvocationHIPWarpMatchBuiltIn::BuiltIn
FunctionInvocationHIPWarpMatchBuiltIn::FunctionSelector(
    const Type &type, std::vector<const Type *> *params, const Type &t_type) {
  assert(params != NULL);
  params->clear();

  assert(hip_warp_match_func_table != NULL);
  VectorFilter filter(hip_warp_match_func_table);

  int rnd = rnd_upto(filter.get_max_prob(), &filter);
  enum BuiltIn func = (enum BuiltIn)filter.lookup(rnd);

  switch (func) {
    case kMatchAny:
    case kMatchAnySync:
      params->push_back(&t_type);
      break;
    case kMatchAll:
    case kMatchAllSync:
      params->push_back(&t_type);
      params->push_back(&Type::get_simple_type(eInt));  // pred
      break;
    default:
      assert(false);
  }
  return func;
}

void FunctionInvocationHIPWarpMatchBuiltIn::InitTables() {
  hip_warp_match_func_table = new DistributionTable();
  hip_warp_match_func_table->add_entry(kMatchAny, 10);
  hip_warp_match_func_table->add_entry(kMatchAll, 10);
  hip_warp_match_func_table->add_entry(kMatchAnySync, 10);
  hip_warp_match_func_table->add_entry(kMatchAllSync, 10);
}

FunctionInvocationHIPWarpMatchBuiltIn *
FunctionInvocationHIPWarpMatchBuiltIn::clone() const {
  FunctionInvocationHIPWarpMatchBuiltIn *fi =
      new FunctionInvocationHIPWarpMatchBuiltIn(built_in_, type_, t_type_);
  for (const Expression *expr : param_value) {
    fi->param_value.push_back(expr->clone());
  }
  return fi;
}

void FunctionInvocationHIPWarpMatchBuiltIn::OutputFuncName(
    std::ostream &out) const {
  out << kWarpMatchNames[built_in_];
}

void FunctionInvocationHIPWarpMatchBuiltIn::Output(std::ostream &out) const {
  OutputFuncName(out);
  out << '(';

  for (size_t idx = 0; idx < param_value.size(); ++idx) {
    // Only thing this function needs to do now is insert the '&'
    bool is_pred =
        (idx == 1 && (built_in_ == kMatchAll || built_in_ == kMatchAllSync));

    if (is_pred) {
      out << "&(";
      param_value[idx]->Output(out);
      out << ")";
    } else {
      param_value[idx]->Output(out);
    }

    if (idx < param_value.size() - 1) {
      out << ", ";
    }
  }
  out << ')';
}

const Type &FunctionInvocationHIPWarpMatchBuiltIn::GetParameterType(
    size_t idx) const {
  // Index 0 is always the T val, Index 1 is always the int* pred
  if (idx == 0) return t_type_;
  return Type::get_simple_type(eInt);
}
// warp match end

// WARP Shuffle start

FunctionInvocationHIPWarpShuffleBuiltIn *
FunctionInvocationHIPWarpShuffleBuiltIn::make_random(CGContext &cg_context,
                                                     const Type &type) {
  std::vector<const Type *> param_types;
  enum BuiltIn func = FunctionSelector(type, &param_types);

  if (func == kIdentity) {
    return NULL;
  }

  FunctionInvocationHIPWarpShuffleBuiltIn *fi =
      new FunctionInvocationHIPWarpShuffleBuiltIn(func, type);

  for (size_t i = 0; i < param_types.size(); ++i) {
    fi->param_value.push_back(
        Expression::make_random(cg_context, param_types[i]));
  }
  return fi;
}

enum FunctionInvocationHIPWarpShuffleBuiltIn::BuiltIn
FunctionInvocationHIPWarpShuffleBuiltIn::FunctionSelector(
    const Type &type, std::vector<const Type *> *params) {
  assert(params != NULL);
  params->clear();

  // support any scalar int
  if (type.eType != eSimple || type.simple_type == eFloat) {
    return kIdentity;
  }

  assert(hip_warp_shuffle_func_table != NULL);
  VectorFilter filter(hip_warp_shuffle_func_table);

  int rnd = rnd_upto(filter.get_max_prob(), &filter);
  enum BuiltIn func = (enum BuiltIn)filter.lookup(rnd);

  // T var arg
  params->push_back(&type);

  // lane/delta/mask arg
  if (func == kShflUp || func == kShflDown || func == kShflUpSync ||
      func == kShflDownSync) {
    params->push_back(&Type::get_simple_type(eUInt));
  } else {
    params->push_back(&Type::get_simple_type(eInt));
  }

  return func;
}

void FunctionInvocationHIPWarpShuffleBuiltIn::InitTables() {
  hip_warp_shuffle_func_table = new DistributionTable();
  hip_warp_shuffle_func_table->add_entry(kShfl, 10);
  hip_warp_shuffle_func_table->add_entry(kShflUp, 10);
  hip_warp_shuffle_func_table->add_entry(kShflDown, 10);
  hip_warp_shuffle_func_table->add_entry(kShflXor, 10);
  hip_warp_shuffle_func_table->add_entry(kShflSync, 10);
  hip_warp_shuffle_func_table->add_entry(kShflUpSync, 10);
  hip_warp_shuffle_func_table->add_entry(kShflDownSync, 10);
  hip_warp_shuffle_func_table->add_entry(kShflXorSync, 10);
}

FunctionInvocationHIPWarpShuffleBuiltIn *
FunctionInvocationHIPWarpShuffleBuiltIn::clone() const {
  FunctionInvocationHIPWarpShuffleBuiltIn *fi =
      new FunctionInvocationHIPWarpShuffleBuiltIn(built_in_, type_);
  for (const Expression *expr : param_value) {
    fi->param_value.push_back(expr->clone());
  }
  return fi;
}

void FunctionInvocationHIPWarpShuffleBuiltIn::OutputFuncName(
    std::ostream &out) const {
  out << kWarpShuffleNames[built_in_];
}

const Type &FunctionInvocationHIPWarpShuffleBuiltIn::GetParameterType(
    size_t idx) const {
  // T var arg
  if (idx == 0) return type_;

  // lane/delta/mask arg
  if (built_in_ == kShflUp || built_in_ == kShflDown ||
      built_in_ == kShflUpSync || built_in_ == kShflDownSync) {
    return Type::get_simple_type(eUInt);
  }
  return Type::get_simple_type(eInt);
}

// warp shuffle end

// WARP Reduce start

FunctionInvocationHIPWarpReduceBuiltIn *
FunctionInvocationHIPWarpReduceBuiltIn::make_random(CGContext &cg_context,
                                                    const Type &type) {
  std::vector<const Type *> param_types;
  enum BuiltIn func = FunctionSelector(type, &param_types);

  if (func == kIdentity) {
    return NULL;
  }

  FunctionInvocationHIPWarpReduceBuiltIn *fi =
      new FunctionInvocationHIPWarpReduceBuiltIn(func, type);

  for (size_t i = 0; i < param_types.size(); ++i) {
    fi->param_value.push_back(
        Expression::make_random(cg_context, param_types[i]));
  }
  return fi;
}

enum FunctionInvocationHIPWarpReduceBuiltIn::BuiltIn
FunctionInvocationHIPWarpReduceBuiltIn::FunctionSelector(
    const Type &type, std::vector<const Type *> *params) {
  assert(params != NULL);
  params->clear();

  // support all scalar int types
  if (type.eType != eSimple || type.simple_type == eFloat) {
    return kIdentity;
  }

  assert(hip_warp_reduce_func_table != NULL);
  VectorFilter filter(hip_warp_reduce_func_table);

  int rnd = rnd_upto(filter.get_max_prob(), &filter);
  enum BuiltIn func = (enum BuiltIn)filter.lookup(rnd);

  params->push_back(&type);
  return func;
}

void FunctionInvocationHIPWarpReduceBuiltIn::InitTables() {
  hip_warp_reduce_func_table = new DistributionTable();
  hip_warp_reduce_func_table->add_entry(kReduceAddSync, 10);
  hip_warp_reduce_func_table->add_entry(kReduceMinSync, 10);
  hip_warp_reduce_func_table->add_entry(kReduceMaxSync, 10);
  hip_warp_reduce_func_table->add_entry(kReduceAndSync, 10);
  hip_warp_reduce_func_table->add_entry(kReduceOrSync, 10);
  hip_warp_reduce_func_table->add_entry(kReduceXorSync, 10);
}

FunctionInvocationHIPWarpReduceBuiltIn *
FunctionInvocationHIPWarpReduceBuiltIn::clone() const {
  FunctionInvocationHIPWarpReduceBuiltIn *fi =
      new FunctionInvocationHIPWarpReduceBuiltIn(built_in_, type_);
  for (const Expression *expr : param_value) {
    fi->param_value.push_back(expr->clone());
  }
  return fi;
}

void FunctionInvocationHIPWarpReduceBuiltIn::OutputFuncName(
    std::ostream &out) const {
  out << kWarpReduceNames[built_in_];
}

const Type &FunctionInvocationHIPWarpReduceBuiltIn::GetParameterType(
    size_t idx) const {
  return type_;  // T var
}

// WARP Reduce end

// atomics
// ==========================================
// HIP ATOMIC GENERATION
// ==========================================

FunctionInvocationHIPAtomicBuiltIn *
FunctionInvocationHIPAtomicBuiltIn::make_random(CGContext &cg_context,
                                                const Type &type) {
  // explicitly block other int types that are not supported
  bool valid_atomic = (type.simple_type == eInt || type.simple_type == eUInt ||
                       type.simple_type == eLong || type.simple_type == eULong ||
                       type.simple_type == eLongLong || type.simple_type == eULongLong);
                       
  if (!valid_atomic) {
    return NULL;
  }

  std::vector<const Type *> param_types;
  enum BuiltIn func = FunctionSelector(type, &param_types);

  if (func == kIdentity) {
    return NULL;
  }

  FunctionInvocationHIPAtomicBuiltIn *fi =
      new FunctionInvocationHIPAtomicBuiltIn(func, type);

  for (size_t i = 0; i < param_types.size(); ++i) {
    if (i == 0) {
      // First arg is always the address. We fetch a local/global variable 
      // of matching type and flag its address as taken.
      const Type *base_type = param_types[i];
      CVQualifiers qfer;
      qfer.add_qualifiers(false, false);
      std::vector<const Variable *> invalid_vars;

      Variable *addr_var =
          VariableSelector::select(Effect::WRITE, cg_context, base_type, &qfer,
                                   invalid_vars, eExact, eParentLocal);

      if (!addr_var || addr_var->isBitfield_ ||
          addr_var->type->simple_type != base_type->simple_type) {
        delete fi;
        return NULL;
      }

      ExpressionVariable *addr_expr =
          new ExpressionVariable(*addr_var, param_types[i]);
      Bookkeeper::record_address_taken(addr_var);
      Bookkeeper::record_volatile_access(addr_var, -1, false);

      fi->param_value.push_back(addr_expr);
    } else {
      // Remaining args (val, compare) are generated randomly
      fi->param_value.push_back(
          Expression::make_random(cg_context, param_types[i]));
    }
  }
  return fi;
}

enum FunctionInvocationHIPAtomicBuiltIn::BuiltIn
FunctionInvocationHIPAtomicBuiltIn::FunctionSelector(
    const Type &type, std::vector<const Type *> *params) {
  assert(params != NULL);
  params->clear();

  assert(hip_atomic_func_table != NULL);
  VectorFilter filter(hip_atomic_func_table);

  enum BuiltIn func;
  
  // Reroll until we get a valid match for our type
  // (Inc and Dec ONLY support unsigned int per standard HIP rules)
  do {
    int rnd = rnd_upto(filter.get_max_prob(), &filter);
    func = (enum BuiltIn)filter.lookup(rnd);
  } while ((func == kAtomicInc || func == kAtomicIncSystem ||
            func == kAtomicDec || func == kAtomicDecSystem) &&
           type.simple_type != eUInt);

  // Parameter 0: address variable type
  params->push_back(&type);

  // Parameter 1 & 2: val / compare
  if (func == kAtomicCAS || func == kAtomicCASSystem) {
    params->push_back(&type);  // compare
    params->push_back(&type);  // val
  } else {
    // All other operations take exactly 1 additional parameter.
    // Add/Sub/Min/Max/Bitwise: operand val.
    // Inc/Dec: ceiling/floor wrap val.
    params->push_back(&type); 
  }

  return func;
}

void FunctionInvocationHIPAtomicBuiltIn::InitTables() {
  hip_atomic_func_table = new DistributionTable();
  hip_atomic_func_table->add_entry(kAtomicAdd, 10);
  hip_atomic_func_table->add_entry(kAtomicAddSystem, 10);
  hip_atomic_func_table->add_entry(kAtomicSub, 10);
  hip_atomic_func_table->add_entry(kAtomicSubSystem, 10);
  hip_atomic_func_table->add_entry(kAtomicMin, 10);
  hip_atomic_func_table->add_entry(kAtomicMinSystem, 10);
  hip_atomic_func_table->add_entry(kAtomicMax, 10);
  hip_atomic_func_table->add_entry(kAtomicMaxSystem, 10);
  hip_atomic_func_table->add_entry(kAtomicExch, 10);
  hip_atomic_func_table->add_entry(kAtomicExchSystem, 10);
  hip_atomic_func_table->add_entry(kAtomicCAS, 10);
  hip_atomic_func_table->add_entry(kAtomicCASSystem, 10);
  hip_atomic_func_table->add_entry(kAtomicAnd, 10);
  hip_atomic_func_table->add_entry(kAtomicAndSystem, 10);
  hip_atomic_func_table->add_entry(kAtomicOr, 10);
  hip_atomic_func_table->add_entry(kAtomicOrSystem, 10);
  hip_atomic_func_table->add_entry(kAtomicXor, 10);
  hip_atomic_func_table->add_entry(kAtomicXorSystem, 10);
  hip_atomic_func_table->add_entry(kAtomicInc, 10);
  hip_atomic_func_table->add_entry(kAtomicIncSystem, 10);
  hip_atomic_func_table->add_entry(kAtomicDec, 10);
  hip_atomic_func_table->add_entry(kAtomicDecSystem, 10);
}

FunctionInvocationHIPAtomicBuiltIn *
FunctionInvocationHIPAtomicBuiltIn::clone() const {
  FunctionInvocationHIPAtomicBuiltIn *fi =
      new FunctionInvocationHIPAtomicBuiltIn(built_in_, type_);
  for (const Expression *expr : param_value) {
    fi->param_value.push_back(expr->clone());
  }
  return fi;
}

void FunctionInvocationHIPAtomicBuiltIn::OutputFuncName(
    std::ostream &out) const {
  out << kAtomicNames[built_in_];
}

void FunctionInvocationHIPAtomicBuiltIn::Output(std::ostream &out) const {
  OutputFuncName(out);
  out << '(';

  for (size_t idx = 0; idx < param_value.size(); ++idx) {
    // Wrap the first parameter in an address-of operator
    if (idx == 0) {
      out << "&(";
      param_value[idx]->Output(out);
      out << ")";
    } else {
      param_value[idx]->Output(out);
    }

    if (idx < param_value.size() - 1) {
      out << ", ";
    }
  }
  out << ')';
}

const Type &FunctionInvocationHIPAtomicBuiltIn::GetParameterType(
    size_t idx) const {
  return type_;
}
}  // namespace HIPSmith