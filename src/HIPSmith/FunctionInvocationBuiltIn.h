#ifndef _HIPSMITH_HIPFUNCTIONINVOCATION_H_
#define _HIPSMITH_HIPFUNCTIONINVOCATION_H_

#include <map>
#include <ostream>
#include <vector>

#include "CommonMacros.h"
#include "FunctionInvocation.h"
#include "SafeOpFlags.h"

class CGContext;
class Expression;
class Type;

namespace HIPSmith {

// Base class for HIP built-in functions
class FunctionInvocationHIPBuiltIn : public FunctionInvocation {
 public:
  enum BuiltInType { kSyncPredicates = 0, kWarpVote, kWarpMatch, kWarpShuffle };

  FunctionInvocationHIPBuiltIn(enum BuiltInType built_in_type, const Type& type)
      : FunctionInvocation(eHIPBuiltin, SafeOpFlags::make_dummy_flags()),
        type_(type),
        built_in_type_(built_in_type) {}
  FunctionInvocationHIPBuiltIn(FunctionInvocationHIPBuiltIn&& other) = default;
  FunctionInvocationHIPBuiltIn& operator=(
      FunctionInvocationHIPBuiltIn&& other) = default;
  virtual ~FunctionInvocationHIPBuiltIn() {}

  static FunctionInvocationHIPBuiltIn* make_random(CGContext& cg_context,
                                                   const Type& type);
  static void InitTables();

  const Type& get_type() const override { return type_; }
  void Output(std::ostream& out) const override;
  void indented_output(std::ostream& out, int indent) const override;
  bool safe_invocation() const override { return true; }

  virtual void OutputFuncName(std::ostream& out) const = 0;
  virtual const Type& GetParameterType(size_t idx) const = 0;

  enum BuiltInType GetBuiltInType() const { return built_in_type_; }

 protected:
  const Type& type_;

 private:
  enum BuiltInType built_in_type_;
  DISALLOW_COPY_AND_ASSIGN(FunctionInvocationHIPBuiltIn);
};

// Specific implementation for the sync predicate functions
class FunctionInvocationHIPSyncBuiltIn : public FunctionInvocationHIPBuiltIn {
 public:
  enum BuiltIn {
    kIdentity = 0,  // Sentinel
    kSyncThreadsCount,
    kSyncThreadsAnd,
    kSyncThreadsOr
  };

  FunctionInvocationHIPSyncBuiltIn(enum BuiltIn built_in, const Type& type)
      : FunctionInvocationHIPBuiltIn(kSyncPredicates, type),
        built_in_(built_in) {}
  FunctionInvocationHIPSyncBuiltIn(FunctionInvocationHIPSyncBuiltIn&& other) =
      default;
  FunctionInvocationHIPSyncBuiltIn& operator=(
      FunctionInvocationHIPSyncBuiltIn&& other) = default;
  virtual ~FunctionInvocationHIPSyncBuiltIn() {}

  static FunctionInvocationHIPSyncBuiltIn* make_random(CGContext& cg_context,
                                                       const Type& type);
  static enum BuiltIn FunctionSelector(const Type& type,
                                       std::vector<const Type*>* params);
  static void InitTables();

  FunctionInvocationHIPSyncBuiltIn* clone() const override;
  void OutputFuncName(std::ostream& out) const override;
  const Type& GetParameterType(size_t idx) const override;

  enum BuiltIn GetBuiltIn() const { return built_in_; }

 private:
  enum BuiltIn built_in_;
  DISALLOW_COPY_AND_ASSIGN(FunctionInvocationHIPSyncBuiltIn);
};

// Specific implementation for the Warp Vote functions
class FunctionInvocationHIPWarpVoteBuiltIn
    : public FunctionInvocationHIPBuiltIn {
 public:
  enum BuiltIn {
    kIdentity = 0,  // Sentinel
    kAll,
    kAny,
    kBallot,
    kActiveMask,
    kAllSync,
    kAnySync,
    kBallotSync
  };

  FunctionInvocationHIPWarpVoteBuiltIn(enum BuiltIn built_in, const Type& type)
      : FunctionInvocationHIPBuiltIn(kWarpVote, type), built_in_(built_in) {}
  FunctionInvocationHIPWarpVoteBuiltIn(
      FunctionInvocationHIPWarpVoteBuiltIn&& other) = default;
  FunctionInvocationHIPWarpVoteBuiltIn& operator=(
      FunctionInvocationHIPWarpVoteBuiltIn&& other) = default;
  virtual ~FunctionInvocationHIPWarpVoteBuiltIn() {}

  static FunctionInvocationHIPWarpVoteBuiltIn* make_random(
      CGContext& cg_context, const Type& type);
  static enum BuiltIn FunctionSelector(const Type& type,
                                       std::vector<const Type*>* params);
  static void InitTables();

  FunctionInvocationHIPWarpVoteBuiltIn* clone() const override;
  void OutputFuncName(std::ostream& out) const override;
  const Type& GetParameterType(size_t idx) const override;

  enum BuiltIn GetBuiltIn() const { return built_in_; }

 private:
  enum BuiltIn built_in_;
  DISALLOW_COPY_AND_ASSIGN(FunctionInvocationHIPWarpVoteBuiltIn);
};

class FunctionInvocationHIPWarpMatchBuiltIn
    : public FunctionInvocationHIPBuiltIn {
 public:
  enum BuiltIn {
    kIdentity = 0,  // Sentinel
    kMatchAny,
    kMatchAll,
    kMatchAnySync,
    kMatchAllSync
  };

  FunctionInvocationHIPWarpMatchBuiltIn(enum BuiltIn built_in, const Type& type,
                                        const Type& t_type)
      : FunctionInvocationHIPBuiltIn(kWarpMatch, type),
        built_in_(built_in),
        t_type_(t_type) {}
  FunctionInvocationHIPWarpMatchBuiltIn(
      FunctionInvocationHIPWarpMatchBuiltIn&& other) = default;
  FunctionInvocationHIPWarpMatchBuiltIn& operator=(
      FunctionInvocationHIPWarpMatchBuiltIn&& other) = default;
  virtual ~FunctionInvocationHIPWarpMatchBuiltIn() {}

  static FunctionInvocationHIPWarpMatchBuiltIn* make_random(
      CGContext& cg_context, const Type& type);
  static enum BuiltIn FunctionSelector(const Type& type,
                                       std::vector<const Type*>* params,
                                       const Type& t_type);
  static void InitTables();

  FunctionInvocationHIPWarpMatchBuiltIn* clone() const override;
  void OutputFuncName(std::ostream& out) const override;
  void Output(std::ostream& out) const override;
  const Type& GetParameterType(size_t idx) const override;

  enum BuiltIn GetBuiltIn() const { return built_in_; }

 private:
  enum BuiltIn built_in_;
  const Type& t_type_; 
  DISALLOW_COPY_AND_ASSIGN(FunctionInvocationHIPWarpMatchBuiltIn);
};

// shuffle start
class FunctionInvocationHIPWarpShuffleBuiltIn
    : public FunctionInvocationHIPBuiltIn {
 public:
  enum BuiltIn {
    kIdentity = 0,  // Sentinel
    kShfl,
    kShflUp,
    kShflDown,
    kShflXor,
    kShflSync,
    kShflUpSync,
    kShflDownSync,
    kShflXorSync
  };

  FunctionInvocationHIPWarpShuffleBuiltIn(enum BuiltIn built_in, const Type& type)
      : FunctionInvocationHIPBuiltIn(kWarpShuffle, type),
        built_in_(built_in) {}
  FunctionInvocationHIPWarpShuffleBuiltIn(
      FunctionInvocationHIPWarpShuffleBuiltIn&& other) = default;
  FunctionInvocationHIPWarpShuffleBuiltIn& operator=(
      FunctionInvocationHIPWarpShuffleBuiltIn&& other) = default;
  virtual ~FunctionInvocationHIPWarpShuffleBuiltIn() {}

  static FunctionInvocationHIPWarpShuffleBuiltIn* make_random(
      CGContext& cg_context, const Type& type);
  static enum BuiltIn FunctionSelector(const Type& type,
                                       std::vector<const Type*>* params);
  static void InitTables();

  FunctionInvocationHIPWarpShuffleBuiltIn* clone() const override;
  void OutputFuncName(std::ostream& out) const override;
  const Type& GetParameterType(size_t idx) const override;

  enum BuiltIn GetBuiltIn() const { return built_in_; }

 private:
  enum BuiltIn built_in_;
  DISALLOW_COPY_AND_ASSIGN(FunctionInvocationHIPWarpShuffleBuiltIn);
};

// shuffle end

}  // namespace HIPSmith

#endif  // _HIPSMITH_HIPFUNCTIONINVOCATION_H_