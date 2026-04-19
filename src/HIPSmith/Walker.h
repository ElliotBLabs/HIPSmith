// Generic walker for traversing the statements of a function.
// This handles the structural traversal of the AST for HIPSmith.

#ifndef _WALKER_H_
#define _WALKER_H_

#include <iostream>
#include <memory>
#include <stack>
#include <vector>

#include "Block.h"
#include "CommonMacros.h"
#include "Function.h"
#include "Statement.h"

namespace HIPSmith {
namespace Walker {
namespace Internal {

class BlockWalker;

// Simple factory method for creating a BlockWalker at the start of a block.
BlockWalker *CreateBlockWalker(Block *block);

// Create a BlockWalker at a specified position in the block.
BlockWalker *CreateBlockWalkerAtStatement(Block *block, Statement *statement);

// Generic type list, simply holds a compile time list of types.
template <eStatementType... types>
struct TypeList;

// All of the statement types that will be traversed by the Walker in HIPSmith.
// Note: eCLStatement has been removed.
typedef TypeList<eAssign, eBlock, eFor, eIfElse, eInvoke, eReturn, eContinue,
                 eBreak, eGoto, eArrayOp>
    StatementList;

// Base of the Walker hierarchy.
class WalkerBase {
 public:
  WalkerBase() : block_(NULL), statement_(NULL) {}
  virtual ~WalkerBase() {}
  bool AdvanceOne();
  bool AdvanceToStatement(Statement *statement);
  Block *block_;
  Statement *statement_;

 private:
  std::vector<Statement *>::iterator block_it_;
};

// Basic declaration, templated with our list of statements.
template <class TypeList>
class Walker;

// Basic implementation declaration, templated with a statement.
template <eStatementType Type>
class WalkerImpl;

// End of the recursive Walker hierarchy.
template <>
class Walker<TypeList<>> : public virtual WalkerBase {};

// Recursive structure for the walker.
template <eStatementType Type, eStatementType... Types>
class Walker<TypeList<Type, Types...>> : public WalkerImpl<Type>,
                                         public Walker<TypeList<Types...>> {};

// Specialisation for For statements.
template <>
class WalkerImpl<eFor> : public virtual WalkerBase {
 public:
  bool Advance(Statement *statement);
  std::unique_ptr<BlockWalker> for_body_;
};

// Specialisation for the if-then-else implementation.
template <>
class WalkerImpl<eIfElse> : public virtual WalkerBase {
 public:
  bool Advance(Statement *statement);
  std::unique_ptr<BlockWalker> if_body_;
  std::unique_ptr<BlockWalker> else_body_;
};

// Goto implementation.
template <>
class WalkerImpl<eGoto> : public virtual WalkerBase {
 public:
  bool Advance(Statement *statement);
  Statement *destination_;
  bool goto_destination_is_forward_;
};

// Generic template for the Walker of a given statement.
template <eStatementType S>
class WalkerImpl : public virtual WalkerBase {
 public:
  bool Advance(Statement *statement);
};

template <>
bool WalkerImpl<eBlock>::Advance(Statement *statement);

template <eStatementType S>
bool WalkerImpl<S>::Advance(Statement *statement) {
  return true;
}

// Simple Walker specifically for walking through a block.
class BlockWalker : public Walker<StatementList> {
 public:
  BlockWalker() {}
  explicit BlockWalker(Block *block) { WalkerBase::block_ = block; }
  bool AdvanceBlock() {
    return WalkerBase::AdvanceOne() && AdvanceSelector(WalkerBase::statement_);
  }
  bool AdvanceSelector(Statement *statement);
};

}  // namespace Internal

// Traverses the blocks in a function.
class FunctionWalker {
 public:
  explicit FunctionWalker(Function *function) : function_(function) {
    assert(function_);
    block_walker_.reset(Internal::CreateBlockWalker(function_->body));
  }
  FunctionWalker(FunctionWalker &&other) = default;
  FunctionWalker &operator=(FunctionWalker &&other) = default;
  virtual ~FunctionWalker() {}

  static FunctionWalker *CreateFunctionWalkerAtStatement(Function *function,
                                                         Statement *statement);

  virtual bool Advance();
  virtual bool Next();

  Statement *GetCurrentStatement() const {
    return block_walker_->WalkerBase::statement_;
  }
  eStatementType GetCurrentStatementType() const {
    return block_walker_->WalkerBase::statement_->get_type();
  }
  Block *GetCurrentBlock() const { return block_walker_->WalkerBase::block_; }

  FunctionWalker *GetGotoDestination() {
    return CreateFunctionWalkerAtStatement(
        function_, block_walker_->Internal::WalkerImpl<eGoto>::destination_);
  }
  bool GotoDestinationIsForward() {
    return block_walker_
        ->Internal::WalkerImpl<eGoto>::goto_destination_is_forward_;
  }

  bool operator==(const FunctionWalker &other) const {
    return function_ == other.function_ &&
           GetCurrentStatement() == other.GetCurrentStatement();
  }
  bool operator!=(const FunctionWalker &other) const {
    return !(*this == other);
  }

  int EnteredBranch() const { return blocks_entered_; }
  int ExitedBranch() const { return blocks_exited_; }

 protected:
  bool MustEnterBranch();
  void EnterBranch(Internal::BlockWalker *block_walker);

  Function *function_;
  std::unique_ptr<Internal::BlockWalker> block_walker_;
  std::stack<std::unique_ptr<Internal::BlockWalker>> nested_blocks_;
  int blocks_entered_;
  int blocks_exited_;

 private:
  FunctionWalker() {}
  DISALLOW_COPY_AND_ASSIGN(FunctionWalker);
};

}  // namespace Walker
}  // namespace HIPSmith

#endif  // _WALKER_H_