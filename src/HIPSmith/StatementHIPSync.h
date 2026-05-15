#ifndef _HIPSMITH_STATEMENT_HIP_SYNC_H_
#define _HIPSMITH_STATEMENT_HIP_SYNC_H_

#include "HIPSmith/HIPStatement.h"

class Expression;

namespace HIPSmith {

class StatementHIPSync : public HIPStatement {
 public:
  enum HIPSyncStatementType {
    kThreadFence = 0,
    kThreadFenceBlock,
    kThreadFenceSystem,
    kSyncThreads,
    kNumSyncFenceTypes  // Used for random selection
  };

  StatementHIPSync(Block* block, HIPSyncStatementType type);
  virtual ~StatementHIPSync();

  static StatementHIPSync* make_random(CGContext& cg_context);

  virtual void Output(std::ostream& out, FactMgr* fm,
                      int indent) const override;

  // Required Csmith AST traversals
  virtual bool visit_facts(std::vector<const Fact*>& inputs,
                           CGContext& cg_context) const override;
  virtual void get_blocks(std::vector<const Block*>& blks) const override {}
  virtual void get_exprs(std::vector<const Expression*>& exps) const override;

 private:
  HIPSyncStatementType sync_type_;
};

}  // namespace HIPSmith

#endif  // _HIPSMITH_STATEMENT_HIP_SYNC_H_