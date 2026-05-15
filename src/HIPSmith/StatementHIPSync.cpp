#include "HIPSmith/StatementHIPSync.h"

#include "CGContext.h"
#include "Constant.h"
#include "Expression.h"
#include "OutputMgr.h"
#include "Type.h"
#include "random.h"

namespace HIPSmith {

StatementHIPSync::StatementHIPSync(Block* block, HIPSyncStatementType type)
    : HIPStatement(kSync, block), sync_type_(type) {}

StatementHIPSync::~StatementHIPSync() {}

StatementHIPSync* StatementHIPSync::make_random(CGContext& cg_context) {
  HIPSyncStatementType type =
      (HIPSyncStatementType)rnd_upto(kNumSyncFenceTypes);

  return new StatementHIPSync(cg_context.get_current_block(), type);
}

void StatementHIPSync::Output(std::ostream& out, FactMgr* fm,
                              int indent) const {
  output_tab(out, indent);

  switch (sync_type_) {
    case kThreadFence:
      out << "__threadfence();";
      break;
    case kThreadFenceBlock:
      out << "__threadfence_block();";
      break;
    case kThreadFenceSystem:
      out << "__threadfence_system();";
      break;
    case kSyncThreads:
      out << "__syncthreads();";
      break;
    default:
      assert(false && "Unknown HIPSyncStatementType");
  }

  out << std::endl;
}

bool StatementHIPSync::visit_facts(std::vector<const Fact*>& inputs,
                                   CGContext& cg_context) const {
  return true;
}

void StatementHIPSync::get_exprs(std::vector<const Expression*>& exps) const {}

}  // namespace HIPSmith