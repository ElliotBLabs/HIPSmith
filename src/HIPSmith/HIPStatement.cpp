#include "HIPSmith/HIPStatement.h"

#include "Block.h"
#include "CGContext.h"
#include "HIPSmith/HIPOptions.h"
#include "HIPSmith/StatementHIPSync.h"
#include "ProbabilityTable.h"
#include "VectorFilter.h"

namespace HIPSmith {
namespace {
DistributionTable *hip_stmt_table = NULL;
}  // namespace

void HIPStatement::InitProbabilityTable() {
  hip_stmt_table = new DistributionTable();
  hip_stmt_table->add_entry(kSync, 10);
}

HIPStatement *HIPStatement::make_random(CGContext &cg_context,
                                        enum HIPStatementType st) {
  if (st == kNone) {
    assert(hip_stmt_table != NULL);

    // remember this is our total weight of all statement probabilities
    int num = rnd_upto(10);
    st = (HIPStatementType)VectorFilter(hip_stmt_table).lookup(num);
  }

  HIPStatement *stmt = NULL;
  switch (st) {
    case kSync:
      if (HIPOptions::hip_sync()) {
        stmt = StatementHIPSync::make_random(cg_context);
      }
      break;
    default:
      assert(false);
  }
  return stmt;
}

// exposed csmith hook
Statement *make_random_st(CGContext &cg_context) {
  return HIPStatement::make_random(cg_context, HIPStatement::kNone);
}

}  // namespace HIPSmith