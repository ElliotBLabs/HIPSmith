#ifndef _HIPSMITH_HIPSTATEMENT_H_
#define _HIPSMITH_HIPSTATEMENT_H_

#include "CommonMacros.h"
#include "Statement.h"

class CGContext;

namespace HIPSmith {

class HIPStatement : public Statement {
 public:
  enum HIPStatementType {
    kNone = 0,  // Sentinel value
    kSync,
  };

  HIPStatement(HIPStatementType type, Block* block)
      : Statement(eHIPStatement, block), hip_statement_type_(type) {}
  HIPStatement(HIPStatement&& other) = default;
  HIPStatement& operator=(HIPStatement&& other) = default;
  virtual ~HIPStatement() {}

  // Factory for creating a random HIP statement.
  static HIPStatement* make_random(CGContext& cg_context,
                                   enum HIPStatementType st);

  // Initialise the probability table for selecting a random statement.
  static void InitProbabilityTable();

  enum HIPStatementType GetHIPStatementType() const {
    return hip_statement_type_;
  }

 private:
  HIPStatementType hip_statement_type_;

  DISALLOW_COPY_AND_ASSIGN(HIPStatement);
};

// Hook method called by Csmith's Statement::make_random
Statement* make_random_st(CGContext& cg_context);

}  // namespace HIPSmith

#endif  // _HIPSMITH_HIPSTATEMENT_H_