// Root of expressions specific to HIP.
// Acts as a hook by which HIPSmith can inject behaviour into csmith.

#ifndef _HIPSMITH_HIPEXPRESSION_H_
#define _HIPSMITH_HIPEXPRESSION_H_

#include "CommonMacros.h"
#include "Expression.h"

class CGContext;
class CVQualifiers;
class DistributionTable;
class Type;
class VectorFilter;

namespace HIPSmith {

// All HIP related expressions derive from this.
// The type of HIP expression to use is handled here, instead of in
// base Csmith's Expression class. So we have our own probability table.
class HIPExpression : public Expression {
 public:
  // Dynamic type information.
  enum HIPExpressionType {
    kNone = 0,  // Sentinel value.
    kVector     // Restricted exclusively to vectors for HIPSmith.
  };

  explicit HIPExpression(HIPExpressionType type)
      : Expression(eHIPExpression), hip_expression_type_(type) {}

  // Must be careful with the move constructors. These will call the copy
  // constructor in Expression.
  HIPExpression(HIPExpression &&other) = default;
  HIPExpression &operator=(HIPExpression &&other) = default;
  virtual ~HIPExpression() {}

  // Factory for creating a random HIP expression. This will typically be
  // called by Expression::make_random.
  static Expression *make_random(CGContext &cg_context, const Type *type,
                                 const CVQualifiers *qfer,
                                 enum HIPExpressionType tt);

  // Create the random probability table, should be called once on startup.
  static void InitProbabilityTable();

  // Getter for hip_expression_type_
  enum HIPExpressionType GetHIPExpressionType() const {
    return hip_expression_type_;
  }

 private:
  HIPExpressionType hip_expression_type_;

  // Table used for deciding on which HIP expression to generate.
  // static DistributionTable *hip_expr_table_;

  DISALLOW_COPY_AND_ASSIGN(HIPExpression);
};

// Hook method for csmith.
Expression *make_random(CGContext &cg_context, const Type *type,
                        const CVQualifiers *qfer);

}  // namespace HIPSmith

#endif  // _HIPSMITH_HIPEXPRESSION_H_