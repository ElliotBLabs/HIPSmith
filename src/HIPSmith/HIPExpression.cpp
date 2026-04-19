#include "HIPSmith/HIPExpression.h"

#include "CGContext.h"
#include "CGOptions.h"
#include "HIPSmith/ExpressionVector.h"
#include "HIPSmith/HIPOptions.h"
#include "ProbabilityTable.h"
#include "Type.h"
#include "VectorFilter.h"
#include "random.h"

class CVQualifiers;

namespace HIPSmith {
namespace {
DistributionTable *hip_expr_table = NULL;
}  // namespace

Expression *HIPExpression::make_random(CGContext &cg_context, const Type *type,
                                       const CVQualifiers *qfer,
                                       enum HIPExpressionType tt) {
  // kNone is used for not specifying an expression type, as it does not require
  // recursive generation.
  // If the expected type is a vector type, then automatically set the term type
  // to a vector expression.
  if (tt == kNone && type->eType == eVector) tt = kVector;

  if (tt == kNone) {
    // The probability for Expression picking a HIPExpression is fixed upstream.
    // If HIPExpression is picked, we perform a lookup here.
    assert(hip_expr_table != NULL);

    // We roll out of 100 since we gave kVector a weight of 100 in the table.
    int num = rnd_upto(100);
    tt = type->eType == eVector
             ? kVector
             : (HIPExpressionType)VectorFilter(hip_expr_table).lookup(num);

    // Restrictions on vector types.
    if (tt == kVector) {
      if (!HIPOptions::vectors() ||
          (type->eType != eSimple && type->eType != eVector) ||
          (cg_context.expr_depth + 2 > CGOptions::max_expr_depth()))
        return NULL;
    }
  }

  Expression *expr = NULL;
  switch (tt) {
    case kVector:
      // Route directly to your vector expression generator
      expr = ExpressionVector::make_random(cg_context, type, qfer, 0);
      break;
    default:
      assert(false);
  }
  return expr;
}

void HIPExpression::InitProbabilityTable() {
  hip_expr_table = new DistributionTable();
  hip_expr_table->add_entry(kVector, 100);
  ExpressionVector::InitProbabilityTable();
}

Expression *make_random(CGContext &cg_context, const Type *type,
                        const CVQualifiers *qfer) {
  return HIPExpression::make_random(cg_context, type, qfer,
                                    HIPExpression::kNone);
}

}  // namespace HIPSmith