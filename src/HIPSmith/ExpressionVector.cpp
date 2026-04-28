#include "HIPSmith/ExpressionVector.h"

#include <map>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "CGContext.h"
#include "CGOptions.h"
#include "Constant.h"
#include "Expression.h"
#include "ExpressionFuncall.h"
#include "ExpressionVariable.h"
#include "ProbabilityTable.h"
#include "Type.h"
#include "VectorFilter.h"
#include "random.h"

namespace HIPSmith {
namespace {
DistributionTable *vector_expr_table = NULL;
}  // namespace

ExpressionVector *ExpressionVector::make_random(CGContext &cg_context,
                                                const Type *type,
                                                const CVQualifiers *qfer,
                                                int size) {
  // If we have been forced in here, but the expression depth is too high,
  // return a plain constant casted to the vector type.
  if (cg_context.expr_depth + 2 > CGOptions::max_expr_depth()) {
    assert(type->eType == eVector);
    std::vector<std::unique_ptr<const Expression>> exprs;
    exprs.emplace_back(
        Constant::make_random(&Vector::DemoteVectorTypeToType(type)));
    return new ExpressionVector(std::move(exprs), *type, type->vector_length_);
  }

  // Check the type. If type is not a vector type, then the expression should
  // produce a single value, so the result must extract ONE component.
  // THE FIX: If the requested type is ALREADY a vector, we MUST match its exact
  // size.
  if (type->eType == eVector) {
    size = type->vector_length_;
  } else if (!size) {
    size = Vector::GetRandomVectorLength(0);
  }

  assert(vector_expr_table != NULL);
  int num = rnd_upto(30);
  enum VectorExprType vec_expr_type =
      (VectorExprType)VectorFilter(vector_expr_table).lookup(num);
  // Create a series of sub-expressions.
  std::vector<std::unique_ptr<const Expression>> exprs;
  if (vec_expr_type == kLiteral) {
    // before this was more fun as could put vectors inside vectors but not in
    // hip :(
    // 1. Get the underlying scalar type (e.g., 'ulong')
    const Type *scalar_type = &Vector::DemoteVectorTypeToType(type);
    // 2. Generate exactly 'size' number of scalar arguments
    for (int i = 0; i < size; ++i) {
      if (rnd_flipcoin(50)) {
        // 50% chance to generate a plain constant
        exprs.emplace_back(Constant::make_random(scalar_type));
      } else {
        // 50% chance to generate a scalar expression
        exprs.emplace_back(
            Expression::make_random(cg_context, scalar_type, qfer));
      }
    }
  } else if (vec_expr_type == kVariable) {
    const Type *vec_type = Vector::PromoteTypeToVectorType(type, size);
    exprs.emplace_back(
        ExpressionVariable::make_random(cg_context, vec_type, qfer));
  } else if (vec_expr_type == kSIMD) {
    // Produce an expression that performs a series of operations on vectors. We
    // borrow from csmith's expression generation where possible.
    const Type *vec_type = Vector::PromoteTypeToVectorType(type, size);
    exprs.emplace_back(Expression::make_random(cg_context, vec_type, qfer,
                                               false, false, eFunction));
    assert(exprs.back()->get_type().eType == eVector);
  }
  // else /*kBuiltIn*/ {
  //   const Type *vec_type = Vector::PromoteTypeToVectorType(type, size);
  //   exprs.emplace_back(new ExpressionFuncall(
  //       *FunctionInvocationBuiltIn::make_random(cg_context, *vec_type)));
  // }

  // Handle single component extraction if the expected return type is scalar.
  ExpressionVector *expr_vec = NULL;
  if (type->eType != eVector) {
    // We need to return a scalar, so we extract exactly one random component (0
    // to size-1).
    int access_index = rnd_upto(size);
    expr_vec =
        new ExpressionVector(std::move(exprs), *type, size, access_index);
  } else {
    // Returning the whole vector, no component extraction.
    expr_vec = new ExpressionVector(std::move(exprs), *type, size);
  }

  return expr_vec;
}

ExpressionVector *ExpressionVector::make_constant(const Type *type, int value) {
  assert(type->eType == eVector);
  std::vector<std::unique_ptr<const Expression>> exprs;
  exprs.emplace_back(Constant::make_int(value));
  return new ExpressionVector(std::move(exprs), *type, type->vector_length_);
}

void ExpressionVector::InitProbabilityTable() {
  vector_expr_table = new DistributionTable();
  vector_expr_table->add_entry(kLiteral, 10);
  vector_expr_table->add_entry(kVariable, 10);
  vector_expr_table->add_entry(kSIMD, 10);
  vector_expr_table->add_entry(kBuiltIn, 10);
}

ExpressionVector *ExpressionVector::clone() const {
  std::vector<std::unique_ptr<const Expression>> exprs;
  for (const auto &expr : exprs_) exprs.emplace_back(expr->clone());

  if (is_component_access_) {
    return new ExpressionVector(std::move(exprs), type_, size_,
                                component_access_index_);
  } else {
    return new ExpressionVector(std::move(exprs), type_, size_);
  }
}

void ExpressionVector::get_eval_to_subexps(
    std::vector<const Expression *> &subs) const {
  subs.push_back(this);
  for (const std::unique_ptr<const Expression> &expr : exprs_)
    expr->get_eval_to_subexps(subs);
}

void ExpressionVector::get_referenced_ptrs(
    std::vector<const Variable *> &ptrs) const {
  for (const std::unique_ptr<const Expression> &expr : exprs_)
    expr->get_referenced_ptrs(ptrs);
}

unsigned ExpressionVector::get_complexity() const {
  unsigned complexity = 1;
  for (const std::unique_ptr<const Expression> &expr : exprs_)
    complexity += expr->get_complexity();
  return complexity;
}

void ExpressionVector::Output(std::ostream &out) const {
  if (exprs_.empty()) {
    // FALLBACK: Empty array. Force zero-initialization.
    out << "make_";
    Vector::OutputVectorType(out, &type_, size_);
    out << "(";
    for (int i = 0; i < size_; ++i) out << "0" << (i < size_ - 1 ? ", " : "");
    out << ")";
  } else if (exprs_.size() == 1) {
    // WE HAVE 1 ARGUMENT. Is it a scalar or a whole vector?
    if (exprs_[0]->get_type().eType == eVector &&
        exprs_[0]->get_type().vector_length_ == size_) {
      // It is ALREADY a vector of the correct size (kVariable or kSIMD).
      // Do NOT wrap it in make_T! Just output it directly in parentheses.
      out << "(";
      exprs_[0]->Output(out);
      out << ")";
    } else {
      // It is a scalar (e.g., from the depth-limit bailout).
      // We must use make_T to broadcast it across all lanes.
      out << "make_";
      Vector::OutputVectorType(out, &type_, size_);
      out << "(";
      for (int i = 0; i < size_; ++i) {
        exprs_[0]->Output(out);
        if (i < size_ - 1) out << ", ";
      }
      out << ")";
    }
  } else {
    // NORMAL CASE (kLiteral): We have exact scalars to pack into a vector.
    out << "make_";
    Vector::OutputVectorType(out, &type_, size_);
    out << "(";
    for (unsigned idx = 0; idx < exprs_.size(); ++idx) {
      exprs_[idx]->Output(out);
      if (exprs_.size() - idx > 1) out << ", ";
    }
    out << ")";
  }

  // Output a single component access if required (e.g. .x, .y)
  if (is_component_access_) {
    out << '.';
    out << Vector::GetComponentChar(size_, component_access_index_);
  }
}

}  // namespace HIPSmith