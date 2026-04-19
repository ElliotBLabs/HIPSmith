// Vector based expressions.
// Expressions are produced such that an intermediate rvalue vector is produced,
// which is then transformed into the expected return type.
// The intermediate vector is produced by a sequence of sub-expressions (e.g.
// scalar, result of binary operation on two premade vectors, etc).
// Once the intermediate vector is made, it can be transformed into a scalar
// by accessing a single component. (Note: HIP does not support multi-component
// swizzling).
//
// Example: Calling make_random with return type 'int':
//
// make_int4(5, 10, 1 + 2, some_vec.x)    Intermediate int4 made from four
// sub-expressions. make_int4(5, 10, 1 + 2, some_vec.x).w  Component accessed to
// give the requested 'int'.
//

#ifndef _HIPSMITH_EXPRESSIONVECTOR_H_
#define _HIPSMITH_EXPRESSIONVECTOR_H_

#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "CGContext.h"
#include "CVQualifiers.h"
#include "HIPSmith/HIPExpression.h"
#include "HIPSmith/Vector.h"
#include "Type.h"

class Expression;

namespace HIPSmith {

class ExpressionVector : public HIPExpression {
 public:
  // Type of vector expression.
  enum VectorExprType { kLiteral = 0, kVariable, kSIMD, kBuiltIn };

  // Constructor for when we are returning the full vector (no component
  // access).
  ExpressionVector(std::vector<std::unique_ptr<const Expression>>&& exprs,
                   const Type& type, int size)
      : HIPExpression(kVector),
        exprs_(std::forward<std::vector<std::unique_ptr<const Expression>>>(
            exprs)),
        type_(type),
        size_(size),
        is_component_access_(false),
        component_access_index_(-1) {}  // -1 indicates no access

  // Constructor for when we are extracting a single scalar component (e.g., .x,
  // .y).
  ExpressionVector(std::vector<std::unique_ptr<const Expression>>&& exprs,
                   const Type& type, int size, int access_index)
      : HIPExpression(kVector),
        exprs_(std::forward<std::vector<std::unique_ptr<const Expression>>>(
            exprs)),
        type_(type),
        size_(size),
        is_component_access_(true),
        component_access_index_(access_index) {}

  ExpressionVector(ExpressionVector&& other) = default;
  ExpressionVector& operator=(ExpressionVector&& other) = default;
  virtual ~ExpressionVector() {}

  // Create an expression that produces a vector value.
  // The expression can be a vector literal, vector variable, an operation on
  // one or two vectors or calling a built-in vector function.
  // The type of the expression will match 'type', if type is a scalar,
  // a single component of the vector will be accessed to match.
  // size will determine the length of the vector produced (before component
  // access), it must be a valid HIP vector length (1, 2, 3, or 4), or 0 (for a
  // random length).
  static ExpressionVector* make_random(CGContext& cg_context, const Type* type,
                                       const CVQualifiers* qfer, int size);

  // Create a vector of constant elements.
  static ExpressionVector* make_constant(const Type* type, int value);

  // Initialise table for selecting vector expression type. Should be called
  // once on start-up.
  static void InitProbabilityTable();

  // Implementations of pure virtual methods in Expression. Most of these are
  // trivial, as the expression evaluates to a runtime constant.
  ExpressionVector* clone() const;
  const Type& get_type() const { return type_; }
  CVQualifiers get_qualifiers() const { return CVQualifiers(true, false); }
  void get_eval_to_subexps(std::vector<const Expression*>& subs) const;
  void get_referenced_ptrs(std::vector<const Variable*>& ptrs) const;
  unsigned get_complexity() const;
  void Output(std::ostream& out) const;

  const std::vector<std::unique_ptr<const Expression>>& GetExpressions() const {
    return exprs_;
  }

 private:
  // Vector of expressions that when combined will form a HIP vector.
  std::vector<std::unique_ptr<const Expression>> exprs_;
  // Basic type of the expression, can be a scalar.
  const Type& type_;
  // Vector length (typically 1, 2, 3, or 4 for HIP).
  const int size_;

  // Vector accesses.
  bool is_component_access_;    // Is the access for a single scalar component
                                // (i.e. vec.x).
  int component_access_index_;  // Which component (0 for x, 1 for y, 2 for z, 3
                                // for w).

  DISALLOW_COPY_AND_ASSIGN(ExpressionVector);
};

}  // namespace HIPSmith

#endif  // _HIPSMITH_EXPRESSIONVECTOR_H_