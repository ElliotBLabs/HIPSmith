#include "HIPSmith/Vector.h"

#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Block.h"
#include "CGContext.h"
#include "Constant.h"
#include "HIPSmith/HIPOptions.h"
#include "Type.h"
#include "VariableSelector.h"
#include "random.h"

class ArrayVariable;
class CVQualifiers;
class Expression;
class Variable;

namespace HIPSmith {
namespace {
// Valid vector lengths for HIP.
const size_t kSizes[4] = {1, 2, 3, 4};
const size_t kSizesCount = 4;

// Outputs for single component accesses.
const char kCompSmall[4] = {'x', 'y', 'z', 'w'};
}  // namespace

std::map<std::pair<enum eSimpleType, unsigned>, const Type *>
    Vector::vector_types_;

Vector *Vector::CreateVectorVariable(const CGContext &cg_context, Block *blk,
                                     const std::string &name, const Type *type,
                                     const Expression *init,
                                     const CVQualifiers *qfer,
                                     const Variable *isFieldVarOf) {
  assert(type != NULL);
  assert((type->eType == eSimple || type->eType == eVector) &&
         type->simple_type != eVoid);

  // The passed type may be a vector type or a simple type.
  int size;
  if (type->eType == eVector) {
    size = type->vector_length_;
    type = &Vector::DemoteVectorTypeToType(type);
  } else {
    size = GetRandomVectorLength(0);
  }

  // Create vector and push to the back of the appropriate variable list.
  Vector *vector = new Vector(blk, name, type, init, qfer, size, isFieldVarOf);
  vector->add_init_value(Constant::make_random(type));
  blk ? blk->local_vars.push_back(vector)
      : VariableSelector::GetGlobalVariables()->push_back(vector);
  return vector;
}

Vector *Vector::itemize(void) const {
  return itemize({(int)rnd_upto(sizes[0])});
}

Vector *Vector::itemize(const std::vector<int> &const_indices) const {
  assert(collective == NULL);
  assert(const_indices.size() == 1);
  Vector *vec = new Vector(*this);
  VariableSelector::GetAllVariables()->push_back(vec);
  vec->comp_access_index_ = const_indices[0];
  vec->collective = this;
  return vec;
}

void Vector::Output(std::ostream &out) const {
  out << get_actual_name();
  if (collective == NULL) return;

  // We have an itemised vector, output single component access.
  out << '.';
  out << GetComponentChar(sizes[0], comp_access_index_);
}

void Vector::OutputDef(std::ostream &out, int indent) const {
  if (collective != NULL) return;
  output_tab(out, indent);
  OutputDecl(out);
  if (!no_loop_initializer()) {
    out << ';';
    outputln(out);
    return;
  }
  std::vector<std::string> init_strs;
  assert(init);
  init_strs.push_back(init->to_string());
  for (const auto &expr : init_values) init_strs.push_back(expr->to_string());

  out << " = " << build_initializer_str(init_strs) << ';';
  outputln(out);
}

void Vector::OutputDecl(std::ostream &out) const {
  // Trying to print all qualifiers prints the type as well. We don't allow
  // vector pointers regardless.
  qfer.OutputFirstQuals(out);
  OutputVectorType(out, type, sizes[0]);
  out << ' ' << get_actual_name();
}

void Vector::hash(std::ostream &out) const {
  if (collective != NULL || !CGOptions::compute_hash()) return;

  // Create temporary itemised vector for printing.
  Vector *vec = itemize({0});
  for (unsigned comp = 0; comp < sizes[0]; ++comp) {
    vec->comp_access_index_ = comp;
    output_tab(out, 1);
    out << "transparent_crc(";
    vec->Output(out);
    out << ", \"";
    vec->Output(out);
    out << "\", print_hash_value);";
    outputln(out);
  }
}

std::string Vector::build_initializer_str(
    const std::vector<std::string> &init_strings) const {
  /*static*/ unsigned long seed = 0xAB;
  std::string ret;
  ret.reserve(1000);

  // Build the HIP specific constructor prefix (e.g., make_int4)
  ret.append("make_");
  std::stringstream ss_type;
  OutputVectorType(ss_type, type, sizes[0]);
  ret.append(ss_type.str());
  ret.append("(");

  // Populate the constructor strictly with scalar elements.
  // Nested initialization is removed for HIP compatibility.
  for (unsigned idx = 0; idx < sizes[0]; ++idx) {
    unsigned long rnd_index = ((seed * seed + (idx + 7) * (idx + 13)) * 487);
    if (/*++*/ seed >= 0x7AB) seed = 0xAB;

    ret.append(init_strings[rnd_index % init_strings.size()]);

    if (idx < sizes[0] - 1) ret.append(", ");
  }
  ret.append(")");
  return ret;
}

int Vector::GetRandomVectorLength(int max) {
  if (!max) max = kSizes[kSizesCount - 1];  // defaults to max HIP size of 4
  int size_idx = kSizesCount - 1;
  while (size_idx >= 0 && kSizes[size_idx] > (unsigned)max) --size_idx;
  return size_idx >= 0 ? kSizes[rnd_upto(size_idx + 1)] : 0;
}

const Type *Vector::PromoteTypeToVectorType(const Type *type, int size) {
  if (!size) size = GetRandomVectorLength(0);
  if (type->eType == eVector && type->vector_length_ == size) return type;
  auto it = vector_types_.find(std::make_pair(type->simple_type, size));
  assert(it != vector_types_.end() && "Unknown vector type.");
  return it->second;
}

const Type &Vector::DemoteVectorTypeToType(const Type *type) {
  return type->eType == eVector ? Type::get_simple_type(type->simple_type)
                                : *type;
}

char Vector::GetComponentChar(int vector_size, int index) {
  return kCompSmall[index];
}

void Vector::OutputVectorType(std::ostream &out, const Type *type,
                              int vector_size) {
  const Type &demoted = DemoteVectorTypeToType(type);

  // no double support at the moment in HIP?
  if (demoted.eType == eSimple) {
    if (demoted.simple_type == eFloat) {
      out << "float";
    } else {
      // map c primitives to hip
      int bytes = demoted.SizeInBytes();
      std::string prefix = demoted.is_signed() ? "" : "u";

      if (bytes == 1)
        out << prefix << "char";
      else if (bytes == 2)
        out << prefix << "short";
      else if (bytes == 4)
        out << prefix << "int";
      else if (bytes == 8)
        out << prefix << "long";
      else {
        throw std::runtime_error("Error: Unsupported integer byte size (" +
                                 std::to_string(bytes) +
                                 ") for HIP vector generation.");
      }
    }
  } else {
    // is this how we are meant to errpr out?
    throw std::runtime_error(
        "Error: Structs and unions are invalid elements for HIP vectors.");
  }

  // append vector length
  out << vector_size;
}

void Vector::GenerateVectorTypes() {
  for (enum eSimpleType simple = eChar; simple < MAX_SIMPLE_TYPES;
       simple = (eSimpleType)(simple + 1)) {
    for (unsigned size_idx = 0; size_idx < kSizesCount; ++size_idx) {
      Type *t = new Type(simple);
      t->eType = eVector;
      t->vector_length_ = kSizes[size_idx];
      vector_types_[std::make_pair(simple, kSizes[size_idx])] = t;
    }
  }
}



}  // namespace HIPSmith