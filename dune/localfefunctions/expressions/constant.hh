// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include <dune/localfefunctions/expressions/unaryExpr.hh>
#include <dune/localfefunctions/meta.hh>
namespace Dune {

  template <typename Type, typename LinAlg>
  requires std::is_arithmetic_v<Type>
  class ConstantExpr : public LocalFunctionInterface<ConstantExpr<Type, LinAlg>> {
  public:
    using LinearAlgebra = LinAlg;
    explicit ConstantExpr(Type val_) : val{val_} {}

    const Type& value() const { return val; }
    Type& value() { return val; }

    auto clone() const { return ConstantExpr(val); }

    template <typename OtherType, size_t ID = 0>
    auto rebindClone(OtherType&& t, Dune::index_constant<ID>&& id = Dune::index_constant<0>()) const {
      if constexpr (Arithmetic::value == ID)
        return ConstantExpr(static_cast<OtherType>(val));
      else
        return clone();
    }

    template <typename OtherType>
    struct Rebind {
      using other = ConstantExpr<OtherType, LinAlg>;
    };

    static constexpr bool isLeaf = true;
    static constexpr std::array<int, 1> id{arithmetic};
    template <size_t ID_ = 0>
    static constexpr int orderID = ID_ == Arithmetic::value ? linear : constant;

  private:
    Type val;
  };

  template <typename Type, typename LinAlg>
  struct LocalFunctionTraits<ConstantExpr<Type, LinAlg>> {
    static constexpr int valueSize = 1;
    /** \brief Type for the points for evaluation, usually the integration points */
    using DomainType = typename LinAlg::template FixedSizedVector<double, 0>;
  };

}  // namespace Dune
