// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include <dune/localfefunctions/expressions/unaryExpr.hh>
//#include <ikarus/utils/linearAlgebraHelper.hh>
namespace Dune {

  template <typename E1>
  class NegateExpr : public UnaryExpr<NegateExpr, E1> {
  public:
    using Base = UnaryExpr<NegateExpr, E1>;
    using Base::Base;
    using Traits                   = LocalFunctionTraits<NegateExpr>;
    using ctype                    = typename Traits::ctype;
    static constexpr int valueSize = Traits::valueSize;
    using LinearAlgebra            = typename Base::E1Raw::LinearAlgebra;

    template <size_t ID_ = 0>
    static constexpr int orderID = Base::E1Raw::template orderID<ID_>;

    template <typename LFArgs>
    auto evaluateValueOfExpression(const LFArgs& lfArgs) const {
      return Dune::eval(-evaluateFunctionImpl(this->m(), lfArgs));
    }

    template <int DerivativeOrder, typename LFArgs>
    auto evaluateDerivativeOfExpression(const LFArgs& lfArgs) const {
      return Dune::eval(-evaluateDerivativeImpl(this->m(), lfArgs));
    }
  };

  template <typename E1>
  struct LocalFunctionTraits<NegateExpr<E1>> : public LocalFunctionTraits<std::remove_cvref_t<E1>> {};

  template <typename E1>
  requires IsLocalFunction<E1>
  constexpr auto operator-(E1&& u) { return NegateExpr<E1>(std::forward<E1>(u)); }

}  // namespace Dune
