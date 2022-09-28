//
// Created by lex on 4/25/22.
//

#pragma once
#include <dune/localfefunctions/expressions/unaryExpr.hh>
namespace Dune {

  template <typename E1>
  class NegateExpr : public UnaryExpr<NegateExpr, E1> {
  public:
    using Base = UnaryExpr<NegateExpr, E1>;
    using Base::UnaryExpr;
    using Traits                   = LocalFEFunctionTraits<NegateExpr>;
    static constexpr int valueSize = Traits::valueSize;

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
  struct LocalFEFunctionTraits<NegateExpr<E1>> : public LocalFEFunctionTraits<std::remove_cvref_t<E1>> {};

  template <typename E1>
  requires IsLocalFunction<E1>
  constexpr auto operator-(E1&& u) { return LocalFunctionNegate<E1>(std::forward<E1>(u)); }

}  // namespace Ikarus