/*
 * This file is part of the Ikarus distribution (https://github.com/IkarusRepo/Ikarus).
 * Copyright (c) 2022. The Ikarus developers.
 *
 * This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 */

#pragma once
#include <dune/localfefunctions/expressions/binaryExpr.hh>
#include <dune/localfefunctions/expressions/constant.hh>
namespace Dune {

  template <typename E1, typename E2>
  class ScaleExpr : public BinaryExpr<ScaleExpr, E1, E2> {
  public:
    using Base = BinaryExpr<ScaleExpr, E1, E2>;
    using Base::Base;
    using Traits                   = LocalFunctionTraits<ScaleExpr>;
    static constexpr int valueSize = Traits::valueSize;

    template <size_t ID_ = 0>
    static constexpr int orderID
        = std::min(Base::E1Raw::template orderID<ID_> + Base::E2Raw::template orderID<ID_>, nonLinear);

    template <typename LocalFunctionEvaluationArgs_>
    auto evaluateValueOfExpression(const LocalFunctionEvaluationArgs_& localFunctionArgs) const {
      auto res=evaluateFunctionImpl(this->r(), localFunctionArgs);
      res*=this->l().value() ;
      return res;
    }

    template <int DerivativeOrder, typename LocalFunctionEvaluationArgs_>
    auto evaluateDerivativeOfExpression(const LocalFunctionEvaluationArgs_& localFunctionArgs) const {
      auto res=evaluateDerivativeImpl(this->r(), localFunctionArgs);
      res*=this->l().value() ;
      return res;
    }
  };

  template <typename E1, typename E2>
  struct LocalFunctionTraits<ScaleExpr<E1, E2>> : public LocalFunctionTraits<std::remove_cvref_t<E2>> {};

  template <typename E1, typename E2>
  requires(std::is_arithmetic_v<std::remove_cvref_t<E1>>and
               IsLocalFunction<E2> and !IsScaleExpr<E2>) constexpr ScaleExpr<ConstantExpr<E1>, E2>
  operator*(E1&& factor, E2&& u) {
    return ScaleExpr<ConstantExpr<E1>, E2>(ConstantExpr(factor), std::forward<E2>(u));
  }

  template <typename E1, typename E2>
  requires(std::is_arithmetic_v<std::remove_cvref_t<E1>>and
               IsLocalFunction<E2> and !IsScaleExpr<E2>) constexpr ScaleExpr<ConstantExpr<E1>, E2>
  operator*(E2&& u, E1&& factor) {
    return factor * u;
  }

  // Simplification if nested scale expression occur
  template <typename E1, typename E2>
  requires(std::is_arithmetic_v<std::remove_cvref_t<E1>>and IsScaleExpr<E2>) constexpr auto operator*(E1&& factor,
                                                                                                      E2&& u) {
    u.l().value() *= factor;
    return u;
  }

  template <typename E1, typename E2>
  requires(std::is_arithmetic_v<std::remove_cvref_t<E1>>and IsScaleExpr<E2>and IsLocalFunction<E2>) constexpr auto
  operator*(E2&& u, E1&& factor) {
    return operator*(std::forward<E1>(factor), std::forward<E2>(u));
  }

  // Division operator
  template <typename E1, typename E2>
  requires(std::is_arithmetic_v<std::remove_cvref_t<E1>>and IsLocalFunction<E2> and !IsScaleExpr<E2>) constexpr auto
  operator/(E2&& u, E1&& factor) {
    static_assert(std::floating_point<std::remove_cvref_t<E1>>,
                  "Operator/ should only called with floating point types.");

    return operator*(std::forward<E1>(1 / factor), std::forward<E2>(u));
  }

}  // namespace Dune