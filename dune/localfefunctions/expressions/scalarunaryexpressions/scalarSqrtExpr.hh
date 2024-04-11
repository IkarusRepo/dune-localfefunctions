// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <dune/localfefunctions/expressions/scalarunaryexpressions/scalarUnaryExpression.hh>

namespace Dune {

  struct SqrtFunc {
    template <typename ScalarType>
    static ScalarType value(const ScalarType& v) {
      return sqrt(v);
    }

    template <typename ScalarType>
    static ScalarType derivative(const ScalarType& v) {
      return ScalarType(0.5) / sqrt(v);
    }

    template <typename ScalarType>
    static ScalarType secondDerivative(const ScalarType& v) {
      using std::pow;
      return ScalarType(-0.25) / pow(v, 1.5);
    }

    template <typename ScalarType>
    static ScalarType thirdDerivative(const ScalarType& v) {
      using std::pow;
      return ScalarType(0.375) / pow(v, 2.5);
    }
  };
  template <typename E1>
    requires IsLocalFunction<E1>
  constexpr auto sqrt(E1&& u) {
    static_assert(std::remove_cvref_t<E1>::valueSize == 1,
                  "Sqrt expression only defined for scalar valued local functions.");
    return ScalarUnaryExpr<E1, SqrtFunc>(std::forward<E1>(u));
  }
}  // namespace Dune
