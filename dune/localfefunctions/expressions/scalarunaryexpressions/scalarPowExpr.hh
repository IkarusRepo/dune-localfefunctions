// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <dune/localfefunctions/expressions/scalarunaryexpressions/scalarUnaryExpression.hh>

namespace Dune {

  template <int exponent>
  struct PowFunc {
    template <typename ScalarType>
    static ScalarType value(const ScalarType& v) {
      return Dune::power(v, exponent);
    }

    template <typename ScalarType>
    static ScalarType derivative(const ScalarType& v) {
      return Dune::power(v, exponent - 1) * exponent;
    }

    template <typename ScalarType>
    static ScalarType secondDerivative(const ScalarType& v) {
      return Dune::power(v, exponent - 2) * exponent * (exponent - 1);
    }

    template <typename ScalarType>
    static ScalarType thirdDerivative(const ScalarType& v) {
      return Dune::power(v, exponent - 3) * exponent * (exponent - 1) * (exponent - 2);
    }
  };
  template <int exponent, typename E1>
  requires IsLocalFunction<E1>
  constexpr auto pow(E1&& u) {
    static_assert(std::remove_cvref_t<E1>::valueSize == 1,
                  "Pow expression only defined for scalar valued local functions.");
    return ScalarUnaryExpr<E1, PowFunc<exponent>>(std::forward<E1>(u));
  }
}  // namespace Dune
