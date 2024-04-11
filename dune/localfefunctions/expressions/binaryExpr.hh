// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include "rebind.hh"

#include <dune/localfefunctions/localFunctionInterface.hh>

namespace Dune {

  template <template <typename, typename> class Op, typename E1, typename E2>
  struct BinaryExpr : public Dune::LocalFunctionInterface<Op<E1, E2>> {
    using E1Raw         = std::remove_cvref_t<E1>;
    using E2Raw         = std::remove_cvref_t<E2>;
    using LinearAlgebra = typename E1Raw::LinearAlgebra;
    constexpr const E1& l() const { return expr.first; }
    constexpr const E2& r() const { return expr.second; }
    static constexpr bool providesDerivativeTransformations
        = E1Raw::providesDerivativeTransformations and E2Raw::providesDerivativeTransformations;

    constexpr E1& l() { return expr.first; }

    constexpr E2& r() { return expr.second; }

    auto clone() const { return Op<decltype(l().clone()), decltype(r().clone())>(l().clone(), r().clone()); }

    /** Rebind the value type of the underlying local function with the id ID */
    template <typename OtherType, size_t ID = 0>
    auto rebindClone(OtherType&&, Dune::index_constant<ID>&& = Dune::index_constant<0UL>()) const {
      return rebind<Op, E1, E2, OtherType>(l(), r(), Dune::index_constant<ID>());
    }

    /* Creates a tuple of all subtype ids, if l or r is not a tuple, tuple_cat may not work.
     * Thus, we artificially wrap them inside a tuple  */
    static constexpr auto id = Dune::Std::concatArray(E1Raw::id, E2Raw::id);

    /** The function order wrt. the coefficients */
    template <size_t ID_ = 0>
    static constexpr int orderID = Op<E1, E2>::template orderID<ID_>;

    constexpr BinaryExpr(E1&& u, E2&& v)
      requires IsLocalFunction<E1, E2>
        : expr(std::forward<E1>(u), std::forward<E2>(v)) {}

    static constexpr bool isLeaf  = false;
    static constexpr int children = 2;

  private:
    std::pair<E1, E2> expr;
  };

}  // namespace Dune
