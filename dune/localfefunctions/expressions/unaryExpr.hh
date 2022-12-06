// SPDX-FileCopyrightText: 2022 Alexander Müller mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include "rebind.hh"

#include <dune/localfefunctions/localFunctionInterface.hh>

namespace Dune {

  template <template <typename, typename...> class Op, typename E1, typename... Args>
  struct UnaryExpr : public Dune::LocalFunctionInterface<Op<E1, Args...>> {
    std::tuple<E1> expr;

    using E1Raw = std::remove_cvref_t<E1>;

    template <size_t ID_ = 0>
    static constexpr int orderID = Op<E1, Args...>::template orderID<ID_>;

    using LinearAlgebra = typename E1Raw::LinearAlgebra;

    const E1& m() const { return std::get<0>(expr); }

    E1& m() { return std::get<0>(expr); }

    static constexpr auto id = E1Raw::id;

    auto clone() const { return Op<decltype(m().clone()), Args...>(m().clone()); }

    /** Rebind the value type of the underlying local function with the id ID */
    template <typename OtherType, size_t ID = 0>
    auto rebindClone(OtherType&&, Dune::index_constant<ID>&& id_ = Dune::index_constant<0>()) const {
      return rebind<Op, E1, OtherType, ID, Args...>(m(), Dune::index_constant<ID>());
    }

    constexpr explicit UnaryExpr(E1&& u) requires IsLocalFunction<E1> : expr(std::forward<E1>(u)) {}

    static constexpr bool isLeaf  = false;
    static constexpr int children = 1;
  };

}  // namespace Dune
