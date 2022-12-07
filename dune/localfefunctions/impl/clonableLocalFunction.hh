// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include <dune/localfefunctions/meta.hh>
namespace Dune {

  template <typename LFImpl>
  class ClonableLocalFunction {
  public:
    LFImpl clone() const {
      return LFImpl(underlying().basis(), underlying().coeffs, underlying().geometry(),
                    Dune::index_constant<LFImpl::id[0]>());
    }

    template <typename OtherType, size_t ID = 0>
    auto rebindClone(OtherType&& t, Dune::index_constant<ID>&& id = Dune::index_constant<0>()) const {
      if constexpr (LFImpl::id[0] == ID)
        return typename LFImpl::template rebind<OtherType>::other(
            underlying().basis(), convertUnderlying<OtherType>(underlying().coeffs), underlying().geometry(),
            Dune::index_constant<LFImpl::id[0]>());
      else
        return clone();
    }

  private:
    constexpr LFImpl const& underlying() const  // CRTP
    {
      return static_cast<LFImpl const&>(*this);
    }
  };

}  // namespace Dune
