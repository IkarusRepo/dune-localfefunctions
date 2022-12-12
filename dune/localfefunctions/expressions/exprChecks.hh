// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include <dune/localfefunctions/leafNodeCollection.hh>
namespace Dune {

  template <typename LocalFunctionImpl>
  bool checkIfAllLeafNodeHaveTheSameBasisState(const LocalFunctionImpl& lf) {
    using namespace Dune::Indices;
    auto leafNodeCollection = collectLeafNodeLocalFunctions(lf);
    bool isValid            = true;
    if constexpr (leafNodeCollection.size() > 0) {
      std::array<bool,3> isBound;
      for (int i = 0; i < isBound.size(); ++i)
        isBound[i]               = leafNodeCollection.node(_0).basis().isBound(i);


      unsigned int integrationRuleSize = std::ranges::any_of(isBound,[](auto v){return v;}) ? leafNodeCollection.node(_0).basis().integrationPointSize() : 0;
      Dune::Hybrid::forEach(Dune::Hybrid::integralRange(Dune::index_constant<leafNodeCollection.size()>{}),
                            [&]<typename I>(I&& i) {
                              if constexpr (I::value == 0) {  // Skip first value
                              } else {
                                auto nodeBasis = leafNodeCollection.node(i).basis();
                                std::array<bool,3> isBoundOther;
                                for (int j = 0; j < isBound.size(); ++j)
                                  isBoundOther[j]               = nodeBasis.isBound(j);
                                if (isBoundOther != isBound)
                                  isValid = false;
                                else {
                                  if (nodeBasis.integrationPointSize() != integrationRuleSize) isValid = false;
                                }
                              }
                            });
    }
    return isValid;
  }

}  // namespace Dune
