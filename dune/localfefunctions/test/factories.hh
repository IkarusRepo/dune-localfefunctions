//
// Created by lex on 7/29/22.
//

#pragma once

#include <random>

#include <dune/istl/bvector.hh>
#include <dune/localfefunctions/linearAlgebraHelper.hh>

namespace Dune {
  template <typename Manifold>
  class ValueFactory {
    using TargetSpace = Manifold;

  public:
    static void construct(Dune::BlockVector<TargetSpace>& values, const int testPointsSize = 10) {
      values.resize(testPointsSize);
      std::generate(values.begin(), values.end(),
                    []() { return TargetSpace(createRandomVector<typename TargetSpace::CoordinateType>()); });
    }
  };

  template <int size>
  class CornerFactory {
  public:
    static void construct(std::vector<Dune::FieldVector<double, size>>& values, const int corners = 10) {
      values.resize(corners);
      std::generate(values.begin(), values.end(),
                    []() { return createRandomVector<Dune::FieldVector<double, size>>(); });
    }
  };
}  // namespace Dune
