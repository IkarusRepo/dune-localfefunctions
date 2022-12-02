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
                    []() { return TargetSpace(createRandomVector<typename TargetSpace::field_type,TargetSpace::valueSize>()); });
    }
  };

  template <int size>
  class CornerFactory {
  public:
    static void construct(std::vector<Dune::FieldVector<double, size>>& values, const int corners = 10) {
      values.resize(corners);
      std::generate(values.begin(), values.end(),
                    []() {
                      std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution< double> dist(-1, 1);
        auto rand = [&dist, &mt]() { return dist(mt); };
        Dune::FieldVector<double, size> vec;
        std::generate(vec.begin(), vec.end(), rand);
        return vec; });
    }
  };
}  // namespace Dune
