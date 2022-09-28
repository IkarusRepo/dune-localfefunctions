//
// Created by lex on 7/29/22.
//

#pragma once

#include <dune/istl/bvector.hh>
#include <random>

namespace Dune {
  template <typename Manifold>
  class ValueFactory {
    using TargetSpace = Manifold;

  public:
    static void construct(Dune::BlockVector<TargetSpace>& values, const int testPointsSize = 10) {
      values.resize(testPointsSize);
      std::generate(values.begin(),values.end(), []() { return TargetSpace(TargetSpace::CoordinateType::Random()); });
    }
  };

  /** \brief Generates FieldVector with random entries in the range -1..1 */
  template<typename field_type,int n>
  auto randomFieldVector(field_type lower=-1, field_type upper=1)
  {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<field_type> dist(lower, upper);
    auto rand = [&dist,&mt](){
      return dist(mt);
    };
    FieldVector<field_type,n> vec;
    std::generate(vec.begin(), vec.end(), rand);
    return vec;
  }


  template <int size>
  class CornerFactory {
  public:
    static void construct(std::vector<Dune::FieldVector<double,size>>& values, const int corners = 10) {
      values.resize(corners);
      std::generate(values.begin(),values.end(), []() { return randomFieldVector<double,size>(); });
    }
  };
}  // namespace Ikarus
