// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include <array>

#include <dune/common/test/testsuite.hh>
#include <dune/localfefunctions/manifolds/realTuple.hh>
#include <dune/localfefunctions/manifolds/unitVector.hh>
#include <dune/python/istl/bvector.hh>
int main(int argc, char** argv) {
  // check blockvector correctness
  using RT = Dune::RealTuple<double, 3>;
  using UV = Dune::UnitVector<double, 3>;

  auto blockVectorObstacleCourse = []<typename T>(const T&) {
    pybind11::class_<T> cls(pybind11::object{});
    Dune::Python::registerBlockVector(cls);
  };

  blockVectorObstacleCourse(Dune::BlockVector<RT>{});
  blockVectorObstacleCourse(Dune::BlockVector<UV>{});

  return 0;
}
