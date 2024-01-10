// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include "testFacilities.hh"
#include "testexpression.hh"

#include <dune/localfefunctions/expressions.hh>
#include <dune/localfefunctions/manifolds/realTuple.hh>

int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);

  TestSuite t;

  static constexpr int worldDim = 3;
  static constexpr int gridDim  = 3;
  static constexpr int order    = 1;
  using Manifold                = Dune::RealTuple<double, worldDim>;
  const auto geometryType       = Dune::GeometryTypes::cube(3);

  auto [f, nodalPoints, geometry, corners, feCache]
      = Testing::localFunctionTestConstructorNew<Manifold, gridDim, worldDim, order>(geometryType, 1);

  // wrt parentheses does not close at the correct position
  f.evaluateDerivative(
      0, Dune::wrt(Dune::DerivativeDirections::spatialAll, Dune::on(Dune::DerivativeDirections::gridElement)));
  return t.exit();
}
