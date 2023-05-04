// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include "testexpression.hh"

#include <dune/localfefunctions/expressions.hh>
#include <dune/localfefunctions/manifolds/realTuple.hh>

#include <Eigen/Core>

template <int dim>
using RealT = Dune::RealTuple<double, dim>;

auto testStandardLocalFunction() {
  TestSuite t;
  using namespace Dune;
  using namespace Dune::Indices;

  auto expr = [](auto& f) { return f; };

  auto exprTest = [](auto& h, auto&, [[maybe_unused]] auto& fe) {
    TestSuite tL("StandardLocalFunctionSpecialTests");

    static_assert(std::tuple_size_v<decltype(collectNonArithmeticLeafNodes(h))> == 1);
    using HRawType = std::remove_cvref_t<decltype(h)>;
    static_assert(countNonArithmeticLeafNodes<HRawType>() == 1);
    tL.check(HRawType::id[0] == 0);

    static_assert(HRawType::order() == linear);

    return tL;
  };

  using Expr        = decltype(expr);
  using ExprTest    = decltype(exprTest);
  using ManiFoldIDP = ManiFoldTemplateIDPair<RealT, _0>;

  using FunctionConstructor = decltype(singleStandardLocalFunction);

  t.subTest(testExpressionsOnLine<Expr, ExprTest, FunctionConstructor, true, ManiFoldIDP>(expr, exprTest,
                                                                                          singleStandardLocalFunction));
  t.subTest(testExpressionsOnTriangle<Expr, ExprTest, FunctionConstructor, true, ManiFoldIDP>(
      expr, exprTest, singleStandardLocalFunction));
  t.subTest(testExpressionsOnQuadrilateral<Expr, ExprTest, FunctionConstructor, true, ManiFoldIDP>(
      expr, exprTest, singleStandardLocalFunction));
  t.subTest(testExpressionsOnHexahedron<Expr, ExprTest, FunctionConstructor, true, ManiFoldIDP>(
      expr, exprTest, singleStandardLocalFunction));
  return t;
}

int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  TestSuite t;

  using namespace std::chrono;
  using namespace std;
  auto start = high_resolution_clock::now();
  t.subTest(testStandardLocalFunction());
  auto stop     = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "The test execution took: " << duration.count() << endl;
  return t.exit();
}
