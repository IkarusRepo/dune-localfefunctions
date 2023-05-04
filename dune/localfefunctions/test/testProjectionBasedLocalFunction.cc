// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include "testexpression.hh"

#include <dune/localfefunctions/expressions.hh>
#include <dune/localfefunctions/manifolds/unitVector.hh>

template <int dim>
using UnitT = Dune::UnitVector<double, dim>;

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

    static_assert(HRawType::order() == nonlinear);

    return tL;
  };

  using Expr        = decltype(expr);
  using ExprTest    = decltype(exprTest);
  using ManiFoldIDP = ManiFoldTemplateIDPair<UnitT, _0>;

  using FC = decltype(singleProjectionBasedLocalFunction);

  t.subTest(testExpressionsOnCustomGeometry<1, 1, 2, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::line, expr, exprTest, singleProjectionBasedLocalFunction));
  t.subTest(testExpressionsOnCustomGeometry<1, 2, 2, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::line, expr, exprTest, singleProjectionBasedLocalFunction));
  t.subTest(testExpressionsOnCustomGeometry<1, 1, 3, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::line, expr, exprTest, singleProjectionBasedLocalFunction));
  t.subTest(testExpressionsOnCustomGeometry<1, 2, 3, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::line, expr, exprTest, singleProjectionBasedLocalFunction));
  t.subTest(testExpressionsOnCustomGeometry<2, 1, 2, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::quadrilateral, expr, exprTest, singleProjectionBasedLocalFunction));
  t.subTest(testExpressionsOnCustomGeometry<2, 2, 2, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::quadrilateral, expr, exprTest, singleProjectionBasedLocalFunction));
  t.subTest(testExpressionsOnCustomGeometry<2, 1, 3, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::quadrilateral, expr, exprTest, singleProjectionBasedLocalFunction));
  t.subTest(testExpressionsOnCustomGeometry<2, 2, 3, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::quadrilateral, expr, exprTest, singleProjectionBasedLocalFunction));
  t.subTest(testExpressionsOnCustomGeometry<3, 1, 2, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::hexahedron, expr, exprTest, singleProjectionBasedLocalFunction));
  t.subTest(testExpressionsOnCustomGeometry<3, 2, 2, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::hexahedron, expr, exprTest, singleProjectionBasedLocalFunction));
  t.subTest(testExpressionsOnCustomGeometry<3, 1, 3, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::hexahedron, expr, exprTest, singleProjectionBasedLocalFunction));
  t.subTest(testExpressionsOnCustomGeometry<3, 2, 3, Expr, ExprTest, FC, true, ManiFoldIDP>(
      Dune::GeometryTypes::hexahedron, expr, exprTest, singleProjectionBasedLocalFunction));
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
