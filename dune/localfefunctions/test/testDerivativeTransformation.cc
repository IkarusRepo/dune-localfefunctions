// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include "testexpression.hh"

#include "dune/localfefunctions/manifolds/unitVector.hh"
#include <dune/localfefunctions/derivativetransformators.hh>
#include <dune/localfefunctions/expressions.hh>
#include <dune/localfefunctions/linearAlgebraHelper.hh>
#include <dune/localfefunctions/manifolds/realTuple.hh>

template <typename Manifold, int domainDim, int worldDim, int order>
auto localFunctionTestConstructorNew(const Dune::GeometryType& geometryType, size_t nNodalTestPointsI = 1) {
  using namespace Dune;
  using namespace Dune::Indices;
  const auto& refElement = Dune::ReferenceElements<double, domainDim>::general(geometryType);

  std::vector<Dune::FieldVector<double, worldDim>> corners;
  CornerFactory<worldDim>::construct(corners, refElement.size(domainDim));
  auto geometry = std::make_shared<const Dune::MultiLinearGeometry<double, domainDim, worldDim>>(refElement, corners);

  auto feCache    = std::make_shared<FECache<domainDim, order>>();
  const auto& fe  = feCache->get(geometryType);
  auto localBasis = Dune::CachedLocalBasis(fe.localBasis());

  auto vBlockedLocal0 = createVectorOfNodalValues<Manifold, domainDim, order>(geometryType, nNodalTestPointsI);

  const auto& rule = Dune::QuadratureRules<double, domainDim>::rule(fe.type(), 2);
  localBasis.bind(rule, bindDerivatives(0, 1));
  if constexpr (Dune::Std::IsSpecializationTypeAndNonTypes<Dune::RealTuple, Manifold>::value) {
    auto f = Dune::StandardLocalFunction(localBasis, vBlockedLocal0, geometry);
    return std::make_tuple(f, vBlockedLocal0, geometry, corners, feCache);
  } else if constexpr (Dune::Std::IsSpecializationTypeAndNonTypes<Dune::UnitVector, Manifold>::value) {
    auto f = Dune::ProjectionBasedLocalFunction(localBasis, vBlockedLocal0, geometry);
    return std::make_tuple(f, vBlockedLocal0, geometry, corners, feCache);
  }
}

template <typename Manifold, int gridDim, int worldDim, int order, typename FirstOrderTransForm>
auto testTranform(const Dune::GeometryType& geometryType, FirstOrderTransForm&&) {
  std::stringstream ss;
  ss << geometryType;
  std::string geometryTypeString = ss.str();

  TestSuite t("testTranform, gridDim: " + std::to_string(gridDim) + " worldDim: " + std::to_string(worldDim) + " order "
              + std::to_string(order) + " FirstOrderTransForm: " + Dune::className<FirstOrderTransForm>() + " on "
              + geometryTypeString);

  Eigen::MatrixXd nodalPointsEigen;
  constexpr auto vecSize = Manifold::valueSize;
  //  // define a lambda that copies the entries of a blocked vector into a flat one
  auto copyToEigenMatrix = [&](auto&& nodalPointsBlocked, auto&& nodalPointsEigenL) {
    nodalPointsEigenL.resize(nodalPointsBlocked.size(), vecSize);
    for (size_t i = 0; i < nodalPointsBlocked.size(); ++i) {
#if DUNE_LOCALFEFUNCTIONS_USE_EIGEN == 1
      nodalPointsEigenL.template block<1, vecSize>(i, 0) = nodalPointsBlocked[i].getValue().transpose();
#else
      for (int j = 0; j < vecSize; ++j) {
        nodalPointsEigenL(i, j) = nodalPointsBlocked[i].getValue()[j];
      }
#endif
    }
  };

  typename Dune::DefaultLinearAlgebra::VariableSizedVector<double> N;
  typename Dune::DefaultLinearAlgebra::VarFixSizedMatrix<double, gridDim> dN;
  typename Dune::DefaultLinearAlgebra::VarFixSizedMatrix<double, gridDim> dNTransformed;
  auto [f, nodalPoints, geometry, corners, feCache]
      = localFunctionTestConstructorNew<Manifold, gridDim, worldDim, order>(geometryType);
  copyToEigenMatrix(nodalPoints, nodalPointsEigen);
  using Dune::eval;
  using Dune::transpose;
  using Dune::transposeEvaluated;
  auto& localBasis = f.basis();
  for (auto [gpIndex, gp] : f.viewOverIntegrationPoints()) {
    auto Jf  = f.evaluateDerivative(gp.position(), Dune::wrt(Dune::DerivativeDirections::spatialAll),
                                    Dune::on(Dune::DerivativeDirections::gridElement, FirstOrderTransForm{}));
    auto Jf_ = f.evaluateDerivative(gpIndex, Dune::wrt(Dune::DerivativeDirections::spatialAll),
                                    Dune::on(Dune::DerivativeDirections::gridElement, FirstOrderTransForm{}));
    localBasis.evaluateJacobian(gp.position(), dN);
    localBasis.evaluateFunction(gp.position(), N);

    if constexpr (gridDim == worldDim) {
#if DUNE_LOCALFEFUNCTIONS_USE_EIGEN == 1
      auto Jgeo     = transposeEvaluated(maybeToEigen(geometry->jacobianTransposed(gp.position())));
      dNTransformed = dN * Jgeo.inverse();
#else
      auto Jgeo = geometry->jacobianInverseTransposed(gp.position());
      dNTransformed.resize(dN.size());
      for (size_t i = 0; i < dN.size(); ++i)
        Jgeo.mv(dN[i], dNTransformed[i]);
#endif
    } else if constexpr (gridDim == 2 and worldDim == 3) {
      const auto j = maybeToEigen(transpose(geometry->jacobianTransposed(gp.position())));
      if constexpr (std::is_same_v<Dune::DefaultFirstOrderTransformFunctor, FirstOrderTransForm>)
        Dune::calcCartesianDerivatives(dN, j, dNTransformed);
      else if constexpr (std::is_same_v<Dune::GramSchmidtFirstOrderTransformFunctor, FirstOrderTransForm>)
        Dune::calcCartesianDerivativesByGramSchmidt(dN, j, dNTransformed);
    } else if constexpr ((gridDim == 1 and worldDim == 3) or (gridDim == 1 and worldDim == 2)) {
      if constexpr (std::is_same_v<Dune::GramSchmidtFirstOrderTransformFunctor, FirstOrderTransForm>) {
        const auto j = Dune::transposeEvaluated(maybeToEigen(geometry->jacobianTransposed(gp.position())));
        Dune::calcCartesianDerivativesByGramSchmidt(dN, j, dNTransformed);
      }
    }
#if DUNE_LOCALFEFUNCTIONS_USE_EIGEN == 1
    auto JExpected
        = Dune::transposeEvaluated(dNTransformed.transpose() * nodalPointsEigen);  // the base vectors are the columns
    if constexpr (Dune::Std::IsSpecializationTypeAndNonTypes<
                      Dune::UnitVector, Manifold>::value) {  // Project derivative onto tangent space
      Eigen::Vector<double, vecSize> val = nodalPointsEigen.transpose() * N;
      auto P                             = Dune::UnitVector<double, vecSize>::derivativeOfProjectionWRTposition(val);
      JExpected                          = P * JExpected;
    }
    t.check(Jf.isApprox(JExpected)) << "(JExpected-Jf).norm(): " << (JExpected - Jf).norm() << "JExpected: \n"
                                    << JExpected << " \nJ:\n"
                                    << Jf;
    t.check(Jf.isApprox(Jf_)) << "(Jf_-Jf).norm(): " << (Jf_ - Jf).norm() << "Jf_: \n" << Jf_ << " \nJ:\n" << Jf;
#else
    Dune::FieldMatrix<double, vecSize, gridDim> JExpected = 0;

    for (int i = 0; i < gridDim; ++i)
      for (int j = 0; j < vecSize; ++j)
        for (size_t k = 0; k < dN.size(); ++k)
          JExpected[j][i] += dNTransformed[k][i] * nodalPointsEigen(k, j);

    if constexpr (Dune::Std::IsSpecializationTypeAndNonTypes<
                      Dune::UnitVector, Manifold>::value) {  // Project derivative onto tangent space
      Dune::FieldVector<double, vecSize> val(0.0);
      for (size_t i = 0; i < N.size(); ++i)
        for (size_t k = 0; k < vecSize; ++k)
          val[k] += nodalPointsEigen(i, k) * N[i];

      auto P = Dune::UnitVector<double, vecSize>::derivativeOfProjectionWRTposition(val);
      JExpected.leftmultiply(P);
    }

    static_assert(Dune::Rows<decltype(JExpected)>::value == Dune::Rows<decltype(Jf)>::value);
    static_assert(Dune::Cols<decltype(JExpected)>::value == Dune::Cols<decltype(Jf)>::value);
    t.check((JExpected - Jf).frobenius_norm() < 1e-12)
        << "(JExpected-Jf).norm(): " << (JExpected - Jf).frobenius_norm() << "JExpected: \n"
        << JExpected << " \nJ:\n"
        << Jf;
    t.check((Jf_ - Jf).frobenius_norm() < 1e-12) << "(Jf_-Jf).norm(): " << (Jf_ - Jf).frobenius_norm() << "Jf_: \n"
                                                 << Jf_ << " \nJ:\n"
                                                 << Jf;
#endif
  }

  return t;
}

template <typename Manifold>
auto testTransformation() {
  using namespace Dune::GeometryTypes;
  TestSuite t("testDerivativeTransformation on " + Dune::className<Manifold>());
  t.subTest(testTranform<Manifold, 1, 1, 1>(line, Dune::DefaultFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 2, 2, 1>(quadrilateral, Dune::DefaultFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 2, 2, 1>(triangle, Dune::DefaultFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 3, 3, 1>(hexahedron, Dune::DefaultFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 1, 1, 1>(line, Dune::DefaultFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 2, 2, 2>(quadrilateral, Dune::DefaultFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 2, 2, 2>(triangle, Dune::DefaultFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 3, 3, 2>(hexahedron, Dune::DefaultFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 1, 1, 2>(line, Dune::DefaultFirstOrderTransformFunctor{}));

  // codim!=0 test
  t.subTest(testTranform<Manifold, 2, 3, 1>(quadrilateral, Dune::DefaultFirstOrderTransformFunctor{}));

  t.subTest(testTranform<Manifold, 1, 1, 1>(line, Dune::GramSchmidtFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 2, 2, 1>(quadrilateral, Dune::GramSchmidtFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 2, 2, 1>(triangle, Dune::GramSchmidtFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 3, 3, 1>(hexahedron, Dune::GramSchmidtFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 1, 1, 1>(line, Dune::GramSchmidtFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 2, 2, 2>(quadrilateral, Dune::GramSchmidtFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 2, 2, 2>(triangle, Dune::GramSchmidtFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 3, 3, 2>(hexahedron, Dune::GramSchmidtFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 1, 1, 2>(line, Dune::GramSchmidtFirstOrderTransformFunctor{}));

  // codim!=0 tests
  t.subTest(testTranform<Manifold, 1, 2, 1>(line, Dune::GramSchmidtFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 1, 3, 1>(line, Dune::GramSchmidtFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 2, 3, 1>(quadrilateral, Dune::GramSchmidtFirstOrderTransformFunctor{}));
  t.subTest(testTranform<Manifold, 2, 3, 1>(triangle, Dune::GramSchmidtFirstOrderTransformFunctor{}));

  return t;
}

int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  TestSuite t;
  using Manifold  = Dune::RealTuple<double, 2>;
  using Manifold2 = Dune::UnitVector<double, 2>;

  t.subTest(testTransformation<Manifold>());
  t.subTest(testTransformation<Manifold2>());

  using Manifold3 = Dune::RealTuple<double, 1>;
  using Manifold4 = Dune::UnitVector<double, 3>;

  t.subTest(testTransformation<Manifold3>());
  t.subTest(testTransformation<Manifold4>());

  using Manifold5 = Dune::RealTuple<double, 5>;
  using Manifold6 = Dune::UnitVector<double, 4>;

  t.subTest(testTransformation<Manifold5>());
  t.subTest(testTransformation<Manifold6>());
}
