// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include <dune/localfefunctions/expressions/expressionHelper.hh>
#include <dune/localfefunctions/expressions/rebind.hh>
#include <dune/localfefunctions/expressions/unaryExpr.hh>
#include <dune/localfefunctions/helper.hh>

namespace Dune {

  template <typename E1>
  class GreenLagrangeStrainsExpr : public UnaryExpr<GreenLagrangeStrainsExpr, E1> {
  public:
    using Base = UnaryExpr<GreenLagrangeStrainsExpr, E1>;
    using Base::Base;
    using Traits        = LocalFunctionTraits<GreenLagrangeStrainsExpr>;
    using LinearAlgebra = typename Base::E1Raw::LinearAlgebra;

    /** \brief Type used for coordinates */
    using ctype                           = typename Traits::ctype;
    static constexpr int strainSize       = Traits::valueSize;
    static constexpr int displacementSize = Base::E1Raw::valueSize;
    static constexpr int gridDim          = Traits::gridDim;

    static_assert(Base::E1Raw::template order<0>() == 1,
                  "Linear strain expression only supported for linear displacement function w.r.t. coefficients.");

    template <size_t ID_ = 0>
    static constexpr int orderID = 2;

    template <typename LFArgs>
    auto evaluateValueOfExpression(const LFArgs& lfArgs) const {
      const auto integrationPointPosition
          = returnIntegrationPointPosition(lfArgs.integrationPointOrIndex, this->m().basis());
      auto referenceJacobian = maybeToEigen(this->m().geometry()->jacobianTransposed(integrationPointPosition));
      static_assert(std::is_same_v<typename decltype(referenceJacobian)::value_type, double>);
      const auto gradArgs = replaceWrt(lfArgs, wrt(DerivativeDirections::spatialAll));
      const auto gradu    = transposeEvaluated(evaluateDerivativeImpl(this->m(), gradArgs));

      if constexpr (std::is_same_v<typename decltype(lfArgs.transformWithArgs)::T, DerivativeDirections::GridElement>)
        referenceJacobian = createScaledIdentityMatrix<double, displacementSize, displacementSize>();

      typename LinearAlgebra::template FixedSizedVector<ctype, strainSize> E;
      // E= 1/2*(H^T * G + G^T * H + H^T * H) with H = gradu
      //         E=
      //         toVoigt(0.5*(2*sym(transpose(gradu)*referenceJacobian)+transpose(referenceJacobian)*referenceJacobian));
      for (int i = 0; i < gridDim; ++i)
        E[i] = inner(row(referenceJacobian, i), row(gradu, i)) + 0.5 * two_norm2(row(gradu, i));

      if constexpr (gridDim == 2) {
        const ctype v1 = inner(row(referenceJacobian, 0), row(gradu, 1));
        const ctype v2 = inner(row(gradu, 0), row(referenceJacobian, 1));
        const ctype v3 = inner(row(gradu, 0), row(gradu, 1));
        E[2]           = v1 + v2 + v3;
      } else if constexpr (gridDim == 3) {
        typename LinearAlgebra::template FixedSizedVector<ctype, gridDim> a1 = row(referenceJacobian, 0);
        a1 += row(gradu, 0);
        typename LinearAlgebra::template FixedSizedVector<ctype, gridDim> a2 = row(referenceJacobian, 1);
        a2 += row(gradu, 1);
        typename LinearAlgebra::template FixedSizedVector<ctype, gridDim> a3 = row(referenceJacobian, 2);
        a3 += row(gradu, 2);
        E[3] = inner(a2, a3);
        E[4] = inner(a1, a3);
        E[5] = inner(a1, a2);
      }

      return E;
    }

    template <int DerivativeOrder, typename LFArgs>
    auto evaluateDerivativeOfExpression(const LFArgs& lfArgs) const {
      if constexpr (DerivativeOrder == 1 and LFArgs::hasSingleCoeff) {
        const auto integrationPointPosition
            = returnIntegrationPointPosition(lfArgs.integrationPointOrIndex, this->m().basis());
        auto referenceJacobian = maybeToEigen(
            this->m().geometry()->jacobianTransposed(integrationPointPosition));  // the rows are X_{,1} and X_{,2}
        if constexpr (std::is_same_v<typename decltype(lfArgs.transformWithArgs)::T, DerivativeDirections::GridElement>)
          referenceJacobian = createScaledIdentityMatrix<double, displacementSize, displacementSize>();
        const auto gradArgs = replaceWrt(lfArgs, wrt(DerivativeDirections::spatialAll));
        const auto gradu
            = transposeEvaluated(evaluateDerivativeImpl(this->m(), gradArgs));  // the rows are u_{,1} and u_{,2}
        const auto gradArgsdI = addWrt(lfArgs, wrt(DerivativeDirections::spatialAll));
        const auto gradUdI    = evaluateDerivativeImpl(this->m(), gradArgsdI);  // derivative of grad u wrt I-th coeff

        typename LinearAlgebra::template FixedSizedMatrix<ctype, strainSize, gridDim> bopI{};
        typename LinearAlgebra::template FixedSizedVector<ctype, gridDim> g1 = row(referenceJacobian, 0);
        g1 += row(gradu, 0);
        if constexpr (displacementSize == 1) {
          coeff(bopI, 0, 0) = getDiagonalEntry(gradUdI[0], 0) * g1[0];
        } else if constexpr (displacementSize == 2) {
          typename DefaultLinearAlgebra::template FixedSizedVector<ctype, gridDim> g2 = row(referenceJacobian, 1);
          g2 += row(gradu, 1);
          const auto& dNIdT1 = getDiagonalEntry(gradUdI[0], 0);
          const auto& dNIdT2 = getDiagonalEntry(gradUdI[1], 0);
          row(bopI, 0)       = dNIdT1 * g1;                // dE11_dCIx,dE11_dCIy
          row(bopI, 1)       = dNIdT2 * g2;                // dE22_dCIx,dE22_dCIy
          row(bopI, 2)       = dNIdT2 * g1 + dNIdT1 * g2;  // 2*dE12_dCIx,2*dE12_dCIy
        } else if constexpr (displacementSize == 3) {
          typename DefaultLinearAlgebra::template FixedSizedVector<ctype, gridDim> g2 = row(referenceJacobian, 1);
          g2 += row(gradu, 1);
          typename DefaultLinearAlgebra::template FixedSizedVector<ctype, gridDim> g3 = row(referenceJacobian, 2);
          g3 += row(gradu, 2);
          const auto& dNIdT1 = getDiagonalEntry(gradUdI[0], 0);
          const auto& dNIdT2 = getDiagonalEntry(gradUdI[1], 0);
          const auto& dNIdT3 = getDiagonalEntry(gradUdI[2], 0);
          row(bopI, 0)       = dNIdT1 * g1;                // dE11_dCIx,dE11_dCIy,dE11_dCIz
          row(bopI, 1)       = dNIdT2 * g2;                // dE22_dCIx,dE22_dCIy,dE22_dCIz
          row(bopI, 2)       = dNIdT3 * g3;                // dE33_dCIx,dE33_dCIy,dE33_dCIz
          row(bopI, 3)       = dNIdT3 * g2 + dNIdT2 * g3;  // dE23_dCIx,dE23_dCIy,dE23_dCIz
          row(bopI, 4)       = dNIdT3 * g1 + dNIdT1 * g3;  // dE13_dCIx,dE13_dCIy,dE13_dCIz
          row(bopI, 5)       = dNIdT2 * g1 + dNIdT1 * g2;  // dE12_dCIx,dE12_dCIy,dE12_dCIz
        }

        return bopI;

      } else if constexpr (DerivativeOrder == 1 and LFArgs::hasOneSpatialAll) {
        DUNE_THROW(Dune::NotImplemented, "Higher spatial derivatives of linear strain expression not implemented.");
        return createZeroMatrix<ctype, strainSize, gridDim>();
      } else if constexpr (DerivativeOrder == 1 and LFArgs::hasOneSpatialSingle) {
        DUNE_THROW(Dune::NotImplemented, "Higher spatial derivatives of linear strain expression not implemented.");
        return createZeroMatrix<ctype, strainSize, 1>();
      } else if constexpr (DerivativeOrder == 2) {
        if constexpr (LFArgs::hasNoSpatial and LFArgs::hasTwoCoeff) {
          const auto& S      = std::get<0>(lfArgs.alongArgs.args);
          using StressVector = std::remove_cvref_t<decltype(S)>;

          static_assert(Rows<StressVector>::value == strainSize);

          const auto gradArgsdIJ         = addWrt(lfArgs, wrt(DerivativeDirections::spatialAll));
          const auto& [gradUdI, gradUdJ] = evaluateSecondOrderDerivativesImpl(this->m(), gradArgsdIJ);
          if constexpr (displacementSize == 1) {
            const auto& dNIdT1 = getDiagonalEntry(gradUdI[0], 0);
            const auto& dNJdT1 = getDiagonalEntry(gradUdJ[0], 0);
            const ctype val    = S[0] * dNIdT1 * dNJdT1;
            return createScaledIdentityMatrix<ctype, displacementSize, displacementSize>(val);
          } else if constexpr (displacementSize == 2) {
            const auto& dNIdT1 = getDiagonalEntry(gradUdI[0], 0);
            const auto& dNIdT2 = getDiagonalEntry(gradUdI[1], 0);
            const auto& dNJdT1 = getDiagonalEntry(gradUdJ[0], 0);
            const auto& dNJdT2 = getDiagonalEntry(gradUdJ[1], 0);
            const ctype val
                = S[0] * dNIdT1 * dNJdT1 + S[1] * dNIdT2 * dNJdT2 + S[2] * (dNIdT1 * dNJdT2 + dNJdT1 * dNIdT2);
            return createScaledIdentityMatrix<ctype, displacementSize, displacementSize>(val);
          } else if constexpr (displacementSize == 3) {
            const auto& dNIdT1 = getDiagonalEntry(gradUdI[0], 0);
            const auto& dNIdT2 = getDiagonalEntry(gradUdI[1], 0);
            const auto& dNIdT3 = getDiagonalEntry(gradUdI[2], 0);
            const auto& dNJdT1 = getDiagonalEntry(gradUdJ[0], 0);
            const auto& dNJdT2 = getDiagonalEntry(gradUdJ[1], 0);
            const auto& dNJdT3 = getDiagonalEntry(gradUdJ[2], 0);
            const ctype val    = S[0] * dNIdT1 * dNJdT1 + S[1] * dNIdT2 * dNJdT2 + S[2] * dNIdT3 * dNJdT3
                              + S[3] * (dNIdT2 * dNJdT3 + dNJdT2 * dNIdT3) + S[4] * (dNIdT1 * dNJdT3 + dNJdT1 * dNIdT3)
                              + S[5] * (dNIdT1 * dNJdT2 + dNJdT1 * dNIdT2);
            return createScaledIdentityMatrix<ctype, displacementSize, displacementSize>(val);
          }
        } else if constexpr (LFArgs::hasOneSpatial and LFArgs::hasSingleCoeff) {
          if constexpr (LFArgs::hasOneSpatialSingle and LFArgs::hasSingleCoeff) {
            DUNE_THROW(Dune::NotImplemented, "Higher spatial derivatives of linear strain expression not implemented.");
            return createZeroMatrix<ctype, strainSize, displacementSize>();
          } else if constexpr (LFArgs::hasOneSpatialAll and LFArgs::hasSingleCoeff) {
            DUNE_THROW(Dune::NotImplemented, "Higher spatial derivatives of linear strain expression not implemented.");
            return std::array<
                typename DefaultLinearAlgebra::template FixedSizedMatrix<ctype, strainSize, displacementSize>,
                gridDim>{};
          }
        }
      } else if constexpr (DerivativeOrder == 3) {
        DUNE_THROW(Dune::NotImplemented, "Higher spatial derivatives of linear strain expression not implemented.");
        if constexpr (LFArgs::hasOneSpatialSingle) {
          return createZeroMatrix<ctype, displacementSize, displacementSize>();
        } else if constexpr (LFArgs::hasOneSpatialAll) {
          return createZeroMatrix<ctype, displacementSize, displacementSize>();
        } else
          static_assert(
              LFArgs::hasOneSpatialSingle or LFArgs::hasOneSpatialAll,
              "Only a spatial single direction or all spatial directions are supported. You should not end up here.");
      } else
        static_assert(DerivativeOrder > 3 or DerivativeOrder < 1,
                      "Only first, second and third order derivatives are supported.");
    }
  };

  template <typename E1>
  struct LocalFunctionTraits<GreenLagrangeStrainsExpr<E1>> {
    using E1Raw = std::remove_cvref_t<E1>;
    /** \brief Size of the function value */
    static constexpr int valueSize = E1Raw::valueSize == 1 ? 1 : (E1Raw::valueSize == 2 ? 3 : 6);
    /** \brief Type for the points for evaluation, usually the integration points */
    using DomainType = std::common_type_t<typename E1Raw::DomainType>;
    /** \brief Type used for coordinates */
    using ctype = std::common_type_t<typename E1Raw::ctype>;
    /** \brief Dimension of the grid */
    static constexpr int gridDim = E1Raw::gridDim;
    /** \brief Dimension of the world where this function is mapped to from the reference element */
    static constexpr int worldDimension = E1Raw::worldDimension;
  };

  template <typename E1>
  requires IsLocalFunction<E1>
  constexpr auto greenLagrangeStrains(E1&& u) { return GreenLagrangeStrainsExpr<E1>(std::forward<E1>(u)); }

}  // namespace Dune
