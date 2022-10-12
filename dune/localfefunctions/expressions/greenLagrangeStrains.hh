//
// Created by lex on 4/25/22.
//

#pragma once
#include "rebind.hh"

#include <dune/localfefunctions/expressions/unaryExpr.hh>
#include <dune/localfefunctions/helper.hh>

namespace Dune {

  template <typename E1,typename E2>
  class GreenLagrangeStrains : public BinaryExpr<GreenLagrangeStrains, E1,E2> {
  public:
    using Base = BinaryExpr<GreenLagrangeStrains, E1,E2>;
    using Base::Base;
    using Traits = LocalFunctionTraits<GreenLagrangeStrains>;

    /** \brief Type used for coordinates */
    using ctype                           = typename Traits::ctype;
    static constexpr int strainSize       = Traits::valueSize;
    static constexpr int displacementSize = Base::E1Raw::valueSize;
    static constexpr int gridDim          = Traits::gridDim;

    static_assert(Base::E1Raw::template order<0>() == 1,
                  "Linear strain expression only supported for linear displacement function w.r.t. coefficients.");

    template <size_t ID_ = 0>
    static constexpr int orderID = Base::E1Raw::template order<ID_>();

    template <typename LFArgs>
    auto evaluateValueOfExpression(const LFArgs &lfArgs) const {
      const auto gradArgs = replaceWrt(lfArgs, wrt(DerivativeDirections::spatialAll));
      const auto referenceJacobian    = evaluateDerivativeImpl(this->l(), gradArgs);
      const auto gradu    = evaluateDerivativeImpl(this->r(), gradArgs);

      Dune::FieldVector<ctype ,strainSize> E;
      for (int i = 0; i < gridDim; ++i)
      E[i] = referenceJacobian[i]*gradu[i]+ 0.5 * gradu[i].two_norm2();

      if constexpr (gridDim == 2)
        E[2] = (referenceJacobian[0]+gradu[0]) * (referenceJacobian[1]+gradu[1]);
      else if constexpr (gridDim == 3) {
        E[3] = (referenceJacobian[1]+gradu[1]) * (referenceJacobian[2]+gradu[2]);
        E[4] = (referenceJacobian[0]+gradu[0]) * (referenceJacobian[2]+gradu[2]);
        E[5] = (referenceJacobian[0]+gradu[0]) * (referenceJacobian[1]+gradu[1]);
      }

      return E;
    }

    template <int DerivativeOrder, typename LFArgs>
    auto evaluateDerivativeOfExpression(const LFArgs &lfArgs) const {
      if constexpr (DerivativeOrder == 1 and LFArgs::hasSingleCoeff) {
        Dune::FieldMatrix<double, strainSize, gridDim> bopI;
        const auto gradArgs = addWrt(lfArgs, wrt(DerivativeDirections::spatialAll));
        const auto gradUdI  = evaluateDerivativeImpl(this->m(), gradArgs);
        if constexpr (displacementSize == 1) {
          coeff(bopI, 0, 0) = gradUdI[0].scalar();
        } else if constexpr (displacementSize == 2) {
          row(bopI, 0)[0] = gradUdI[0].scalar();
          row(bopI, 0)[1] = 0;
          row(bopI, 1)[0] = 0;
          row(bopI, 1)[1] = gradUdI[1].scalar();
          row(bopI, 2)[0] = gradUdI[1].scalar();
          row(bopI, 2)[1] = gradUdI[0].scalar();

        } else if constexpr (displacementSize == 3) {
          row(bopI, 0)[0] = gradUdI[0].scalar();
          row(bopI, 0)[1] = 0;
          row(bopI, 0)[2] = 0;

          row(bopI, 1)[0] = 0;
          row(bopI, 1)[1] = gradUdI[1].scalar();
          row(bopI, 1)[2] = 0;

          row(bopI, 2)[0] = 0;
          row(bopI, 2)[1] = 0;
          row(bopI, 2)[2] = gradUdI[2].scalar();

          row(bopI, 3)[0] = 0;
          row(bopI, 3)[1] = gradUdI[2].scalar();
          row(bopI, 3)[2] = gradUdI[1].scalar();

          row(bopI, 4)[0] = gradUdI[2].scalar();
          row(bopI, 4)[1] = 0;
          row(bopI, 4)[2] = gradUdI[0].scalar();

          row(bopI, 5)[0] = gradUdI[1].scalar();
          row(bopI, 5)[1] = gradUdI[0].scalar();
          row(bopI, 5)[2] = 0;
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
          return createZeroMatrix<ctype, displacementSize, displacementSize>();
        } else if constexpr (LFArgs::hasOneSpatial and LFArgs::hasSingleCoeff) {
          if constexpr (LFArgs::hasOneSpatialSingle and LFArgs::hasSingleCoeff) {
            return createZeroMatrix<ctype, strainSize, displacementSize>();
          } else if constexpr (LFArgs::hasOneSpatialAll and LFArgs::hasSingleCoeff) {
            std::array<Dune::FieldMatrix<ctype, strainSize, displacementSize>, gridDim> res;
            for (int i = 0; i < gridDim; ++i)
              setZero(res[i]);
            return res;
          }
        }
      } else if constexpr (DerivativeOrder == 3) {
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

  template <typename E1,typename E2>
  struct LocalFunctionTraits<GreenLagrangeStrains<E1,E2>> {
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

  template <typename E1,typename E2>
  requires IsLocalFunction<E1,E2>
  constexpr auto greenLagrangeStrains(E1 &&X,E2 &&u) { return GreenLagrangeStrains<E1,E2>(std::forward<E1>(X),std::forward<E2>(u)); }

}  // namespace Dune