//
// Created by lex on 4/25/22.
//

#pragma once
#include "rebind.hh"

#include <dune/localfefunctions/expressions/unaryExpr.hh>
#include <dune/localfefunctions/helper.hh>

namespace Dune {

  template <typename E1>
  class GradExpr : public UnaryLocalFunctionExpression<GradExpr, E1> {
  public:
    using Base = UnaryLocalFunctionExpression<GradExpr, E1>;
    using Base::Base;
    using Traits = LocalFunctionTraits<GradExpr>;

    /** \brief Type used for coordinates */
    using ctype                           = typename Traits::ctype;
    static constexpr int valueSize       = Traits::valueSize;
    static constexpr int gridDim          = Traits::gridDim;

    static_assert(Base::E1Raw::template order<0>() == 1,
                  "Linear strain expression only supported for linear displacement function w.r.t. coefficients.");

    template <size_t ID_ = 0>
    static constexpr int orderID = Base::E1Raw::template order<ID_>();

    template <typename LFArgs>
    auto evaluateValueOfExpression(const LFArgs &lfArgs) const {
      const auto gradArgs = replaceWrt(lfArgs, wrt(DerivativeDirections::spatialAll));
      return evaluateDerivativeImpl(this->m(), gradArgs);
    }

    template <int DerivativeOrder, typename LFArgs>
    auto evaluateDerivativeOfExpression(const LFArgs &lfArgs) const {
      if constexpr (DerivativeOrder == 1 and LFArgs::hasSingleCoeff) {
        Eigen::Matrix<double, valueSize, gridDim> bopI;
        const auto gradArgs = addWrt(lfArgs, wrt(DerivativeDirections::spatialAll));
        const auto gradUdI  = evaluateDerivativeImpl(this->m(), gradArgs);
        if constexpr (displacementSize == 1) {
          bopI(0, 0) = gradUdI[0].diagonal()(0);
        } else if constexpr (displacementSize == 2) {
          bopI.row(0) << gradUdI[0].diagonal()(0), 0;
          bopI.row(1) << 0, gradUdI[1].diagonal()(1);
          bopI.row(2) << gradUdI[1].diagonal()(0), gradUdI[0].diagonal()(1);

        } else if constexpr (displacementSize == 3) {
          bopI.row(0) << gradUdI[0].diagonal()(0), 0, 0;
          bopI.row(1) << 0, gradUdI[1].diagonal()(1), 0;
          bopI.row(2) << 0, 0, gradUdI[2].diagonal()(2);
          bopI.row(3) << 0, gradUdI[2].diagonal()(1), gradUdI[1].diagonal()(2);
          bopI.row(4) << gradUdI[2].diagonal()(0), 0, gradUdI[0].diagonal()(2);
          bopI.row(5) << gradUdI[1].diagonal()(0), gradUdI[0].diagonal()(1), 0;
        }

        return bopI;

      } else if constexpr (DerivativeOrder == 1 and LFArgs::hasOneSpatialAll) {
        DUNE_THROW(Dune::NotImplemented, "Higher spatial derivatives of linear strain expression not implemented.");
        return Eigen::Matrix<ctype, strainSize, gridDim>::Zero().eval();
      } else if constexpr (DerivativeOrder == 1 and LFArgs::hasOneSpatialSingle) {
        DUNE_THROW(Dune::NotImplemented, "Higher spatial derivatives of linear strain expression not implemented.");
        return Eigen::Matrix<ctype, strainSize, 1>::Zero().eval();
      } else if constexpr (DerivativeOrder == 2) {
        if constexpr (LFArgs::hasNoSpatial and LFArgs::hasTwoCoeff) {
          return Eigen::Matrix<ctype, displacementSize, displacementSize>::Zero().eval();
        } else if constexpr (LFArgs::hasOneSpatial and LFArgs::hasSingleCoeff) {
          if constexpr (LFArgs::hasOneSpatialSingle and LFArgs::hasSingleCoeff) {
            return Eigen::Matrix<ctype, strainSize, displacementSize>::Zero().eval();
          } else if constexpr (LFArgs::hasOneSpatialAll and LFArgs::hasSingleCoeff) {
            std::array<Eigen::Matrix<ctype, strainSize, displacementSize>, gridDim> res;
            for (int i = 0; i < gridDim; ++i)
              res[i].setZero();
            return res;
          }
        }
      } else if constexpr (DerivativeOrder == 3) {
        if constexpr (LFArgs::hasOneSpatialSingle) {
          return Eigen::Matrix<ctype, displacementSize, displacementSize>::Zero().eval();
        } else if constexpr (LFArgs::hasOneSpatialAll) {
          return Eigen::Matrix<ctype, displacementSize, displacementSize>::Zero().eval();
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
  struct LocalFunctionTraits<GradExpr<E1>> {
    using E1Raw = std::remove_cvref_t<E1>;
    /** \brief Size of the function value */
    static constexpr int valueSize = E1Raw::valueSize == 1 ? 1 : (E1Raw::valueSize == 2 ? 3 : 6);
    /** \brief Type for the points for evaluation, usually the integration points */
    using DomainType = std::common_type_t<typename E1Raw::DomainType>;
    /** \brief Type used for coordinates */
    using ctype = std::common_type_t<typename E1Raw::ctype>;
    /** \brief Dimension of the grid */
    static constexpr int gridDim = E1Raw::gridDim;
  };

  template <typename E1>
  requires IsLocalFunction<E1>
  constexpr auto grad(E1 &&u) { return GradExpr<E1>(std::forward<E1>(u)); }

}  // namespace Dune