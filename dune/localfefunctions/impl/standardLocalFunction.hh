/*
 * This file is part of the Ikarus distribution (https://github.com/ikarus-project/ikarus).
 * Copyright (c) 2022. The Ikarus developers.
 *
 * This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 */

#pragma once

#include "clonableLocalFunction.hh"

#include <concepts>

#include <dune/common/indices.hh>
#include <dune/localfefunctions/cachedlocalBasis/cachedlocalBasis.hh>
#include <dune/localfefunctions/linalgconcepts.hh>
#include <dune/localfefunctions/localFunctionHelper.hh>
#include <dune/localfefunctions/localFunctionInterface.hh>

#include <Eigen/Core>
#include <Eigen/Dense>
//#include <ikarus/utils/linearAlgebraHelper.hh>

namespace Dune {

  template <typename DuneBasis, typename CoeffContainer, typename Geometry, std::size_t ID = 0,
            typename LinAlg = Dune::DefaultLinearAlgebra>
  class StandardLocalFunction
      : public LocalFunctionInterface<StandardLocalFunction<DuneBasis, CoeffContainer, Geometry, ID, LinAlg>>,
        public ClonableLocalFunction<StandardLocalFunction<DuneBasis, CoeffContainer, Geometry, ID, LinAlg>> {
    using Interface = LocalFunctionInterface<StandardLocalFunction>;

  public:
    friend Interface;
    friend ClonableLocalFunction<StandardLocalFunction>;

    constexpr StandardLocalFunction(const Dune::CachedLocalBasis<DuneBasis>& p_basis, const CoeffContainer& coeffs_,
                                    const std::shared_ptr<const Geometry>& geo,
                                    Dune::template index_constant<ID> = Dune::template index_constant<std::size_t(0)>{})
        : basis_{p_basis},
          coeffs{coeffs_},
          geometry_{geo}  //        ,  coeffsAsMat{Dune::viewAsEigenMatrixFixedDyn(coeffs)}
    {}

    static constexpr bool isLeaf = true;
    static constexpr std::array<int, 1> id{ID};

    using LinearAlgebra = LinAlg;

    template <size_t ID_ = 0>
    static constexpr int orderID = ID_ == ID ? linear : constant;

    template <typename LocalFunctionEvaluationArgs_, typename LocalFunctionImpl_>
    friend auto evaluateDerivativeImpl(const LocalFunctionInterface<LocalFunctionImpl_>& f,
                                       const LocalFunctionEvaluationArgs_& localFunctionArgs);

    template <typename LocalFunctionEvaluationArgs_, typename LocalFunctionImpl_>
    friend auto evaluateFunctionImpl(const LocalFunctionInterface<LocalFunctionImpl_>& f,
                                     const LocalFunctionEvaluationArgs_& localFunctionArgs);

    using Traits = LocalFunctionTraits<StandardLocalFunction>;
    /** \brief Type used for coordinates */
    using ctype = typename Traits::ctype;
    //    /** \brief Dimension of the coeffs */
    static constexpr int valueSize = Traits::valueSize;
    /** \brief Dimension of the correction size of coeffs */
    static constexpr int correctionSize = Traits::correctionSize;
    /** \brief Dimension of the grid */
    static constexpr int gridDim = Traits::gridDim;
    /** \brief Dimension of the world where this function is mapped to from the reference element */
    static constexpr int worldDimension = Traits::worldDimension;
    /** \brief Type for coordinate vector in world space */
    using FunctionReturnType = typename Traits::FunctionReturnType;
    /** \brief The manifold where the function values lives in */
    using Manifold = typename Traits::Manifold;
    /** \brief Type for the directional derivatives */
    using AlongType = typename Traits::AlongType;
    /** \brief Type for the Jacobian matrix */
    using Jacobian = typename Traits::Jacobian;
    /** \brief Type for a column of the Jacobian matrix */
    using JacobianColType = typename Traits::JacobianColType;
    /** \brief Type for the derivatives wrT the coefficients */
    using CoeffDerivMatrix = typename Traits::CoeffDerivMatrix;
    /** \brief Type for ansatz function values */
    using AnsatzFunctionType = typename Traits::AnsatzFunctionType;
    /** \brief Type for the Jacobian of the ansatz function values */
    using AnsatzFunctionJacobian = typename Traits::AnsatzFunctionJacobian;

    const auto& coefficientsRef() const { return coeffs; }
    auto& coefficientsRef() { return coeffs; }
    auto& geometry() const { return geometry_; }

    template <typename OtherType>
    struct rebind {
      using other = StandardLocalFunction<
          DuneBasis, typename Std::Rebind<CoeffContainer, typename Manifold::template rebind<OtherType>::other>::other,
          Geometry, ID, LinAlg>;
    };

    const Dune::CachedLocalBasis<DuneBasis>& basis() const { return basis_; }

  private:
    template <typename DomainTypeOrIntegrationPointIndex, typename TransformArgs>
    FunctionReturnType evaluateFunctionImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                            [[maybe_unused]] const On<TransformArgs>&) const {
      const auto& N = evaluateFunctionWithIPorCoord(ipIndexOrPosition, basis_);
      FunctionReturnType res;
      setZero(res);
      for (int i = 0; i < coeffs.size(); ++i)
        for (int j = 0; j < Rows<FunctionReturnType>::value; ++j) {
          res[j] += coeffs[i].getValue()[j] * N[i];
        }

      return res;
    }

    template <typename DomainTypeOrIntegrationPointIndex, typename TransformArgs>
    Jacobian evaluateDerivativeWRTSpaceAllImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                               const On<TransformArgs>& transArgs) const {
      const auto& dNraw = evaluateDerivativeWithIPorCoord(ipIndexOrPosition, basis_);
      maytransformDerivatives(dNraw, dNTransformed, transArgs, geometry_, ipIndexOrPosition, basis_);
      Jacobian J;
      setZero(J);
      for (int j = 0; j < gridDim; ++j)
        for (int k = 0; k < valueSize; ++k)
          for (int i = 0; i < coeffs.size(); ++i)
            coeff(J, k, j) += coeffs[i].getValue()[k] * coeff(dNTransformed, i, j);
      return J;
    }

    template <typename DomainTypeOrIntegrationPointIndex, typename TransformArgs>
    JacobianColType evaluateDerivativeWRTSpaceSingleImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                                         int spaceIndex, const On<TransformArgs>& transArgs) const {
      const auto& dNraw = evaluateDerivativeWithIPorCoord(ipIndexOrPosition, basis_);
      maytransformDerivatives(dNraw, dNTransformed, transArgs, geometry_, ipIndexOrPosition, basis_);

      JacobianColType Jcol;
      setZero(Jcol);
      for (int j = 0; j < Rows<JacobianColType>::value; ++j) {
        for (int i = 0; i < coeffs.size(); ++i)
          Jcol[j] += coeffs[i].getValue()[j] * coeff(dNTransformed, i, spaceIndex);
      }

      return Jcol;
    }

    template <typename DomainTypeOrIntegrationPointIndex, typename TransformArgs>
    CoeffDerivMatrix evaluateDerivativeWRTCoeffsImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                                     int coeffsIndex, const On<TransformArgs>& transArgs) const {
      const auto& N        = evaluateFunctionWithIPorCoord(ipIndexOrPosition, basis_);
      CoeffDerivMatrix mat = createScaledIdentityMatrix<ctype, valueSize, valueSize>(N[coeffsIndex]);

      return mat;
    }

    template <typename DomainTypeOrIntegrationPointIndex, typename TransformArgs>
    std::array<CoeffDerivMatrix, gridDim> evaluateDerivativeWRTCoeffsANDSpatialImpl(
        const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition, int coeffsIndex,
        const On<TransformArgs>& transArgs) const {
      const auto& dNraw = evaluateDerivativeWithIPorCoord(ipIndexOrPosition, basis_);
      maytransformDerivatives(dNraw, dNTransformed, transArgs, geometry_, ipIndexOrPosition, basis_);
      std::array<CoeffDerivMatrix, gridDim> Warray;
      for (int dir = 0; dir < gridDim; ++dir) {
        setZero(Warray[dir]);
        setDiagonal(Warray[dir], dNTransformed[coeffsIndex][dir]);
      }

      return Warray;
    }

    template <typename DomainTypeOrIntegrationPointIndex, typename TransformArgs>
    CoeffDerivMatrix evaluateDerivativeWRTCoeffsANDSpatialSingleImpl(
        const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition, int coeffsIndex, int spatialIndex,
        const On<TransformArgs>& transArgs) const {
      const auto& dNraw = evaluateDerivativeWithIPorCoord(ipIndexOrPosition, basis_);
      maytransformDerivatives(dNraw, dNTransformed, transArgs, geometry_, ipIndexOrPosition, basis_);
      CoeffDerivMatrix W
          = createScaledIdentityMatrix<ctype, valueSize, valueSize>(dNTransformed[coeffsIndex][spatialIndex]);
      return W;
    }

    mutable AnsatzFunctionJacobian dNTransformed;
    const Dune::CachedLocalBasis<DuneBasis>& basis_;
    CoeffContainer coeffs;
    std::shared_ptr<const Geometry> geometry_;
    //    const decltype(Dune::viewAsEigenMatrixFixedDyn(coeffs)) coeffsAsMat;
  };

  template <typename DuneBasis, typename CoeffContainer, typename Geometry, std::size_t ID, typename LinAlg>
  struct LocalFunctionTraits<StandardLocalFunction<DuneBasis, CoeffContainer, Geometry, ID, LinAlg>> {
    /** \brief Type used for coordinates */
    using ctype = typename CoeffContainer::value_type::ctype;
    /** \brief Dimension of the coeffs */
    static constexpr int valueSize = CoeffContainer::value_type::valueSize;
    /** \brief Dimension of the correction size of coeffs */
    static constexpr int correctionSize = CoeffContainer::value_type::correctionSize;
    /** \brief Dimension of the grid */
    static constexpr int gridDim = Dune::CachedLocalBasis<DuneBasis>::gridDim;
    /** \brief The manifold where the function values lives in */
    using Manifold = typename CoeffContainer::value_type;
    /** \brief Type for the return value */
    using FunctionReturnType = typename Manifold::CoordinateType;
    /** \brief Type for the Jacobian matrix */
    using Jacobian = typename LinAlg::template FixedSizedMatrix<ctype, valueSize, gridDim>;
    /** \brief Type for the derivatives wrt. the coefficients */
    using CoeffDerivMatrix = typename LinAlg::template FixedSizedScaledIdentityMatrix<ctype, valueSize>;
    /** \brief Type for the Jacobian of the ansatz function values */
    using AnsatzFunctionJacobian = typename Dune::CachedLocalBasis<DuneBasis>::JacobianType;
    /** \brief Type for ansatz function values */
    using AnsatzFunctionType = typename Dune::CachedLocalBasis<DuneBasis>::AnsatzFunctionType;
    /** \brief Type for the points for evaluation, usually the integration points */
    using DomainType = typename DuneBasis::Traits::DomainType;
    /** \brief Type for a column of the Jacobian matrix */
    using JacobianColType = typename LinAlg::template FixedSizedVector<ctype, valueSize>;
    /** \brief Type for the directional derivatives */
    using AlongType = typename LinAlg::template FixedSizedVector<ctype, valueSize>;
    /** \brief Dimension of the world where this function is mapped to from the reference element */
    static constexpr int worldDimension = Geometry::coorddimension;
  };

}  // namespace Dune
