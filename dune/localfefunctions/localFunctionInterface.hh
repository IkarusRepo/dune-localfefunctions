// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include "leafNodeCollection.hh"
#include "localFunctionArguments.hh"

#include <concepts>

#include <dune/localfefunctions/cachedlocalBasis/cachedlocalBasis.hh>
#include <dune/localfefunctions/concepts.hh>
#include <dune/localfefunctions/expressions/exprChecks.hh>
#include <dune/localfefunctions/linearAlgebraHelper.hh>

namespace Dune {

  template <typename LocalFunctionImpl>
  class LocalFunctionInterface {
  public:
    using Traits                        = LocalFunctionTraits<LocalFunctionImpl>;
    using DomainType                    = typename Traits::DomainType;
    static constexpr int gridDim        = Traits::gridDim;
    static constexpr int worldDimension = Traits::worldDimension;

    template <typename WrtType>
    static constexpr bool hasTwoCoeff = DerivativeDirections::HasTwoCoeff<WrtType>;
    template <typename WrtType>
    static constexpr bool hasSingleCoeff = DerivativeDirections::HasSingleCoeff<WrtType>;
    template <typename WrtType>
    static constexpr bool hasNoCoeff = DerivativeDirections::HasNoCoeff<WrtType>;

    template <std::size_t ID_ = 0>
    static constexpr auto order(Dune::index_constant<ID_> = Dune::index_constant<ID_>()) {
      return LocalFunctionImpl::template orderID<ID_>;
    }

    /** \brief Return the function value*/
    template <typename DomainTypeOrIntegrationPointIndex, typename Transform = DerivativeDirections::GridElement>
    requires Concepts::IsIntegrationPointIndexOrIntegrationPointPosition<DomainTypeOrIntegrationPointIndex, DomainType>
    auto evaluate(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                  const On<Transform>& transform = {}) const {
      checkIfLocalFunctionCanProvideDerivativeTransformation<Transform>();
      const LocalFunctionEvaluationArgs evalArgs(ipIndexOrPosition, wrt(), along(), transform);
      return evaluateFunctionImpl(*this, evalArgs);
    }

    /** \brief Deligation function to calculate derivatives */
    template <typename... WrtArgs, typename Transform = DerivativeDirections::GridElement, typename... AlongArgs,
              typename DomainTypeOrIntegrationPointIndex>
    auto evaluateDerivative(const DomainTypeOrIntegrationPointIndex& localOrIpId, Wrt<WrtArgs...>&& args,
                            Along<AlongArgs...>&& along, const On<Transform>& transform = {}) const {
      checkIfLocalFunctionCanProvideDerivativeTransformation<Transform>();

      const LocalFunctionEvaluationArgs evalArgs(localOrIpId, std::forward<Wrt<WrtArgs...>>(args),
                                                 std::forward<Along<AlongArgs...>>(along), transform);
      return evaluateDerivativeImpl(*this, evalArgs);
    }

    /** \brief Deligation function to calculate derivatives, without providing along arguments */
    template <typename... WrtArgs, typename Transform = DerivativeDirections::GridElement,
              typename DomainTypeOrIntegrationPointIndex>
    auto evaluateDerivative(const DomainTypeOrIntegrationPointIndex& localOrIpId, Wrt<WrtArgs...>&& args,
                            const On<Transform>& transform = {}) const {
      checkIfLocalFunctionCanProvideDerivativeTransformation<Transform>();

      return evaluateDerivative(localOrIpId, std::forward<Wrt<WrtArgs...>>(args), along(), transform);
    }

    /** Return the view of the integration points of the bound Basis with id I */
    template <std::size_t I = 0>
    auto viewOverIntegrationPoints(Dune::index_constant<I> = Dune::index_constant<I>()) const {
      assert(checkIfAllLeafNodeHaveTheSameBasisState(impl())
             && "The basis of the leaf nodes are not in the same state.");
      auto leafNodeCollection = collectLeafNodeLocalFunctions(impl());
      auto& node              = leafNodeCollection.node(Dune::index_constant<I>());
      return node.basis().viewOverIntegrationPoints();
    }

    template <std::size_t I = 0>
    auto& node(Dune::index_constant<I> = Dune::index_constant<I>()) const {
      return collectLeafNodeLocalFunctions(impl()).node(Dune::index_constant<I>());
    }

    /** \brief Forward the binding to the local basis */
    template <typename IntegrationRule>
    void bind(IntegrationRule&& p_rule, std::set<int>&& ints) {
      impl().basis.bind(std::forward<IntegrationRule>(p_rule), std::forward<std::set<int>>(ints));
    }
    template <std::size_t I = 0>
    auto& geometry() {
      return collectLeafNodeLocalFunctions(impl()).node(Dune::index_constant<I>()).geometry();
    }

  protected:
    /* Default implementation returns Zero expression if they are not overloaded */
    template <typename DomainTypeOrIntegrationPointIndex, typename... AlongArgs,
              typename Transform = DerivativeDirections::GridElement>
    auto evaluateSecondDerivativeWRTCoeffsImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                               const std::array<size_t, 2>& coeffsIndex,
                                               const Along<AlongArgs...>& alongArgs,
                                               const On<Transform>& transform = {}) const {
      return createZeroMatrix<typename LocalFunctionImpl::ctype, LocalFunctionImpl::correctionSize,
                              LocalFunctionImpl::correctionSize>();
    }

    /* Default implementation returns Zero expression if they are not overloaded */
    template <typename DomainTypeOrIntegrationPointIndex, typename... AlongArgs,
              typename Transform = DerivativeDirections::GridElement>
    auto evaluateThirdDerivativeWRTCoeffsTwoTimesAndSpatialSingleImpl(
        const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition, const std::array<size_t, 2>& coeffsIndex,
        const int spatialIndex, const Along<AlongArgs...>& alongArgs, const On<Transform>& transArgs) const {
      return createZeroMatrix<typename LocalFunctionImpl::ctype, LocalFunctionImpl::correctionSize,
                              LocalFunctionImpl::correctionSize>();
    }

    /* Default implementation returns Zero expression if they are not overloaded */
    template <typename DomainTypeOrIntegrationPointIndex, typename Transform = DerivativeDirections::GridElement>
    auto evaluateDerivativeWRTSpaceAllImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                           const On<Transform>& transArgs) const {
      return createZeroMatrix<typename LocalFunctionImpl::Jacobian>();
    }

    /* Default implementation returns Zero expression if they are not overloaded */
    template <typename DomainTypeOrIntegrationPointIndex, typename Transform = DerivativeDirections::GridElement>
    auto evaluateDerivativeWRTCoeffsImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition, int coeffsIndex,
                                         const On<Transform>& transArgs) const {
      return createZeroMatrix<typename LocalFunctionImpl::ctype, LocalFunctionImpl::valueSize,
                              LocalFunctionImpl::correctionSize>();
    }

    /* Default implementation returns Zero expression if they are not overloaded  */
    template <typename DomainTypeOrIntegrationPointIndex, typename Transform = DerivativeDirections::GridElement>
    auto evaluateDerivativeWRTCoeffsANDSpatialImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                                   int coeffsIndex, const On<Transform>& transArgs) const {
      return std::array<Dune::DerivativeDirections::ZeroMatrix, gridDim>();
    }

    /* Default implementation returns Zero expression if they are not overloaded  */
    template <typename DomainTypeOrIntegrationPointIndex, typename Transform = DerivativeDirections::GridElement>
    auto evaluateDerivativeWRTCoeffsANDSpatialSingleImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                                         int coeffsIndex, int spatialIndex,
                                                         const On<Transform>& transArgs) const {
      return createZeroMatrix<typename LocalFunctionImpl::ctype, LocalFunctionImpl::valueSize,
                              LocalFunctionImpl::correctionSize>();
    }
    /* Default implementation returns Zero expression if they are not overloaded  */
    template <typename DomainTypeOrIntegrationPointIndex, typename... AlongArgs,
              typename Transform = DerivativeDirections::GridElement>
    auto evaluateThirdDerivativeWRTCoeffsTwoTimesAndSpatialImpl(
        const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition, const std::array<size_t, 2>& coeffsIndex,
        const Along<AlongArgs...>& alongArgs, const On<Transform>& transArgs) const {
      return createZeroMatrix<typename LocalFunctionImpl::ctype, LocalFunctionImpl::correctionSize,
                              LocalFunctionImpl::correctionSize>();
    }

    /* Default implementation returns Zero expression if they are not overloaded  */
    template <typename DomainTypeOrIntegrationPointIndex, typename Transform = DerivativeDirections::GridElement>
    auto evaluateDerivativeWRTSpaceSingleImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                              int spaceIndex, const On<Transform>& transArgs) const {
      return createZeroVector<typename LocalFunctionImpl::JacobianColType>();
    }

  private:
    template <typename Transform>
    static consteval void checkIfLocalFunctionCanProvideDerivativeTransformation() {
      static_assert((gridDim != worldDimension and std::is_same_v<Transform, DerivativeDirections::ReferenceElement>)
                        or (gridDim == worldDimension),
                    "For local functions that are defined on a geometry with a codimension!=0, the derivatives cannot "
                    "be transformed. Pass Dune::on(DerivativeDirections::referenceElement) instead. ");
    }
    template <typename LocalFunctionEvaluationArgs_, typename LocalFunctionImpl_>
    friend auto evaluateDerivativeImpl(const LocalFunctionInterface<LocalFunctionImpl_>& f,
                                       const LocalFunctionEvaluationArgs_& localFunctionArgs);

    template <typename LocalFunctionEvaluationArgs_, typename LocalFunctionImpl_>
    friend auto evaluateFunctionImpl(const LocalFunctionInterface<LocalFunctionImpl_>& f,
                                     const LocalFunctionEvaluationArgs_& localFunctionArgs);

    template <typename LF>
    //      requires LocalFunction<LF>
    friend auto collectNonArithmeticLeafNodes(LF&& a);

    constexpr LocalFunctionImpl const& impl() const  // CRTP
    {
      return static_cast<LocalFunctionImpl const&>(*this);
    }

    constexpr LocalFunctionImpl& impl()  // CRTP
    {
      return static_cast<LocalFunctionImpl&>(*this);
    }
  };

  template <typename LocalFunctionEvaluationArgs_, typename LocalFunctionImpl>
  auto evaluateFunctionImpl(const LocalFunctionInterface<LocalFunctionImpl>& f,
                            const LocalFunctionEvaluationArgs_& localFunctionArgs) {
    if constexpr (LocalFunctionImpl::isLeaf)
      return f.impl().evaluateFunctionImpl(localFunctionArgs.integrationPointOrIndex,
                                           localFunctionArgs.transformWithArgs);
    else {
      return f.impl().evaluateValueOfExpression(localFunctionArgs);
    }
  }

  template <typename LocalFunctionArguments, typename LocalFunctionImpl>
  auto evaluateDerivativeImpl(const LocalFunctionInterface<LocalFunctionImpl>& f,
                              const LocalFunctionArguments& localFunctionArgs) {
    using namespace Dune::Indices;
    if constexpr (LocalFunctionImpl::isLeaf) {
      if constexpr (LocalFunctionArguments::hasNoCoeff) {
        if constexpr (LocalFunctionArguments::hasOneSpatialSingle) {
          return f.impl().evaluateDerivativeWRTSpaceSingleImpl(localFunctionArgs.integrationPointOrIndex,
                                                               localFunctionArgs.spatialPartialIndices,
                                                               localFunctionArgs.transformWithArgs);
        } else if constexpr (LocalFunctionArguments::hasOneSpatialAll) {
          return f.impl().evaluateDerivativeWRTSpaceAllImpl(localFunctionArgs.integrationPointOrIndex,
                                                            localFunctionArgs.transformWithArgs);
        }
      } else if constexpr (LocalFunctionArguments::hasSingleCoeff) {
        if constexpr (decltype(localFunctionArgs.coeffsIndices[_0][_0])::value != LocalFunctionImpl::id[0])
          return DerivativeDirections::ZeroMatrix();
        else if constexpr (LocalFunctionArguments::hasNoSpatial) {
          return f.impl().evaluateDerivativeWRTCoeffsImpl(localFunctionArgs.integrationPointOrIndex,
                                                          localFunctionArgs.coeffsIndices[_0][1],
                                                          localFunctionArgs.transformWithArgs);
        } else if constexpr (LocalFunctionArguments::hasOneSpatialSingle) {
          return f.impl().evaluateDerivativeWRTCoeffsANDSpatialSingleImpl(
              localFunctionArgs.integrationPointOrIndex, localFunctionArgs.coeffsIndices[_0][1],
              localFunctionArgs.spatialPartialIndices, localFunctionArgs.transformWithArgs);
        } else if constexpr (LocalFunctionArguments::hasOneSpatialAll) {
          return f.impl().evaluateDerivativeWRTCoeffsANDSpatialImpl(localFunctionArgs.integrationPointOrIndex,
                                                                    localFunctionArgs.coeffsIndices[_0][1],
                                                                    localFunctionArgs.transformWithArgs);
        }
      } else if constexpr (LocalFunctionArguments::hasTwoCoeff) {
        if constexpr (LocalFunctionArguments::hasNoSpatial) {
          return f.impl().evaluateSecondDerivativeWRTCoeffsImpl(
              localFunctionArgs.integrationPointOrIndex,
              {localFunctionArgs.coeffsIndices[_0][1], localFunctionArgs.coeffsIndices[_1][1]},
              localFunctionArgs.alongArgs, localFunctionArgs.transformWithArgs);
        } else if constexpr (LocalFunctionArguments::hasOneSpatialSingle) {
          return f.impl().evaluateThirdDerivativeWRTCoeffsTwoTimesAndSpatialSingleImpl(
              localFunctionArgs.integrationPointOrIndex,
              {localFunctionArgs.coeffsIndices[_0][1], localFunctionArgs.coeffsIndices[_1][1]},
              localFunctionArgs.spatialPartialIndices, localFunctionArgs.alongArgs,
              localFunctionArgs.transformWithArgs);
        } else if constexpr (LocalFunctionArguments::hasOneSpatialAll) {
          return f.impl().evaluateThirdDerivativeWRTCoeffsTwoTimesAndSpatialImpl(
              localFunctionArgs.integrationPointOrIndex,
              {localFunctionArgs.coeffsIndices[_0][1], localFunctionArgs.coeffsIndices[_1][1]},
              localFunctionArgs.alongArgs, localFunctionArgs.transformWithArgs);
        }
      }
    } else {
      return f.impl().template evaluateDerivativeOfExpression<LocalFunctionArguments::derivativeOrder>(
          localFunctionArgs);
    }
  }

  template <typename LocalFunctionArguments, typename LocalFunctionImpl>
  auto evaluateFirstOrderDerivativesImpl(const LocalFunctionInterface<LocalFunctionImpl>& f,
                                         const LocalFunctionArguments& localFunctionArgs) {
    if constexpr (LocalFunctionArguments::derivativeOrder == 3) {
      const auto argsForDx              = localFunctionArgs.extractSpatialOrFirstWrtArg();
      const auto [argsForDy, argsForDz] = extractWrtArgsTwoCoeffsToSingleCoeff(localFunctionArgs);
      auto dfdx                         = evaluateDerivativeImpl(f, argsForDx);
      auto dfdy                         = evaluateDerivativeImpl(f, argsForDy);
      auto dfdz                         = evaluateDerivativeImpl(f, argsForDz);
      return std::make_tuple(dfdx, dfdy, dfdz);
    } else if constexpr (LocalFunctionArguments::derivativeOrder == 2) {
      const auto [argsForDx, argsForDy] = extractFirstTwoArgs(localFunctionArgs);
      auto dfdx                         = evaluateDerivativeImpl(f, argsForDx);
      auto dfdy                         = evaluateDerivativeImpl(f, argsForDy);
      return std::make_tuple(dfdx, dfdy);
    }
  }

  template <typename LocalFunctionArguments, typename LocalFunctionImpl>
  auto evaluateSecondOrderDerivativesImpl(const LocalFunctionInterface<LocalFunctionImpl>& f,
                                          const LocalFunctionArguments& localFunctionArgs) {
    if constexpr (LocalFunctionArguments::derivativeOrder == 3) {
      const auto argsForDx              = localFunctionArgs.extractSpatialOrFirstWrtArg();
      const auto [argsForDy, argsForDz] = extractWrtArgsTwoCoeffsToSingleCoeff(localFunctionArgs);
      const auto argsForDxy             = joinWRTArgs(argsForDx, argsForDy);
      const auto argsForDxz             = joinWRTArgs(argsForDx, argsForDz);
      const auto df_dxy                 = evaluateDerivativeImpl(f, argsForDxy);
      const auto df_dxz                 = evaluateDerivativeImpl(f, argsForDxz);
      return std::make_tuple(df_dxy, df_dxz);
    } else
      static_assert(LocalFunctionArguments::derivativeOrder == 3);
  }

}  // namespace Dune
