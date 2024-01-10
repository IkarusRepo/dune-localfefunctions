// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <cstddef>

#include <dune/istl/multitypeblockvector.hh>
#include <dune/localfefunctions/helper.hh>
#include <dune/typetree/typetree.hh>
namespace Dune {

  struct DefaultFirstOrderTransformFunctor;

  template <typename... Args_>
  struct Along {
    using Args = std::tuple<Args_...>;
    Args args;
  };

  template <typename... Args>
  auto along(Args&&... args) {
    return Along<Args&...>{std::forward_as_tuple(std::forward<Args>(args)...)};
  }

  namespace DerivativeDirections {

    struct ZeroMatrix {
      auto operator[](int) const { return ZeroMatrix{}; }
    };

    [[maybe_unused]] static struct SpatialAll {
    } spatialAll;

    [[maybe_unused]] static struct ReferenceElement {
    } referenceElement;

    [[maybe_unused]] static struct GridElement {
    } gridElement;

    struct SpatialPartial {
      size_t index{};
    };

    template <std::size_t I>
    struct SingleCoeff {
      decltype(Dune::TypeTree::treePath(std::declval<Dune::index_constant<I>>(), std::declval<size_t>())) index{};
    };

    template <std::size_t I, std::size_t J>
    struct TwoCoeff {
      std::pair<decltype(Dune::TypeTree::treePath(std::declval<Dune::index_constant<I>>(), std::declval<size_t>())),
                decltype(Dune::TypeTree::treePath(std::declval<Dune::index_constant<J>>(), std::declval<size_t>()))>
          index{};
    };

    inline SpatialPartial spatial(size_t i) { return {i}; }

    template <std::size_t I>
    SingleCoeff<I> coeff(Dune::index_constant<I>, size_t i) {
      using namespace Dune::Indices;
      SingleCoeff<I> coeffs;
      std::get<1>(coeffs.index._data) = i;
      return coeffs;
    }
    template <std::size_t I, std::size_t J>
    TwoCoeff<I, J> coeff(Dune::index_constant<I>, size_t i, Dune::index_constant<J>, size_t j) {
      using namespace Dune::Indices;
      TwoCoeff<I, J> coeffs;
      std::get<1>(coeffs.index.first._data)  = i;
      std::get<1>(coeffs.index.second._data) = j;
      return coeffs;
    }

    inline SingleCoeff<0> coeff(size_t i) {
      using namespace Dune::Indices;
      SingleCoeff<0> coeffs;
      std::get<1>(coeffs.index._data) = i;
      return coeffs;
    }
    inline TwoCoeff<0, 0> coeff(size_t i, size_t j) {
      using namespace Dune::Indices;
      TwoCoeff<0, 0> coeffs;
      std::get<1>(coeffs.index.first._data)  = i;
      std::get<1>(coeffs.index.second._data) = j;
      return coeffs;
    }

    template <typename WrtType>
    auto extractSpatialPartialIndices(WrtType&& wrt) {
      if constexpr (Std::hasType<SpatialPartial, typename std::remove_reference_t<WrtType>::Args>::value)
        return std::get<SpatialPartial>(wrt.args).index;  // returns single int
      else
        return std::array<int, 0>();  // signals no SpatialPartial derivative found
    }

    template <typename Type>
    concept isSpatial
        = std::is_same_v<std::remove_reference_t<Type>, DerivativeDirections::SpatialPartial> or std::is_same_v<
            std::remove_reference_t<Type>, DerivativeDirections::SpatialAll>;

    template <typename Type>
    concept isCoeff = Std::IsSpecializationNonTypes<SingleCoeff, std::remove_reference_t<Type>>::value or Std::
        IsSpecializationNonTypes<TwoCoeff, std::remove_reference_t<Type>>::value;

    struct ConstExprCounter {
      int singleCoeffDerivs{};
      int twoCoeffDerivs{};
      int spatialDerivs{};
      int spatialAllCounter{};

      [[nodiscard]] consteval int orderOfDerivative() const {
        return singleCoeffDerivs + 2 * twoCoeffDerivs + spatialDerivs + spatialAllCounter;
      }
    };

    template <typename WrtType>
    consteval ConstExprCounter countDerivativesType() {
      ConstExprCounter counter{};
      using Tuple               = typename WrtType::Args;
      counter.singleCoeffDerivs = Dune::Std::countTypeSpecialization_v<Tuple, SingleCoeff>;
      counter.twoCoeffDerivs    = Dune::Std::countTypeSpecialization_v<Tuple, TwoCoeff>;
      counter.spatialDerivs     = Dune::Std::countType<Tuple, SpatialPartial>();
      counter.spatialAllCounter = Dune::Std::countType<Tuple, SpatialAll>();
      return counter;
    }

    template <typename WrtType>
    auto extractCoeffIndices(WrtType&& wrt) {
      if constexpr (Std::hasTypeSpecialization<SingleCoeff, typename std::remove_reference_t<WrtType>::Args>())
        return Std::getSpecialization<SingleCoeff>(wrt.args).index;  // returns single int
      else if constexpr (Std::hasTypeSpecialization<TwoCoeff, typename std::remove_reference_t<WrtType>::Args>())
        return Std::getSpecialization<TwoCoeff>(wrt.args).index;  // return std::array<size_t,2>
      else
        return std::array<int, 0>();  // signals no coeff derivative found
    }

    template <typename WrtType>
    concept HasTwoCoeff = (countDerivativesType<WrtType>().twoCoeffDerivs == 1);

    template <typename WrtType>
    concept HasSingleCoeff = (countDerivativesType<WrtType>().singleCoeffDerivs == 1);

    template <typename WrtType>
    concept HasNoCoeff = (countDerivativesType<WrtType>().singleCoeffDerivs == 0
                          and countDerivativesType<WrtType>().twoCoeffDerivs == 0);

    template <typename WrtType>
    concept HasNoSpatial = (countDerivativesType<WrtType>().spatialDerivs == 0
                            and countDerivativesType<WrtType>().spatialAllCounter == 0);

    template <typename WrtType>
    concept HasOneSpatialAll = countDerivativesType<WrtType>()
    .spatialAllCounter == 1;

    template <typename WrtType>
    concept HasOneSpatialSingle = countDerivativesType<WrtType>()
    .spatialDerivs == 1;

    template <typename WrtType>
    concept HasOneSpatial = HasOneSpatialSingle<WrtType> or HasOneSpatialAll<WrtType>;

  }  // namespace DerivativeDirections

  template <typename... Args_>
  struct Wrt {
    using Args = std::tuple<std::remove_cvref_t<Args_>...>;
    Args args;
  };

  template <typename T>
  concept DerivativeDirection = DerivativeDirections::isSpatial<T> or DerivativeDirections::isCoeff<T>;

  template <DerivativeDirection... Args>
  auto wrt(Args&&... args) {
    return Wrt<Args&&...>{std::forward_as_tuple(std::forward<Args>(args)...)};
  }

  template <typename... T_>
  struct On;

  template <typename T_, typename F_>
  struct On<T_, F_> {
    using T = T_;
    using F = F_;
    F f;
  };

  template <typename T_, typename F_, typename F2_>
  struct On<T_, F_, F2_> {
    using T  = T_;
    using F  = F_;
    using F2 = F2_;
    F f;
    F2 f2;
  };

  template <>
  struct On<DerivativeDirections::ReferenceElement, void> {
    using T = DerivativeDirections::ReferenceElement;
  };

  template <>
  struct On<DerivativeDirections::ReferenceElement, void, void> {
    using T = DerivativeDirections::ReferenceElement;
  };

  inline On<DerivativeDirections::ReferenceElement, void> on(DerivativeDirections::ReferenceElement) { return {}; }

  template <typename F = Dune::DefaultFirstOrderTransformFunctor>
  inline On<DerivativeDirections::GridElement, F> on(DerivativeDirections::GridElement, F&& = {}) {
    return {};
  }

  template <typename LF>
  concept IsUnaryExpr = LF::children == 1;

  template <typename LF>
  concept IsBinaryExpr = LF::children == 2;

  using Arithmetic                = Dune::index_constant<100>;
  static constexpr int arithmetic = Arithmetic::value;

  template <typename LF>
  concept IsArithmeticExpr = std::remove_cvref_t<LF>::id[0] == arithmetic;

  template <typename E1, typename E2>
  class ScaleExpr;

  template <typename LocalFunctionImpl>
  class LocalFunctionInterface;

  template <typename LocalFunctionImpl>
  concept LocalFunction = requires {
    typename std::remove_cvref_t<LocalFunctionImpl>::Traits;
    std::remove_cvref_t<LocalFunctionImpl>::Traits::valueSize;
    typename std::remove_cvref_t<LocalFunctionImpl>::Traits::DomainType;
    std::remove_cvref_t<LocalFunctionImpl>::id;
  };

  template <typename LF>
  concept IsScaleExpr = Std::isSpecialization<ScaleExpr, std::remove_cvref_t<LF>>::value;

  template <typename LF>
  concept IsNonArithmeticLeafNode
      = std::remove_cvref_t<LF>::isLeaf == true and !IsArithmeticExpr<std::remove_cvref_t<LF>>;

  template <typename... LF>
  concept IsLocalFunction = (LocalFunction<LF> and ...);

  static constexpr int nonlinear = 1000;
  static constexpr int constant  = 0;
  static constexpr int linear    = 1;
  static constexpr int quadratic = 2;
  static constexpr int cubic     = 3;

  /**
   * \brief In the following several traits for functions are defined
   *        Here we start with the unused general template
   */
  template <typename T, typename = void>
  struct FunctionTraits;

  /**
   * \brief Specialization for general functions
   */
  template <typename R, typename... Args>
  struct FunctionTraits<R (*)(Args...)> {
    using return_type = R;
    template <int i>
    using args_type                        = typename std::tuple_element<i, std::tuple<Args...>>::type;
    static constexpr int numberOfArguments = sizeof...(Args);
  };

  /**
   * \brief Specialization for const member functions
   */
  template <typename R, typename C, typename... Args>
  struct FunctionTraits<R (C::*)(Args...) const> {
    using return_type = R;
    template <int i>
    using args_type                        = typename std::tuple_element<i, std::tuple<Args...>>::type;
    static constexpr int numberOfArguments = sizeof...(Args);
  };

  /**
   * \brief Specialization for non-const member functions
   */
  template <typename R, typename C, typename... Args>
  struct FunctionTraits<R (C::*)(Args...)> {
    using return_type = R;
    template <int i>
    using args_type                        = typename std::tuple_element<i, std::tuple<Args...>>::type;
    static constexpr int numberOfArguments = sizeof...(Args);
  };

  /**
   * \brief Specialization for lambdas using std::void_t to allow a specialization of the original template
   *        The lambda is forwarded using lambdas operator() to the general function traits
   */
  template <typename T>
  struct FunctionTraits<T, Dune::void_t<decltype(&T::operator())>> : public FunctionTraits<decltype(&T::operator())> {};

}  // namespace Dune
