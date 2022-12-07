// SPDX-FileCopyrightText: 2022 Alexander Müller mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include <ranges>
#include <set>
#include <vector>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/localfefunctions/concepts.hh>

#include <Eigen/Core>

namespace Dune {

  /* Helper function to pass integers. These indicate which derivatives should be precomputed */
  template <typename... Ints>
  requires std::conjunction_v<std::is_convertible<int, Ints>...>
  auto bindDerivatives(Ints... ints) { return std::set<int>({std::forward<Ints>(ints)...}); }

  /* Convenient wrapper to store a dune local basis. It is possible to precompute derivatives */
  template <Concepts::LocalBasis DuneLocalBasis>
  class CachedLocalBasis {
    using RangeDuneType    = typename DuneLocalBasis::Traits::RangeType;
    using JacobianDuneType = typename DuneLocalBasis::Traits::JacobianType;

  public:
    constexpr explicit CachedLocalBasis(const DuneLocalBasis& p_basis) : duneLocalBasis{&p_basis} {}
    CachedLocalBasis() = default;

    static constexpr int gridDim = DuneLocalBasis::Traits::dimDomain;
    static_assert(gridDim <= 3, "This local Basis only works for grids with dimensions<=3");
    using DomainType = typename DuneLocalBasis::Traits::DomainType;

    using DomainFieldType = typename DuneLocalBasis::Traits::DomainFieldType;
    using RangeFieldType  = typename DuneLocalBasis::Traits::RangeFieldType;

    using JacobianType         = Dune::BlockVector<Dune::FieldVector<RangeFieldType, gridDim>>;
    using SecondDerivativeType = Dune::BlockVector<Dune::FieldVector<RangeFieldType, gridDim*(gridDim + 1) / 2>>;
    using AnsatzFunctionType   = Dune::BlockVector<RangeFieldType>;

    /* Evaluates the ansatz functions into the given Eigen Vector N */
    void evaluateFunction(const DomainType& local, AnsatzFunctionType& N) const;

    /* Evaluates the ansatz functions derivatives into the given Eigen Matrix dN */
    void evaluateJacobian(const DomainType& local, JacobianType& dN) const;

    /* Evaluates the ansatz functions second derivatives into the given Eigen Matrix ddN */
    void evaluateSecondDerivatives(const DomainType& local, SecondDerivativeType& ddN) const;

    /* Evaluates the ansatz functions and derivatives into the given Eigen Vector/Matrix N,dN */
    void evaluateFunctionAndJacobian(const DomainType& local, AnsatzFunctionType& N,
                                     JacobianType& dN) const;

    /* Returns the number of ansatz functions */
    unsigned int size() const { return duneLocalBasis->size(); }

    /* Returns the polynomial order  */
    unsigned int order() const { return duneLocalBasis->order(); }

    /* Returns the number of integration points if the basis is bound */
    unsigned int integrationPointSize() const {
      if (not Nbound) throw std::logic_error("You have to bind the basis first");
      return Nbound.value().size();
    }

    /* Binds this basis to a given integration rule */
    void bind(const Dune::QuadratureRule<DomainFieldType, gridDim>& p_rule, std::set<int>&& ints);

    /* Returns a reference to the ansatz functions evaluated at the given integration point index
     * The "requires" statement is needed to circumvent implicit conversion from FieldVector<double,1>
     * */
    template <typename IndexType>
    requires std::same_as<IndexType, long unsigned> or std::same_as<IndexType, int>
    const auto& evaluateFunction(IndexType ipIndex) const {
      if (not Nbound) throw std::logic_error("You have to bind the basis first");
      return Nbound.value()[ipIndex];
    }

    /* Returns a reference to the ansatz functions derivatives evaluated at the given integration point index */
    const auto& evaluateJacobian(long unsigned i) const {
      if (not dNbound) throw std::logic_error("You have to bind the basis first");
      return dNbound.value()[i];
    }

    /* Returns a reference to the ansatz functions second derivatives evaluated at the given integration point index */
    const auto& evaluateSecondDerivatives(long unsigned i) const {
      if (not ddNbound) throw std::logic_error("You have to bind the basis first");
      return ddNbound.value()[i];
    }

    /* Returns true if the local basis is currently bound to an integration rule */
    bool isBound() const { return (dNbound and Nbound); }

    struct FunctionAndJacobian {
      long unsigned index{};
      const Dune::QuadraturePoint<DomainFieldType, gridDim>& ip{};
      const Eigen::VectorX<RangeFieldType>& N{};
      const Dune::FieldMatrix<RangeFieldType, Eigen::Dynamic, gridDim>& dN{};
    };

    /* Returns a view over the integration point index, the point itself, and the ansatz function and ansatz function
     * derivatives at the very same point */
    auto viewOverFunctionAndJacobian() const {
      assert(Nbound.value().size() == dNbound.value().size()
             && "Number of intergrationpoint evaluations does not match.");
      if (isBound())
        return std::views::iota(0UL, Nbound.value().size()) | std::views::transform([&](auto&& i_) {
                 return FunctionAndJacobian(i_, rule.value()[i_], getFunction(i_), getJacobian(i_));
               });
      else {
        assert(false && "You need to call bind first");
        __builtin_unreachable();
      }
    }

    const Dune::QuadraturePoint<DomainFieldType, gridDim>& indexToIntegrationPoint(int i) const;

    struct IntegrationPointsAndIndex {
      long unsigned index{};
      const Dune::QuadraturePoint<DomainFieldType, gridDim>& ip{};
    };

    /* Returns a view over the integration point index and the point itself */
    auto viewOverIntegrationPoints() const {  // FIXME dont construct this on the fly
      assert(Nbound && "You have to bind the basis first");
      assert(Nbound.value().size() == dNbound.value().size()
             && "Number of integration point evaluations does not match.");
      if (Nbound and dNbound) {
        auto res = std::views::iota(0UL, Nbound.value().size()) | std::views::transform([&](auto&& i_) {
                     return IntegrationPointsAndIndex({i_, rule.value()[i_]});
                   });
        return res;
      } else {
        assert(false && "You need to call bind first");
        __builtin_unreachable();
      }
    }

  private:
    mutable std::vector<JacobianDuneType> dNdune{};
    mutable std::vector<RangeDuneType> ddNdune{};
    mutable std::vector<RangeDuneType> Ndune{};
    DuneLocalBasis const* duneLocalBasis;  // FIXME pass shared_ptr around
    std::optional<std::set<int>> boundDerivatives;
    std::optional<std::vector<AnsatzFunctionType>> Nbound{};
    std::optional<std::vector<JacobianType>> dNbound{};
    std::optional<std::vector<SecondDerivativeType>> ddNbound{};
    std::optional<Dune::QuadratureRule<DomainFieldType, gridDim>> rule;
  };

}  // namespace Dune

#include "cachedlocalBasis.inl"
