// SPDX-FileCopyrightText: 2022 Alexander Müller mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <dune/common/diagonalmatrix.hh>
#include <dune/common/float_cmp.hh>
#include <dune/localfefunctions/helper.hh>

#include <Eigen/Core>

template <typename Derived, typename OtherDerived>
requires(
    not Dune::Std::IsSpecializationTypeAndNonTypes<Dune::FieldMatrix, Derived>::value
    and not Dune::Std::IsSpecializationTypeAndNonTypes<Dune::DiagonalMatrix, Derived>::value
    and not Dune::Std::IsSpecializationTypeAndNonTypes<Dune::ScaledIdentityMatrix, Derived>::value
    and not Dune::Std::IsSpecializationTypeAndNonTypes<Dune::FieldVector, Derived>::value
    and std::convertible_to<
        Derived,
        Eigen::EigenBase<
            Derived> const&> and std::convertible_to<OtherDerived, Eigen::EigenBase<OtherDerived> const&>) bool isApproxSame(Derived const&
                                                                                                                                 val,
                                                                                                                             OtherDerived const&
                                                                                                                                 other,
                                                                                                                             double
                                                                                                                                 prec) {
  if constexpr (requires {
                  val.isApprox(other, prec);
                  (val - other).isMuchSmallerThan(1, prec);
                })
    return val.isApprox(other, prec) or (val - other).isZero(prec);
  else if constexpr (requires { val.isApprox(other, prec); })
    return val.isApprox(other, prec);
  else  // Dune::DiagonalMatrix branch
    return val.diagonal().isApprox(other.diagonal(), prec) or (val.diagonal() - other.diagonal()).isZero(prec);
}

template <typename field_type, int rows, int cols>
bool isApproxSame(const Dune::FieldMatrix<field_type, rows, cols>& a,
                  const Dune::FieldMatrix<field_type, rows, cols>& b, double prec) {
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      if (not Dune::FloatCmp::eq(a[i][j], b[i][j], prec))
        if (std::abs(a[i][j] - b[i][j]) > prec) return false;

  return true;
}

template <typename field_type, int rows>
bool isApproxSame(const Dune::FieldVector<field_type, rows>& a, const Dune::FieldMatrix<field_type, rows, 1>& b,
                  double prec) {
  for (int i = 0; i < rows; ++i)
    if (not Dune::FloatCmp::eq(a[i], b[i][0], prec))
      if (std::abs(a[i] - b[i][0]) > prec) return false;

  return true;
}

template <typename field_type, int rows>
bool isApproxSame(const Dune::FieldMatrix<field_type, rows, 1>& a, const Dune::FieldVector<field_type, rows>& b,
                  double prec) {
  for (int i = 0; i < rows; ++i)
    if (not Dune::FloatCmp::eq(a[i][0], b[i], prec))
      if (std::abs(a[i][0] - b[i]) > prec) return false;

  return true;
}

template <typename field_type, int rows>
bool isApproxSame(const Dune::DiagonalMatrix<field_type, rows>& a, const Dune::DiagonalMatrix<field_type, rows>& b,
                  double prec) {
  for (int i = 0; i < rows; ++i)
    if (not Dune::FloatCmp::eq(a.diagonal()[i], b.diagonal()[i], prec))
      if (std::abs(a.diagonal()[i] - b.diagonal()[i]) > prec) return false;

  return true;
}
template <typename field_type, int rows>
bool isApproxSame(const Dune::ScaledIdentityMatrix<field_type, rows>& a,
                  const Dune::ScaledIdentityMatrix<field_type, rows>& b, double prec) {
  if (not Dune::FloatCmp::eq(a.scalar(), b.scalar(), prec))
    if (std::abs(a.scalar() - b.scalar()) > prec) return false;

  return true;
}

template <typename field_type, int rows>
bool isApproxSame(const Dune::ScaledIdentityMatrix<field_type, rows>& a,
                  const Dune::FieldMatrix<field_type, rows, rows>& b, double prec) {
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < rows; ++j) {
      if (i == j) {
        if (not Dune::FloatCmp::eq(a.diagonal(i), b[i][i], prec))
          if (std::abs(a.diagonal(i) - b[i][i]) > prec) return false;  // Check diagonal of FieldMatrix
      } else if (not Dune::FloatCmp::eq(b[i][j], 0.0, prec))
        return false;  // Check off-diagonal of FieldMatrix
    }
  return true;
}
