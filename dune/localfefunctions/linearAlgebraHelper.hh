

/*
 * This file is part of the Ikarus distribution (https://github.com/IkarusRepo/Ikarus).
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
#if ENABLE_TESTING
#  include <autodiff/forward/dual/dual.hpp>
#endif
#include <iosfwd>
#include <random>

#include <dune/common/diagonalmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/transpose.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/multitypeblockvector.hh>
#include <dune/istl/scaledidmatrix.hh>
#include <dune/localfefunctions/concepts.hh>
#include <dune/localfefunctions/helper.hh>
#include <dune/localfefunctions/manifolds/manifoldInterface.hh>
#include <dune/localfefunctions/meta.hh>
#include <dune/matrix-vector/transpose.hh>

#include <Eigen/Core>

namespace Dune {

#if ENABLE_TESTING
  /** \brief  eval overload for autodiff scalars */
  template <typename T>
  requires autodiff::detail::isDual<T> || autodiff::detail::isExpr<T> || autodiff::detail::isArithmetic<T>
  auto eval(T&& t) { return autodiff::detail::eval(t); }

  template <>
  struct IsNumber<autodiff::HigherOrderDual<2, double>> : public std::true_type {};

  /** \brief  Multiply with scalar and autodiff types */
  template <typename T, int rows, int cols>
  requires autodiff::detail::isDual<T> || autodiff::detail::isExpr<T> || autodiff::detail::isArithmetic<T>
  auto operator*(T&& t, const Dune::FieldMatrix<decltype(autodiff::detail::eval(t)), rows, cols>& A) {
    return autodiff::detail::eval(t) * A;
  }

  /** \brief  Multiply with scalar and autodiff types */
  template <typename T, int rows, int cols>
  requires autodiff::detail::isDual<T> || autodiff::detail::isExpr<T> || autodiff::detail::isArithmetic<T>
  auto operator*(const Dune::FieldMatrix<decltype(autodiff::detail::eval(std::declval<T>())), rows, cols>& A, T&& t) {
    return autodiff::detail::eval(t) * A;
  }

  /** \brief  Transpose for scalars and autodiff types */
  template <typename T>
  requires autodiff::detail::isDual<T> || autodiff::detail::isExpr<T> || autodiff::detail::isArithmetic<T>
  auto transpose(T&& t) { return t; }
  template <typename T1, typename T2>
  requires(
      autodiff::detail::isDual<
          T1> || autodiff::detail::isExpr<T1> || autodiff::detail::isArithmetic<T1> || autodiff::detail::isDual<T2> || autodiff::detail::isExpr<T2> || autodiff::detail::isArithmetic<T2>) struct
      PromotionTraits<T1, T2> {
    using PromotedType = decltype(autodiff::eval(std::declval<T1>() + std::declval<T2>()));
  };
#endif

  template <typename T>
  auto transposeEvaluated(const T& A) {
    if constexpr (Std::IsSpecializationTypeAndNonTypes<Dune::FieldVector, T>::value) {
      Dune::FieldMatrix<typename T::value_type, 1, T::dimension> aT;
      for (int i = 0; i < T::dimension; ++i)
        aT[0][i] = A[i];

      return aT;
    } else {
      typename Dune::MatrixVector::TransposeHelper<T>::TransposedType AT;
      Dune::MatrixVector::transpose(A, AT);
      return AT;
    }
  }

  /** \brief Get the requested column of fieldmatrix */
  template <typename field_type, int rows, int cols>
  auto col(const Dune::FieldMatrix<field_type, rows, cols>& mat, const int requestedCol) {
    Dune::FieldVector<field_type, rows> col;

    for (int i = 0; i < rows; ++i)
      col[i] = mat[i][requestedCol];

    return col;
  }

  /** \brief Computes the inner product (Frobenius) of two matrices  */
  template <typename field_type, typename field_type2, int rows, int cols>
  auto inner(const Dune::FieldMatrix<field_type, rows, cols>& a, const Dune::FieldMatrix<field_type2, rows, cols>& b) {
    using ScalarResultType = typename Dune::PromotionTraits<field_type, field_type2>::PromotedType;
    ScalarResultType res{0};

    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        res += a[i][j] * b[i][j];

    return res;
  }

  /** \brief Computes the inner product (Frobenius) of two vector, no complex conjugate here, used .dot instead!  */
  template <typename field_type, typename field_type2, int rows>
  auto inner(const Dune::FieldVector<field_type, rows>& a, const Dune::FieldMatrix<field_type2, rows, 1>& b) {
    using ScalarResultType = typename Dune::PromotionTraits<field_type, field_type2>::PromotedType;

    ScalarResultType res{0};
    for (int i = 0; i < rows; ++i)
      res += a[i] * b[i][0];
    return res;
  }

  /** \brief Computes the matrix vector product  */
  template <typename field_type, typename field_type2, int rows, int cols>
  auto operator*(const Dune::FieldMatrix<field_type, rows, cols>& b, const Dune::FieldVector<field_type2, cols>& a) {
    using ScalarResultType = typename Dune::PromotionTraits<field_type, field_type2>::PromotedType;

    Dune::FieldVector<ScalarResultType, rows> y;
    b.mv(a, y);
    return y;
  }

  /** \brief Computes the inner product (Frobenius) of two vector, no complex conjugate here, used .dot instead!  */
  template <typename field_type, typename field_type2, int rows>
  auto inner(const Dune::FieldMatrix<field_type, rows, 1>& b, const Dune::FieldVector<field_type2, rows>& a) {
    return inner(a, b);
  }

  /** \brief Computes the inner product (Frobenius) of two vector, no complex conjugate here, used .dot instead!  */
  template <typename field_type, typename field_type2, int rows>
  auto inner(const Dune::FieldVector<field_type, rows>& a, const Dune::FieldVector<field_type2, rows>& b) {
    field_type res{0};
    for (int i = 0; i < rows; ++i)
      res += a[i] * b[i];
    return res;
  }

  /** \brief Computes norm squared (Frobenius) of the matrix  */
  template <typename field_type, int rows, int cols>
  auto two_norm2(const Dune::FieldMatrix<field_type, rows, cols>& a) {
    return a.frobenius_norm2();
  }

  /** \brief Outer product between two vector */
  template <typename field_type, int rows1, int rows2>
  auto outer(const Dune::FieldVector<field_type, rows1>& a, const Dune::FieldVector<field_type, rows2>& b) {
    Dune::FieldMatrix<field_type, rows1, rows2> res;

    for (int i = 0; i < rows1; ++i)
      for (int j = 0; j < rows2; ++j)
        res[i][j] = a[i] * b[j];

    return res;
  }

  /** \brief Outer product between two vector */
  template <typename field_type, int rows1, int rows2>
  auto outer(const Dune::FieldMatrix<field_type, rows1, 1>& a, const Dune::FieldMatrix<field_type, rows2, 1>& b) {
    Dune::FieldMatrix<field_type, rows1, rows2> res;

    for (int i = 0; i < rows1; ++i)
      for (int j = 0; j < rows2; ++j)
        res[i][j] = a[i][0] * b[j][0];

    return res;
  }

  /** \brief Outer product between two vector */
  template <typename field_type, int rows1, int rows2>
  auto outer(const Dune::FieldVector<field_type, rows1>& a, const Dune::FieldMatrix<field_type, rows2, 1>& b) {
    Dune::FieldMatrix<field_type, rows1, rows2> res;

    for (int i = 0; i < rows1; ++i)
      for (int j = 0; j < rows2; ++j)
        res[i][j] = a[i] * b[j][0];

    return res;
  }

  /** \brief Outer product between two vector */
  template <typename field_type, int rows1, int rows2>
  auto outer(const Dune::FieldMatrix<field_type, rows1, 1>& a, const Dune::FieldVector<field_type, rows2>& b) {
    Dune::FieldMatrix<field_type, rows1, rows2> res;

    for (int i = 0; i < rows1; ++i)
      for (int j = 0; j < rows2; ++j)
        res[i][j] = a[i][0] * b[j];

    return res;
  }

  template <typename field_type, typename field_type2, int rows1>
  requires(std::is_arithmetic_v<field_type>and std::is_arithmetic_v<field_type2>) auto operator*(
      field_type a, const Dune::ScaledIdentityMatrix<field_type2, rows1>& b) {
    using ScalarResultType = typename Dune::PromotionTraits<field_type, field_type2>::PromotedType;
    Dune::ScaledIdentityMatrix<ScalarResultType, rows1> c = b;
    c.scalar() *= a;
    return c;
  }

  template <typename field_type, typename field_type2, int rows1>
  requires(std::is_arithmetic_v<field_type>and std::is_arithmetic_v<field_type2>) auto operator*(
      const Dune::ScaledIdentityMatrix<field_type2, rows1>& b, field_type a) {
    return a * b;
  }

  template <typename field_type, typename field_type2, int rows1>
  requires std::is_arithmetic_v<field_type>
  auto operator*(field_type a, const Dune::DiagonalMatrix<field_type2, rows1>& b) {
    using ScalarResultType = typename Dune::PromotionTraits<field_type, field_type2>::PromotedType;

    Dune::DiagonalMatrix<ScalarResultType, rows1> c = b;
    c *= a;
    return c;
  }

  template <typename field_type, int rows1>
  auto operator*(const Dune::DiagonalMatrix<field_type, rows1>& b, field_type a) {
    return a * b;
  }

  /** \brief  This multiplies a vector from left to a matrix
   *  y = x^T A
   * */
  template <typename field_type, int rows, int cols>
  auto leftMultiplyTranspose(const Dune::FieldVector<field_type, rows>& x,
                             const Dune::FieldMatrix<field_type, rows, cols>& A) {
    Dune::FieldMatrix<field_type, cols, 1> y;
    Dune::FieldMatrix<field_type, 1, cols> yT;
    A.mtv(x, y);  // y=A^T*x
    Dune::MatrixVector::transpose(y, yT);
    return yT;
  }

  template <typename field_type, int size>
  auto leftMultiplyTranspose(const Dune::FieldVector<field_type, size>& x,
                             const Dune::ScaledIdentityMatrix<field_type, size>& A) {
    Dune::FieldMatrix<field_type, 1, size> yT;
    for (int i = 0; i < size; ++i)
      yT[0][i] = x[i] * A.scalar();

    return yT;
  }

  template <typename field_type, int size>
  auto leftMultiplyTranspose(const Dune::ScaledIdentityMatrix<field_type, size>& B,
                             const Dune::ScaledIdentityMatrix<field_type, size>& A) {
    auto y = B;
    y.scalar() *= A.scalar();

    return y;
  }

  template <typename field_type, int size>
  auto leftMultiplyTranspose(const Dune::FieldVector<field_type, size>& x,
                             const Dune::DiagonalMatrix<field_type, size>& A) {
    Dune::FieldMatrix<field_type, size, 1> y;
    Dune::FieldMatrix<field_type, 1, size> yT;

    A.mtv(x, y);
    Dune::MatrixVector::transpose(y, yT);
    return yT;
  }

  template <typename field_type, int size>
  auto leftMultiplyTranspose(const Dune::FieldVector<field_type, size>& x,
                             const Dune::FieldVector<field_type, size>& a) {
    Dune::FieldMatrix<field_type, 1, 1> y(x * a);

    return y;
  }

  template <typename field_type, int size>
  auto leftMultiplyTranspose(const Dune::FieldVector<field_type, size>&, Dune::DerivativeDirections::ZeroMatrix) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename field_type, int rows, int cols>
  auto leftMultiplyTranspose(const Dune::FieldMatrix<field_type, rows, cols>& A,
                             const Dune::FieldVector<field_type, rows>& x) {
    Dune::FieldMatrix<field_type, cols, 1> y;
    A.mtv(x, y);  // y=A x
    return y;
  }

  template <typename field_type, int rows, int cols1, int cols2>
  auto leftMultiplyTranspose(const Dune::FieldMatrix<field_type, rows, cols1>& B,
                             const Dune::FieldMatrix<field_type, rows, cols2>& A) {
    Dune::FieldMatrix<field_type, cols1, rows> BT;
    Dune::MatrixVector::transpose(B, BT);

    return BT * A;
  }

  template <typename field_type, int rows, int cols1>
  auto leftMultiplyTranspose(const Dune::FieldMatrix<field_type, rows, cols1>& B,
                             const Dune::DiagonalMatrix<field_type, rows>& A) {
    Dune::FieldMatrix<field_type, cols1, rows> BT;
    Dune::MatrixVector::transpose(B, BT);

    return BT * Dune::FieldMatrix<field_type, rows, rows>(A);
  }

  template <typename field_type, int rows>
  auto operator+(const Dune::FieldMatrix<field_type, rows, rows>& B, const Dune::DiagonalMatrix<field_type, rows>& A) {
    auto y = B;

    for (int i = 0; i < rows; ++i)
      y[i][i] += A.diagonal()[i];

    return y;
  }

  template <typename field_type, int rows>
  auto operator+(const Dune::FieldMatrix<field_type, rows, rows>& B,
                 const Dune::ScaledIdentityMatrix<field_type, rows>& A) {
    auto y = B;

    for (int i = 0; i < rows; ++i)
      y[i][i] += A.scalar();

    return y;
  }

  template <typename field_type, int rows>
  auto operator-(const Dune::FieldMatrix<field_type, rows, rows>& B,
                 const Dune::ScaledIdentityMatrix<field_type, rows>& A) {
    auto y = B;

    for (int i = 0; i < rows; ++i)
      y[i][i] -= A.scalar();

    return y;
  }

  template <typename field_type, int rows>
  auto operator+(const Dune::DiagonalMatrix<field_type, rows>& B, const Dune::DiagonalMatrix<field_type, rows>& A) {
    auto y = B;

    for (int i = 0; i < rows; ++i)
      y.diagonal()[i] += A.diagonal()[i];

    return y;
  }

  template <typename field_type, int rows>
  auto operator+(const Dune::ScaledIdentityMatrix<field_type, rows>& B,
                 const Dune::ScaledIdentityMatrix<field_type, rows>& A) {
    auto y = B;
    y.scalar() += A.scalar();

    return y;
  }

  template <typename field_type, int rows>
  Dune::DiagonalMatrix<field_type, rows> leftMultiplyTranspose(const Dune::DiagonalMatrix<field_type, rows>& B,
                                                               const Dune::DiagonalMatrix<field_type, rows>& A) {
    Dune::DiagonalMatrix<field_type, rows> C = B;
    for (int i = 0; i < rows; ++i) {
      C[i][i] *= A[i][i];
    }
    return C;
  }

  /** \brief Generates FieldVector with random entries in the range -1..1 */
  template <typename FieldVectorT>
  auto createRandomVector(typename FieldVectorT::value_type lower = -1, typename FieldVectorT::value_type upper = 1) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<typename FieldVectorT::value_type> dist(lower, upper);
    auto rand = [&dist, &mt]() { return dist(mt); };
    FieldVectorT vec;
    std::generate(vec.begin(), vec.end(), rand);
    return vec;
  }

  /** \brief Return a segment of a FieldVector from lower up to lower+size-1 */
  template <int lower, int size, typename field_type, int n>
  static FieldVector<field_type, size> segment(const FieldVector<field_type, n>& v) {
    FieldVector<field_type, size> res;
    std::copy(v.begin() + lower, v.begin() + lower + size, res.begin());
    return res;
  }

  /** \brief Return a static sized segment of a Eigen::Vector as FieldVector from lower up to lower+size-1 */
  template <int size, typename field_type>
  static FieldVector<field_type, size> segmentToDune(const Eigen::VectorX<field_type>& v, int lower) {
    FieldVector<field_type, size> res;
    std::copy(v.begin() + lower, v.begin() + lower + size, res.begin());
    return res;
  }

  /** \brief Return a static sized block of a EigenMatrix  (lower1...lower1+size1-1,lower2...lower2+size2-1) as
   * FieldMatrix */
  template <int size1, int size2, typename field_type>
  static auto blockToDune(const Eigen::MatrixX<field_type>& v, int lower1, int lower2) {
    assert(lower1 + size1 <= v.rows() && lower2 + size2 <= v.cols() && "Size mismatch for Block!");
    FieldMatrix<field_type, size1, size2> res;

    for (int i = lower1; i < lower1 + size1; ++i)
      for (int j = lower2; j < lower2 + size2; ++j)
        res[i - lower1][j - lower2] = v(i, j);
    return res;
  }

  /** \brief sets a matrix to zero */
  template <typename field_type, int rows, int cols>
  void setZero(Dune::FieldMatrix<field_type, rows, cols>& a) {
    a = 0;
  }

  /** \brief sets a vector to zero */
  template <typename field_type, int rows>
  void setZero(Dune::FieldVector<field_type, rows>& a) {
    a = 0;
  }

  /** \brief sets a matrix to zero with given templates*/
  template <typename field_type, int rows, int cols>
  auto createZeroMatrix() {
    return Dune::FieldMatrix<field_type, rows, cols>(0);
  }

  /** \brief sets a matrix to zero with given templates*/
  template <typename field_type, int rows>
  auto createZeroVector() {
    return Dune::FieldVector<field_type, rows>(0);
  }

  template <typename field_type, int rows, int cols>
  auto leftMultiplyTranspose(const Dune::FieldMatrix<field_type, rows, cols>&,
                             const Dune::DerivativeDirections::ZeroMatrix&) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename field_type, int rows, int cols>
  auto leftMultiplyTranspose(const Dune::DerivativeDirections::ZeroMatrix&,
                             const Dune::FieldMatrix<field_type, rows, cols>&) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename field_type, int rows>
  auto leftMultiplyTranspose(const Dune::DerivativeDirections::ZeroMatrix&,
                             const Dune::DiagonalMatrix<field_type, rows>&) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename field_type, int rows>
  auto leftMultiplyTranspose(const Dune::DerivativeDirections::ZeroMatrix&,
                             const Dune::ScaledIdentityMatrix<field_type, rows>&) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename field_type, int rows>
  auto leftMultiplyTranspose(const Dune::DiagonalMatrix<field_type, rows>&,
                             const Dune::DerivativeDirections::ZeroMatrix& A) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename field_type, int rows>
  auto leftMultiplyTranspose(const Dune::ScaledIdentityMatrix<field_type, rows>&,
                             const Dune::DerivativeDirections::ZeroMatrix& A) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename field_type, int rows, int cols>
  auto createOnesMatrix() {
    return Dune::FieldMatrix<field_type, rows, cols>(1);
  }

  template <typename field_type, int rows>
  auto createOnesVector() {
    return Dune::FieldVector<field_type, rows>(1);
  }

  /** \brief Creates an identity matrix with given FieldMatrix */
  template <typename FieldMatrixT>
  auto createZeroMatrix() {
    return Dune::FieldMatrix<typename FieldMatrixT::value_type, FieldMatrixT::rows, FieldMatrixT::cols>(0);
  }

  /** \brief Creates an identity matrix with given templates */
  template <int rows, int cols, typename field_type>
  auto createScaledIdentityMatrix(field_type val) {
    return Dune::ScaledIdentityMatrix<field_type, rows>(val);
  }

  /** \brief Creates an identity matrix with given FieldMatrix */
  template <typename FieldMatrixT>
  auto createScaledIdentityMatrix(typename FieldMatrixT::value_type val = typename FieldMatrixT::value_type{1}) {
    return Dune::ScaledIdentityMatrix<typename FieldMatrixT::value_type, FieldMatrixT::rows>(val);
  }

  /** \brief Returns a reference to a row. This is easily done since Dune  simply returns a reference to the stacked
   * FieldVector But again to stay compatible with a possible Eigen replacement we can not use operator[] for both
   * cases*/
  template <typename field_type, int rows, int cols>
  auto& row(Dune::FieldMatrix<field_type, rows, cols>& a, int row) {
    return a[row];
  }

  /** \brief Dummy eval function to support eigen*/
  template <typename field_type, int rows, int cols>
  auto& eval(const Dune::FieldMatrix<field_type, rows, cols>& a) {
    return a;
  }

  /** \brief Dummy eval function to support eigen*/
  template <typename field_type, int rows>
  auto& eval(const Dune::FieldVector<field_type, rows>& a) {
    return a;
  }

  /** \brief Dummy eval function to support eigen*/
  template <typename field_type, int rows>
  auto& eval(const Dune::DiagonalMatrix<field_type, rows>& a) {
    return a;
  }

  /** \brief Dummy eval function to support eigen*/
  template <typename field_type, int rows>
  auto& eval(const Dune::ScaledIdentityMatrix<field_type, rows>& a) {
    return a;
  }

  /** \brief Access coeffs of Dune::FieldMatrix to allow replacement by Eigen Matrices
   * Dune::FieldMatrix does not have a operator() like a(i,j) like Eigen. Since i can't implement these since they need
   * to be member functions
   * Dune FieldMatrix provides [i][j] for element access but this is not provided by Eigen::Matrix. Thus we fallback to
   * have this implemented by an indirection using a free function ...*/
  template <typename field_type, int rows, int cols>
  auto& coeff(Dune::FieldMatrix<field_type, rows, cols>& a, int row, int col) {
    return a[row][col];
  }

  template <typename field_type, int rows, int cols>
  auto& coeff(const Dune::FieldMatrix<field_type, rows, cols>& a, int row, int col) {
    return a[row][col];
  }

  template <typename field_type, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic>
  auto& coeff(Eigen::Matrix<field_type, rows, cols>& a, int row, int col) {
    return a(row, col);
  }

  template <typename field_type, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic>
  auto& coeff(const Eigen::Matrix<field_type, rows, cols>& a, int row, int col) {
    return a(row, col);
  }

  template <typename RangeFieldType, int size>
  auto& coeff(Dune::BlockVector<Dune::FieldVector<RangeFieldType, size>>& a, int row, int col) {
    return a[row][col];
  }

  template <typename RangeFieldType, int size>
  auto& coeff(const Dune::BlockVector<Dune::FieldVector<RangeFieldType, size>>& a, int row, int col) {
    return a[row][col];
  }

  template <typename field_type, int rows>
  void setIdentity(Dune::DiagonalMatrix<field_type, rows>& a) {
    a = Dune::DiagonalMatrix<field_type, rows>(1);
  }

  /** \brief Computes the two_norm as free function */
  template <typename field_type, int rows>
  auto two_norm(const Dune::FieldVector<field_type, rows>& a) {
    return a.two_norm();
  }

  /** \brief Computes the two_norm as free function */
  template <typename field_type, int rows, int cols>
  auto two_norm(const Dune::FieldMatrix<field_type, rows, cols>& a) {
    return a.frobenius_norm();
  }

  /** \brief Computes the two_norm2 as free function */
  template <typename field_type, int rows>
  auto two_norm2(const Dune::FieldVector<field_type, rows>& a) {
    return a.two_norm2();
  }

  /** \brief Orthonormalizes all Matrix columns */
  template <typename Derived>
  auto orthonormalizeMatrixColumns(const Eigen::MatrixBase<Derived>& A) {
    // Gram Schmidt Ortho
    auto Q = A.eval();

    Q.col(0).normalize();

    for (int colIndex = 1; colIndex < Q.cols(); colIndex++) {
      Q.col(colIndex) -= Q.leftCols(colIndex) * (Q.leftCols(colIndex).transpose() * A.col(colIndex));
      Q.col(colIndex).normalize();
    }

    return Q;
  }

  /** \brief View Dune::BlockVector as a Eigen::Vector */
  template <typename ValueType>
  auto viewAsFlatEigenVector(Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<Eigen::VectorX<typename ValueType::field_type>> vec(&blockedVector.begin()->begin().operator*(),
                                                                   blockedVector.size() * blockedVector[0].size());

    return vec;
  }

  /** \brief View Dune::BlockVector as a Eigen::Matrix with dynamic rows and fixed columns depending on the size of  the
   * ValueType*/
  template <typename ValueType>
  auto viewAsEigenMatrixAsDynFixed(Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<Eigen::Matrix<typename ValueType::field_type, Eigen::Dynamic, ValueType::valueSize, Eigen::RowMajor>>
        vec(&blockedVector.begin()->begin().operator*(), blockedVector.size(), blockedVector[0].size());

    return vec;
  }

  /** \brief Const view Dune::BlockVector as a Eigen::Matrix with dynamic rows and fixed columns depending on the size
   * of  the ValueType */
  template <typename ValueType>
  auto viewAsEigenMatrixAsDynFixed(const Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<
        const Eigen::Matrix<typename ValueType::field_type, Eigen::Dynamic, ValueType::valueSize, Eigen::RowMajor>>
        vec(&blockedVector.begin()->begin().operator*(), blockedVector.size(), blockedVector[0].size());

    return vec;
  }

  /** \brief View Dune::BlockVector as a Eigen::Matrix with fixed rows depending on the size of  the ValueType and
   * dynamics columns */
  template <typename ValueType>
  auto viewAsEigenMatrixFixedDyn(Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<Eigen::Matrix<typename ValueType::field_type, ValueType::valueSize, Eigen::Dynamic>> vec(
        &blockedVector.begin()->begin().operator*(), blockedVector[0].size(), blockedVector.size());

    return vec;
  }

  /** \brief View Dune::BlockVector as a Eigen::Matrix with fixed rows depending on the size of  the ValueType and
   * dynamics columns */
  template <typename ScalarType>
  auto viewAsEigenMatrixFixedDyn(Dune::BlockVector<Dune::FieldVector<ScalarType, 1>>& blockedVector) {
    Eigen::Map<Eigen::Matrix<typename Dune::FieldVector<ScalarType, 1>::field_type,
                             Dune::FieldVector<ScalarType, 1>::valueSize, Eigen::Dynamic>>
        vec((&blockedVector.begin()->begin().operator*())[0], blockedVector[0].size(), blockedVector.size());

    return vec;
  }

  /** \brief Const view Dune::BlockVector as a Eigen::Matrix with fixed rows depending on the size of  the ValueType and
   * dynamics columns */
  template <typename ValueType>
  auto viewAsEigenMatrixFixedDyn(const Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<const Eigen::Matrix<typename ValueType::field_type, ValueType::valueSize, Eigen::Dynamic>> vec(
        &blockedVector.begin()->begin().operator*(), blockedVector[0].size(), blockedVector.size());

    return vec;
  }

  /** \brief View Dune::BlockVector as a Eigen::Matrix with fixed rows depending on the size of  the ValueType and
   * dynamics columns */
  template <typename ScalarType>
  auto viewAsEigenMatrixFixedDyn(const Dune::BlockVector<Dune::FieldVector<ScalarType, 1>>& blockedVector) {
    Eigen::Map<const Eigen::Matrix<typename Dune::FieldVector<ScalarType, 1>::field_type,
                                   Dune::FieldVector<ScalarType, 1>::valueSize, Eigen::Dynamic>>
        vec((&blockedVector.begin()->begin().operator*())[0], blockedVector[0].size(), blockedVector.size());

    return vec;
  }

  /** \brief View Dune::BlockVector as a Eigen::Vector */
  template <typename ValueType>
  auto viewAsFlatEigenVector(const Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<const Eigen::VectorX<typename ValueType::field_type>> vec(
        &blockedVector.begin()->begin().operator*(), blockedVector.size() * blockedVector[0].size());

    return vec;
  }

  /* Returns the total correction size of a block vector with a Manifold as type */
  template <typename Type>
  size_t correctionSize(const Dune::BlockVector<Type>& a) requires requires {
    Type::correctionSize;
  }
  { return a.size() * Type::correctionSize; }

  /* Returns the total value size of a block vector with a Manifold as type */
  template <typename Type>
  size_t valueSize(const Dune::BlockVector<Type>& a) requires requires {
    Type::valueSize;
  }
  { return a.size() * Type::valueSize; }

  /* Enables the += operator for Dune::BlockVector += Eigen::Vector */
  template <typename Type, typename Derived>
  Dune::BlockVector<Type>& operator+=(Dune::BlockVector<Type>& a, const Eigen::MatrixBase<Derived>& b) requires(
      Dune::Concepts::AddAssignAble<Type, decltype(segmentToDune<Type::correctionSize>(b, 0))>and requires() {
        Type::correctionSize;
      }) {
    assert(correctionSize(a) == static_cast<size_t>(b.size()) && " The passed vector has wrong size");
    for (auto i = 0U; i < a.size(); ++i)
      a[i] += segmentToDune<Type::correctionSize>(b, i * Type::correctionSize);
    return a;
  }

  /* Enables the -= operator for Dune::BlockVector += Eigen::Vector */
  template <typename Type, typename Derived>
  Dune::BlockVector<Type>& operator-=(Dune::BlockVector<Type>& a, const Eigen::MatrixBase<Derived>& b) requires(
      Dune::Concepts::AddAssignAble<Type, decltype(b.template segment<Type::correctionSize>(0))>and requires() {
        Type::correctionSize;
      }) {
    return a += (-b);
  }

  /* Enables the += operator for Dune::MultiTypeBlockVector += Eigen::Vector */
  template <typename... Types, typename Derived>
  Dune::MultiTypeBlockVector<Types...>& operator+=(Dune::MultiTypeBlockVector<Types...>& a,
                                                   const Eigen::MatrixBase<Derived>& b) {
    using namespace Dune::Indices;
    size_t posStart = 0;
    Dune::Hybrid::forEach(Dune::Hybrid::integralRange(Dune::index_constant<a.size()>()), [&](const auto i) {
      const size_t size = correctionSize(a[i]);
      a[i] += b(Eigen::seqN(posStart, size));
      posStart += size;
    });

    return a;
  }

  /* Enables the += operator for Dune::BlockVector += Eigen::Vector */
  template <typename Type, typename Derived>
  Dune::BlockVector<Type>& addInEmbedding(Dune::BlockVector<Type>& a, const Eigen::MatrixBase<Derived>& b)

  {
    auto& bE = b.derived().eval();  // useless copy
    assert(valueSize(a) == static_cast<size_t>(bE.size()) && " The passed vector has wrong size");
    for (auto i = 0U; i < a.size(); ++i)
      a[i].addInEmbedding(segmentToDune<Type::valueSize>(bE, i * Type::valueSize));
    return a;
  }

  /** \brief Adding free norm function to Eigen types */
  template <typename Derived>
  requires(!std::floating_point<Derived>) auto norm(const Eigen::MatrixBase<Derived>& v) { return v.norm(); }

  /** \brief Helper Free Function to have the same interface as for Eigen Vector Types */
  auto norm(const std::floating_point auto& v) { return std::abs(v); }

  /** \brief Eigen::DiagonalMatrix Product Missing in Eigen*/
  template <typename Scalar, int size>
  auto operator*(const Eigen::DiagonalMatrix<Scalar, size>& a, const Eigen::DiagonalMatrix<Scalar, size>& b) {
    return (a.diagonal().cwiseProduct(b.diagonal())).asDiagonal();
  }

  template <typename Scalar, int size>
  auto operator+=(Dune::DiagonalMatrix<Scalar, size>& a, const Dune::DiagonalMatrix<Scalar, size>& b) {
    a.diagonal() += b.diagonal();
    return a;
  }

  template <typename Scalar, int size>
  auto operator+=(Dune::FieldMatrix<Scalar, size>& a, const Dune::DiagonalMatrix<Scalar, size>& b) {
    for (int i = 0; i < size; ++i) {
      a[i][i] += b.diagonal()[i];
    }
    return a;
  }

  template <typename Scalar, int size>
  auto operator+=(Dune::FieldMatrix<Scalar, size>& a, const Dune::ScaledIdentityMatrix<Scalar, size>& b) {
    for (int i = 0; i < size; ++i) {
      a[i][i] += b.scalar();
    }
    return a;
  }

  template <typename Scalar, int size>
  Dune::FieldMatrix<Scalar, size>& operator-=(Dune::FieldMatrix<Scalar, size>& a,
                                              const Dune::ScaledIdentityMatrix<Scalar, size>& b) {
    for (int i = 0; i < size; ++i)
      a[i][i] -= b.scalar();

    return a;
  }

  template <typename Scalar, int size>
  auto operator*(const Dune::FieldMatrix<Scalar, size, size>& a, const Dune::ScaledIdentityMatrix<Scalar, size>& b) {
    Dune::FieldMatrix<Scalar, size, size> c = a;

    for (int i = 0; i < size; ++i) {
      c[i][i] += b.scalar();
    }
    return a;
  }

  template <typename Scalar, int rows, int cols>
  auto operator*(const Dune::FieldVector<Scalar, rows>& x, const Dune::FieldMatrix<Scalar, 1, cols>& A) {
    Dune::FieldMatrix<Scalar, rows, cols> c;

    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        c[i][j] += x[i] * A[0][j];

    return c;
  }

  template <typename Scalar, int size>
  auto operator*(const Dune::ScaledIdentityMatrix<Scalar, size>& b, const Dune::FieldVector<Scalar, size>& a) {
    return b.scalar() * a;
  }

  /** \brief Eigen::Matrix + Eigen::DiagonalMatrix addition missing in Eigen*/
  template <typename Derived, typename Scalar, int size>
  auto operator+(const Eigen::MatrixBase<Derived>& a, const Eigen::DiagonalMatrix<Scalar, size>& b) {
    auto c = a.derived().eval();
    c.diagonal() += b.diagonal();
    return c;
  }

  /** \brief Eigen::DiagonalMatrix + Eigen::Matrix addition missing in Eigen*/
  template <typename Derived, typename Scalar, int size>
  auto operator+(const Eigen::DiagonalMatrix<Scalar, size>& a, const Eigen::MatrixBase<Derived>& b) {
    return b + a;
  }

  template <typename Scalar, int size>
  Dune::DiagonalMatrix<Scalar, size> operator-(const Eigen::DiagonalMatrix<Scalar, size>& a) {
    return (-a.diagonal()).asDiagonal();
  }

  template <typename Scalar, int size>
  Dune::DiagonalMatrix<Scalar, size> operator-(const Dune::DiagonalMatrix<Scalar, size>& a) {
    Dune::DiagonalMatrix<Scalar, size> b = a;
    b.diagonal() *= -1;
    return b;
  }

  template <typename Scalar, int size>
  auto operator-(const Dune::ScaledIdentityMatrix<Scalar, size>& a) {
    Dune::ScaledIdentityMatrix<Scalar, size> b = a;
    b.scalar() *= -1;
    return b;
  }

  template <typename Scalar, int size>
  auto& operator+(const Eigen::DiagonalMatrix<Scalar, size>& a, Dune::DerivativeDirections::ZeroMatrix) {
    return a.derived();
  }

  template <typename Scalar, int size>
  auto& operator+(const Dune::DiagonalMatrix<Scalar, size>& a, Dune::DerivativeDirections::ZeroMatrix) {
    return a;
  }

  template <typename Scalar, int size>
  auto& operator+(const Dune::ScaledIdentityMatrix<Scalar, size>& a, Dune::DerivativeDirections::ZeroMatrix) {
    return a;
  }

  template <typename Scalar, int rows, int cols>
  auto& operator+(const Dune::FieldMatrix<Scalar, rows, cols>& a, Dune::DerivativeDirections::ZeroMatrix) {
    return a;
  }

  template <typename Scalar, int size>
  auto operator+(Dune::DerivativeDirections::ZeroMatrix, const Eigen::DiagonalMatrix<Scalar, size>& a) {
    return a.derived();
  }

  template <typename Scalar, int size>
  auto operator+(Dune::DerivativeDirections::ZeroMatrix, const Dune::DiagonalMatrix<Scalar, size>& a) {
    return a;
  }

  template <typename Scalar, int size>
  auto operator+(Dune::DerivativeDirections::ZeroMatrix, const Dune::ScaledIdentityMatrix<Scalar, size>& a) {
    return a;
  }

  template <typename Scalar, int size>
  auto operator*(const Eigen::DiagonalMatrix<Scalar, size>& a, Dune::DerivativeDirections::ZeroMatrix) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename Scalar, int size>
  auto operator*(const Dune::DiagonalMatrix<Scalar, size>& a, Dune::DerivativeDirections::ZeroMatrix) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename Scalar, int size>
  auto operator*(Dune::DerivativeDirections::ZeroMatrix, const Eigen::DiagonalMatrix<Scalar, size>& a) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename Scalar, int size>
  auto operator*(Dune::DerivativeDirections::ZeroMatrix, const Dune::DiagonalMatrix<Scalar, size>& a) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename Scalar, int size>
  Eigen::DiagonalMatrix<Scalar, size> operator-(const Eigen::DiagonalMatrix<Scalar, size>& a,
                                                Dune::DerivativeDirections::ZeroMatrix) {
    return a.derived();
  }

  template <typename Scalar, int size>
  Dune::DiagonalMatrix<Scalar, size> operator-(const Dune::DiagonalMatrix<Scalar, size>& a,
                                               Dune::DerivativeDirections::ZeroMatrix) {
    return a;
  }

  template <typename Scalar, int size>
  auto operator-(const Dune::ScaledIdentityMatrix<Scalar, size>& a, Dune::DerivativeDirections::ZeroMatrix) {
    return a;
  }

  template <typename Scalar, int size>
  Eigen::DiagonalMatrix<Scalar, size> operator-(Dune::DerivativeDirections::ZeroMatrix,
                                                const Eigen::DiagonalMatrix<Scalar, size>& a) {
    return -a.derived();
  }

  template <typename Scalar, int size>
  Dune::DiagonalMatrix<Scalar, size> operator-(Dune::DerivativeDirections::ZeroMatrix,
                                               const Dune::DiagonalMatrix<Scalar, size>& a) {
    return -a;
  }

  template <typename Scalar, int size>
  auto& transpose(const Dune::DiagonalMatrix<Scalar, size>& a) {
    return a;
  }

  template <typename Derived, typename Derived2>
  auto operator+(const Eigen::MatrixBase<Derived>& a, const Eigen::DiagonalWrapper<Derived2>& b) {
    auto c = a.derived().eval();
    c.diagonal() += b.diagonal();
    return c;
  }

  template <typename Derived>
  auto operator+(const Eigen::MatrixBase<Derived>& a, Dune::DerivativeDirections::ZeroMatrix) {
    return a.derived();
  }

  template <typename Derived>
  auto operator+(Dune::DerivativeDirections::ZeroMatrix, const Eigen::MatrixBase<Derived>& a) {
    return a.derived();
  }

  Dune::DerivativeDirections::ZeroMatrix operator+(Dune::DerivativeDirections::ZeroMatrix,
                                                   Dune::DerivativeDirections::ZeroMatrix);

  Dune::DerivativeDirections::ZeroMatrix operator-(Dune::DerivativeDirections::ZeroMatrix,
                                                   Dune::DerivativeDirections::ZeroMatrix);

  template <typename Derived>
  auto operator*(const Eigen::MatrixBase<Derived>& a, Dune::DerivativeDirections::ZeroMatrix) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <std::floating_point T>
  auto operator*(const T&, Dune::DerivativeDirections::ZeroMatrix) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <std::floating_point T>
  auto operator*(Dune::DerivativeDirections::ZeroMatrix b, const T& a) {
    return a * b;
  }

  template <typename Derived>
  auto operator*(Dune::DerivativeDirections::ZeroMatrix, const Eigen::MatrixBase<Derived>& a) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  template <typename Derived>
  auto operator-(const Eigen::MatrixBase<Derived>& a, Dune::DerivativeDirections::ZeroMatrix) {
    return a.derived();
  }

  template <typename Derived>
  Derived operator-(Dune::DerivativeDirections::ZeroMatrix, const Eigen::MatrixBase<Derived>& a) {
    return -a.derived();
  }

  template <typename Derived, typename Derived2>
  auto operator+(const Eigen::DiagonalWrapper<Derived>& a, const Eigen::MatrixBase<Derived2>& b) {
    return b + a;
  }

  template <typename Derived>
  auto operator+(const Eigen::DiagonalWrapper<Derived>& a, Dune::DerivativeDirections::ZeroMatrix) {
    return a;
  }

  Dune::DerivativeDirections::ZeroMatrix operator-(Dune::DerivativeDirections::ZeroMatrix);

  template <typename Derived>
  auto operator+(Dune::DerivativeDirections::ZeroMatrix, const Eigen::DiagonalWrapper<Derived>& a) {
    return a;
  }

  template <typename Scalar, int size>
  std::ostream& operator<<(std::ostream& os, const Dune::DiagonalMatrix<Scalar, size>& a) {
    os << Dune::FieldMatrix<Scalar, size, size>(a);
    return os;
  }

  /** \brief Returns the symmetric part of a matrix*/
  template <typename Derived>
  Derived sym(const Eigen::MatrixBase<Derived>& A) {
    return 0.5 * (A + A.transpose());
  }

  /** \brief Returns the skew part of a matrix*/
  template <typename Derived>
  Derived skew(const Eigen::MatrixBase<Derived>& A) {
    return 0.5 * (A - A.transpose());
  }

  /** \brief Evaluates Eigen expressions */
  template <typename Derived>
  auto eval(const Eigen::EigenBase<Derived>& A) {
    if constexpr (static_cast<bool>(
                      Eigen::internal::is_diagonal<Derived>::ret))  // workaround needed since Eigen::DiagonalWrapper
                                                                    // does not has a eval function
    {
      using Scalar = typename Derived::Scalar;
      using namespace Eigen;
      constexpr int diag_size = EIGEN_SIZE_MIN_PREFER_DYNAMIC(Derived::RowsAtCompileTime, Derived::ColsAtCompileTime);
      constexpr int max_diag_size
          = EIGEN_SIZE_MIN_PREFER_FIXED(Derived::MaxRowsAtCompileTime, Derived::MaxColsAtCompileTime);

      return Eigen::DiagonalMatrix<Scalar, diag_size, max_diag_size>(A.derived().diagonal());
    } else
      return A.derived().eval();
  }

  /** \brief Does nothing if type is not an Eigen type but our manifolds type or floatingin point instead*/
  template <typename Type>
  requires Dune::Concepts::Manifold<Type> or std::floating_point<Type>
  auto eval(const Type& A) { return A; }

  /** \brief  eval overload for std::array  */
  template <typename Type, std::size_t d>
  auto eval(std::array<Type, d>&& t) {
    return t;
  }

  Dune::DerivativeDirections::ZeroMatrix transpose(const Dune::DerivativeDirections::ZeroMatrix&);
  Dune::DerivativeDirections::ZeroMatrix eval(const Dune::DerivativeDirections::ZeroMatrix&);

  template <typename Derived>
  auto transpose(const Eigen::EigenBase<Derived>& A) {
    if constexpr (requires {
                    A.derived().transpose();
                  })  // workaround needed since Dune::DiagonalMatrix has no transpose function
      return A.derived().transpose();
    else if constexpr (Dune::Std::IsSpecializationTypeAndNonTypes<Dune::DiagonalMatrix, Derived>::value
                       or Std::isSpecialization<Eigen::DiagonalWrapper, Derived>::value)
      return A.derived();
    else
      static_assert((requires { A.derived().transpose(); })
                    or Dune::Std::IsSpecializationTypeAndNonTypes<Dune::DiagonalMatrix, Derived>::value
                    or Std::isSpecialization<Eigen::DiagonalWrapper, Derived>::value);
  }

  template <typename To, typename From>
  requires std::convertible_to<typename From::ctype, To>
  auto convertUnderlying(const Dune::BlockVector<From>& from) {
    Dune::BlockVector<typename From::template Rebind<To>::other> to;
    to.resize(from.size());
    for (std::size_t i = 0; i < to.size(); ++i)
      to[i] = from[i];

    return to;
  }

  /* Enables the += operator for std::array if the underlying objects are addable  */
  template <typename Type, typename Type2, std::size_t d>
  std::array<Type, d> operator+(const std::array<Type, d>& a,
                                const std::array<Type2, d>& b) requires Concepts::AddAble<Type, Type2> {
    std::array<Type, d> res;
    for (size_t i = 0U; i < d; ++i)
      res[i] = a[i] + b[i];
    return res;
  }

  /* Enables the - operator for std::array if the underlying objects are negate able  */
  template <std::size_t d, typename Type>
  std::array<Type, d> operator-(const std::array<Type, d>& a)  // requires Concepts::NegateAble<Type>
  {
    std::array<Type, d> res;
    for (size_t i = 0U; i < d; ++i)
      res[i] = -a[i];
    return res;
  }

  /* Enables the transposition for std::array if the underlying objects are transposable able  */
  template <std::size_t d, typename Type>
  auto transpose(const std::array<Type, d>& a) requires Concepts::TransposeAble<Type> {
    std::array<decltype(transpose(a[0])), d> res;
    for (size_t i = 0U; i < d; ++i)
      res[i] = transpose(a[i]);
    return res;
  }

  template <std::size_t d, typename Scalar, typename Type>
  requires Concepts::MultiplyAble<Scalar, Type>
  auto operator*(Scalar b, const std::array<Type, d>& a) {
    std::array<std::remove_cvref_t<decltype(eval(b * a[0]))>, d> res;
    for (size_t i = 0U; i < d; ++i)
      res[i] = b * a[i];
    return res;
  }

  template <std::size_t d, typename Scalar, typename Type>
  requires Concepts::MultiplyAble<Scalar, Type>
  auto operator*=(std::array<Type, d>& a, Scalar b) {
    for (size_t i = 0U; i < d; ++i)
      a[i] *= b;
    return a;
  }

  /* Method to print to cout the matrix in a format that can directly be copied to maple*/
  template <typename Derived>
  void printForMaple(const Eigen::EigenBase<Derived>& A) {
    Eigen::IOFormat mapleFmt(Eigen::FullPrecision, 0, ", ", "|\n", "<", ">", "<", ">");
    if constexpr (std::convertible_to<Derived, const Eigen::MatrixBase<Derived>&>) {
      std::cout << "\n" << A.derived().format(mapleFmt) << std::endl;
    } else {  // branch for Dune::DiagonalMatrix
      using Scalar = typename Derived::Scalar;
      using namespace Eigen;
      constexpr int diag_size = EIGEN_SIZE_MIN_PREFER_DYNAMIC(Derived::RowsAtCompileTime, Derived::ColsAtCompileTime);
      std::cout << "\n"
                << Dune::FieldMatrix<Scalar, diag_size, diag_size>(A.derived().diagonal().asDiagonal()).format(mapleFmt)
                << std::endl;
    }
  }

  namespace Impl {
    constexpr std::tuple<std::array<std::array<int, 2>, 1>, std::array<std::array<int, 2>, 3>,
                         std::array<std::array<int, 2>, 6>>
        voigtIndices = {{{{0, 0}}}, {{{0, 0}, {1, 1}, {0, 1}}}, {{{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}}}};
  }

  /*
   * This class returns the indices, if you go through a symmetric matrix with index (Voigt) notation
   * 1D: 0,0
   * 2D: 0,0 1,1 0,1
   * 3D: 0,0 1,1 2,2 1,2 0,2 0,1
   */
  template <int dim>
  constexpr auto voigtNotationContainer = std::get<dim - 1>(Impl::voigtIndices);

}  // namespace Dune