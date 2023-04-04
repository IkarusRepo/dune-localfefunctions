// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <dune/common/diagonalmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/localfefunctions/linalgconcepts.hh>

#include <Eigen/Core>

namespace Dune {

  /** \brief Creates a Dune::FieldVector from a given Eigen::Vector */
  template <typename ScalarType, int size>
  Dune::FieldVector<ScalarType, size> toFieldVector(const Eigen::Vector<ScalarType, size>& vec) {
    Dune::FieldVector<ScalarType, size> fieldvec;
    for (int i = 0; i < size; ++i)
      fieldvec[i] = vec[i];
    return fieldvec;
  }

  /** \brief Creates a Dune::FieldVector from a given Eigen::Matrix. The matrix has fixed dynamic size The matrix needs
   * to have a single column. */
  template <typename ScalarType, int maxRows, int maxCols>
  Dune::FieldVector<ScalarType, maxRows> toFieldVector(
      const Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, 0, maxRows, maxCols>& vec) {
    assert(vec.cols() == 1 && "The passed matrix needs to have a single column.");
    Dune::FieldVector<ScalarType, maxRows> fieldvec{0.0};

    for (int i = 0; i < vec.rows(); ++i)
      fieldvec[i] = vec(i, 0);
    return fieldvec;
  }

  /** \brief Views a const dune fieldvector as a const Eigen::Vector as Map, no copies take place! */
  template <typename ScalarType, int size>
  Eigen::Map<const Eigen::Vector<ScalarType, size>> toEigenVectorMap(const Dune::FieldVector<ScalarType, size>& vec) {
    return {vec.data(), size};
  }

  /** \brief No-op of  toEigenVectorMap(const Dune::FieldVector<ScalarType, size>&*/
  template <typename ScalarType, int size>
  auto toEigenVectorMap(const Eigen::Vector<ScalarType, size>& vec) {
    return vec;
  }

  /** \brief Creates a Eigen::Vector from a given Dune::FieldVector  */
  template <typename ScalarType, int size>
  Eigen::Vector<ScalarType, size> toEigen(const Dune::FieldVector<ScalarType, size>& vec) {
    Eigen::Vector<ScalarType, size> eigenVector;
    for (int i = 0; i < size; ++i)
      eigenVector(i) = vec[i];
    return eigenVector;
  }

  /** \brief Does nothing if Matrix is already eigen */
  template <typename Derived>
  auto& toEigen(const Eigen::MatrixBase<Derived>& mat) {
    return mat.derived();
  }

  /** \brief Creates a Eigen::Vector from a given Dune::FieldMatrix with one column  */
  template <typename ScalarType, int size>
  Eigen::Vector<ScalarType, size> toEigen(const Dune::FieldMatrix<ScalarType, size, 1>& vec) {
    Eigen::Vector<ScalarType, size> eigenVector;
    for (int i = 0; i < size; ++i)
      eigenVector(i) = vec[i][0];
    return eigenVector;
  }

  /** \brief Depending on the Selected LinearAlgebra the Dune::Fieldmatrix is converted to an Eigen::Matrix */
  template <typename ScalarType, int rows, int cols, typename LinAlg = DefaultLinearAlgebra>
  auto maybeToEigen(const Dune::FieldMatrix<ScalarType, rows, cols>& vec) {
#if DUNE_LOCALFEFUNCTIONS_USE_EIGEN == 1
    if constexpr (std::is_same_v<LinAlg, EigenLinearAlgebra>)
      return toEigen(vec);
    else
#endif
      return vec;
  }

  /** \brief Creates a Eigen::Matrix from a given Dune::FieldMatrix  */
  template <typename ScalarType, int size1, int size2>
  auto toDune(const Eigen::Matrix<ScalarType, size1, size2>& mat) {
    if constexpr (size2 == 1) {
      Dune::FieldVector<ScalarType, size1> duneVector;
      for (int i = 0; i < size1; ++i)
        duneVector[i] = mat(i, 0);
      return duneVector;
    } else {
      Dune::FieldMatrix<ScalarType, size1, size2> duneMatrix;
      for (int i = 0; i < size1; ++i)
        for (int j = 0; j < size2; ++j)
          duneMatrix[i][j] = mat(i, j);
      return duneMatrix;
    }
  }

  /** \brief Depending on the Selected LinearAlgebra the Eigen::Matrix is converted to an Dune::FieldMatrix */
  template <typename ScalarType, int rows, int cols, typename LinAlg = DefaultLinearAlgebra>
  auto maybeToDune(const Eigen::Matrix<ScalarType, rows, cols>& mat) {
#if DUNE_LOCALFEFUNCTIONS_USE_EIGEN == 1
    if constexpr (std::is_same_v<LinAlg, EigenLinearAlgebra>)
      return mat;
    else
#endif
      return toDune(mat);
  }

  /** \brief Views a  dune fieldvector as a Eigen::Vector, no copies take place! */
  template <typename ScalarType, int size>
  Eigen::Map<Eigen::Vector<ScalarType, size>> toEigen(Dune::FieldVector<ScalarType, size>& vec) {
    return {vec.data(), size};
  }

  /** \brief Creates a Eigen::Matrix from a given Dune::FieldMatrix  */
  template <typename ScalarType, int size1, int size2>
  Eigen::Matrix<ScalarType, size1, size2> toEigen(const Dune::FieldMatrix<ScalarType, size1, size2>& mat) {
    Eigen::Matrix<ScalarType, size1, size2> eigenmatrix;
    for (int i = 0; i < size1; ++i)
      for (int j = 0; j < size2; ++j)
        eigenmatrix(i, j) = mat[i][j];
    return eigenmatrix;
  }

  /** \brief Creates a Eigen::Matrix from a given Dune::DiagonalMatrix. This should return Eigen::DiagonalMatrix but
   * Eigen::DiagonalMatrix does not contain e.g. a transpose method. And therefore we would need to specialize user
   * code. Maybe someone wants to do a PR at Eigen? */
  template <typename ScalarType, int size1>
  Eigen::Matrix<ScalarType, size1, size1> toEigen(const Dune::DiagonalMatrix<ScalarType, size1>& mat) {
    Eigen::Matrix<ScalarType, size1, size1> eigenmatrix;
    eigenmatrix.setZero();
    for (int i = 0; i < size1; ++i)
      eigenmatrix(i, i) = mat.diagonal(i);
    return eigenmatrix;
  }
  
   /** \brief Depending on the Selected LinearAlgebra the Dune::Fieldmatrix is converted to an Eigen::Matrix */
  template <typename ScalarType, int rows, typename LinAlg = DefaultLinearAlgebra>
  auto maybeToEigen(const Dune::DiagonalMatrix<ScalarType, rows>& mat) {
#if DUNE_LOCALFEFUNCTIONS_USE_EIGEN == 1
    if constexpr (std::is_same_v<LinAlg, EigenLinearAlgebra>)
      return toEigen(mat);
    else
#endif
      return mat;
  }

  template <typename ScalarType, int cols>
  auto toEigen(const Dune::BlockVector<Dune::FieldVector<ScalarType, cols>>& mat) {
    Eigen::Matrix<ScalarType, Eigen::Dynamic, cols> matE;
    matE.resize(mat.size(), Eigen::NoChange);
    for (auto i = 0U; i < mat.size(); ++i)
      for (auto j = 0U; j < cols; ++j)
        matE(i, j) = mat[i][j];
    return matE;
  }

  template <typename ScalarType>
  requires std::is_arithmetic_v<ScalarType>
  auto toEigen(const Dune::BlockVector<ScalarType>& vec) {
    Eigen::Vector<ScalarType, Eigen::Dynamic> vecE;
    vecE.resize(vec.size());
    for (auto i = 0U; i < vec.size(); ++i)
      vecE(i) = vec[i];
    return vecE;
  }

  template <typename ScalarType, int cols>
  void resize(Dune::BlockVector<Dune::FieldVector<ScalarType, cols>>& mat, size_t newSize) {
    mat.resize(newSize);
  }

  template <typename ScalarType, int cols>
  void resize(Eigen::Matrix<ScalarType, Eigen::Dynamic, cols>& mat, size_t newSize) {
    mat.resize(newSize, Eigen::NoChange);
  }

}  // namespace Dune
