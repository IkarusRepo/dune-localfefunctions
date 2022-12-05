// SPDX-FileCopyrightText: 2022 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later
#pragma once

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/scaledidmatrix.hh>
#if DUNE_LOCALFEFUNCTIONS_USE_EIGEN == 1
#include <Eigen/Core>
#endif
namespace Dune
{
  struct DuneLinearAlgebra
  {
    template<typename ScalarType,int rows>
    using FixedSizedVector= Dune::FieldVector<ScalarType,rows>;
    template<typename ScalarType,int rows,int cols>
    using FixedSizedMatrix= Dune::FieldMatrix<ScalarType,rows,cols>;

    template<typename ScalarType,int rows>
    using FixedSizedScaledIdentityMatrix= Dune::ScaledIdentityMatrix<ScalarType,rows>;

    template<typename ScalarType,int rows>
    static auto createZeroVector()
    {
      return Dune::FieldVector<ScalarType, rows>(0);
    }

    template<typename ScalarType,int rows,int cols>
    static auto createZeroMatrix() {
      return Dune::FieldMatrix<ScalarType, rows, cols>(0);
    }

    template<typename ScalarType,int rows>
    static auto createOnesVector()
    {
      return Dune::FieldVector<ScalarType, rows>(1);
    }

    template<typename ScalarType,int rows,int cols>
    static auto createOnesMatrix()
    {
      return Dune::FieldMatrix<ScalarType, rows,cols>(1);
    }

    template<typename ScalarType,int rows>
    static auto createScaledIdentityMatrix(ScalarType val)
    {
      return Dune::ScaledIdentityMatrix<ScalarType, rows>(val);
    }
  };

#if DUNE_LOCALFEFUNCTIONS_USE_EIGEN == 1
  struct EigenLinearAlgebra
  {
    template<typename ScalarType,int rows>
    using FixedSizedVector= Eigen::Vector<ScalarType,rows>;
    template<typename ScalarType,int rows,int cols>
    using FixedSizedMatrix= Eigen::Matrix<ScalarType,rows,cols>;

    template<typename ScalarType,int rows>
    using FixedSizedScaledIdentityMatrix= Eigen::Matrix<ScalarType,rows,rows>;

    template<typename ScalarType,int rows>
    static auto createZeroVector()
    {
      return Eigen::Vector<ScalarType, rows>::Zero();
    }

    template<typename ScalarType,int rows>
    static auto createOnesVector()
    {
      return Eigen::Vector<ScalarType, rows>::Ones();
    }

    template<typename ScalarType,int rows,int cols>
    static auto createZeroMatrix() {
      return Eigen::Matrix<ScalarType, rows, cols>::Zero();
    }

    template<typename ScalarType,int rows,int cols>
    static auto createOnesMatrix()
    {
      return Eigen::Matrix<ScalarType, rows,cols>::Ones();
    }

    template<typename ScalarType,int rows>
    static Eigen::Matrix<ScalarType, rows,rows> createScaledIdentityMatrix(const ScalarType& val)
    {
      return (Eigen::Matrix<ScalarType, rows,rows>::Identity()*val).eval();
    }

  };
#endif
#if DUNE_LOCALFEFUNCTIONS_USE_EIGEN == 1
  using DefaultLinearAlgebra = EigenLinearAlgebra;
#else
  using DefaultLinearAlgebra = DuneLinearAlgebra;
#endif


//  concept DuneOrEigenMatrix
}
