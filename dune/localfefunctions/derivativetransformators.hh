// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <dune/localfefunctions/linearAlgebraHelper.hh>

#if DUNE_LOCALFEFUNCTIONS_USE_EIGEN == 1
namespace Dune {
  template <typename ScalarType, int Options, int MaxRowsAtCompileTime, int MaxColsAtCompileTime>
  void calcCartesianDerivatives(
      const Eigen::Matrix<ScalarType, Eigen::Dynamic, 2> &dN,
      const Eigen::Matrix<ScalarType, 3, 2, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime> &A1andA2,
      Eigen::Matrix<ScalarType, Eigen::Dynamic, 2> &dNTransformed) noexcept {
    const Eigen::Vector<ScalarType, 3> A1   = A1andA2.col(0);
    const Eigen::Vector<ScalarType, 3> A2   = A1andA2.col(1);
    const Eigen::Vector<ScalarType, 3> Axi1 = A1.normalized();
    const Eigen::Vector<ScalarType, 3> Axi2 = A2.normalized();

    Eigen::Vector<ScalarType, 3> A3 = A1.cross(A2);
    ScalarType dA                   = A3.norm();
    A3                              = 1.0 / dA * A3;

    const Eigen::Vector<ScalarType, 3> Axi1bar = (0.5 * (Axi1 + Axi2)).normalized();
    const Eigen::Vector<ScalarType, 3> Axi2bar = A3.cross(Axi1bar).normalized();

    const Eigen::Vector<ScalarType, 3> A1loc = .70710678118654752440 * (Axi1bar - Axi2bar);
    const Eigen::Vector<ScalarType, 3> A2loc = .70710678118654752440 * (Axi1bar + Axi2bar);

    Eigen::Matrix<ScalarType, 2, 2> invJT;
    invJT << A1.dot(A1loc), A1.dot(A2loc), A2.dot(A1loc), A2.dot(A2loc);

    dNTransformed = dN * invJT.inverse().eval();
  }

  template <typename ScalarType, int worldDim, int GridDim, int Options, int MaxWorldDim, int MaxGridDim>
  void calcCartesianDerivativesByGramSchmidt(
      const Eigen::Matrix<ScalarType, Eigen::Dynamic, GridDim> &dN,
      const Eigen::Matrix<double, worldDim, GridDim, Options, MaxWorldDim, MaxGridDim> &A1andA2,
      Eigen::Matrix<ScalarType, Eigen::Dynamic, GridDim> &dNTransformed) noexcept {
    const Eigen::Matrix<ScalarType, worldDim, GridDim> A1andA2Ortho = Dune::orthonormalizeMatrixColumns(A1andA2);

    Eigen::Matrix<ScalarType, GridDim, GridDim> invJT = A1andA2.transpose() * A1andA2Ortho;

    dNTransformed = dN * invJT.inverse().eval();
  }

  struct DefaultFirstOrderTransformFunctor {
    template <typename DerivativeMatrix, typename Geometry, typename LocalCoord>
    void operator()(const Geometry &geo, const LocalCoord &gp, const DerivativeMatrix &dN,
                    DerivativeMatrix &dNTransformed) const {
      if constexpr (Geometry::coorddimension == Geometry::mydimension) {
        const auto jInv = toEigen(geo.jacobianTransposed(gp)).eval().inverse().transpose().eval();
        dNTransformed   = dN * jInv;
      } else if constexpr (Geometry::mydimension == 2
                           and Geometry::coorddimension == 3) {  // two-dimensional grid element in 3D space
        const auto j = toEigen(geo.jacobianTransposed(gp)).transpose().eval();
        calcCartesianDerivatives(dN, j, dNTransformed);
      }
    }
  };

  struct GramSchmidtFirstOrderTransformFunctor {
    template <typename DerivativeMatrix, typename Geometry, typename LocalCoord>
    void operator()(const Geometry &geo, const LocalCoord &gp, const DerivativeMatrix &dN,
                    DerivativeMatrix &dNTransformed) const {
      if constexpr (Geometry::coorddimension == Geometry::mydimension) {
        const auto jInv = toEigen(geo.jacobianTransposed(gp)).eval().inverse().transpose().eval();
        dNTransformed   = dN * jInv;
      } else {
        const auto j = toEigen(geo.jacobianTransposed(gp)).transpose().eval();
        calcCartesianDerivativesByGramSchmidt(dN, j, dNTransformed);
      }
    }
  };

}  // namespace Dune

#else
namespace Dune {

  template <typename ScalarType>
  void calcCartesianDerivatives(const Dune::BlockVector<Dune::FieldVector<ScalarType, 2> > &dN,
                                const Dune::FieldMatrix<ScalarType, 3, 2> &A1andA2,
                                Dune::BlockVector<Dune::FieldVector<ScalarType, 2> > &dNTransformed) noexcept {
    const Dune::FieldMatrix<ScalarType, 2, 3> A1andA2T = Dune::transpose(A1andA2);
    const Dune::FieldVector<ScalarType, 3> &A1         = A1andA2T[0];
    const Dune::FieldVector<ScalarType, 3> &A2         = A1andA2T[1];
    const Dune::FieldVector<ScalarType, 3> Axi1        = A1 / A1.two_norm();
    const Dune::FieldVector<ScalarType, 3> Axi2        = A2 / A2.two_norm();

    Dune::FieldVector<ScalarType, 3> A3 = cross(A1, A2);
    const ScalarType dA                 = A3.two_norm();
    A3                                  = 1.0 / dA * A3;

    Dune::FieldVector<ScalarType, 3> Axi1bar = (0.5 * (Axi1 + Axi2));
    Axi1bar /= Axi1bar.two_norm();
    Dune::FieldVector<ScalarType, 3> Axi2bar = Dune::cross(A3, Axi1bar);
    Axi2bar /= Axi2bar.two_norm();
    const Dune::FieldVector<ScalarType, 3> A1loc = sqrt(2.0) / 2.0 * (Axi1bar - Axi2bar);
    const Dune::FieldVector<ScalarType, 3> A2loc = sqrt(2.0) / 2.0 * (Axi1bar + Axi2bar);

    Dune::FieldMatrix<ScalarType, 2, 2> invJT;
    invJT[0][0] = A1 * A1loc;
    invJT[0][1] = A1 * A2loc;
    invJT[1][0] = A2 * A1loc;
    invJT[1][1] = A2 * A2loc;

    dNTransformed.resize(dN.size());
    for (size_t i = 0; i < dN.size(); ++i)
      invJT.mv(dN[i], dNTransformed[i]);
  }

  template <typename ScalarType, int worldDim, int GridDim>
  void calcCartesianDerivativesByGramSchmidt(
      const Dune::BlockVector<Dune::FieldVector<ScalarType, GridDim> > &dN,
      const Dune::FieldMatrix<ScalarType, worldDim, GridDim> &A1andA2,
      Dune::BlockVector<Dune::FieldVector<ScalarType, GridDim> > &dNTransformed) noexcept {
    const Dune::FieldMatrix<ScalarType, worldDim, GridDim> A1andA2Ortho = Dune::orthonormalizeMatrixColumns(A1andA2);

    Dune::FieldMatrix<ScalarType, GridDim, GridDim> invJT = (transpose(A1andA2) * A1andA2Ortho);
    invJT.invert();
    dNTransformed.resize(dN.size());
    for (size_t i = 0; i < dN.size(); ++i)
      invJT.mv(dN[i], dNTransformed[i]);
  }

  struct DefaultFirstOrderTransformFunctor {
    template <typename DerivativeMatrix, typename Geometry, typename LocalCoord>
    void operator()(const Geometry &geo, const LocalCoord &gp, const DerivativeMatrix &dN,
                    DerivativeMatrix &dNTransformed) const {
      if constexpr (Geometry::coorddimension == Geometry::mydimension) {
        const auto jInv = geo.jacobianInverseTransposed(gp);
        dNTransformed.resize(dN.size());
        for (size_t i = 0; i < dN.size(); ++i)
          jInv.mv(dN[i], dNTransformed[i]);
      } else if constexpr (Geometry::mydimension == 2
                           and Geometry::coorddimension == 3) {  // two-dimensional grid element in 3D space
        const auto j = maybeToEigen(transpose(geo.jacobianTransposed(gp)));
        calcCartesianDerivatives(dN, j, dNTransformed);
      }
    }
  };

  struct GramSchmidtFirstOrderTransformFunctor {
    template <typename DerivativeMatrix, typename Geometry, typename LocalCoord>
    void operator()(const Geometry &geo, const LocalCoord &gp, const DerivativeMatrix &dN,
                    DerivativeMatrix &dNTransformed) const {
      if constexpr (Geometry::coorddimension == Geometry::mydimension) {
        const auto jInv = geo.jacobianInverseTransposed(gp);
        dNTransformed.resize(dN.size());
        for (size_t i = 0; i < dN.size(); ++i)
          jInv.mv(dN[i], dNTransformed[i]);
      } else {
        const auto j = transpose(geo.jacobianTransposed(gp));
        calcCartesianDerivativesByGramSchmidt(dN, j, dNTransformed);
      }
    }
  };

}  // namespace Dune
#endif
