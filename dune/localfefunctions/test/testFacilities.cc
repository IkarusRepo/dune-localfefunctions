// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "testFacilities.hh"

#include <Eigen/Core>

namespace Testing {

  /*
   * This function returns the polynom fitted onto the data passed in the least square sense.
   * It also returns the least square error.
   */
  std::tuple<Dune::Functions::Polynomial<double>, double> polyfit(const Eigen::Ref<const Eigen::VectorXd>& x,
                                                                  const Eigen::Ref<const Eigen::VectorXd>& y,
                                                                  const int deg) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Ones(x.size(), deg + 1);
    for (int j = 1; j < deg + 1; ++j)
      A.col(j) = A.col(j - 1).cwiseProduct(x);

    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(A.rows(), A.cols());
    qr.compute(A);
    Eigen::VectorXd coeffs = qr.solve(y);

    std::vector<double> coeffsSTD(coeffs.begin(), coeffs.end());
    Dune::Functions::Polynomial<double> poly(std::move(coeffsSTD));
    return std::make_tuple(poly, (A * coeffs - y).norm());
  }

  std::tuple<Dune::Functions::Polynomial<double>, decltype(Eigen::seq(0, 0))> findLineSegment(const Eigen::VectorXd& x,
                                                                                              const Eigen::VectorXd& y,
                                                                                              int segmentSize) {
    Eigen::VectorXd errors = Eigen::VectorXd::Zero(x.size() - segmentSize);
    std::vector<Dune::Functions::Polynomial<double>> lines;
    for (int i = 0; i < errors.size(); ++i) {
      auto range = Eigen::seq(i, i + segmentSize);

      auto [poly, error] = polyfit(x(range), y(range), 1);
      errors(i)          = error;
      lines.push_back(poly);
    }
    std::forward_iterator auto minEle = std::ranges::min_element(errors.begin(), errors.end());
    auto index                        = std::distance(errors.begin(), minEle);
    auto range                        = Eigen::seq(index, index + segmentSize);
    return std::make_tuple(lines[index], range);
  }

  double getSlope(const std::function<double(double)>& ftfunc, int slopeOfReference) {
    std::vector<double> t;
    std::ranges::transform(Eigen::VectorXd::LinSpaced(100, -8, -2), std::back_inserter(t),
                           [](double x) { return pow(10, x); });
    Eigen::Map<Eigen::VectorXd> data(t.data(), t.size());
    std::vector<double> ftevaluated;
    std::ranges::transform(t, std::back_inserter(ftevaluated), ftfunc);
    Eigen::Map<Eigen::VectorXd> yE(ftevaluated.data(), ftevaluated.size());

    std::vector<double> fexpectedSlope;
    std::ranges::transform(t, std::back_inserter(fexpectedSlope),
                           [slopeOfReference](auto tL) { return Dune::power(tL, slopeOfReference); });
    const int rangeSize      = 10;
    const auto [poly, range] = findLineSegment(data.array().log10(), yE.array().log10(), rangeSize);

    if (yE(range).lpNorm<Eigen::Infinity>() < 1e-10)
      return std::numeric_limits<double>::infinity();  // If the error is zero everywhere the function is linear for
                                                       // this case we return infinity
    return poly.coefficients()[1];
  }
}  // namespace Testing
