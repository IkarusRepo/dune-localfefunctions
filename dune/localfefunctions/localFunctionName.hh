// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include <regex>
#include <string>

#include <dune/common/classname.hh>

namespace Dune {

  /** Pretty printing the name of a local function expression */
  template <typename LF>
  auto localFunctionName(const LF& lf) {
    std::string name = Dune::className(lf);
    std::regex regexp("Dune::StandardLocalFunction<(([a-zA-Z0-9_:<, ]*>){12})");
    name = regex_replace(name, regexp, "SLF");

    regexp = "Dune::ProjectionBasedLocalFunction<(([a-zA-Z0-9_:<, ]*>){12})";
    name   = regex_replace(name, regexp, "PBLF");

    regexp = "Dune::";
    name   = regex_replace(name, regexp, "");

    regexp = "InnerProductExpr";
    name   = regex_replace(name, regexp, "Dot");

    regexp = "NegateExpr";
    name   = regex_replace(name, regexp, "Negate");

    regexp = "ScaleExpr";
    name   = regex_replace(name, regexp, "Scale");

    regexp = "SumExpr";
    name   = regex_replace(name, regexp, "Sum");

    regexp = "const";
    name   = regex_replace(name, regexp, "");

    regexp = "ConstantExpr<(([a-zA-Z0-9_:<, ]*>))";
    name   = regex_replace(name, regexp, "Constant");

    regexp = "SqrtExpr";
    name   = regex_replace(name, regexp, "Sqrt");

    regexp = "NormSquaredExpr";
    name   = regex_replace(name, regexp, "NormSquared");

    regexp = "LinearStrainExpr";
    name   = regex_replace(name, regexp, "LinearStrains");

    regexp = "GreenLagrangeStrainsExpr";
    name   = regex_replace(name, regexp, "GreenLagrangeStrains");

    regexp = "[ \\t]+$";  // remove trailing white space
    name   = regex_replace(name, regexp, "");

    return name;
  }
}  // namespace Dune
