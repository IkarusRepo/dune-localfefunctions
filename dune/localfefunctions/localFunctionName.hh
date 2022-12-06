/*
 * This file is part of the Ikarus distribution (https://github.com/ikarus-project/ikarus).
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
