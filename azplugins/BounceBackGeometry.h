// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file BounceBackGeometry.h
 * \brief Declaration of valid bounce back geometries.
 */

#ifndef AZPLUGINS_BOUNCE_BACK_GEOMETRY_H_
#define AZPLUGINS_BOUNCE_BACK_GEOMETRY_H_

#include "hoomd/mpcd/BoundaryCondition.h"
#include "hoomd/mpcd/SlitGeometry.h"

#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{
namespace detail
{

//! Export boundary enum to python
//void export_boundary(pybind11::module& m);

//! Export SlitGeometry to python
//void export_SlitGeometry(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins

#endif // NVCC
#endif // MPCD_STREAMING_GEOMETRY_H_
