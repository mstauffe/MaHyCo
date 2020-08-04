#ifndef VARIABLESLAGREMAP_H
#define VARIABLESLAGREMAP_H

#include <stddef.h>  // for size_t

#include <Kokkos_Core.hpp>                // for KOKKOS_LAMBDA
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>  // for OpenMP::impl_is_initialized
#include <algorithm>                      // for copy
#include <array>                          // for array
#include <string>                         // for allocator, string
#include <vector>                         // for vector
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D, CartesianM...
#include "mesh/MeshGeometry.h"     // for MeshGeometry
#include "mesh/PvdFileWriter2D.h"  // for PvdFileWriter2D

#include "types/Types.h"  // for RealArray1D, RealArray2D
#include "utils/Timer.h"  // for Timer

namespace variableslagremaplib {

class VariablesLagRemap {
 public:
  Kokkos::View<double*> deltaxLagrange;
 private:
  CartesianMesh2D* mesh;
  int nbCells, nbFaces;
  VariablesLagRemap(
		    CartesianMesh2D* aCartesianMesh2D):
    mesh(aCartesianMesh2D),   
    nbCells(mesh->getNbCells()),
    nbFaces(mesh->getNbFaces()),
      deltaxLagrange("deltaxLagrange", nbFaces){}
      };
} // namespace variableslagremaplib
#endif  // VARIABLESLAGREMAP_H
