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

using namespace nablalib;

namespace variableslagremaplib {

class VariablesLagRemap {

 private:
  CartesianMesh2D* mesh;
  int nbNodes, nbCells, nbFaces, nbFacesOfCell;
  
 public: 
  Kokkos::View<RealArray1D<nbequamax>*> Phi;
  Kokkos::View<RealArray1D<nbequamax>*> PhiDual;
  Kokkos::View<double*> deltaxLagrange;
  Kokkos::View<RealArray1D<dim>*> XLagrange;
  Kokkos::View<RealArray1D<dim>*> XfLagrange;
  Kokkos::View<RealArray1D<dim>*> XcLagrange;
  Kokkos::View<RealArray1D<dim>*> Xf;
  Kokkos::View<RealArray1D<dim>*> faceNormal;
  Kokkos::View<RealArray1D<dim>**> outerFaceNormal;
  Kokkos::View<double*> faceLength; 
  Kokkos::View<double*> faceLengthLagrange;
  Kokkos::View<double*> faceNormalVelocity;
  Kokkos::View<RealArray1D<nbequamax>*> ULagrange;
  Kokkos::View<RealArray1D<nbequamax>*> UDualLagrange;
  Kokkos::View<RealArray1D<nbequamax>*> Uremap2;
  Kokkos::View<RealArray1D<nbequamax>*> UDualremap2;
  Kokkos::View<double*> vLagrange;
  Kokkos::View<int*> mixte;
  Kokkos::View<int*> pure;
  bool x_then_y_n, x_then_y_nplus1;
  
  VariablesLagRemap(CartesianMesh2D* aCartesianMesh2D):
    mesh(aCartesianMesh2D),   
    nbNodes(mesh->getNbNodes()),
    nbCells(mesh->getNbCells()),
    nbFaces(mesh->getNbFaces()),
    nbFacesOfCell(CartesianMesh2D::MaxNbFacesOfCell),
    Phi("Phi", nbCells),
    PhiDual("PhiDual", nbNodes),
    deltaxLagrange("deltaxLagrange", nbFaces),
    XLagrange("XLagrange", nbNodes),
    XfLagrange("XfLagrange", nbFaces),
    XcLagrange("XcLagrange", nbCells),
    Xf("Xf", nbFaces),
    faceNormal("faceNormal", nbFaces),
    outerFaceNormal("outerFaceNormal", nbCells, nbFacesOfCell),
    faceLength("faceLength", nbFaces),
    faceLengthLagrange("faceLengthLagrange", nbFaces),
    faceNormalVelocity("faceNormalVelocity", nbFaces),
    ULagrange("ULagrange", nbCells),
    Uremap2("Uremap2", nbCells),
    UDualLagrange("UDualLagrange", nbNodes),
    UDualremap2("UDualremap2", nbNodes),
    vLagrange("vLagrange", nbCells),
    mixte("mixte", nbCells),
    pure("pure", nbCells),
    x_then_y_n(true),
    x_then_y_nplus1(true)
    {
      std::cout << "Nombre de mailles:  " <<  nbCells
		<< "Nombre de noeuds: " << nbNodes
		<< "Nombre de faces:  " << nbFaces << std::endl;}
    };
} // namespace variableslagremaplib
#endif  // VARIABLESLAGREMAP_H
