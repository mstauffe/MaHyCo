#include <Kokkos_Core.hpp>
#include <algorithm>  // for copy
#include <array>      // for array
#include <iostream>   // for operator<<, basic_ostream::operat...
#include <vector>     // for allocator, vector

#include "Remap.h"          // for Remap, Remap::Opt...
#include "../remap/UtilesRemap-Impl.h"  // for Remap::computeRemapFlux
#include "mesh/CartesianMesh2D.h"   // for CartesianMesh2D
#include "types/MathFunctions.h"    // for dot, matVectProduct, norm
#include "types/MultiArray.h"       // for operator<<
#include "utils/Utils.h"            // for indexOf

RealArray1D<nbequamax> Remap::computeBoundaryFluxes(
    int proj, int cCells, RealArray1D<dim> exy) {
  RealArray1D<nbequamax> phiFace_fFaces = Uzero;
  int nbCellX = cstmesh->X_EDGE_ELEMS;
  int nbCellY = cstmesh->Y_EDGE_ELEMS;
  if (cdl->bottomFluxBC == 1 && cCells < nbCellX && exy[1] == 1) {
    // cellules Bottom
    int cId(cCells);
    int fbBottomFaceOfCellC(mesh->getBottomFaceOfCell(cId));
    size_t fbId(fbBottomFaceOfCellC);
    int fbFaces(utils::indexOf(mesh->getFaces(), fbId));
    int fbFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), fbId));

    int ftTopFaceOfCellC(mesh->getTopFaceOfCell(cId));
    size_t ftId(ftTopFaceOfCellC);
    int ftFaces(utils::indexOf(mesh->getFaces(), ftId));
    int ftFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), ftId));

    std::cout << " Bottom cell   " << cCells << std::endl;
    if (proj == 1) phiFace_fFaces = phiFace1(ftFaces);
    if (proj == 2) phiFace_fFaces = phiFace2(ftFaces);
    return Remap::computeRemapFlux(
        options->projectionOrder, limiteurs->projectionAvecPlateauPente,
        varlp->faceNormalVelocity(fbFaces),
	varlp->faceNormal(fbFaces),
	varlp->faceLength(fbFaces),
        phiFace_fFaces,
	varlp->outerFaceNormal(cCells, fbFacesOfCellC), exy,
        gt->deltat_n);
  }
  if (cdl->topFluxBC == 1 && cCells <= nbCellX * (nbCellY - 1) &&
      cCells < nbCellX * nbCellY && exy[1] == 1) {
    // cellules top
    int cId(cCells);
    int fbBottomFaceOfCellC(mesh->getBottomFaceOfCell(cId));
    size_t fbId(fbBottomFaceOfCellC);
    int fbFaces(utils::indexOf(mesh->getFaces(), fbId));
    int fbFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), fbId));

    int ftTopFaceOfCellC(mesh->getTopFaceOfCell(cId));
    size_t ftId(ftTopFaceOfCellC);
    int ftFaces(utils::indexOf(mesh->getFaces(), ftId));
    int ftFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), ftId));
    std::cout << " Top cell   " << cCells << std::endl;

    if (proj == 1) phiFace_fFaces = phiFace1(fbFaces);
    if (proj == 2) phiFace_fFaces = phiFace2(fbFaces);
    return Remap::computeRemapFlux(
        options->projectionOrder, limiteurs->projectionAvecPlateauPente,
        varlp->faceNormalVelocity(ftFaces),
	varlp->faceNormal(ftFaces),
	varlp->faceLength(ftFaces),
        phiFace_fFaces,
	varlp->outerFaceNormal(cCells, ftFacesOfCellC), exy,
        gt->deltat_n);
  }
  if (cdl->leftFluxBC == 1 && exy[0] == 1) {
    // cellules de gauche - a optimiser
    for (int icCells = 0; icCells < nbCellX * nbCellY;
         icCells = icCells + nbCellX) {
      if (icCells == cCells) {
        int cId(cCells);
        int frRightFaceOfCellC(mesh->getRightFaceOfCell(cId));
        size_t frId(frRightFaceOfCellC);
        int frFaces(utils::indexOf(mesh->getFaces(), frId));
        int frFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), frId));

        int flLeftFaceOfCellC(mesh->getLeftFaceOfCell(cId));
        size_t flId(flLeftFaceOfCellC);
        int flFaces(utils::indexOf(mesh->getFaces(), flId));
        int flFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), flId));

        std::cout << " Left cell   " << cCells << std::endl;

        if (proj == 1) phiFace_fFaces = phiFace1(frFaces);
        if (proj == 2) phiFace_fFaces = phiFace2(frFaces);
        return Remap::computeRemapFlux(
            options->projectionOrder, limiteurs->projectionAvecPlateauPente,
            varlp->faceNormalVelocity(flFaces),
	    varlp->faceNormal(flFaces),
            varlp->faceLength(flFaces), phiFace_fFaces,
            varlp->outerFaceNormal(cCells, flFacesOfCellC), exy, gt->deltat_n);
      }
    }
  }
  if (cdl->rightFluxBC == 1 && exy[0] == 1) {
    // cellules de droite
    for (int icCells = nbCellX - 1; icCells < nbCellX * nbCellY;
         icCells = icCells + nbCellX) {
      if (icCells == cCells) {
        int cId(cCells);
        int frRightFaceOfCellC(mesh->getRightFaceOfCell(cId));
        size_t frId(frRightFaceOfCellC);
        int frFaces(utils::indexOf(mesh->getFaces(), frId));
        int frFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), frId));

        int flLeftFaceOfCellC(mesh->getLeftFaceOfCell(cId));
        size_t flId(flLeftFaceOfCellC);
        int flFaces(utils::indexOf(mesh->getFaces(), flId));
        int flFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), flId));

        if (proj == 1) phiFace_fFaces = phiFace1(flFaces);
        if (proj == 2) phiFace_fFaces = phiFace2(flFaces);      
        //
        return Remap::computeRemapFlux(
            options->projectionOrder, limiteurs->projectionAvecPlateauPente,
            varlp->faceNormalVelocity(frFaces),
	    varlp->faceNormal(frFaces),
            varlp->faceLength(frFaces), phiFace_fFaces,
            varlp->outerFaceNormal(cCells, frFacesOfCellC),
	    exy, gt->deltat_n);
      }
    }
  }
  return phiFace_fFaces;  // Uzero;
}
