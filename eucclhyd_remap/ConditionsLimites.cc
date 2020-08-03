#include <Kokkos_Core.hpp>
#include <algorithm>  // for copy
#include <array>      // for array
#include <iostream>   // for operator<<, basic_ostream::operat...
#include <vector>     // for allocator, vector

#include "EucclhydRemap.h"          // for EucclhydRemap, EucclhydRemap::Opt...
#include "UtilesRemap-Impl.h"       // for EucclhydRemap::computeRemapFlux
#include "mesh/CartesianMesh2D.h"   // for CartesianMesh2D
#include "types/MathFunctions.h"    // for dot, matVectProduct, norm
#include "types/MultiArray.h"       // for operator<<
#include "utils/Utils.h"            // for indexOf

/**
 * Job computeBoundaryNodeVelocities called @4.0 in executeTimeLoopN method.
 * In variables: G, Mnode, bottomBC, bottomBCValue, leftBC, leftBCValue,
 * rightBC, rightBCValue, topBC, topBCValue Out variables: Vnode_nplus1
 */
void EucclhydRemap::computeBoundaryNodeVelocities() noexcept {
  auto leftNodes(mesh->getLeftNodes());
  int nbLeftNodes(mesh->getNbLeftNodes());
  Kokkos::parallel_for("computeBoundaryNodeVelocities", nbLeftNodes,
                       KOKKOS_LAMBDA(const int& pLeftNodes) {
                         int pId(leftNodes[pLeftNodes]);
                         int pNodes(pId);
                         Vnode_nplus1(pNodes) = nodeVelocityBoundaryCondition(
                             cdl->leftBC, cdl->leftBCValue, Mnode(pNodes),
                             G(pNodes));
                       });
  auto rightNodes(mesh->getRightNodes());
  int nbRightNodes(mesh->getNbRightNodes());
  Kokkos::parallel_for("computeBoundaryNodeVelocities", nbRightNodes,
                       KOKKOS_LAMBDA(const int& pRightNodes) {
                         int pId(rightNodes[pRightNodes]);
                         int pNodes(pId);
                         Vnode_nplus1(pNodes) = nodeVelocityBoundaryCondition(
                             cdl->rightBC, cdl->rightBCValue, Mnode(pNodes),
                             G(pNodes));
                       });
  auto topNodes(mesh->getTopNodes());
  int nbTopNodes(mesh->getNbTopNodes());
  Kokkos::parallel_for("computeBoundaryNodeVelocities", nbTopNodes,
                       KOKKOS_LAMBDA(const int& pTopNodes) {
                         int pId(topNodes[pTopNodes]);
                         int pNodes(pId);
                         Vnode_nplus1(pNodes) = nodeVelocityBoundaryCondition(
                             cdl->topBC, cdl->topBCValue, Mnode(pNodes),
                             G(pNodes));
                       });
  auto bottomNodes(mesh->getBottomNodes());
  int nbBottomNodes(mesh->getNbBottomNodes());
  Kokkos::parallel_for("computeBoundaryNodeVelocities", nbBottomNodes,
                       KOKKOS_LAMBDA(const int& pBottomNodes) {
                         int pId(bottomNodes[pBottomNodes]);
                         int pNodes(pId);
                         Vnode_nplus1(pNodes) = nodeVelocityBoundaryCondition(
                             cdl->bottomBC, cdl->bottomBCValue, Mnode(pNodes),
                             G(pNodes));
                       });
  auto topLeftNode(mesh->getTopLeftNode());
  int nbTopLeftNode(mesh->getNbTopLeftNode());
  Kokkos::parallel_for("computeBoundaryNodeVelocities", nbTopLeftNode,
                       KOKKOS_LAMBDA(const int& pTopLeftNode) {
                         int pId(topLeftNode[pTopLeftNode]);
                         int pNodes(pId);
                         Vnode_nplus1(pNodes) =
                             nodeVelocityBoundaryConditionCorner(
                                 cdl->topBC, cdl->topBCValue, cdl->leftBC,
                                 cdl->leftBCValue, Mnode(pNodes), G(pNodes));
                       });
  auto topRightNode(mesh->getTopRightNode());
  int nbTopRightNode(mesh->getNbTopRightNode());
  Kokkos::parallel_for("computeBoundaryNodeVelocities", nbTopRightNode,
                       KOKKOS_LAMBDA(const int& pTopRightNode) {
                         int pId(topRightNode[pTopRightNode]);
                         int pNodes(pId);
                         Vnode_nplus1(pNodes) =
                             nodeVelocityBoundaryConditionCorner(
                                 cdl->topBC, cdl->topBCValue, cdl->rightBC,
                                 cdl->rightBCValue, Mnode(pNodes), G(pNodes));
                       });
  auto bottomLeftNode(mesh->getBottomLeftNode());
  int nbBottomLeftNode(mesh->getNbBottomLeftNode());
  Kokkos::parallel_for("computeBoundaryNodeVelocities", nbBottomLeftNode,
                       KOKKOS_LAMBDA(const int& pBottomLeftNode) {
                         int pId(bottomLeftNode[pBottomLeftNode]);
                         int pNodes(pId);
                         Vnode_nplus1(pNodes) =
                             nodeVelocityBoundaryConditionCorner(
                                 cdl->bottomBC, cdl->bottomBCValue, cdl->leftBC,
                                 cdl->leftBCValue, Mnode(pNodes), G(pNodes));
                       });
  auto bottomRightNode(mesh->getBottomRightNode());
  int nbBottomRightNode(mesh->getNbBottomRightNode());
  Kokkos::parallel_for(
      "computeBoundaryNodeVelocities", nbBottomRightNode,
      KOKKOS_LAMBDA(const int& pBottomRightNode) {
        int pId(bottomRightNode[pBottomRightNode]);
        int pNodes(pId);
        Vnode_nplus1(pNodes) = nodeVelocityBoundaryConditionCorner(
            cdl->bottomBC, cdl->bottomBCValue, cdl->rightBC, cdl->rightBCValue,
            Mnode(pNodes), G(pNodes));
      });
}
KOKKOS_INLINE_FUNCTION
RealArray1D<dim> EucclhydRemap::nodeVelocityBoundaryCondition(
    int BC, RealArray1D<dim> BCValue, RealArray2D<dim, dim> Mp,
    RealArray1D<dim> Gp) {
  if (BC == 200)
    return (dot(Gp, BCValue) /
            (dot(MathFunctions::matVectProduct(Mp, BCValue),
                                BCValue)) * BCValue);
  else if (BC == 201)
    return BCValue;
  else if (BC == 202)
    return MathFunctions::matVectProduct(inverse(Mp), Gp);

  return zeroVect;  // inutile juste pour eviter le warning de compilation
}

KOKKOS_INLINE_FUNCTION
RealArray1D<dim> EucclhydRemap::nodeVelocityBoundaryConditionCorner(
    int BC1, RealArray1D<dim> BCValue1, int BC2, RealArray1D<dim> BCValue2,
    RealArray2D<dim, dim> Mp, RealArray1D<dim> Gp) {
  if (BC1 == 200 && BC2 == 200) {
    if (MathFunctions::fabs(
            MathFunctions::fabs(dot(BCValue1, BCValue2)) -
            MathFunctions::norm(BCValue1) * MathFunctions::norm(BCValue2)) <
        1.0E-8)
      return (
          dot(Gp, BCValue1) /
              (dot(MathFunctions::matVectProduct(Mp, BCValue1),
                                  BCValue1)) * BCValue1);
    else {
      return zeroVect;
    }
  } else if (BC1 == 201 && BC2 == 201) {
    return (0.5 * (BCValue1 + BCValue2));
  } else if (BC1 == 202 && BC2 == 202) {
    return MathFunctions::matVectProduct(inverse(Mp), Gp);
  } else {
    { return zeroVect; }
  }
}

RealArray1D<nbequamax> EucclhydRemap::computeBoundaryFluxes(
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
    return computeRemapFlux(
        options->projectionOrder, limiteurs->projectionAvecPlateauPente,
        faceNormalVelocity(fbFaces), faceNormal(fbFaces), faceLength(fbFaces),
        phiFace_fFaces, outerFaceNormal(cCells, fbFacesOfCellC), exy,
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
    return computeRemapFlux(
        options->projectionOrder, limiteurs->projectionAvecPlateauPente,
        faceNormalVelocity(ftFaces), faceNormal(ftFaces), faceLength(ftFaces),
        phiFace_fFaces, outerFaceNormal(cCells, ftFacesOfCellC), exy,
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
        return computeRemapFlux(
            options->projectionOrder, limiteurs->projectionAvecPlateauPente,
            faceNormalVelocity(flFaces), faceNormal(flFaces),
            faceLength(flFaces), phiFace_fFaces,
            outerFaceNormal(cCells, flFacesOfCellC), exy, gt->deltat_n);
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

        if (proj == 1 &&
            (cCells == dbgcell1 || cCells == dbgcell2 || cCells == dbgcell3)) {
          std::cout << " AP 1 Right cell   " << exy << " " << cCells << " "
                    << flFaces << " " << phiFace1(flFaces) << " " << frFaces
                    << " " << phiFace_fFaces << std::endl;
          std::cout << " faceNormalVelocity " << faceNormalVelocity(flFaces)
                    << " " << faceNormalVelocity(frFaces) << std::endl;
          std::cout << " faceLength " << faceLength(flFaces) << " "
                    << faceLength(frFaces) << std::endl;
          std::cout << " outerFaceNormal(cCells,frFacesOfCellC) "
                    << outerFaceNormal(cCells, frFacesOfCellC)
                    << " frFacesOfCellC " << frFacesOfCellC << std::endl;
        }

        if (proj == 2 &&
            (cCells == dbgcell1 || cCells == dbgcell2 || cCells == dbgcell3)) {
          std::cout << " AP 2 Right cell   " << exy << " " << cCells << " "
                    << flFaces << " " << phiFace2(flFaces) << " " << frFaces
                    << " " << phiFace_fFaces << std::endl;
          std::cout << " faceNormalVelocity " << faceNormalVelocity(flFaces)
                    << " " << faceNormalVelocity(frFaces) << std::endl;
          std::cout << " faceLength " << faceLength(flFaces) << " "
                    << faceLength(frFaces) << std::endl;
          std::cout << " outerFaceNormal(cCells,frFacesOfCellC) "
                    << outerFaceNormal(cCells, frFacesOfCellC)
                    << " frFacesOfCellC " << frFacesOfCellC << std::endl;
        }
        //
        return computeRemapFlux(
            options->projectionOrder, limiteurs->projectionAvecPlateauPente,
            faceNormalVelocity(frFaces), faceNormal(frFaces),
            faceLength(frFaces), phiFace_fFaces,
            outerFaceNormal(cCells, frFacesOfCellC), exy, gt->deltat_n);
      }
    }
  }
  return phiFace_fFaces;  // Uzero;
}
