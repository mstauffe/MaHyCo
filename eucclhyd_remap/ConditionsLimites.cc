#include <Kokkos_Core.hpp>
#include <algorithm>  // for copy
#include <array>      // for array
#include <iostream>   // for operator<<, basic_ostream::operat...
#include <vector>     // for allocator, vector

#include "Eucclhyd.h"          // for Eucclhyd, Eucclhyd::Opt...
#include "../remap/UtilesRemap-Impl.h"  // for Remap::computeRemapFlux
#include "mesh/CartesianMesh2D.h"   // for CartesianMesh2D
#include "types/MathFunctions.h"    // for dot, matVectProduct, norm
#include "types/MultiArray.h"       // for operator<<
#include "utils/Utils.h"            // for indexOf

/**
 * Job computeBoundaryNodeVelocities called @4.0 in executeTimeLoopN method.
 * In variables: G, Mnode, bottomBC, bottomBCValue, leftBC, leftBCValue,
 * rightBC, rightBCValue, topBC, topBCValue Out variables: Vnode_nplus1
 */
void Eucclhyd::computeBoundaryNodeVelocities() noexcept {
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
RealArray1D<dim> Eucclhyd::nodeVelocityBoundaryCondition(
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
RealArray1D<dim> Eucclhyd::nodeVelocityBoundaryConditionCorner(
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


