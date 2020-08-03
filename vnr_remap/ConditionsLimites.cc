#include <Kokkos_Core.hpp>
#include "VnrRemap.h"               // for VnrRemap
#include "../includes/Freefunctions.h"
#include "mesh/CartesianMesh2D.h"   // for CartesianMesh2D
#include "utils/Utils.h"  // for Indexof

/**
 * Job updateVelocity called @2.0 in executeTimeLoopN method.
 * In variables: C, Q_nplus1, deltat_n, deltat_nplus1, m, p_n, u_n
 * Out variables: u_nplus1
 */
void VnrRemap::updateVelocityBoundaryConditions() noexcept
{
  const double dt(0.5 * (gt->deltat_nplus1 + gt->deltat_n));
  if (cdl->bottomBC == cdl->symmetry)
    {
      const auto bottomNodes(mesh->getBottomNodes());
      const size_t nbBottomNodes(bottomNodes.size());
      Kokkos::parallel_for(nbBottomNodes, KOKKOS_LAMBDA(const size_t& pBottomNodes)
			   {
			     const Id pId(bottomNodes[pBottomNodes]);
			     const size_t pNodes(pId);
			     RealArray1D<2> reduction1({0.0, 0.0});
			     {
			       const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
			       const size_t nbCellsOfNodeP(cellsOfNodeP.size());
			       for (size_t cCellsOfNodeP=0; cCellsOfNodeP<nbCellsOfNodeP; cCellsOfNodeP++)
				 {
				   const Id cId(cellsOfNodeP[cCellsOfNodeP]);
				   const size_t cCells(cId);
				   const size_t pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
				   reduction1 = sumR1(reduction1, (p_n(cCells) + Q_n(cCells)) * (
						C(cCells,pNodesOfCellC)
						+ symmetricVector(C(cCells,pNodesOfCellC), {1.0, 0.0})));
				 }
			     }
			     u_nplus1(pNodes) = u_n(pNodes) + dt / (m(pNodes)) * reduction1;
			   });
    }
  if (cdl->topBC == cdl->symmetry)
    {
      const auto topNodes(mesh->getTopNodes());
      const size_t nbTopNodes(topNodes.size());
      Kokkos::parallel_for(nbTopNodes, KOKKOS_LAMBDA(const size_t& pTopNodes)
			   {
			     const Id pId(topNodes[pTopNodes]);
			     const size_t pNodes(pId);
			     RealArray1D<2> reduction2({0.0, 0.0});
			     {
			       const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
			       const size_t nbCellsOfNodeP(cellsOfNodeP.size());
			       for (size_t cCellsOfNodeP=0; cCellsOfNodeP<nbCellsOfNodeP; cCellsOfNodeP++)
				 {
				   const Id cId(cellsOfNodeP[cCellsOfNodeP]);
				   const size_t cCells(cId);
				   const size_t pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
				   reduction2 = sumR1(reduction2, (p_n(cCells) + Q_n(cCells)) * (
						C(cCells,pNodesOfCellC)
						+ symmetricVector(C(cCells,pNodesOfCellC), {1.0, 0.0})));
				 }
			     }
			     u_nplus1(pNodes) = u_n(pNodes) + dt / (m(pNodes)) * reduction2;
			   });
    }
  if (cdl->leftBC == cdl->symmetry)
    {
      const auto leftNodes(mesh->getLeftNodes());
      const size_t nbLeftNodes(leftNodes.size());
      Kokkos::parallel_for(nbLeftNodes, KOKKOS_LAMBDA(const size_t& pLeftNodes)
			   {
			     const Id pId(leftNodes[pLeftNodes]);
			     const size_t pNodes(pId);
			     RealArray1D<2> reduction3({0.0, 0.0});
			     {
			       const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
			       const size_t nbCellsOfNodeP(cellsOfNodeP.size());
			       for (size_t cCellsOfNodeP=0; cCellsOfNodeP<nbCellsOfNodeP; cCellsOfNodeP++)
				 {
				   const Id cId(cellsOfNodeP[cCellsOfNodeP]);
				   const size_t cCells(cId);
				   const size_t pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
				   reduction3 = sumR1(reduction3, (p_n(cCells) + Q_n(cCells)) * (
						C(cCells,pNodesOfCellC)
						+ symmetricVector(C(cCells,pNodesOfCellC), {0.0, 1.0})));
				 }
			     }
			     u_nplus1(pNodes) = u_n(pNodes) + dt / (m(pNodes)) * reduction3;
			   });
    }
  if (cdl->rightBC == cdl->symmetry)
    {
      const auto rightNodes(mesh->getRightNodes());
      const size_t nbRightNodes(rightNodes.size());
      Kokkos::parallel_for(nbRightNodes, KOKKOS_LAMBDA(const size_t& pRightNodes)
			   {
			     const Id pId(rightNodes[pRightNodes]);
			     const size_t pNodes(pId);
			     RealArray1D<2> reduction4({0.0, 0.0});
			     {
			       const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
			       const size_t nbCellsOfNodeP(cellsOfNodeP.size());
			       for (size_t cCellsOfNodeP=0; cCellsOfNodeP<nbCellsOfNodeP; cCellsOfNodeP++)
				 {
				   const Id cId(cellsOfNodeP[cCellsOfNodeP]);
				   const size_t cCells(cId);
				   const size_t pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
				   reduction4 = sumR1(reduction4, (p_n(cCells) + Q_n(cCells)) * (
						C(cCells,pNodesOfCellC)
						+ symmetricVector(C(cCells,pNodesOfCellC), {0.0, 1.0})));
				 }
			     }
			     u_nplus1(pNodes) = u_n(pNodes) + dt / (m(pNodes)) * reduction4;
			   });
    }
  if (cdl->topBC == cdl->symmetry && cdl->leftBC == cdl->symmetry)
    {
      const auto topLeftNode(mesh->getTopLeftNode());
      const size_t nbTopLeftNode(mesh->getNbTopLeftNode());
      Kokkos::parallel_for("computeBoundaryNodeVelocities", nbTopLeftNode,
			   KOKKOS_LAMBDA(const int& pTopLeftNode) {
			     int pId(topLeftNode[pTopLeftNode]);
			     int pNodes(pId);
			     u_nplus1(pNodes) = zeroVect;
			   });
    }
  if (cdl->topBC == cdl->symmetry && cdl->rightBC == cdl->symmetry)
    {
      const auto topRightNode(mesh->getTopRightNode());
      const size_t nbTopRightNode(mesh->getNbTopRightNode());
      Kokkos::parallel_for("computeBoundaryNodeVelocities", nbTopRightNode,
			   KOKKOS_LAMBDA(const int& pTopRightNode) {
			     int pId(topRightNode[pTopRightNode]);
			     int pNodes(pId);
			     u_nplus1(pNodes) = zeroVect;
			   });
    }
  if (cdl->bottomBC == cdl->symmetry && cdl->leftBC == cdl->symmetry)
    {
      const auto bottomLeftNode(mesh->getBottomLeftNode());
      const size_t nbBottomLeftNode(mesh->getNbBottomLeftNode());
      Kokkos::parallel_for("computeBoundaryNodeVelocities", nbBottomLeftNode,
			   KOKKOS_LAMBDA(const int& pBottomLeftNode) {
			     int pId(bottomLeftNode[pBottomLeftNode]);
			     int pNodes(pId);
			     u_nplus1(pNodes) = zeroVect;
			   });
    }
  if (cdl->bottomBC == cdl->symmetry && cdl->rightBC == cdl->symmetry)
    {
      const auto bottomRightNode(mesh->getBottomRightNode());
      const size_t nbBottomRightNode(mesh->getNbBottomRightNode());
      Kokkos::parallel_for(
			   "computeBoundaryNodeVelocities", nbBottomRightNode,
			   KOKKOS_LAMBDA(const int& pBottomRightNode) {
			     int pId(bottomRightNode[pBottomRightNode]);
			     int pNodes(pId);
			     u_nplus1(pNodes) = zeroVect;			     
			   });
    }
}
