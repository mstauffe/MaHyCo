#include <math.h>  // for floor, sqrt

#include <Kokkos_Core.hpp>  // for deep_copy
#include <algorithm>        // for copy
#include <array>            // for array
#include <iostream>         // for operator<<, basic_ostream::ope...
#include <vector>           // for allocator, vector

#include "Init.h"

#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "types/MathFunctions.h"   // for max, norm, dot
#include "types/MultiArray.h"      // for operator<<
#include "utils/Utils.h"           // for indexOf

namespace initlib {
/**
 * Job initVpAndFpc called @1.0 in simulate method.
 * In variables: zeroVect
 * Out variables: m_node_force_n0, m_node_velocity_n0
 */
void Initialisations::initVpAndFpc() noexcept {
  Kokkos::parallel_for(
      "initVpAndFpc", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
        int pId(pNodes);
        {
          auto cellsOfNodeP(mesh->getCellsOfNode(pId));
          for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
               cCellsOfNodeP++) {
            m_node_velocity_n0(pNodes) = zeroVect;
            m_node_force_n0(pNodes, cCellsOfNodeP) = zeroVect;
          }
        }
      });
}
/**
 * Job initPseudo called @3.0 in simulate method.
 * In variables: m_density_n0
 * Out variables: m_divu_n0, m_tau_density_n0
 */
void Initialisations::initPseudo() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    m_tau_density_n0(cCells) = 1 / m_density_n0(cCells);
    m_divu_n0(cCells) = 0.0;
    m_pseudo_viscosity_n0(cCells) = 0.0;
    for (int imat = 0; imat < nbmatmax; imat++) {
      m_tau_density_env_n0(cCells)[imat] = 1 / m_density_env_n0(cCells)[imat];
      m_pseudo_viscosity_env_n0(cCells)[imat] = 0.0;
    }
  });
}
/**
 * Job initSubVol called @2.0 in simulate method.
 * In variables: m_node_coord_n0, m_cell_coord_n0
 * Out variables: m_node_cellvolume_n0
 */
void Initialisations::initSubVol() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    const Id cId(cCells);
    {
      const auto nodesOfCellC(mesh->getNodesOfCell(cId));
      const size_t nbNodesOfCellC(nodesOfCellC.size());
      for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
           pNodesOfCellC++) {
        const Id pMinus1Id(
            nodesOfCellC[(pNodesOfCellC - 1 + nbNodesOfCell) % nbNodesOfCell]);
        const Id pId(nodesOfCellC[pNodesOfCellC]);
        const Id pPlus1Id(
            nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) % nbNodesOfCell]);
        const size_t pMinus1Nodes(pMinus1Id);
        const size_t pNodes(pId);
        const size_t pPlus1Nodes(pPlus1Id);
        const RealArray1D<2> x1(m_cell_coord_n0(cCells));
        const RealArray1D<2> x2(
            0.5 * (m_node_coord_n0(pMinus1Nodes) + m_node_coord_n0(pNodes)));
        const RealArray1D<2> x3(m_node_coord_n0(pNodes));
        const RealArray1D<2> x4(
            0.5 * (m_node_coord_n0(pPlus1Nodes) + m_node_coord_n0(pNodes)));
        m_node_cellvolume_n0(cCells, pNodesOfCellC) =
            0.5 * (crossProduct2d(x1, x2) + crossProduct2d(x2, x3) +
                   crossProduct2d(x3, x4) + crossProduct2d(x4, x1));
      }
    }
    m_euler_volume_n0(cCells) = cstmesh->X_EDGE_LENGTH * cstmesh->Y_EDGE_LENGTH;
  });
}
} // namespace initlib
