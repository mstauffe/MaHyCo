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
 * Job initCellInternalEnergy called @2.0 in simulate method.
 * In variables: NohTestCase, SedovTestCase, SodCase, TriplePoint, X,
 * X_EDGE_LENGTH, m_cell_coord_n0, Y_EDGE_LENGTH, testCase, threshold Out
 * variables: m_internal_energy_n0
 */
void Initialisations::initCellInternalEnergy() noexcept {
  Kokkos::parallel_for("initCellInternalEnergy", nbCells,
                       KOKKOS_LAMBDA(const int& cCells) {
                         for (int imat = 0; imat < nbmatmax; imat++) {
                           m_internal_energy_env_n0(cCells)[imat] = 0.0;
                         }
                       });

  if (test->Nom == test->NohTestCase || test->Nom == test->BiNohTestCase ||
      test->Nom == test->UnitTestCase || test->Nom == test->BiUnitTestCase) {
    double p0(1.);
    double rho0(1.);
    double gamma(1.4);
    double e0 = p0 / ((gamma - 1.0) * rho0);
    Kokkos::parallel_for("initCellInternalEnergy", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           m_internal_energy_n0(cCells) = e0;
                           for (int imat = 0; imat < nbmatmax; imat++)
                             m_internal_energy_env_n0(cCells)[imat] = e0;
                         });

    
  } else if (test->Nom == test->SodCaseX || test->Nom == test->SodCaseY) {
    Kokkos::parallel_for("initCellInternalEnergy", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           double pInit;
                           double rhoInit;
                           double eInit;
                           double r(0.);
                           if (test->Nom == test->SodCaseX)
                             r = m_cell_coord_n0(cCells)[0];
                           if (test->Nom == test->SodCaseY)
                             r = m_cell_coord_n0(cCells)[1];
                           if (r <= 0.5) {
                             pInit = 1.0;
                             rhoInit = 1.0;
                           } else {
                             pInit = 0.1;
                             rhoInit = 0.125;
                           }
                           eInit = pInit / ((eos->gammap[0] - 1.0) * rhoInit);
                           m_internal_energy_n0(cCells) = eInit;
                           m_internal_energy_env_n0(cCells)[0] = eInit;
                         });
  } else if (test->Nom == test->BiSodCaseX || test->Nom == test->BiSodCaseY) {
    Kokkos::parallel_for("initCellInternalEnergy", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           double pInit;
                           double rhoInit;
                           double eInit;
                           double r = 0.;
                           // r=
                           // sqrt(m_cell_coord_n0(cCells)[0]*m_cell_coord_n0(cCells)[0]+m_cell_coord_n0(cCells)[1]*m_cell_coord_n0(cCells)[1]);
                           // en rayon
                           if (test->Nom == test->BiSodCaseX)
                             r = m_cell_coord_n0(cCells)[0];
                           if (test->Nom == test->BiSodCaseY)
                             r = m_cell_coord_n0(cCells)[1];
                           if (r <= 0.5) {
                             pInit = 1.0;
                             rhoInit = 1.0;
                             eInit = pInit / ((eos->gammap[0] - 1.0) * rhoInit);
                             m_internal_energy_env_n0(cCells)[0] = eInit;
                             m_internal_energy_env_n0(cCells)[1] = 0.;
                             m_internal_energy_n0(cCells) = eInit;
                             // std::cout << " cell " << cCells << "  e1= " <<
                             // m_internal_energy_env_n0(cCells)[0]
                             // << "  e2= " <<
                             // m_internal_energy_env_n0(cCells)[0] <<
                             // std::endl;
                           } else {
                             pInit = 0.1;
                             rhoInit = 0.125;
                             eInit = pInit / ((eos->gammap[1] - 1.0) * rhoInit);
                             m_internal_energy_env_n0(cCells)[0] = 0.;
                             m_internal_energy_env_n0(cCells)[1] = eInit;
                             m_internal_energy_n0(cCells) = eInit;
                             // std::cout << " cell " << cCells << "  e1= " <<
                             // m_internal_energy_env_n0(cCells)[0]
                             // << "  e2= " <<
                             // m_internal_energy_env_n0(cCells)[0] <<
                             // std::endl;
                           }
                         });
  } else if (test->Nom == test->BiShockBubble) {
    Kokkos::parallel_for(
        "initCellInternalEnergy", nbCells, KOKKOS_LAMBDA(const int& cCells) {
          double pInit;
          double rhoInit;
          double eInit;
          // Air partout
          rhoInit = 1.0;
          pInit = 1.e5;
          eInit = pInit / ((eos->gammap[0] - 1.0) * rhoInit);
          m_internal_energy_env_n0(cCells)[0] = eInit;
          m_internal_energy_env_n0(cCells)[1] = 0.;
          m_internal_energy_n0(cCells) = eInit;
          // bulle surchargera l'aire
          // centre de la bulle
          RealArray1D<dim> Xb = {{0.320, 0.04}};
          double rb = 0.025;
          double r = sqrt((m_cell_coord_n0(cCells)[0] - Xb[0]) *
                              (m_cell_coord_n0(cCells)[0] - Xb[0]) +
                          (m_cell_coord_n0(cCells)[1] - Xb[1]) *
                              (m_cell_coord_n0(cCells)[1] - Xb[1]));

          if (r <= rb) {
            pInit = 1.e5;
            rhoInit = 0.182;
            eInit = pInit / ((eos->gammap[0] - 1.0) * rhoInit);
            m_internal_energy_env_n0(cCells)[0] = 0.;
            m_internal_energy_env_n0(cCells)[1] = eInit;
            m_internal_energy_n0(cCells) = eInit;
          }
        });
  } else if (test->Nom == test->SedovTestCase ||
             test->Nom == test->BiSedovTestCase) {
    Kokkos::parallel_for(
        "initCellInternalEnergy", nbCells, KOKKOS_LAMBDA(const int& cCells) {
          int cId(cCells);
          bool isCenterCell = false;
          double pInit = 1.e-6;
          double rhoInit = 1.;
          double rmin = options->threshold;  // depot sur 1 maille
          double e1 = pInit / ((eos->gammap[0] - 1.0) * rhoInit);
          {
            auto nodesOfCellC(mesh->getNodesOfCell(cId));
            for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
                 pNodesOfCellC++) {
              int pId(nodesOfCellC[pNodesOfCellC]);
              int pNodes(pId);
              if (MathFunctions::norm(m_node_coord_n0(pNodes)) < rmin)
                isCenterCell = true;
            }
          }
          if (isCenterCell) {
            double total_energy_deposit = 0.244816;
            double dx = cstmesh->X_EDGE_LENGTH;
            double dy = cstmesh->Y_EDGE_LENGTH;
            m_internal_energy_n0(cCells) =
                e1 + total_energy_deposit / (dx * dy);
          } else {
            m_internal_energy_n0(cCells) = e1;
          }
          m_internal_energy_env_n0(cCells)[0] = m_internal_energy_n0(cCells);
        });
  } else if (test->Nom == test->TriplePoint) {
    Kokkos::parallel_for("initCellInternalEnergy", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           double pInit;
                           double rhoInit;
                           double eInit;
                           if (m_cell_coord_n0(cCells)[0] <= 0.01) {
                             pInit = 1.0;
                             rhoInit = 1.0;
                           } else {
                             if (m_cell_coord_n0(cCells)[1] <= 0.015) {
                               pInit = 0.1;
                               rhoInit = 1.;
                             } else {
                               pInit = 0.1;
                               rhoInit = 0.1;
                             }
                           }
                           eInit = pInit / ((eos->gammap[0] - 1.0) * rhoInit);
                           m_internal_energy_n0(cCells) = eInit;
                           m_internal_energy_env_n0(cCells)[0] = eInit;
                         });
  } else if (test->Nom == test->BiTriplePoint) {
    Kokkos::parallel_for(
        "initCellInternalEnergy", nbCells, KOKKOS_LAMBDA(const int& cCells) {
          double pInit;
          double rhoInit;
          double eInit;
          if (m_cell_coord_n0(cCells)[0] <= 0.01) {
            pInit = 1.0;  // 1.e5; // 1.0;
            rhoInit = 1.0;
            eInit = pInit / ((eos->gammap[0] - 1.0) * rhoInit);
            m_internal_energy_n0(cCells) = eInit;
            m_internal_energy_env_n0(cCells)[0] = eInit;
          } else {
            if (m_cell_coord_n0(cCells)[1] <= 0.015) {
              pInit = 0.1;  // 1.e4; // 0.1;
              rhoInit = 1.;
              eInit = pInit / ((eos->gammap[1] - 1.0) * rhoInit);
              m_internal_energy_n0(cCells) = eInit;
              m_internal_energy_env_n0(cCells)[1] = eInit;
            } else {
              pInit = 0.1;  // 1.e4; // 0.1;
              rhoInit = 0.1;
              eInit = pInit / ((eos->gammap[2] - 1.0) * rhoInit);
              m_internal_energy_n0(cCells) = eInit;
              m_internal_energy_env_n0(cCells)[2] = eInit;
            }
          }
        });
  }
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
