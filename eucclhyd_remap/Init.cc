#include <math.h>  // for floor, sqrt

#include <Kokkos_Core.hpp>  // for deep_copy
#include <algorithm>        // for copy
#include <array>            // for array
#include <iostream>         // for operator<<, basic_ostream::ope...
#include <vector>           // for allocator, vector

#include "Eucclhyd.h"              // for Eucclhyd, Eucclhyd::...
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "types/MathFunctions.h"   // for max, norm, dot
#include "types/MultiArray.h"      // for operator<<
#include "utils/Utils.h"           // for indexOf

void Eucclhyd::initBoundaryConditions() noexcept {
  if (test->Nom == test->SodCaseX || test->Nom == test->SodCaseY ||
      test->Nom == test->BiSodCaseX || test->Nom == test->BiSodCaseY) {
    // maillage 200 5 0.005 0.02
    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->symmetry;
    cdl->rightBCValue = ey;

    cdl->topBC = cdl->symmetry;
    cdl->topBCValue = ex;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;
  } else if (test->Nom == test->BiShockBubble) {
    // maillage 520 64 0.00125 0.00125
    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->imposedVelocity;
    cdl->rightBCValue = {{-124.824, 0.0}};
    cdl->rightFluxBC = 1;
    cdl->rightFluxBCValue = {
        {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -124.824, 0.0, 250000}};

    cdl->topBC = cdl->symmetry;
    cdl->topBCValue = ex;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;

  } else if (test->Nom == test->SedovTestCase ||
             test->Nom == test->BiSedovTestCase) {
    // const ℕ leftBC = symmetry; const ℝ[2] leftBCValue = ey;
    // const ℕ rightBC = imposedVelocity; const ℝ[2] rightBCValue = zeroVect;
    // const ℕ topBC = imposedVelocity; const ℝ[2] topBCValue = zeroVect;
    // const ℕ bottomBC = symmetry; const ℝ[2] bottomBCValue = ex;
    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->imposedVelocity;
    cdl->rightBCValue = zeroVect;

    cdl->topBC = cdl->imposedVelocity;
    cdl->topBCValue = zeroVect;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;

  } else if (test->Nom == test->TriplePoint ||
             test->Nom == test->BiTriplePoint) {
    // maillage 140 60 0.0005 0.0005
    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->symmetry;
    cdl->rightBCValue = ey;

    cdl->topBC = cdl->symmetry;
    cdl->topBCValue = ex;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;
  } else if (test->Nom == test->NohTestCase ||
             test->Nom == test->BiNohTestCase) {
    // const ℕ leftBC = symmetry; const ℝ[2] leftBCValue = ey;
    // const ℕ rightBC = imposedVelocity; const ℝ[2] rightBCValue = zeroVect;
    // const ℕ topBC = imposedVelocity; const ℝ[2] topBCValue = zeroVect;
    // const ℕ bottomBC = symmetry; const ℝ[2] bottomBCValue = ex;

    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->imposedVelocity;
    cdl->rightBCValue = zeroVect;

    cdl->topBC = cdl->imposedVelocity;
    cdl->topBCValue = zeroVect;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;
  }
  cdl->FluxBC =
      cdl->leftFluxBC + cdl->rightFluxBC + cdl->bottomFluxBC + cdl->topFluxBC;
}
/**
 * Job initMeshGeometryForCells called @1.0 in simulate method.
 * In variables: X
 * Out variables: m_cell_coord, m_cell_coord_x, m_cell_coord_y,
 * m_cell_perimeter, v
 */
void Eucclhyd::initMeshGeometryForCells() noexcept {
  Kokkos::parallel_for(
      "initMeshGeometryForCells", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        int cId(cCells);
        double reduction11 = 0.0;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            reduction11 =
                reduction11 + (crossProduct2d(m_node_coord(pNodes),
                                              m_node_coord(pPlus1Nodes)));
          }
        }
        double vol = 0.5 * reduction11;
        RealArray1D<dim> reduction12 = zeroVect;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            reduction12 =
                reduction12 +
                ((crossProduct2d(m_node_coord(pNodes),
                                 m_node_coord(pPlus1Nodes)) *
                  ((m_node_coord(pNodes) + m_node_coord(pPlus1Nodes)))));
          }
        }
        RealArray1D<dim> xc = (1.0 / (6.0 * vol) * reduction12);
        m_cell_coord(cCells) = xc;
        m_cell_coord_x(cCells) = xc[0];
        m_cell_coord_y(cCells) = xc[1];
        m_euler_volume(cCells) = vol;
        particules->m_particlecell_euler_volume(cCells) = vol;
        double reduction13 = 0.0;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            reduction13 = reduction13 +
                          (MathFunctions::norm((m_node_coord(pNodes) -
                                                m_node_coord(pPlus1Nodes))));
          }
        }
        m_cell_perimeter(cCells) = reduction13;
      });
}

/**
 * Job initVpAndFpc called @1.0 in simulate method.
 * In variables: zeroVect
 * Out variables: m_node_force_n0, m_node_velocity_n0
 */
void Eucclhyd::initVpAndFpc() noexcept {
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
 * X_EDGE_LENGTH, m_cell_coord, Y_EDGE_LENGTH, testCase, threshold Out
 * variables: m_internal_energy_n0
 */
void Eucclhyd::initCellInternalEnergy() noexcept {
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
                             r = m_cell_coord(cCells)[0];
                           if (test->Nom == test->SodCaseY)
                             r = m_cell_coord(cCells)[1];
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
                           // sqrt(m_cell_coord(cCells)[0]*m_cell_coord(cCells)[0]+m_cell_coord(cCells)[1]*m_cell_coord(cCells)[1]);
                           // en rayon
                           if (test->Nom == test->BiSodCaseX)
                             r = m_cell_coord(cCells)[0];
                           if (test->Nom == test->BiSodCaseY)
                             r = m_cell_coord(cCells)[1];
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
          double r = sqrt((m_cell_coord(cCells)[0] - Xb[0]) *
                              (m_cell_coord(cCells)[0] - Xb[0]) +
                          (m_cell_coord(cCells)[1] - Xb[1]) *
                              (m_cell_coord(cCells)[1] - Xb[1]));

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
              if (MathFunctions::norm(m_node_coord(pNodes)) < rmin)
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
                           if (m_cell_coord(cCells)[0] <= 0.01) {
                             pInit = 1.0;
                             rhoInit = 1.0;
                           } else {
                             if (m_cell_coord(cCells)[1] <= 0.015) {
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
          if (m_cell_coord(cCells)[0] <= 0.01) {
            pInit = 1.0;  // 1.e5; // 1.0;
            rhoInit = 1.0;
            eInit = pInit / ((eos->gammap[0] - 1.0) * rhoInit);
            m_internal_energy_n0(cCells) = eInit;
            m_internal_energy_env_n0(cCells)[0] = eInit;
          } else {
            if (m_cell_coord(cCells)[1] <= 0.015) {
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
 * Job initCellVelocity called @2.0 in simulate method.
 * In variables: NohTestCase, SedovTestCase, SodCase, m_cell_coord, testCase
 * Out variables: m_cell_velocity_n0
 */
void Eucclhyd::initCellVelocity() noexcept {
  Kokkos::parallel_for(
      "initCellVelocity", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        if (test->Nom == test->NohTestCase ||
            test->Nom == test->BiNohTestCase) {
          double u0(0.);
          double n1 = m_cell_coord(cCells)[0];
          double n2 = m_cell_coord(cCells)[1];
          double normVect = MathFunctions::sqrt(n1 * n1 + n2 * n2);
          m_cell_velocity_n0(cCells)[0] = -u0 * n1 / normVect;
          m_cell_velocity_n0(cCells)[1] = -u0 * n2 / normVect;
        } else if (test->Nom == test->SedovTestCase ||
                   test->Nom == test->SodCaseX || test->Nom == test->SodCaseY ||
                   test->Nom == test->TriplePoint ||
                   test->Nom == test->BiSedovTestCase ||
                   test->Nom == test->BiSodCaseX ||
                   test->Nom == test->BiSodCaseY ||
                   test->Nom == test->BiTriplePoint) {
          m_cell_velocity_n0(cCells)[0] = 0.0;
          m_cell_velocity_n0(cCells)[1] = 0.0;
        } else if (test->Nom == test->BiShockBubble) {
          if (m_cell_coord(cCells)[0] >= 0.60) {
            m_cell_velocity_n0(cCells)[0] = -124.824;
            m_cell_velocity_n0(cCells)[1] = 0.0;
          }
        } else if (test->Nom == test->UnitTestCase ||
                   test->Nom == test->BiUnitTestCase) {
          m_cell_velocity_n0(cCells)[0] = 1.0;
          m_cell_velocity_n0(cCells)[1] = 0.0;
        }
      });
}

/**
 * Job initDensity called @2.0 in simulate method.
 * In variables: SodCase, m_cell_coord, testCase
 * Out variables: m_density_n0, m_fracvol_env, m_mass_fraction_env
 */
void Eucclhyd::initDensity() noexcept {
  Kokkos::parallel_for("initDensity", nbCells,
                       KOKKOS_LAMBDA(const int& cCells) {
                         for (int imat = 0; imat < nbmatmax; imat++) {
                           m_fracvol_env(cCells)[imat] = 0.0;
                           m_mass_fraction_env(cCells)[imat] = 0.0;
                           m_density_env_n0(cCells)[imat] = 0.0;
                         }
                       });
  if (test->Nom == test->UnitTestCase) {
    Kokkos::parallel_for("initDensity", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           if (m_cell_coord(cCells)[0] <= 0.5) {
                             m_fracvol_env(cCells)[0] = 1.;
                             m_fracvol_env(cCells)[1] = 0.;
                             m_mass_fraction_env(cCells)[0] = 1.;
                             m_mass_fraction_env(cCells)[1] = 0.;
                             m_density_n0(cCells) = 1.0;
                             m_density_env_n0(cCells)[0] = 1.0;
                             m_density_env_n0(cCells)[1] = 0.;
                           } else {
                             m_fracvol_env(cCells)[0] = 1.;
                             m_fracvol_env(cCells)[1] = 0.;
                             m_mass_fraction_env(cCells)[0] = 1.;
                             m_mass_fraction_env(cCells)[1] = 0.;
                             m_density_n0(cCells) = 1.;
                             m_density_env_n0(cCells)[0] = 1.;
                             m_density_env_n0(cCells)[1] = 0.;
                           }
                         });
  } else if (test->Nom == test->BiUnitTestCase) {
    Kokkos::parallel_for("initDensity", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           if (m_cell_coord(cCells)[0] <= 0.5) {
                             m_fracvol_env(cCells)[0] = 1.;
                             m_fracvol_env(cCells)[1] = 0.;
                             m_mass_fraction_env(cCells)[0] = 1.;
                             m_mass_fraction_env(cCells)[1] = 0.;
                             m_density_n0(cCells) = 1.0;
                             m_density_env_n0(cCells)[0] = 1.0;
                             m_density_env_n0(cCells)[1] = 0.;
                           } else {
                             m_fracvol_env(cCells)[0] = 0.;
                             m_fracvol_env(cCells)[1] = 1.;
                             m_mass_fraction_env(cCells)[0] = 0.;
                             m_mass_fraction_env(cCells)[1] = 1.;
                             m_density_n0(cCells) = 1.;
                             m_density_env_n0(cCells)[0] = 0.;
                             m_density_env_n0(cCells)[1] = 1.0;
                           }
                         });
  } else if (test->Nom == test->SodCaseX || test->Nom == test->SodCaseY) {
    Kokkos::parallel_for("initDensity", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           m_fracvol_env(cCells)[0] = 1.;
                           m_fracvol_env(cCells)[1] = 0.;
                           m_mass_fraction_env(cCells)[0] = 1.;
                           m_mass_fraction_env(cCells)[1] = 0.;
                           double r(0.);
                           if (test->Nom == test->SodCaseX)
                             r = m_cell_coord(cCells)[0];
                           if (test->Nom == test->SodCaseY)
                             r = m_cell_coord(cCells)[1];
                           if (r <= 0.5) {
                             m_density_n0(cCells) = 1.0;
                             m_density_env_n0(cCells)[0] = 1.0;
                             m_density_env_n0(cCells)[1] = 1.0;
                           } else {
                             m_density_n0(cCells) = 0.125;
                             m_density_env_n0(cCells)[0] = 0.125;
                             m_density_env_n0(cCells)[1] = 0.125;
                           }
                         });
  } else if (test->Nom == test->BiSodCaseX || test->Nom == test->BiSodCaseY) {
    Kokkos::parallel_for("initDensity", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           double r(0.);
                           if (test->Nom == test->BiSodCaseX)
                             r = m_cell_coord(cCells)[0];
                           if (test->Nom == test->BiSodCaseY)
                             r = m_cell_coord(cCells)[1];
                           if (r <= 0.5) {
                             m_fracvol_env(cCells)[0] = 1.;
                             m_fracvol_env(cCells)[1] = 0.;
                             m_mass_fraction_env(cCells)[0] = 1.;
                             m_mass_fraction_env(cCells)[1] = 0.;
                             m_density_n0(cCells) = 1.0;
                             m_density_env_n0(cCells)[0] = 1.0;
                             m_density_env_n0(cCells)[1] = 0.;
                           } else {
                             m_fracvol_env(cCells)[0] = 0.;
                             m_fracvol_env(cCells)[1] = 1.;
                             m_mass_fraction_env(cCells)[0] = 0.;
                             m_mass_fraction_env(cCells)[1] = 1.;
                             m_density_n0(cCells) = 0.125;
                             m_density_env_n0(cCells)[0] = 0.;
                             m_density_env_n0(cCells)[1] = 0.125;
                           }
                         });
  } else if (test->Nom == test->BiShockBubble) {
    Kokkos::parallel_for(
        "initDensity", nbCells, KOKKOS_LAMBDA(const int& cCells) {
          double pInit;
          double rhoInit;
          double eInit;
          // Air partout
          m_density_env_n0(cCells)[0] = 1.0;
          m_density_n0(cCells) = 1.0;
          m_fracvol_env(cCells)[0] = 1.;
          m_fracvol_env(cCells)[1] = 0.;

          m_mass_fraction_env(cCells)[0] = 1.;
          m_mass_fraction_env(cCells)[1] = 0.;
          // bulle surchargera l'aire
          // centre de la bulle
          RealArray1D<dim> Xb = {{0.320, 0.04}};
          double rb = 0.025;
          double r = sqrt((m_cell_coord(cCells)[0] - Xb[0]) *
                              (m_cell_coord(cCells)[0] - Xb[0]) +
                          (m_cell_coord(cCells)[1] - Xb[1]) *
                              (m_cell_coord(cCells)[1] - Xb[1]));
          if (r <= rb) {
            m_density_env_n0(cCells)[0] = 0.0;
            m_density_env_n0(cCells)[1] = 0.182;
            m_density_n0(cCells) = 0.182;

            m_fracvol_env(cCells)[0] = 0.;
            m_fracvol_env(cCells)[1] = 1.;

            m_mass_fraction_env(cCells)[0] = 0.;
            m_mass_fraction_env(cCells)[1] = 1.;
          }
        });
  } else if (test->Nom == test->TriplePoint) {
    Kokkos::parallel_for("initDensity", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           m_fracvol_env(cCells)[0] = 1.;
                           m_fracvol_env(cCells)[1] = 0.;

                           m_mass_fraction_env(cCells)[0] = 1.;
                           m_mass_fraction_env(cCells)[1] = 0.;
                           if (m_cell_coord(cCells)[0] <= 0.01) {
                             // std::cout << " cell " << cCells << "  x= " <<
                             // m_cell_coord(cCells)[0] << " y= " <<
                             // m_cell_coord(cCells)[1] << std::endl;
                             m_density_n0(cCells) = 1.0;
                           } else {
                             if (m_cell_coord(cCells)[1] <= 0.015) {
                               // std::cout << " cell cas 2  " << cCells << " x=
                               // " << m_cell_coord(cCells)[0] << "  y= " <<
                               // m_cell_coord(cCells)[1]
                               // << std::endl;
                               m_density_n0(cCells) = 1.0;
                             } else {
                               // std::cout << " cell cas 3  " << cCells << " x=
                               // " << m_cell_coord(cCells)[0] << "  y= " <<
                               // m_cell_coord(cCells)[1]
                               // << std::endl;
                               m_density_n0(cCells) = 0.1;
                             }
                           }
                         });
  } else if (test->Nom == test->BiTriplePoint) {
    Kokkos::parallel_for("initDensity", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           if (m_cell_coord(cCells)[0] <= 0.01) {
                             // std::cout << " cell cas 1 " << cCells << "  x= "
                             // << m_cell_coord(cCells)[0]
                             //          << "  y= " << m_cell_coord(cCells)[1]
                             //          << std::endl;
                             m_density_n0(cCells) = 1.0;
                             m_fracvol_env(cCells)[0] = 1.;
                             m_mass_fraction_env(cCells)[0] = 1.;
                             m_density_env_n0(cCells)[0] = 1.0;

                           } else {
                             if (m_cell_coord(cCells)[1] <= 0.015) {
                               // std::cout << "cell cas 2  " <<cCells << "  x=
                               // " <<m_cell_coord(cCells)[0]
                               //          << "  y= " << m_cell_coord(cCells)[1]
                               //          << std::endl;
                               m_density_n0(cCells) = 1.0;
                               m_fracvol_env(cCells)[1] = 1.;
                               m_mass_fraction_env(cCells)[1] = 1.;
                               m_density_env_n0(cCells)[1] = 1.0;
                             } else {
                               // std::cout << "cell cas 3  " << cCells << "  x=
                               // " <<m_cell_coord(cCells)[0]
                               //          << "  y= " << m_cell_coord(cCells)[1]
                               //          << std::endl;
                               m_density_n0(cCells) = 0.1;
                               m_fracvol_env(cCells)[2] = 1.;
                               m_mass_fraction_env(cCells)[2] = 1.;
                               m_density_env_n0(cCells)[2] = 0.1;
                             }
                           }
                         });
  } else if (test->Nom == test->SedovTestCase) {
    Kokkos::parallel_for("initDensity", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           m_density_n0(cCells) = 1.0;
                           m_density_env_n0(cCells)[0] = 1.0;
                         });
  }
  if (options->nbmat == 1) {
    Kokkos::parallel_for("initDensity", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           m_fracvol_env(cCells)[0] = 1.;
                           m_fracvol_env(cCells)[1] = 0.;
                           m_fracvol_env(cCells)[2] = 0.;

                           m_mass_fraction_env(cCells)[0] = 1.;
                           m_mass_fraction_env(cCells)[1] = 0.;
                           m_mass_fraction_env(cCells)[2] = 0.;
                         });
  }
  Kokkos::parallel_for(
      "initDensity", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        // pour les sorties au temps 0:
        m_fracvol_env1(cCells) = m_fracvol_env(cCells)[0];
        m_fracvol_env2(cCells) = m_fracvol_env(cCells)[1];
        m_fracvol_env3(cCells) = m_fracvol_env(cCells)[2];
        m_x_cell_velocity(cCells) = m_cell_velocity_n0(cCells)[0];
        m_y_cell_velocity(cCells) = m_cell_velocity_n0(cCells)[1];
        // indicateur mailles mixtes
        int matcell(0);
        int imatpure(-1);
        for (int imat = 0; imat < nbmatmax; imat++)
          if (m_fracvol_env(cCells)[imat] > options->threshold) {
            matcell++;
            imatpure = imat;
          }

        if (matcell > 1) {
          varlp->mixte(cCells) = 1;
          varlp->pure(cCells) = -1;
        } else {
          varlp->mixte(cCells) = 0;
          varlp->pure(cCells) = imatpure;
        }
      });
}

/**
 * Job initMeshGeometryForFaces called @2.0 in simulate method.
 * In variables: X, m_cell_coord, ex, ey, threshold
 * Out variables: Xf, faceLength, faceNormal, outerFaceNormal
 */
void Eucclhyd::initMeshGeometryForFaces() noexcept {
  auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "initMeshGeometryForFaces", nbFaces, KOKKOS_LAMBDA(const int& fFaces) {
        size_t fId(faces[fFaces]);
        int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
        int n1Id(n1FirstNodeOfFaceF);
        int n1Nodes(n1Id);
        int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
        int n2Id(n2SecondNodeOfFaceF);
        int n2Nodes(n2Id);
        RealArray1D<dim> X_face =
            (0.5 * ((m_node_coord(n1Nodes) + m_node_coord(n2Nodes))));
        RealArray1D<dim> face_vec =
            (m_node_coord(n2Nodes) - m_node_coord(n1Nodes));
        varlp->Xf(fFaces) = X_face;
        varlp->faceLength(fFaces) = MathFunctions::norm(face_vec);
        {
          auto cellsOfFaceF(mesh->getCellsOfFace(fId));
          for (int cCellsOfFaceF = 0; cCellsOfFaceF < cellsOfFaceF.size();
               cCellsOfFaceF++) {
            int cId(cellsOfFaceF[cCellsOfFaceF]);
            int cCells(cId);
            int fFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), fId));
            varlp->outerFaceNormal(cCells, fFacesOfCellC) =
                (((X_face - m_cell_coord(cCells))) /
                 MathFunctions::norm((X_face - m_cell_coord(cCells))));
          }
        }
        RealArray1D<dim> face_normal;
        if (MathFunctions::fabs(dot(face_vec, ex)) < options->threshold)
          face_normal = ex;
        else
          face_normal = ey;
        varlp->faceNormal(fFaces) = face_normal;
        // std::cout << nbFaces << " "
        //	  <<  varlp->faceNormal(fFaces) << std::endl;
      });
}
/**
 * Job setUpTimeLoopN called @3.0 in simulate method.
 * In variables: m_node_force_n0, m_cell_velocity_n0, m_node_velocity_n0,
 * m_internal_energy_n0, m_density_n0 Out variables: m_node_force_n,
 * m_cell_velocity_n, m_node_velocity_n, m_internal_energy_n, m_density_n
 */
void Eucclhyd::setUpTimeLoopN() noexcept {
  deep_copy(m_node_velocity_n, m_node_velocity_n0);
  deep_copy(m_density_n, m_density_n0);
  deep_copy(m_density_env_n, m_density_env_n0);
  deep_copy(m_cell_velocity_n, m_cell_velocity_n0);
  deep_copy(m_internal_energy_n, m_internal_energy_n0);
  deep_copy(m_internal_energy_env_n, m_internal_energy_env_n0);
  deep_copy(m_node_force_n, m_node_force_n0);

  if (test->Nom == test->SodCaseX || test->Nom == test->SodCaseY ||
      test->Nom == test->BiSodCaseX || test->Nom == test->BiSodCaseY) {
    // const ℝ δt_init = 1.0e-4;
    gt->deltat_init = 1.0e-4;
    gt->deltat_n = gt->deltat_init;
  } else if (test->Nom == test->BiShockBubble) {
    gt->deltat_init = 1.e-7;
    gt->deltat_n = 1.0e-7;
  } else if (test->Nom == test->SedovTestCase ||
             test->Nom == test->BiSedovTestCase) {
    // const ℝ δt_init = 1.0e-4;
    gt->deltat_init = 1.0e-4;
    gt->deltat_n = 1.0e-4;
  } else if (test->Nom == test->NohTestCase ||
             test->Nom == test->BiNohTestCase) {
    // const ℝ δt_init = 1.0e-4;
    gt->deltat_init = 1.0e-4;
    gt->deltat_n = 1.0e-4;
  } else if (test->Nom == test->TriplePoint ||
             test->Nom == test->BiTriplePoint) {
    // const ℝ δt_init = 1.0e-5; avec donnees adimensionnées
    gt->deltat_init = 1.0e-5;  // avec pression de 1.e5 / 1.e-8
    gt->deltat_n = 1.0e-5;
  }
  m_global_total_energy_0 = 0.;
  Kokkos::parallel_for("init_m_global_total_energy_0", nbCells,
                       KOKKOS_LAMBDA(const int& cCells) {
                         m_total_energy_0(cCells) =
                             (m_density_n0(cCells) * m_euler_volume(cCells)) *
                             (m_internal_energy_n0(cCells) +
                              0.5 * (m_cell_velocity_n0(cCells)[0] *
                                         m_cell_velocity_n0(cCells)[0] +
                                     m_cell_velocity_n0(cCells)[1] *
                                         m_cell_velocity_n0(cCells)[1]));
                         m_global_masse_0(cCells) =
                             (m_density_n0(cCells) * m_euler_volume(cCells));
                       });
  double reductionE(0.), reductionM(0.);
  {
    Kokkos::Sum<double> reducerE(reductionE);
    Kokkos::parallel_reduce("reductionE", nbCells,
                            KOKKOS_LAMBDA(const int& cCells, double& x) {
                              reducerE.join(x, m_total_energy_0(cCells));
                            },
                            reducerE);
    Kokkos::Sum<double> reducerM(reductionM);
    Kokkos::parallel_reduce("reductionM", nbCells,
                            KOKKOS_LAMBDA(const int& cCells, double& x) {
                              reducerM.join(x, m_global_masse_0(cCells));
                            },
                            reducerM);
  }
  m_global_total_energy_0 = reductionE;
  m_total_masse_0 = reductionM;
}
