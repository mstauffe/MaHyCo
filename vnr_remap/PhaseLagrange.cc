#include "Vnr.h"

using namespace nablalib;

#include "../includes/Freefunctions.h"
#include "types/MathFunctions.h"  // for max, min, dot, matVectProduct
#include "utils/Utils.h"          // for Indexof

/**
 * Job computeCellMass called @3.0 in simulate method.
 * In variables: X_EDGE_LENGTH, Y_EDGE_LENGTH, m_density_n0
 * Out variables: m_cell_mass
 */
void Vnr::computeCellMass() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    int nbmat = options->nbmat;
    m_cell_mass(cCells) =
        cstmesh->X_EDGE_LENGTH * cstmesh->Y_EDGE_LENGTH * init->m_density_n0(cCells);
    for (int imat = 0; imat < nbmat; ++imat) {
      m_cell_mass_env(cCells)[imat] =
          m_mass_fraction_env(cCells)[imat] * m_cell_mass(cCells);
    }
  });
}

/**
 * Job computeNodeMass called @4.0 in simulate method.
 * In variables: m_cell_mass
 * Out variables: m
 */
void Vnr::computeNodeMass() noexcept {
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    const Id pId(pNodes);
    const auto cells_of_node(mesh->getCellsOfNode(pId));
    double reduction0(0.0);
    {
      const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
      const size_t nbCellsOfNodeP(cellsOfNodeP.size());
      for (size_t cCellsOfNodeP = 0; cCellsOfNodeP < nbCellsOfNodeP;
           cCellsOfNodeP++) {
        const Id cId(cellsOfNodeP[cCellsOfNodeP]);
        const size_t cCells(cId);
        reduction0 = sumR0(reduction0, m_cell_mass(cCells));
      }
    }
    m_node_mass(pNodes) = reduction0 / cells_of_node.size();
  });
}
/**
 * Job computeArtificialViscosity called @1.0 in executeTimeLoopN method.
 * In variables: m_node_cellvolume_n, m_speed_velocity_n, m_divu_n, gamma,
 * m_tau_density_n Out variables: m_pseudo_viscosity_nplus1
 */
void Vnr::computeArtificialViscosity() noexcept {
  double reductionP(numeric_limits<double>::min());
  Kokkos::Max<double> reducer(reductionP);
  Kokkos::parallel_reduce("reductionP", nbCells,
                          KOKKOS_LAMBDA(const int& cCells, double& x) {
                            reducer.join(x, m_pressure_n(cCells));
                          },
                          reducer);
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    const Id cId(cCells);
    if (m_divu_n(cCells) < 0.0) {
      double reduction0(0.0);
      {
        const auto nodesOfCellC(mesh->getNodesOfCell(cId));
        const size_t nbNodesOfCellC(nodesOfCellC.size());
        for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
             pNodesOfCellC++) {
          reduction0 =
              sumR0(reduction0, m_node_cellvolume_n(cCells, pNodesOfCellC));
        }
      }
      double reduction1(0.0);
      {
        const auto nodesOfCellC(mesh->getNodesOfCell(cId));
        const size_t nbNodesOfCellC(nodesOfCellC.size());
        for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
             pNodesOfCellC++) {
          reduction1 =
              sumR0(reduction1, m_node_cellvolume_n(cCells, pNodesOfCellC));
        }
      }
      m_pseudo_viscosity_nplus1(cCells) =
          1.0 / m_tau_density_nplus1(cCells) *
          (-0.5 * std::sqrt(reduction0) * m_speed_velocity_n(cCells) *
               m_divu_nplus1(cCells) +
           (eos->gamma + 1) / 2.0 * reduction1 * m_divu_nplus1(cCells) *
               m_divu_nplus1(cCells));
    } else
      m_pseudo_viscosity_nplus1(cCells) = 0.0;
    //
    // limitation par la pression
    // permet de limiter un exces de pseudo lié à des erreurs d'arrondis sur
    // m_tau_density_nplus1
    if (m_pseudo_viscosity_nplus1(cCells) > reductionP) {
      std::cout << cCells << " pseudo " << m_pseudo_viscosity_nplus1(cCells)
                << " pression " << m_pressure_n(cCells) << std::endl;
      m_pseudo_viscosity_nplus1(cCells) = reductionP;
    }
    // pour chaque matériau
    for (int imat = 0; imat < options->nbmat; ++imat)
      m_pseudo_viscosity_env_nplus1(cCells)[imat] =
          m_fracvol_env(cCells)[imat] * m_pseudo_viscosity_nplus1(cCells);
  });
}

/**
 * Job computeCornerNormal called @1.0 in executeTimeLoopN method.
 * In variables: m_node_coord_n
 * Out variables: C
 */
void Vnr::computeCornerNormal() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    const Id cId(cCells);
    {
      const auto nodesOfCellC(mesh->getNodesOfCell(cId));
      const size_t nbNodesOfCellC(nodesOfCellC.size());
      for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
           pNodesOfCellC++) {
        const Id pId(nodesOfCellC[pNodesOfCellC]);
        const Id pPlus1Id(
            nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) % nbNodesOfCell]);
        const Id pMinus1Id(
            nodesOfCellC[(pNodesOfCellC - 1 + nbNodesOfCell) % nbNodesOfCell]);
        const size_t pNodes(pId);
        const size_t pPlus1Nodes(pPlus1Id);
        const size_t pMinus1Nodes(pMinus1Id);
        m_cqs(cCells, pNodesOfCellC) =
            computeLpcNpc(m_node_coord_n(pNodes), m_node_coord_n(pPlus1Nodes),
                          m_node_coord_n(pMinus1Nodes));
      }
    }
  });
}
/**
 * Job computeNodeVolume called @1.0 in executeTimeLoopN method.
 * In variables: m_node_cellvolume_n
 * Out variables: V
 */
void Vnr::computeNodeVolume() noexcept {
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    const Id pId(pNodes);
    double reduction0(0.0);
    {
      const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
      const size_t nbCellsOfNodeP(cellsOfNodeP.size());
      for (size_t cCellsOfNodeP = 0; cCellsOfNodeP < nbCellsOfNodeP;
           cCellsOfNodeP++) {
        const Id cId(cellsOfNodeP[cCellsOfNodeP]);
        const size_t cCells(cId);
        const size_t pNodesOfCellC(
            utils::indexOf(mesh->getNodesOfCell(cId), pId));
        reduction0 =
            sumR0(reduction0, m_node_cellvolume_n(cCells, pNodesOfCellC));
      }
    }
    m_node_volume(pNodes) = reduction0;
  });
}
/**
 * Job updateVelocity called @2.0 in executeTimeLoopN method.
 * In variables: C, m_pseudo_viscosity_nplus1, deltat_n, deltat_nplus1, m,
 * m_pressure_n, m_node_velocity_n Out variables: m_node_velocity_nplus1
 */
void Vnr::updateVelocity() noexcept {
  const double dt(0.5 * (gt->deltat_nplus1 + gt->deltat_n));
  {
    const auto innerNodes(mesh->getInnerNodes());
    const size_t nbInnerNodes(mesh->getNbInnerNodes());
    Kokkos::parallel_for(
        nbInnerNodes, KOKKOS_LAMBDA(const size_t& pInnerNodes) {
          const Id pId(innerNodes[pInnerNodes]);
          const size_t pNodes(pId);
          RealArray1D<2> reduction0({0.0, 0.0});
          {
            const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            const size_t nbCellsOfNodeP(cellsOfNodeP.size());
            for (size_t cCellsOfNodeP = 0; cCellsOfNodeP < nbCellsOfNodeP;
                 cCellsOfNodeP++) {
              const Id cId(cellsOfNodeP[cCellsOfNodeP]);
              const size_t cCells(cId);
              const size_t pNodesOfCellC(
                  utils::indexOf(mesh->getNodesOfCell(cId), pId));
              reduction0 = sumR1(reduction0, (m_pressure_n(cCells) +
                                              m_pseudo_viscosity_n(cCells)) *
                                                 m_cqs(cCells, pNodesOfCellC));
            }
          }
          m_node_velocity_nplus1(pNodes) =
              m_node_velocity_n(pNodes) + dt / m_node_mass(pNodes) * reduction0;
          m_x_velocity(pNodes) = m_node_velocity_nplus1(pNodes)[0];
          m_y_velocity(pNodes) = m_node_velocity_nplus1(pNodes)[1];
        });
  }
}
/**
 * Job updatePosition called @3.0 in executeTimeLoopN method.
 * In variables: m_node_coord_n, deltat_nplus1, m_node_velocity_nplus1
 * Out variables: m_node_coord_nplus1
 */
void Vnr::updatePosition() noexcept {
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    m_node_coord_nplus1(pNodes) =
        m_node_coord_n(pNodes) +
        gt->deltat_nplus1 * m_node_velocity_nplus1(pNodes);
  });
}
/**
 * Job initCellPos called @1.0 in simulate method.
 * In variables: m_node_coord_nplus1
 * Out variables: m_cell_coord_nplus1
 */
void Vnr::updateCellPos() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    const Id cId(cCells);
    RealArray1D<2> reduction0({0.0, 0.0});
    {
      const auto nodesOfCellC(mesh->getNodesOfCell(cId));
      const size_t nbNodesOfCellC(nodesOfCellC.size());
      for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
           pNodesOfCellC++) {
        const Id pId(nodesOfCellC[pNodesOfCellC]);
        const size_t pNodes(pId);
        reduction0 = sumR1(reduction0, m_node_coord_nplus1(pNodes));
      }
    }
    m_cell_coord_nplus1(cCells) = 0.25 * reduction0;
  });
}
/**
 * Job computeSubVol called @4.0 in executeTimeLoopN method.
 * In variables: m_node_coord_nplus1, m_cell_coord_nplus1
 * Out variables: m_node_cellvolume_nplus1
 */
void Vnr::computeSubVol() noexcept {
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
        const RealArray1D<2> x1(m_cell_coord_nplus1(cCells));
        const RealArray1D<2> x2(0.5 * (m_node_coord_nplus1(pMinus1Nodes) +
                                       m_node_coord_nplus1(pNodes)));
        const RealArray1D<2> x3(m_node_coord_nplus1(pNodes));
        const RealArray1D<2> x4(0.5 * (m_node_coord_nplus1(pPlus1Nodes) +
                                       m_node_coord_nplus1(pNodes)));
        m_node_cellvolume_nplus1(cCells, pNodesOfCellC) =
            0.5 * (crossProduct2d(x1, x2) + crossProduct2d(x2, x3) +
                   crossProduct2d(x3, x4) + crossProduct2d(x4, x1));
      }
    }
  });
}
/**
 * Job updateRho called @5.0 in executeTimeLoopN method.
 * In variables: m_node_cellvolume_nplus1, m_cell_mass
 * Out variables: m_density_nplus1
 */
void Vnr::updateRho() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    const Id cId(cCells);
    double reduction0(0.0);
    {
      const auto nodesOfCellC(mesh->getNodesOfCell(cId));
      const size_t nbNodesOfCellC(nodesOfCellC.size());
      for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
           pNodesOfCellC++) {
        reduction0 =
            sumR0(reduction0, m_node_cellvolume_nplus1(cCells, pNodesOfCellC));
      }
    }
    varlp->vLagrange(cCells) = reduction0;
    m_density_nplus1(cCells) = 0.;
    for (int imat = 0; imat < options->nbmat; ++imat) {
      if (m_fracvol_env(cCells)[imat] > options->threshold)
        m_density_env_nplus1(cCells)[imat] =
            m_cell_mass_env(cCells)[imat] /
            (m_fracvol_env(cCells)[imat] * reduction0);
      // ou 1/rhon_nplus1 += m_mass_fraction_env(cCells)[imat] /
      // m_density_env_nplus1[imat];
      m_density_nplus1(cCells) +=
          m_fracvol_env(cCells)[imat] * m_density_env_nplus1(cCells)[imat];
    }
  });
}

/**
 * Job computeTau called @6.0 in executeTimeLoopN method.
 * In variables: m_density_n, m_density_nplus1
 * Out variables: m_tau_density_nplus1
 */
void Vnr::computeTau() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    m_tau_density_nplus1(cCells) =
        0.5 * (1.0 / m_density_nplus1(cCells) + 1.0 / m_density_n(cCells));
    for (int imat = 0; imat < options->nbmat; ++imat) {
      m_tau_density_env_nplus1(cCells)[imat] = 0.;
      if ((m_density_env_nplus1(cCells)[imat] > options->threshold) &&
          (m_density_env_n(cCells)[imat] > options->threshold))
        m_tau_density_env_nplus1(cCells)[imat] =
            0.5 * (1.0 / m_density_env_nplus1(cCells)[imat] +
                   1.0 / m_density_env_n(cCells)[imat]);
    }
  });
}

/**
 * Job updateEnergy called @6.0 in executeTimeLoopN method.
 * In variables: m_pseudo_viscosity_nplus1, m_internal_energy_n, gamma,
 * m_pressure_n, m_density_n, m_density_nplus1 Out variables:
 * m_internal_energy_nplus1
 */
void Vnr::updateEnergy() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    m_internal_energy_nplus1(cCells) = 0.;
    for (int imat = 0; imat < options->nbmat; ++imat) {
      m_internal_energy_env_nplus1(cCells)[imat] = 0.;
      if ((m_density_env_nplus1(cCells)[imat] > options->threshold) &&
          (m_density_env_n(cCells)[imat] > options->threshold)) {
        // calcul du DV a changer utiliser divU
        double pseudo(0.);
        if ((options->pseudo_centree == 1) &&
            ((m_pseudo_viscosity_env_nplus1(cCells)[imat] +
              m_pseudo_viscosity_env_n(cCells)[imat]) *
                 (1.0 / m_density_env_nplus1(cCells)[imat] -
                  1.0 / m_density_env_n(cCells)[imat]) >
             0.)) {
          pseudo = 0.5 * (m_pseudo_viscosity_env_nplus1(cCells)[imat] +
                          m_pseudo_viscosity_env_n(cCells)[imat]);
        }
        if (options->pseudo_centree ==
            0) {  // test sur la positivité du travail dans le calcul de
                  // m_pseudo_viscosity_nplus1(cCells)
          pseudo = m_pseudo_viscosity_env_nplus1(cCells)[imat];
        }
        const double den(1 + 0.5 * (eos->gammap[imat] - 1.0) *
                                 m_density_env_nplus1(cCells)[imat] *
                                 (1.0 / m_density_env_nplus1(cCells)[imat] -
                                  1.0 / m_density_env_n(cCells)[imat]));
        const double num(m_internal_energy_env_n(cCells)[imat] -
                         (0.5 * m_pressure_env_n(cCells)[imat] + pseudo) *
                             (1.0 / m_density_env_nplus1(cCells)[imat] -
                              1.0 / m_density_env_n(cCells)[imat]));
        m_internal_energy_env_nplus1(cCells)[imat] = num / den;
        m_internal_energy_nplus1(cCells) +=
            m_mass_fraction_env(cCells)[imat] *
            m_internal_energy_env_nplus1(cCells)[imat];
      }
    }
  });
}

/**
 * Job computeDivU called @7.0 in executeTimeLoopN method.
 * In variables: deltat_nplus1, m_density_n, m_density_nplus1,
 * m_tau_density_nplus1 Out variables: m_divu_nplus1
 */
void Vnr::computeDivU() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    m_divu_nplus1(cCells) =
        1.0 / gt->deltat_nplus1 *
        (1.0 / m_density_nplus1(cCells) - 1.0 / m_density_n(cCells)) /
        m_tau_density_nplus1(cCells);
    // a changer comme le calcul du DV, utiliser les
    // m_cqs(cCells,pNodesOfCellC)
  });
}
/**
 * Job computeEOS called in executeTimeLoopN method.
 */
void Vnr::computeEOS() {
  for (int imat = 0; imat < options->nbmat; ++imat) {
    if (eos->Nom[imat] == eos->PerfectGas) computeEOSGP(imat);
    if (eos->Nom[imat] == eos->Void) computeEOSVoid(imat);
    if (eos->Nom[imat] == eos->StiffenedGas) computeEOSSTIFG(imat);
    if (eos->Nom[imat] == eos->Murnhagan) computeEOSMur(imat);
    if (eos->Nom[imat] == eos->SolidLinear) computeEOSSL(imat);
  }
}
/**
 * Job computeEOSGP called @1.0 in executeTimeLoopN method.
 * In variables: eos, eosPerfectGas, m_internal_energy_n, gammap, m_density_n
 * Out variables: c, p
 */
void Vnr::computeEOSGP(int imat) {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    m_pressure_env_nplus1(cCells)[imat] =
        (eos->gammap[imat] - 1.0) * m_density_env_nplus1(cCells)[imat] *
        m_internal_energy_env_nplus1(cCells)[imat];
    m_speed_velocity_env_nplus1(cCells)[imat] =
        std::sqrt(eos->gammap[imat] * (eos->gammap[imat] - 1.0) *
                  m_internal_energy_env_nplus1(cCells)[imat]);
  });
}
/**
 * Job computeEOSVoid called in executeTimeLoopN method.
 * In variables: eos, eosPerfectGas, m_internal_energy_n, gammap, m_density_n
 * Out variables: c, p
 */
void Vnr::computeEOSVoid(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    m_pressure_env_nplus1(cCells)[imat] = m_pressure_env_n(cCells)[imat];
    m_speed_velocity_env_nplus1(cCells)[imat] =
        m_speed_velocity_env_n(cCells)[imat];
    m_internal_energy_env_nplus1(cCells)[imat] =
        m_internal_energy_env_n(cCells)[imat];
    m_internal_energy_nplus1(cCells) = 0.;
  });
}
/**
 * Job computeEOSSTIFG
 * In variables: m_internal_energy_n, m_density_n
 * Out variables: c, p
 */
void Vnr::computeEOSSTIFG(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    std::cout << " Pas encore programmée" << std::endl;
  });
}
/**
 * Job computeEOSMur called @1.0 in executeTimeLoopN method.
 * In variables: m_internal_energy_n, m_density_n
 * Out variables: c, p
 */
void Vnr::computeEOSMur(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    std::cout << " Pas encore programmée" << std::endl;
  });
}
/**
 * Job computeEOSSL called @1.0 in executeTimeLoopN method.
 * In variables: m_internal_energy_n, m_density_n
 * Out variables: c, p
 */
void Vnr::computeEOSSL(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    std::cout << " Pas encore programmée" << std::endl;
  });
}
/**
 * Job computeEOS called in executeTimeLoopN method.
 */
void Vnr::computePressionMoyenne() noexcept {
  for (int cCells = 0; cCells < nbCells; cCells++) {
    m_pressure_nplus1(cCells) = 0.;
    for (int imat = 0; imat < options->nbmat; ++imat) {
      m_pressure_nplus1(cCells) +=
          m_fracvol_env(cCells)[imat] * m_pressure_env_nplus1(cCells)[imat];
      m_speed_velocity_nplus1(cCells) =
          MathFunctions::max(m_speed_velocity_nplus1(cCells),
                             m_speed_velocity_env_nplus1(cCells)[imat]);
    }
    // NONREG GP A SUPPRIMER
    if (m_density_nplus1(cCells) > 0.) {
      m_speed_velocity_nplus1(cCells) =
          std::sqrt(eos->gammap[0] * (eos->gammap[0] - 1.0) *
                    m_internal_energy_nplus1(cCells));
    }
    for (int imat = 0; imat < options->nbmat; ++imat)
      if (eos->Nom[imat] == eos->Void)
        m_internal_energy_nplus1(cCells) +=
            m_mass_fraction_env(cCells)[imat] *
            m_internal_energy_env_nplus1(cCells)[imat];
  }
}
