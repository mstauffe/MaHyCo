#include "Vnr.h"

using namespace nablalib;

#include <iomanip>  // for operator<<, setw, setiosflags
#include "../includes/Freefunctions.h"
#include "utils/Utils.h"  // for __RESET__, __BOLD__, __GREEN__

/**
 * Job computeDeltaT called @1.0 in executeTimeLoopN method.
 * In variables: m_node_cellvolume_n, m_speed_velocity_n, deltat_n
 * Out variables: deltat_nplus1
 */
void Vnr::computeDeltaT() noexcept {
  double reduction0;
  double Aveccfleuler(0.);
  double cfl(0.1);
  if (options->AvecProjection == 1) {
    // cfl euler
    Aveccfleuler = 1;
    cfl = 0.05;  // explication à trouver, permet de passer les cas euler ?
  }
  Kokkos::parallel_reduce(
      nbCells,
      KOKKOS_LAMBDA(const size_t& cCells, double& accu) {
        const Id cId(cCells);
        double uc(0.0);
        double reduction1(0.0);
        {
          const auto nodesOfCellC(mesh->getNodesOfCell(cId));
          const size_t nbNodesOfCellC(nodesOfCellC.size());
          for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pNodes(pId);
            reduction1 =
                sumR0(reduction1, m_node_cellvolume_n(cCells, pNodesOfCellC));
            uc += std::sqrt(m_node_velocity_n(pNodes)[0] * m_node_velocity_n(pNodes)[0] +
                            m_node_velocity_n(pNodes)[1] * m_node_velocity_n(pNodes)[1]) *
                  0.25;
          }
        }
        // 0.05 a expliquer
        accu =
            minR0(accu, cfl * std::sqrt(reduction1) /
                            (Aveccfleuler * uc + m_speed_velocity_n(cCells)));
      },
      KokkosJoiner<double>(reduction0, numeric_limits<double>::max(), &minR0));
  gt->deltat_nplus1 = std::min(reduction0, 1.05 * gt->deltat_n);
}

/**
 * Job computeTime called @2.0 in executeTimeLoopN method.
 * In variables: deltat_nplus1, t_n
 * Out variables: t_nplus1
 */
void Vnr::computeTime() noexcept { gt->t_nplus1 = gt->t_n + gt->deltat_nplus1; }

/**
 * Job SetUpTimeLoopN called @4.0 in simulate method.
 * In variables: m_node_cellvolume_n0, m_node_coord_n0, m_speed_velocity_n0,
 * m_cell_coord_n0, deltat_init, m_divu_n0, m_internal_energy_n0, m_pressure_n0,
 * m_density_n0, m_tau_density_n0, m_node_velocity_n0 Out variables:
 * m_node_cellvolume_n, m_node_coord_n, m_speed_velocity_n, m_cell_coord_n, deltat_n,
 * m_divu_n, m_internal_energy_n, m_pressure_n, m_density_n, m_tau_density_n,
 * m_node_velocity_n
 */
void Vnr::setUpTimeLoopN() noexcept {
  gt->deltat_n = gt->deltat_init;
  deep_copy(m_node_coord_n, m_node_coord_n0);
  deep_copy(m_node_cellvolume_n, m_node_cellvolume_n0);
  deep_copy(m_density_n, m_density_n0);
  deep_copy(m_density_env_n, m_density_env_n0);
  deep_copy(m_pressure_n, m_pressure_n0);
  deep_copy(m_pressure_env_n, m_pressure_env_n0);
  deep_copy(m_pseudo_viscosity_n, m_pseudo_viscosity_n0);
  deep_copy(m_pseudo_viscosity_env_n, m_pseudo_viscosity_env_n0);
  deep_copy(m_tau_density_n, m_tau_density_n0);
  deep_copy(m_tau_density_env_n, m_tau_density_env_n0);
  deep_copy(m_divu_n, m_divu_n0);
  deep_copy(m_speed_velocity_n, m_speed_velocity_n0);
  deep_copy(m_speed_velocity_env_n, m_speed_velocity_env_n0);
  deep_copy(m_internal_energy_n, m_internal_energy_n0);
  deep_copy(m_internal_energy_env_n, m_internal_energy_env_n0);
  deep_copy(m_node_velocity_n, m_node_velocity_n0);
  deep_copy(m_cell_coord_n, m_cell_coord_n0);

  // if (test->Nom == test->SodCaseX || test->Nom == test->SodCaseY ||
  //     test->Nom == test->BiSodCaseX || test->Nom == test->BiSodCaseY) {
  //   // const ℝ δt_init = 1.0e-4;
  //   gt->deltat_init = 1.0e-4;
  //   gt->deltat_n = gt->deltat_init;
  // } else if (test->Nom == test->BiShockBubble) {
  //   gt->deltat_init = 1.e-7;
  //   gt->deltat_n = 1.0e-7;
  // } else if (test->Nom == test->SedovTestCase ||
  //            test->Nom == test->BiSedovTestCase) {
  //   // const ℝ δt_init = 1.0e-4;
  //   gt->deltat_init = 1.0e-4;
  //   gt->deltat_n = 1.0e-4;
  // } else if (test->Nom == test->NohTestCase ||
  //            test->Nom == test->BiNohTestCase) {
  //   // const ℝ δt_init = 1.0e-4;
  //   gt->deltat_init = 1.0e-4;
  //   gt->deltat_n = 1.0e-4;
  // } else if (test->Nom == test->TriplePoint ||
  //            test->Nom == test->BiTriplePoint) {
  //   // const ℝ δt_init = 1.0e-5; avec donnees adimensionnées
  //   gt->deltat_init = 1.0e-5;  // avec pression de 1.e5 / 1.e-8
  //   gt->deltat_n = 1.0e-5;
  // }
}

/**
 * Job ExecuteTimeLoopN called @5.0 in simulate method.
 * In variables: C, m_pseudo_viscosity_nplus1, m_node_cellvolume_n,
 * m_node_cellvolume_nplus1, m_node_coord_n, m_node_coord_nplus1,
 * m_speed_velocity_n, m_cell_mass, m_cell_coord_nplus1, deltat_n, deltat_nplus1,
 * m_divu_n, m_internal_energy_n, m_internal_energy_nplus1, gamma, m,
 * m_pressure_n, m_density_n, m_density_nplus1, t_n, m_tau_density_n,
 * m_tau_density_nplus1, m_node_velocity_n, m_node_velocity_nplus1 Out variables: C,
 * m_pseudo_viscosity_nplus1, m_node_cellvolume_nplus1, V, m_node_coord_nplus1,
 * m_speed_velocity_nplus1, deltat_nplus1, m_divu_nplus1,
 * m_internal_energy_nplus1, m_pressure_nplus1, m_density_nplus1, t_nplus1,
 * m_tau_density_nplus1, m_node_velocity_nplus1
 */
void Vnr::executeTimeLoopN() noexcept {
  n = 0;
  bool continueLoop = true;
  do {
    global_timer.start();
    cpu_timer.start();
    n++;
    if (n != 1)
      std::cout << "[" << __CYAN__ << __BOLD__ << setw(3) << n
                << __RESET__ "] t = " << __BOLD__
                << setiosflags(std::ios::scientific) << setprecision(8)
                << setw(16) << gt->t_n << __RESET__;
    if (options->sansLagrange == 0) {
      computeCornerNormal();  // @1.0
      computeDeltaT();        // @1.0
      computeNodeVolume();    // @1.0
    } else {
      gt->deltat_nplus1 = gt->deltat_n;
    }
    computeTime();    // @2.0
    dumpVariables();  // @2.0

    if (options->sansLagrange == 0) {
      updateVelocity();                    // @2.0
      updateVelocityBoundaryConditions();  // @2.0
    } else {
      std::swap(m_node_velocity_n, m_node_velocity_nplus1);
    }

    updatePosition();  // @3.0
    updateCellPos();
    computeSubVol();  // @4.0
    updateRho();      // @5.0
    updatePeriodicBoundaryConditions();

    if (options->sansLagrange == 0) {
      computeTau();                  // @6.0
      computeDivU();                 // @7.0
      computeArtificialViscosity();  // @1.0
      updateEnergy();                // @6.0
      computeEOS();                  // @7.0
      computePressionMoyenne();      // @7.0
    } else {
      std::swap(m_internal_energy_n, m_internal_energy_nplus1);
    }
    if (options->AvecProjection == 1) {
      computeVariablesForRemap();
      computeCellQuantitesForRemap();
      computeFaceQuantitesForRemap();
      remap->computeGradPhiFace1();
      remap->computeGradPhi1();
      remap->computeUpwindFaceQuantitiesForProjection1();
      remap->computeUremap1();
      remap->computeDualUremap1();
      remap->computeGradPhiFace2();
      remap->computeGradPhi2();
      remap->computeUpwindFaceQuantitiesForProjection2();
      remap->computeUremap2();
      remap->computeDualUremap2();
      remapVariables();
      computeNodeMass();         // avec la masse des mailles recalculée dans
                                 // remapVariables()
      computeEOS();              // rappel EOS apres projection
      computePressionMoyenne();  // rappel Pression moyenne apres projection
    }

    // Evaluate loop condition with variables at time n
    continueLoop =
        (n + 1 < gt->max_time_iterations && gt->t_nplus1 < gt->final_time);

    // if (gt->t_nplus1 > 0.05) limiteurs->projectionAvecPlateauPente = 1;
    if (continueLoop) {
      // Switch variables to prepare next iteration
      std::swap(varlp->x_then_y_nplus1, varlp->x_then_y_n);
      std::swap(gt->t_nplus1, gt->t_n);
      std::swap(gt->deltat_nplus1, gt->deltat_n);
      std::swap(m_density_nplus1, m_density_n);
      std::swap(m_density_env_nplus1, m_density_env_n);
      std::swap(m_pressure_nplus1, m_pressure_n);
      std::swap(m_pressure_env_nplus1, m_pressure_env_n);
      std::swap(m_pseudo_viscosity_nplus1, m_pseudo_viscosity_n);
      std::swap(m_pseudo_viscosity_env_nplus1, m_pseudo_viscosity_env_n);
      std::swap(m_tau_density_nplus1, m_tau_density_n);
      std::swap(m_tau_density_env_nplus1, m_tau_density_env_n);
      std::swap(m_divu_nplus1, m_divu_n);
      std::swap(m_speed_velocity_nplus1, m_speed_velocity_n);
      std::swap(m_speed_velocity_env_nplus1, m_speed_velocity_env_n);
      std::swap(m_internal_energy_nplus1, m_internal_energy_n);
      std::swap(m_internal_energy_env_nplus1, m_internal_energy_env_n);
      std::swap(m_node_velocity_nplus1, m_node_velocity_n);
      if (options->AvecProjection == 0) {
        std::swap(m_cell_coord_nplus1, m_cell_coord_n);
        std::swap(m_node_cellvolume_nplus1, m_node_cellvolume_n);
        std::swap(m_node_coord_nplus1, m_node_coord_n);
      }
    }

    std::cout << " DT  = " << gt->deltat_nplus1 << std::endl;
    cpu_timer.stop();
    global_timer.stop();

    // Timers display
    if (!writer.isDisabled())
      std::cout << " {CPU: " << __BLUE__ << cpu_timer.print(true)
                << __RESET__ ", IO: " << __BLUE__ << io_timer.print(true)
                << __RESET__ "} ";
    else
      std::cout << " {CPU: " << __BLUE__ << cpu_timer.print(true)
                << __RESET__ ", IO: " << __RED__ << "none" << __RESET__ << "} ";

    // Progress
    std::cout << utils::progress_bar(n, gt->max_time_iterations, gt->t_n,
                                     gt->final_time, 25);
    std::cout << __BOLD__ << __CYAN__
              << utils::Timer::print(
                     utils::eta(n, gt->max_time_iterations, gt->t_n,
                                gt->final_time, gt->deltat_n, global_timer),
                     true)
              << __RESET__ << "\r";
    std::cout.flush();

    cpu_timer.reset();
    io_timer.reset();
  } while (continueLoop);
  // force a last output at the end
  dumpVariables();
}

void Vnr::dumpVariables() noexcept {
  nbCalls++;
  if (!writer.isDisabled() &&
      (gt->t_n >= lastDump + gt->output_time || gt->t_n == 0.)) {
    cpu_timer.stop();
    io_timer.start();
    std::map<string, double*> cellVariables;
    std::map<string, double*> nodeVariables;
    std::map<string, double*> partVariables;
    cellVariables.insert(
        pair<string, double*>("Pressure", m_pressure_n.data()));
    cellVariables.insert(pair<string, double*>("Density", m_density_n.data()));
    cellVariables.insert(
        pair<string, double*>("Energy", m_internal_energy_n.data()));
    if (options->nbmat > 1 && options->AvecProjection == 1)
      cellVariables.insert(
          pair<string, double*>("fracvol1", m_fracvol_env1.data()));
    if (options->nbmat > 2 && options->AvecProjection == 1)
      cellVariables.insert(
          pair<string, double*>("fracvol2", m_fracvol_env2.data()));
    nodeVariables.insert(
        pair<string, double*>("VitesseX", m_x_velocity.data()));
    nodeVariables.insert(
        pair<string, double*>("VitesseY", m_y_velocity.data()));
    auto quads = mesh->getGeometry()->getQuads();
    writer.writeFile(nbCalls, gt->t_n, nbNodes, m_node_coord_n.data(), nbCells,
                     quads.data(), cellVariables, nodeVariables);
    lastDump = gt->t_n;
    std::cout << " time = " << gt->t_n << " sortie demandée " << std::endl;
    io_timer.stop();
    cpu_timer.start();
  }
}

void Vnr::simulate() {
  std::cout << "\n"
            << __BLUE_BKG__ << __YELLOW__ << __BOLD__ << "\tStarting Vnr ..."
            << __RESET__ << "\n\n";

  std::cout << "[" << __GREEN__ << "MESH" << __RESET__
            << "]      X=" << __BOLD__ << cstmesh->X_EDGE_ELEMS << __RESET__
            << ", Y=" << __BOLD__ << cstmesh->Y_EDGE_ELEMS << __RESET__
            << ", X length=" << __BOLD__ << cstmesh->X_EDGE_LENGTH << __RESET__
            << ", Y length=" << __BOLD__ << cstmesh->Y_EDGE_LENGTH << __RESET__
            << std::endl;

  if (Kokkos::hwloc::available()) {
    std::cout << "[" << __GREEN__ << "TOPOLOGY" << __RESET__
              << "]  NUMA=" << __BOLD__
              << Kokkos::hwloc::get_available_numa_count()

              << __RESET__ << ", Cores/NUMA=" << __BOLD__
              << Kokkos::hwloc::get_available_cores_per_numa() << __RESET__
              << ", Threads/Core=" << __BOLD__
              << Kokkos::hwloc::get_available_threads_per_core() << __RESET__
              << std::endl;
  } else {
    std::cout << "[" << __GREEN__ << "TOPOLOGY" << __RESET__
              << "]  HWLOC unavailable cannot get topological informations"
              << std::endl;
  }

  // std::cout << "[" << __GREEN__ << "KOKKOS" << __RESET__ << "]    " <<
  // __BOLD__ << (is_same<MyLayout,Kokkos::LayoutLeft>::value?"Left":"Right")"
  // << __RESET__ << " layout" << std::endl;

  if (!writer.isDisabled())
    std::cout << "[" << __GREEN__ << "OUTPUT" << __RESET__
              << "]    VTK files stored in " << __BOLD__
              << writer.outputDirectory() << __RESET__ << " directory"
              << std::endl;
  else
    std::cout << "[" << __GREEN__ << "OUTPUT" << __RESET__ << "]    "
              << __BOLD__ << "Disabled" << __RESET__ << std::endl;

  initBoundaryConditions();
  initCellPos();  // @1.0
  init();         // @2.0
  initSubVol();   // @2.0
  initMeshGeometryForFaces();
  remap->FacesOfNode();  // pour la conectivité Noeud-face
  computeCellMass();     // @3.0
  if (options->sansLagrange == 0) initDeltaT();          // @3.0
  if (options->sansLagrange == 0) initInternalEnergy();  // @3.0
  if (options->sansLagrange == 0) initPseudo();          // @3.0
  setUpTimeLoopN();                                      // @4.0
  computeNodeMass();                                     // @4.0
  executeTimeLoopN();                                    // @5.0

  std::cout << __YELLOW__ << "\n\tDone ! Took " << __MAGENTA__ << __BOLD__
            << global_timer.print() << __RESET__ << std::endl;
}
