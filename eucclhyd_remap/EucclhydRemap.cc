#include <stdlib.h>  // for exit

#include <Kokkos_Core.hpp>  // for finalize
#include <iomanip>          // for operator<<, setw, setiosflags
#include <iostream>         // for operator<<, basic_ostream, cha...
#include <limits>           // for numeric_limits
#include <map>              // for map
#include <utility>          // for pair, swap

#include "../remap/Remap.h"
#include "Eucclhyd.h"
#include "types/MathFunctions.h"  // for min
#include "utils/Utils.h"          // for __RESET__, __BOLD__, __GREEN__

using namespace nablalib;

/**
 * Job dumpVariables called @2.0 in executeTimeLoopN method.
 * In variables: m_cell_coord_x, m_cell_coord_y, m_internal_energy_n, m, p,
 * m_density_n, t_n, v Out variables:
 */
KOKKOS_INLINE_FUNCTION
void Eucclhyd::dumpVariables() noexcept {
  // std::cout << " Deltat = " << deltat_n << std::endl;
  // std::cout << " ---------------------------"
  //           << " Energie totale(t=0) = " << m_global_total_energy_0
  //           << " Energie totale(Lagrange) = " << m_global_total_energy_L
  //           << " Energie totale(time) = " << m_global_total_energy_T <<
  //           std::endl;
  // std::cout << " ---------------------------"
  //           << " Masse totale(t=0) = " << m_total_masse_0
  //           << " Masse totale(Lagrange) = " << m_total_masse_L
  //           << " Masse totale(time) = " << m_total_masse_T << std::endl;
  nbCalls++;
  if (!writer.isDisabled() &&
      (gt->t_n >= lastDump + gt->output_time || gt->t_n == 0.)) {
    cpu_timer.stop();
    io_timer.start();
    std::map<string, double*> cellVariables;
    std::map<string, double*> nodeVariables;
    std::map<string, double*> partVariables;
    if (so->pression)
      cellVariables.insert(pair<string, double*>("Pressure", m_pressure.data()));
    if (so->densite)
      cellVariables.insert(pair<string, double*>("Density", m_density_n.data()));
    if (so->energie_interne)
      cellVariables.insert(pair<string, double*>("Energy", m_internal_energy_n.data()));
    if (so->fraction_volumique) {
      cellVariables.insert(pair<string, double*>("F1", m_fracvol_env1.data()));
      cellVariables.insert(pair<string, double*>("F2", m_fracvol_env2.data()));
      cellVariables.insert(pair<string, double*>("F3", m_fracvol_env3.data()));
    }
    if (so->interface) {
	cellVariables.insert(pair<string, double*>("interface12", m_interface12.data()));
	cellVariables.insert(pair<string, double*>("interface23", m_interface23.data()));
	cellVariables.insert(pair<string, double*>("interface13", m_interface13.data()));
    }	
    if (so->vitesse) {
      cellVariables.insert(pair<string, double*>("VelocityX", m_x_cell_velocity.data()));
      cellVariables.insert(pair<string, double*>("VelocityY", m_y_cell_velocity.data()));
    }    

    partVariables.insert(pair<string, double*>(
        "VolumePart", particules->m_particle_volume.data()));
    partVariables.insert(pair<string, double*>(
        "VxPart", particules->m_particle_velocity_n[0].data()));
    partVariables.insert(pair<string, double*>(
        "VyPart", particules->m_particle_velocity_n[1].data()));
    auto quads = mesh->getGeometry()->getQuads();
    writer.writeFile(nbCalls, gt->t_n, nbNodes, m_node_coord.data(), nbCells,
                     quads.data(), cellVariables, nodeVariables);
    writer_particle.writeFile(nbCalls, gt->t_n, particules->nbPart,
                              particules->m_particle_coord_n.data(), 0,
                              quads.data(), cellVariables, partVariables);
    lastDump = gt->t_n;
    std::cout << " time = " << gt->t_n << " sortie demandée " << std::endl;
    io_timer.stop();
    cpu_timer.start();
  }
}

/**
 * Job executeTimeLoopN called @4.0 in simulate method.
 * In variables: m_node_force_n, m_node_force_nplus1, G, M, m_node_dissipation,
 * ULagrange, Uremap1, Uremap2, m_cell_velocity_extrap, m_cell_velocity_n,
 * m_node_velocity_n, m_node_velocity_nplus1, X, XLagrange, m_cell_coord,
 * m_cell_coordLagrange, m_cell_coord_x, m_cell_coord_y, Xf, bottomBC,
 * bottomBCValue, c, cfl, deltat_n, deltat_nplus1, m_cell_deltat,
 * deltaxLagrange, eos, eosPerfectGas, m_internal_energy_n, faceLength,
 * faceNormal, faceNormalVelocity, gamma, gradPhi1, gradPhi2, gradPhiFace1,
 * gradPhiFace2, m_velocity_gradient, m_pressure_gradient, leftBC, leftBCValue,
 * lminus, m_lpc, lplus, m, nminus, nplus, outerFaceNormal, p,
 * m_pressure_extrap, m_cell_perimeter, phiFace1, phiFace2, projectionLimiterId,
 * projectionOrder, m_density_n, rightBC, rightBCValue, spaceOrder, t_n, topBC,
 * topBCValue, v, vLagrange, x_then_y_n Out variables: m_node_force_nplus1, G,
 * M, m_node_dissipation, ULagrange, Uremap1, Uremap2, m_cell_velocity_extrap,
 * m_cell_velocity_nplus1, m_node_velocity_nplus1, XLagrange, XcLagrange, c,
 * deltat_nplus1, m_cell_deltat, deltaxLagrange, m_internal_energy_nplus1,
 * faceNormalVelocity, gradPhi1, gradPhi2, gradPhiFace1, gradPhiFace2,
 * m_velocity_gradient, m_pressure_gradient, m, p, m_pressure_extrap, phiFace1,
 * phiFace2, m_density_nplus1, t_nplus1, vLagrange, x_then_y_nplus1
 */
KOKKOS_INLINE_FUNCTION
void Eucclhyd::executeTimeLoopN() noexcept {
  n = 0;
  bool continueLoop = true;
  do {
    global_timer.start();
    cpu_timer.start();
    n++;
    if (n != 1)
      std::cout << "[" << __CYAN__ << __BOLD__ << setw(3) << n
                << __RESET__ "] time = " << __BOLD__
                << setiosflags(std::ios::scientific) << setprecision(8)
                << setw(16) << gt->t_n << __RESET__;

    if (options->AvecParticules == 1) switchalpharho_rho();
    computeEOS();
    computePressionMoyenne();
    if (options->AvecParticules == 1) switchrho_alpharho();
    computeGradients();                         // @1.0
    computeMass();                              // @1.0
    computeDissipationMatrix();                 // @2.0
    computem_cell_deltat();                     // @2.0
    dumpVariables();                            // @2.0
    extrapolateValue();                         // @2.0
    computeG();                                 // @3.0
    computeNodeDissipationMatrixAndG();         // @3.0
    computedeltat();                            // @3.0
    computeBoundaryNodeVelocities();            // @4.0
    computeNodeVelocity();                      // @4.0
    updateTime();                               // @4.0
    computeFaceVelocity();                      // @5.0
    computeLagrangePosition();                  // @5.0
    computeSubCellForce();                      // @5.0
    computeLagrangeVolumeAndCenterOfGravity();  // @6.0
    computeFacedeltaxLagrange();                // @7.0
    updateCellCenteredLagrangeVariables();      // @7.0

    if (options->AvecParticules == 1) {
      PreparecellvariablesForParticles();
      particules->updateParticlePosition();
      particules->updateParticleCoefficients();
      particules->updateParticleVelocity();
      particules->updateParticleRetroaction();
    }
    if (options->AvecProjection == 1) {
      remap->computeGradPhiFace1();                        // @8.0
      remap->computeGradPhi1();                            // @9.0
      remap->computeUpwindFaceQuantitiesForProjection1();  // @10.0
      remap->computeUremap1();                             // @11.0
      remap->computeGradPhiFace2();                        // @12.0
      remap->computeGradPhi2();                            // @13.0
      remap->computeUpwindFaceQuantitiesForProjection2();  // @14.0
      remap->computeUremap2();                             // @15.0
      remapCellcenteredVariable();                         // @16.0
    }

    // Evaluate loop condition with variables at time n
    continueLoop =
        (n + 1 < gt->max_time_iterations && gt->t_nplus1 < gt->final_time);
    std::cout << " DT  = " << gt->deltat_nplus1 << std::endl;
    if (continueLoop) {
      // Switch variables to prepare next iteration
      std::swap(varlp->x_then_y_nplus1, varlp->x_then_y_n);
      std::swap(gt->t_nplus1, gt->t_n);
      std::swap(gt->deltat_nplus1, gt->deltat_n);
      std::swap(m_node_velocity_nplus1, m_node_velocity_n);
      std::swap(m_density_nplus1, m_density_n);
      std::swap(m_density_env_nplus1, m_density_env_n);
      std::swap(m_cell_velocity_nplus1, m_cell_velocity_n);
      std::swap(m_internal_energy_nplus1, m_internal_energy_n);
      std::swap(m_internal_energy_env_nplus1, m_internal_energy_env_n);
      std::swap(m_node_force_nplus1, m_node_force_n);
      std::swap(m_node_force_env_nplus1, m_node_force_env_n);
      if (options->AvecParticules == 1) {
        std::cout << " swap " << std::endl;
        std::swap(particules->m_particle_velocity_nplus1,
                  particules->m_particle_velocity_n);
        std::swap(particules->m_particle_coord_nplus1,
                  particules->m_particle_coord_n);
        std::cout << " fin swap " << std::endl;
      }
      if (options->AvecProjection == 0) {
        std::swap(varlp->vLagrange, m_euler_volume);
        std::swap(varlp->XLagrange, m_node_coord);
        std::swap(varlp->XfLagrange, varlp->Xf);
        std::swap(varlp->faceLengthLagrange, varlp->faceLength);
        std::swap(varlp->XcLagrange, m_cell_coord);
      }
    }
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
                                     gt->final_time, 30);

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
}

/**
 * Job computedeltat called @3.0 in executeTimeLoopN method.
 * In variables: cfl, deltat_n, m_cell_deltat
 * Out variables: deltat_nplus1
 */
KOKKOS_INLINE_FUNCTION
void Eucclhyd::computedeltat() noexcept {
  double reduction10(numeric_limits<double>::max());
  {
    Kokkos::Min<double> reducer(reduction10);
    Kokkos::parallel_reduce("reduction10", nbCells,
                            KOKKOS_LAMBDA(const int& cCells, double& x) {
                              reducer.join(x, m_cell_deltat(cCells));
                            },
                            reducer);
  }
  gt->deltat_nplus1 =
      MathFunctions::min(gt->cfl * reduction10, gt->deltat_n * 1.05);
  if (gt->deltat_nplus1 < gt->deltat_min) {
    std::cerr << "Fin de la simulation par pas de temps minimum "
      << gt->deltat_nplus1 << " < " << gt->deltat_min << " et " << reduction10 << std::endl;
    Kokkos::finalize();
    exit(1);
  }
}
/**
 * Job setUpTimeLoopN called @3.0 in simulate method.
 * In variables: m_node_force_n0, m_cell_velocity_n0, m_node_velocity_n0,
 * m_internal_energy_n0, m_density_n0 Out variables: m_node_force_n,
 * m_cell_velocity_n, m_node_velocity_n, m_internal_energy_n, m_density_n
 */
void Eucclhyd::setUpTimeLoopN() noexcept {
  deep_copy(m_node_coord, init->m_node_coord_n0);
  deep_copy(m_cell_coord, init->m_cell_coord_n0);
  deep_copy(m_euler_volume, init->m_euler_volume_n0);
  deep_copy(m_density_n, init->m_density_n0);
  deep_copy(m_density_env_n, init->m_density_env_n0);
  deep_copy(m_internal_energy_n, init->m_internal_energy_n0);
  deep_copy(m_internal_energy_env_n, init->m_internal_energy_env_n0);
  deep_copy(m_node_velocity_n, init->m_node_velocity_n0);
  deep_copy(m_mass_fraction_env, init->m_mass_fraction_env_n0);
  deep_copy(m_fracvol_env, init->m_fracvol_env_n0);
  // specfiques à eucclhyd
  deep_copy(m_cell_perimeter, init->m_cell_perimeter_n0);
  deep_copy(m_cell_velocity_n, init->m_cell_velocity_n0);
  deep_copy(m_node_force_n, init->m_node_force_n0);

  if (test->Nom == test->SodCaseX || test->Nom == test->SodCaseY ||
      test->Nom == test->BiSodCaseX || test->Nom == test->BiSodCaseY) {
    // const ℝ δt_init = 1.0e-4;
    gt->deltat_init = 1.0e-4;
  } else if (test->Nom == test->BiShockBubble) {
    gt->deltat_init = 1.e-7;
  } else if (test->Nom == test->SedovTestCase ||
             test->Nom == test->BiSedovTestCase) {
    // const ℝ δt_init = 1.0e-4;
    gt->deltat_init = 1.0e-4;
  } else if (test->Nom == test->NohTestCase ||
             test->Nom == test->BiNohTestCase) {
    // const ℝ δt_init = 1.0e-4;
    gt->deltat_init = 1.0e-4;
  } else if (test->Nom == test->TriplePoint ||
             test->Nom == test->BiTriplePoint) {
    // const ℝ δt_init = 1.0e-5; avec donnees adimensionnées
    gt->deltat_init = 1.0e-5;  // avec pression de 1.e5 / 1.e-8
  }
  gt->deltat_n = gt->deltat_init;
  m_global_total_energy_0 = 0.;
  Kokkos::parallel_for(
      "init_m_global_total_energy_0", nbCells,
      KOKKOS_LAMBDA(const int& cCells) {
        m_total_energy_0(cCells) =
            (init->m_density_n0(cCells) * m_euler_volume(cCells)) *
            (init->m_internal_energy_n0(cCells) +
             0.5 * (init->m_cell_velocity_n0(cCells)[0] *
                        init->m_cell_velocity_n0(cCells)[0] +
                    init->m_cell_velocity_n0(cCells)[1] *
                        init->m_cell_velocity_n0(cCells)[1]));
        m_global_masse_0(cCells) =
            (init->m_density_n0(cCells) * m_euler_volume(cCells));
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
  // pour les sorties au temps 0
  Kokkos::parallel_for(
      "initDensity", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        // pour les sorties au temps 0:
        m_fracvol_env1(cCells) = init->m_fracvol_env_n0(cCells)[0];
        m_fracvol_env2(cCells) = init->m_fracvol_env_n0(cCells)[1];
        m_fracvol_env3(cCells) = init->m_fracvol_env_n0(cCells)[2];
        m_x_cell_velocity(cCells) = init->m_cell_velocity_n0(cCells)[0];
        m_y_cell_velocity(cCells) = init->m_cell_velocity_n0(cCells)[1];

        m_cell_coord_x(cCells) = init->m_cell_coord_n0(cCells)[0];
        m_cell_coord_y(cCells) = init->m_cell_coord_n0(cCells)[1];
        // a deplacer pour valider le schéma particule
        particules->m_particlecell_euler_volume(cCells) =
            init->m_euler_volume_n0(cCells);
      });
}

/**
 * Job updateTime called @4.0 in executeTimeLoopN method.
 * In variables: deltat_nplus1, t_n
 * Out variables: t_nplus1
 */
KOKKOS_INLINE_FUNCTION
void Eucclhyd::updateTime() noexcept {
  gt->t_nplus1 = gt->t_n + gt->deltat_nplus1;
}

void Eucclhyd::simulate() {
  std::cout << "\n"
            << __BLUE_BKG__ << __YELLOW__ << __BOLD__
            << "\tStarting Eucclhyd ..." << __RESET__ << "\n\n";

  std::cout << "[" << __GREEN__ << "MESH" << __RESET__
            << "]      X=" << __BOLD__ << cstmesh->X_EDGE_ELEMS << __RESET__
            << ", Y=" << __BOLD__ << cstmesh->Y_EDGE_ELEMS << __RESET__
            << ", X length=" << __BOLD__ << cstmesh->X_EDGE_LENGTH << __RESET__
            << ", Y length=" << __BOLD__ << cstmesh->Y_EDGE_LENGTH << __RESET__
            << std::endl;

  if (Kokkos::hwloc::available()) {
    std::cout << "[" << __GREEN__ << "TOPOLOGY" << __RESET__
              << "]  NUMA=" << __BOLD__
              << Kokkos::hwloc::get_available_numa_count() << __RESET__
              << ", Cores/NUMA=" << __BOLD__
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

  init->initMeshGeometryForCells();  // @1.0
  init->initVpAndFpc();              // @1.0
  init->initBoundaryConditions();
  init->initVar();                   // @2.0
  init->initMeshGeometryForFaces();  // @2.0
  if (options->AvecParticules == 1) {
    particules->initPart();
    particules->updateParticleCoefficients();
    switchrho_alpharho();  // on travaille avec alpharho sauf pour l'EOS
  }
  setUpTimeLoopN();  // @3.0

  computeCornerNormal();  // @1.0

  executeTimeLoopN();  // @4.0
  std::cout << __YELLOW__ << "\n\tDone ! Took " << __MAGENTA__ << __BOLD__
            << global_timer.print() << __RESET__ << std::endl;
}
