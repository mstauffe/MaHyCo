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
 * Job init called @2.0 in simulate method.
 * In variables: m_cell_coord_n0, gamma
 * Out variables: m_speed_velocity_n0, m_pressure_n0, m_density_n0,
 * m_node_velocity_n0
 */
void Initialisations::initVar() noexcept {
  Kokkos::parallel_for("initDensity", nbCells,
                       KOKKOS_LAMBDA(const int& cCells) {
                         for (int imat = 0; imat < nbmatmax; imat++) {
                           m_fracvol_env_n0(cCells)[imat] = 0.0;
                           m_mass_fraction_env_n0(cCells)[imat] = 0.0;
                           m_density_env_n0(cCells)[imat] = 0.0;
                           m_pressure_env_n0(cCells)[imat] = 0.0;
			   m_internal_energy_env_n0(cCells)[imat] = 0.0;
                         }
                       });
  if (test->Nom == test->UnitTestCase) {

    initVarUnitTest();
    
  } else if (test->Nom == test->BiUnitTestCase) {

    initVarBiUnitTest();

  } else if (test->Nom == test->SodCaseX || test->Nom == test->SodCaseY) {
    
    initVarSOD();

  } else if (test->Nom == test->BiSodCaseX || test->Nom == test->BiSodCaseY) {
    
    initVarBiSOD();

  } else if (test->Nom == test->BiShockBubble) {
    
    initVarShockBubble();
    
  } else if (test->Nom == test->TriplePoint) {
    
    initVarTriplePoint() ;
    
  } else if (test->Nom == test->BiTriplePoint) {
    
    initVarBiTriplePoint();
    
  } else if (test->Nom == test->SedovTestCase) {

    initVarSEDOV();
    
  } else if (test->Nom == test->AdvectionX || test->Nom == test->AdvectionY) {
    
    initVarAdvection();

  } else if (test->Nom == test->BiAdvectionX ||
             test->Nom == test->BiAdvectionY) {
    
    initVarBiAdvection();
    
  } else if (test->Nom == test->BiAdvectionVitX ||
             test->Nom == test->BiAdvectionVitY) {
    
    initVarBiAdvectionVitesse();
    
  } else {
    std::cout << "Cas test inconnu " << std::endl;
    exit(1);
  }
  if (options->nbmat == 1) {
    Kokkos::parallel_for("initDensity", nbCells,
                         KOKKOS_LAMBDA(const int& cCells) {
                           m_fracvol_env_n0(cCells)[0] = 1.;
                           m_fracvol_env_n0(cCells)[1] = 0.;
                           m_fracvol_env_n0(cCells)[2] = 0.;

                           m_mass_fraction_env_n0(cCells)[0] = 1.;
                           m_mass_fraction_env_n0(cCells)[1] = 0.;
                           m_mass_fraction_env_n0(cCells)[2] = 0.;
                         });
  }
  Kokkos::parallel_for("init", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    // indicateur mailles mixtes
    int matcell(0);
    int imatpure(-1);
    for (int imat = 0; imat < nbmatmax; imat++)
      if (m_fracvol_env_n0(cCells)[imat] > options->threshold) {
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
 * Job initInternalEnergy called @3.0 in simulate method.
 * In variables: gamma, m_pressure_n0, m_density_n0
 * Out variables: m_internal_energy_n0
 */
void Initialisations::initInternalEnergy() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    for (int imat = 0; imat < nbmatmax; imat++)
      m_internal_energy_env_n0(cCells)[imat] =
          m_pressure_env_n0(cCells)[imat] /
          ((eos->gammap[imat] - 1.0) * m_density_env_n0(cCells)[imat]);
    m_internal_energy_n0(cCells) =
        m_pressure_n0(cCells) / ((eos->gamma - 1.0) * m_density_n0(cCells));
  });
}
} // namespace initlib


