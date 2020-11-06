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

void Initialisations::initVarShockBubble() noexcept {
  Kokkos::parallel_for("initVarShockBubble", nbCells,
    KOKKOS_LAMBDA(const int& cCells) {
    double pInit;
    double rhoInit;
    double eInit;
    // Air partout
    rhoInit = 1.0;
    pInit = 1.e5;
    eInit = pInit / ((eos->gammap[0] - 1.0) * rhoInit);
    m_density_env_n0(cCells)[0] = rhoInit;
    m_density_n0(cCells) = rhoInit;
    
    m_pressure_env_n0(cCells)[0] = pInit;
    m_pressure_env_n0(cCells)[1] = 0.0;
    m_pressure_n0(cCells) = pInit;
      
    m_internal_energy_env_n0(cCells)[0] = eInit;
    m_internal_energy_env_n0(cCells)[1] = 0.;
    m_internal_energy_n0(cCells) = eInit;
    
    m_fracvol_env_n0(cCells)[0] = 1.;
    m_fracvol_env_n0(cCells)[1] = 0.;
    
    m_mass_fraction_env_n0(cCells)[0] = 1.;
    m_mass_fraction_env_n0(cCells)[1] = 0.;
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
      
      m_density_env_n0(cCells)[0] = 0.0;
      m_density_env_n0(cCells)[1] = rhoInit;
      m_density_n0(cCells) = rhoInit;
      
      m_pressure_env_n0(cCells)[0] = 0.0;
      m_pressure_env_n0(cCells)[1] = pInit;
      m_pressure_n0(cCells) = pInit;
    
      m_internal_energy_env_n0(cCells)[0] = 0.;
      m_internal_energy_env_n0(cCells)[1] = eInit;
      m_internal_energy_n0(cCells) = eInit;
      
      m_fracvol_env_n0(cCells)[0] = 0.;
      m_fracvol_env_n0(cCells)[1] = 1.;
      
      m_mass_fraction_env_n0(cCells)[0] = 0.;
      m_mass_fraction_env_n0(cCells)[1] = 1.;
    }
    if (m_cell_coord_n0(cCells)[0] >= 0.60) {
      m_cell_velocity_n0(cCells)[0] = -124.824;
      m_cell_velocity_n0(cCells)[1] = 0.0;
    }
  });
}

}  // namespace initlib
