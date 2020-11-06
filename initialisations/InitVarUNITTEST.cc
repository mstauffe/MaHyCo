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

void Initialisations::initVarUnitTest() noexcept {
  Kokkos::parallel_for("initDensity", nbCells,
    KOKKOS_LAMBDA(const int& cCells) {
    if (m_cell_coord_n0(cCells)[0] <= 0.5) {
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_fracvol_env_n0(cCells)[1] = 0.;
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[1] = 0.;
      m_density_n0(cCells) = 1.0;
      m_density_env_n0(cCells)[0] = 1.0;
      m_density_env_n0(cCells)[1] = 0.;
    } else {
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_fracvol_env_n0(cCells)[1] = 0.;
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[1] = 0.;
      m_density_n0(cCells) = 1.;
      m_density_env_n0(cCells)[0] = 1.;
      m_density_env_n0(cCells)[1] = 0.;
    }
    m_cell_velocity_n0(cCells)[0] = 1.0;
    m_cell_velocity_n0(cCells)[1] = 0.0;
  });
}
void Initialisations::initVarBiUnitTest() noexcept {
  Kokkos::parallel_for("initDensity", nbCells,
    KOKKOS_LAMBDA(const int& cCells) {
    if (m_cell_coord_n0(cCells)[0] <= 0.5) {
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_fracvol_env_n0(cCells)[1] = 0.;
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[1] = 0.;
      m_density_n0(cCells) = 1.0;
      m_density_env_n0(cCells)[0] = 1.0;
      m_density_env_n0(cCells)[1] = 0.;
    } else {
      m_fracvol_env_n0(cCells)[0] = 0.;
      m_fracvol_env_n0(cCells)[1] = 1.;
      m_mass_fraction_env_n0(cCells)[0] = 0.;
      m_mass_fraction_env_n0(cCells)[1] = 1.;
      m_density_n0(cCells) = 1.;
      m_density_env_n0(cCells)[0] = 0.;
      m_density_env_n0(cCells)[1] = 1.0;
    }    
    m_cell_velocity_n0(cCells)[0] = 1.0;
    m_cell_velocity_n0(cCells)[1] = 0.0;
  });
}

}  // namespace initlib
