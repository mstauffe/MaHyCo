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

void Initialisations::initVarAdvection() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    double r(0.);
    if (test->Nom == test->AdvectionX) r = m_cell_coord_n0(cCells)[0];
    if (test->Nom == test->AdvectionY) r = m_cell_coord_n0(cCells)[1];
    if (r < 0.3) {
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_density_n0(cCells) = 0.;
      m_pressure_n0(cCells) = 0.0;
      m_density_env_n0(cCells)[0] = 1.;
      m_pressure_env_n0(cCells)[0] = 0.0;
    } else if ((r > 0.3) && (r < 0.5)) {
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_density_n0(cCells) = 1.;
      m_pressure_n0(cCells) = 0.0;
      m_density_env_n0(cCells)[0] = 10.0;
      m_pressure_env_n0(cCells)[0] = 0.0;
    } else if (r > 0.5) {
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_density_n0(cCells) = 0.;
      m_pressure_n0(cCells) = 0.0;
      m_density_env_n0(cCells)[0] = 1.;
      m_pressure_env_n0(cCells)[0] = 0.0;
    }
    m_speed_velocity_env_n0(cCells)[0] = 1.;
    m_speed_velocity_env_n0(cCells)[1] = 1.;
    m_speed_velocity_n0(cCells) = min(m_speed_velocity_env_n0(cCells)[0],
                                      m_speed_velocity_env_n0(cCells)[1]);
    m_internal_energy_env_n0(cCells)[0] = 1.;
    m_internal_energy_env_n0(cCells)[1] = 1.;
    m_internal_energy_n0(cCells) = 1.;
  });
  const RealArray1D<dim> ex = {{1.0, 0.0}};
  const RealArray1D<dim> ey = {{0.0, 1.0}};
  RealArray1D<dim> u;
  if (test->Nom == test->AdvectionX) u = ex;
  if (test->Nom == test->AdvectionY) u = ey;
  for (size_t pNodes = 0; pNodes < nbNodes; pNodes++) {
    m_node_velocity_n0(pNodes) = u;
  }
}
void Initialisations::initVarBiAdvection() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    double r(0.);
    if (test->Nom == test->BiAdvectionX) r = m_cell_coord_n0(cCells)[0];
    if (test->Nom == test->BiAdvectionY) r = m_cell_coord_n0(cCells)[1];
    if (r < 0.3) {
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_density_n0(cCells) = 1.;
      m_pressure_n0(cCells) = 0.0;
      m_density_env_n0(cCells)[0] = 1.0;
      m_pressure_env_n0(cCells)[0] = 0.0;
    } else if ((r > 0.3) && (r < 0.5)) {
      m_fracvol_env_n0(cCells)[1] = 1.;
      m_mass_fraction_env_n0(cCells)[1] = 1.;
      m_density_n0(cCells) = 1.;
      m_pressure_n0(cCells) = 0.0;
      m_density_env_n0(cCells)[1] = 1.0;
      m_pressure_env_n0(cCells)[1] = 0.0;
    } else if (r > 0.5) {
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_density_n0(cCells) = 1.;
      m_pressure_n0(cCells) = 0.0;
      m_density_env_n0(cCells)[0] = 1.0;
      m_pressure_env_n0(cCells)[0] = 0.0;
    }
    m_speed_velocity_env_n0(cCells)[0] = 1.;
    m_speed_velocity_env_n0(cCells)[1] = 1.;
    m_speed_velocity_n0(cCells) = min(m_speed_velocity_env_n0(cCells)[0],
                                      m_speed_velocity_env_n0(cCells)[1]);
    m_internal_energy_env_n0(cCells)[0] = 1.;
    m_internal_energy_env_n0(cCells)[1] = 1.;
    m_internal_energy_n0(cCells) = 1.;
  });
  const RealArray1D<dim> ex = {{1.0, 0.0}};
  const RealArray1D<dim> ey = {{0.0, 1.0}};
  RealArray1D<dim> u;
  if (test->Nom == test->BiAdvectionX) u = ex;
  if (test->Nom == test->BiAdvectionY) u = ey;
  for (size_t pNodes = 0; pNodes < nbNodes; pNodes++) {
    m_node_velocity_n0(pNodes) = u;
  }
}
void Initialisations::initVarBiAdvectionVitesse() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    double r(0.);
    if (test->Nom == test->BiAdvectionVitX) r = m_cell_coord_n0(cCells)[0];
    if (test->Nom == test->BiAdvectionVitY) r = m_cell_coord_n0(cCells)[1];
    if (r < 0.3) {
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_density_n0(cCells) = 4.;
      m_pressure_n0(cCells) = 0.0;
      m_density_env_n0(cCells)[0] = 4.0;
      m_pressure_env_n0(cCells)[0] = 0.0;
    } else if ((r > 0.3) && (r < 0.5)) {
      m_fracvol_env_n0(cCells)[1] = 1.;
      m_mass_fraction_env_n0(cCells)[1] = 1.;
      m_density_n0(cCells) = 4.;
      m_pressure_n0(cCells) = 0.0;
      m_density_env_n0(cCells)[1] = 1.0;
      m_pressure_env_n0(cCells)[1] = 0.0;
    } else if (r > 0.5) {
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_density_n0(cCells) = 4.;
      m_pressure_n0(cCells) = 0.0;
      m_density_env_n0(cCells)[0] = 4.0;
      m_pressure_env_n0(cCells)[0] = 0.0;
    }
    m_speed_velocity_env_n0(cCells)[0] = 1.;
    m_speed_velocity_env_n0(cCells)[1] = 1.;
    m_speed_velocity_n0(cCells) = min(m_speed_velocity_env_n0(cCells)[0],
                                      m_speed_velocity_env_n0(cCells)[1]);
    m_internal_energy_env_n0(cCells)[0] = 1.;
    m_internal_energy_env_n0(cCells)[1] = 1.;
    m_internal_energy_n0(cCells) = 1.;
  });
  const RealArray1D<dim> ex = {{1.0, 0.0}};
  const RealArray1D<dim> ey = {{0.0, 1.0}};
  RealArray1D<dim> u;
  if (test->Nom == test->BiAdvectionVitX) u = ex;
  if (test->Nom == test->BiAdvectionVitY) u = ey;
  for (size_t pNodes = 0; pNodes < nbNodes; pNodes++) {
    double r(0.);
    if (test->Nom == test->BiAdvectionVitX) r = m_node_coord_n0(pNodes)[0];
    if (test->Nom == test->BiAdvectionVitY) r = m_node_coord_n0(pNodes)[1];
    if (r <= 0.3) {
      m_node_velocity_n0(pNodes) = 0.2 * u;
    } else if ((r > 0.3) && (r <= 0.5)) {
      m_node_velocity_n0(pNodes) = (4. * r - 1.) * u;
    } else if (r > 0.5) {
      m_node_velocity_n0(pNodes) = u;
    }
  }
}
}  // namespace initlib
