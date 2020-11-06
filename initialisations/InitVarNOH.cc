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

void Initialisations::initVarNOH() noexcept {
  Kokkos::parallel_for("initVarNOH", nbCells,
    KOKKOS_LAMBDA(const int& cCells) {
    m_fracvol_env_n0(cCells)[0] = 1.;
    m_mass_fraction_env_n0(cCells)[0] = 1.;
    double p0(1.);
    double rho0(1.);
    double e0 = p0 / ((eos->gamma - 1.0) * rho0);
    m_pressure_n0(cCells) = p0;
    m_pressure_env_n0(cCells)[0] = p0;
    m_density_n0(cCells) = rho0;
    m_density_env_n0(cCells)[0] = rho0;
    m_internal_energy_n0(cCells) = e0;
    for (int imat = 0; imat < nbmatmax; imat++)
      m_internal_energy_env_n0(cCells)[imat] = e0;
    // vitesse
    double u0(0.);
    double n1 = m_cell_coord_n0(cCells)[0];
    double n2 = m_cell_coord_n0(cCells)[1];
    double normVect = MathFunctions::sqrt(n1 * n1 + n2 * n2);
    m_cell_velocity_n0(cCells)[0] = -u0 * n1 / normVect;
    m_cell_velocity_n0(cCells)[1] = -u0 * n2 / normVect;
  });
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    double u0(0.);
    double n1 = m_node_coord_n0(pNodes)[0];
    double n2 = m_node_coord_n0(pNodes)[1];
    double normVect = MathFunctions::sqrt(n1 * n1 + n2 * n2);
    m_node_velocity_n0(pNodes)[0] = -u0 * n1 / normVect;
    m_node_velocity_n0(pNodes)[1] = -u0 * n2 / normVect;
  });
}
void Initialisations::initVarBiNOH() noexcept {
  Kokkos::parallel_for("initVarNOH", nbCells,
    KOKKOS_LAMBDA(const int& cCells) {			 
    m_fracvol_env_n0(cCells)[0] = 1.;
    m_mass_fraction_env_n0(cCells)[0] = 1.;
    double p0(1.);
    double rho0(1.);
    double e0 = p0 / ((eos->gamma - 1.0) * rho0);
    m_pressure_n0(cCells) = p0;
    m_pressure_env_n0(cCells)[0] = p0;
    m_density_n0(cCells) = rho0;
    m_density_env_n0(cCells)[0] = rho0;
    m_internal_energy_n0(cCells) = e0;
    for (int imat = 0; imat < nbmatmax; imat++)
      m_internal_energy_env_n0(cCells)[imat] = e0;
    // vitesse
    double u0(0.);
    double n1 = m_cell_coord_n0(cCells)[0];
    double n2 = m_cell_coord_n0(cCells)[1];
    double normVect = MathFunctions::sqrt(n1 * n1 + n2 * n2);
    m_cell_velocity_n0(cCells)[0] = -u0 * n1 / normVect;
    m_cell_velocity_n0(cCells)[1] = -u0 * n2 / normVect;
  });
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    double u0(0.);
    double n1 = m_node_coord_n0(pNodes)[0];
    double n2 = m_node_coord_n0(pNodes)[1];
    double normVect = MathFunctions::sqrt(n1 * n1 + n2 * n2);
    m_node_velocity_n0(pNodes)[0] = -u0 * n1 / normVect;
    m_node_velocity_n0(pNodes)[1] = -u0 * n2 / normVect;
  });
}  
}  // namespace initlib
