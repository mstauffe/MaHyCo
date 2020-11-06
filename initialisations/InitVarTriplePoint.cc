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

void Initialisations::initVarTriplePoint() noexcept {
  Kokkos::parallel_for("initDensity", nbCells,
    KOKKOS_LAMBDA(const int& cCells) {
    m_fracvol_env_n0(cCells)[0] = 1.;
    m_fracvol_env_n0(cCells)[1] = 0.;
    m_mass_fraction_env_n0(cCells)[0] = 1.;
    m_mass_fraction_env_n0(cCells)[1] = 0.;
    double pInit;
    double rhoInit;
    double eInit;
    if (m_cell_coord_n0(cCells)[0] <= 0.01) {
      // std::cout << " cell " << cCells << "  x= " <<
      // m_cell_coord_n0(cCells)[0] << " y= " <<
      // m_cell_coord_n0(cCells)[1] << std::endl;
      m_density_n0(cCells) = 1.0;
      pInit = 1.0;
      rhoInit = 1.0;
    } else {
      if (m_cell_coord_n0(cCells)[1] <= 0.015) {
	// std::cout << " cell cas 2  " << cCells << " x=
	// " << m_cell_coord_n0(cCells)[0] << "  y= " <<
	// m_cell_coord_n0(cCells)[1]
	// << std::endl;
	m_density_n0(cCells) = 1.0;
	pInit = 0.1;
	rhoInit = 1.;
      } else {
	// std::cout << " cell cas 3  " << cCells << " x=
	// " << m_cell_coord_n0(cCells)[0] << "  y= " <<
	// m_cell_coord_n0(cCells)[1]
	// << std::endl;
	m_density_n0(cCells) = 0.1;
	pInit = 0.1;
	rhoInit = 0.1;
      }
    }
    eInit = pInit / ((eos->gammap[0] - 1.0) * rhoInit);
    m_internal_energy_n0(cCells) = eInit;
    m_internal_energy_env_n0(cCells)[0] = eInit;
    // vitesses      
    m_cell_velocity_n0(cCells)[0] = 0.0;
    m_cell_velocity_n0(cCells)[1] = 0.0;
  });
}
void Initialisations::initVarBiTriplePoint() noexcept {
  Kokkos::parallel_for("initDensity", nbCells,
    KOKKOS_LAMBDA(const int& cCells) { 
    double pInit;
    double rhoInit;
    double eInit;
    if (m_cell_coord_n0(cCells)[0] <= 0.01) {
      // std::cout << " cell cas 1 " << cCells << "  x= "
      // << m_cell_coord_n0(cCells)[0]
      //          << "  y= " << m_cell_coord_n0(cCells)[1]
      //          << std::endl;
      m_density_n0(cCells) = 1.0;
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_density_env_n0(cCells)[0] = 1.0;
      pInit = 1.0;  // 1.e5; // 1.0;
      rhoInit = 1.0;
      eInit = pInit / ((eos->gammap[0] - 1.0) * rhoInit);
      m_internal_energy_n0(cCells) = eInit;
      m_internal_energy_env_n0(cCells)[0] = eInit;     
    } else {
      if (m_cell_coord_n0(cCells)[1] <= 0.015) {
	// std::cout << "cell cas 2  " <<cCells << "  x=
	// " <<m_cell_coord_n0(cCells)[0]
	//          << "  y= " << m_cell_coord_n0(cCells)[1]
	//          << std::endl;
	m_density_n0(cCells) = 1.0;
	m_fracvol_env_n0(cCells)[1] = 1.;
	m_mass_fraction_env_n0(cCells)[1] = 1.;
	m_density_env_n0(cCells)[1] = 1.0;
	pInit = 0.1;  // 1.e4; // 0.1;
	rhoInit = 1.;
	eInit = pInit / ((eos->gammap[1] - 1.0) * rhoInit);
	m_internal_energy_n0(cCells) = eInit;
	m_internal_energy_env_n0(cCells)[1] = eInit;
      } else {
	// std::cout << "cell cas 3  " << cCells << "  x=
	// " <<m_cell_coord_n0(cCells)[0]
	//          << "  y= " << m_cell_coord_n0(cCells)[1]
	//          << std::endl;
	m_density_n0(cCells) = 0.1;
	m_fracvol_env_n0(cCells)[2] = 1.;
	m_mass_fraction_env_n0(cCells)[2] = 1.;
	m_density_env_n0(cCells)[2] = 0.1;
	pInit = 0.1;  // 1.e4; // 0.1;
	rhoInit = 0.1;
	eInit = pInit / ((eos->gammap[2] - 1.0) * rhoInit);
	m_internal_energy_n0(cCells) = eInit;
	m_internal_energy_env_n0(cCells)[2] = eInit;
      }
    }
    // vitesses      
    m_cell_velocity_n0(cCells)[0] = 0.0;
    m_cell_velocity_n0(cCells)[1] = 0.0;
  });
}
}  // namespace initlib
