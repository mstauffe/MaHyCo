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
  // centre en 0.0 , 0.0
  // rayon boule centre 1.
  // rayon externe 1.2
void Initialisations::initVarImplosion() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    // light fluid
    double pInit_l(0.1);
    double rhoInit_l(0.05);
    double eInit_l = pInit_l / ((eos->gamma[0] - 1.0) * rhoInit_l);
    // eavy fluid
    double pInit_e(0.1);
    double rhoInit_e(1.);
    double eInit_e = pInit_e / ((eos->gamma[0] - 1.0) * rhoInit_e);
    // very light fluid
    double pInit_ll(0.001);
    double rhoInit_ll = 0.01*0.05;
    double eInit_ll = pInit_ll / ((eos->gamma[0] - 1.0) * rhoInit_ll);
    // rayon interne et externe
    double ri(1.0), re(1.2);
    // parametres maille
    double rmin(10.), rmax(0.);
    size_t pmin, pmax;
    int cId(cCells);
    const auto nodesOfCellC(mesh->getNodesOfCell(cId));
    const size_t nbNodesOfCellC(nodesOfCellC.size());
    for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
	 pNodesOfCellC++) {
      const Id pId(nodesOfCellC[pNodesOfCellC]);
      const size_t pNodes(pId);
      double rnode = std::sqrt(m_node_coord_n0(pNodes)[0] * m_node_coord_n0(pNodes)[0]
			     + m_node_coord_n0(pNodes)[1] * m_node_coord_n0(pNodes)[1]);
      rmin = std::min(rmin, rnode);
      if (rmin == rnode) pmin = pNodesOfCellC;
      rmax = std::max(rmax, rnode);
      if (rmax == rnode) pmax = pNodesOfCellC;
    }
    //std::cout << " cell " << cCells << " rmax " << rmax << " rmin " << rmin
    //	      << " ri " << ri << " re " << re  << std::endl;

    if (rmax <= ri) {
      // maille pure de light fluid
      m_density_env_n0(cCells)[0] = rhoInit_l;
      m_pressure_env_n0(cCells)[0] = pInit_l;
      m_internal_energy_env_n0(cCells)[0] = eInit_l;
      // std::cout << " cell pure light " << cCells << std::endl;
    } else if ((rmax > ri) && (rmin <= ri)) {
      // maille mixte light fuid and eavy fluid
      double frac_l = (ri - rmin) / (rmax -rmin);
      double rhoInit_le = frac_l * rhoInit_l + (1.-frac_l) * rhoInit_e;
      m_density_env_n0(cCells)[0] = rhoInit_le;
      m_pressure_env_n0(cCells)[0] = pInit_l;
      m_internal_energy_env_n0(cCells)[0] = pInit_l /
	((eos->gamma[0] - 1.0) *rhoInit_le);	
      // std::cout << " cell pure light-eavy " << cCells <<
      // " frac " << frac_l << " et " << rhoInit_le << std::endl;

    } else if ((rmin > ri) && (rmax <= re)) {
      // maille pure de eavy fluid
      m_density_env_n0(cCells)[0] = rhoInit_e;
      m_pressure_env_n0(cCells)[0] = pInit_e;
      m_internal_energy_env_n0(cCells)[0] = eInit_e;
      //std::cout << " cell pure eavy " << cCells << std::endl;
      
    } else if ((rmax > re) && (rmin <= re)) {
      // maille mixte eavy fluid and very light fluid
      double frac_e = (re - rmin) / (rmax -rmin);
      double rhoInit_ell = frac_e * rhoInit_e + (1.-frac_e) * rhoInit_ll;
      double pInit_ell = frac_e * pInit_e + (1.-frac_e) * pInit_ll;
      m_density_env_n0(cCells)[0] = rhoInit_ell;
      m_pressure_env_n0(cCells)[0] = pInit_ell;
      m_internal_energy_env_n0(cCells)[0] = pInit_ell /
	((eos->gamma[0] - 1.0) *rhoInit_ell);	
      // std::cout << " cell pure very light-eavy " << cCells <<
      // " rho " << m_density_env_n0(cCells)[0] << std::endl;	
    } else if (rmin > re) {
      // maille pure very light fluid
      m_density_env_n0(cCells)[0] = rhoInit_ll;
      m_pressure_env_n0(cCells)[0] = pInit_ll;
      m_internal_energy_env_n0(cCells)[0] = eInit_ll;
      //std::cout << " cell pure very light " << cCells << std::endl;
    }
    m_density_n0(cCells) = m_density_env_n0(cCells)[0];
    m_pressure_n0(cCells) = m_pressure_env_n0(cCells)[0];
    m_internal_energy_n0(cCells) = m_internal_energy_env_n0(cCells)[0];
    m_speed_velocity_env_n0(cCells)[0] =
      std::sqrt(eos->gamma[0] * m_density_env_n0(cCells)[0] /
		m_pressure_env_n0(cCells)[0]);
    m_speed_velocity_n0(cCells) = m_speed_velocity_env_n0(cCells)[0];
    
    m_fracvol_env_n0(cCells)[0] = 1.;
    m_mass_fraction_env_n0(cCells)[0] = 1.;

    // if ( m_speed_velocity_n0(cCells) < 0.83666)
    //std::cout << " cell " << cCells << " " << rmin << " " << rmax <<
    //	" " << m_speed_velocity_n0(cCells) << std::endl;

    // vitesses
    m_cell_velocity_n0(cCells) = {0.0, 0.0};
  });
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    m_node_velocity_n0(pNodes) = {0.0, 0.0};
  });
}
void Initialisations::initVarBiImplosion() noexcept {

}

}  // namespace initlib
