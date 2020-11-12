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

void Initialisations::initVarRider(RealArray1D<dim> Xb) noexcept {
  Kokkos::parallel_for(
    "initVarShockBubble", nbCells, KOKKOS_LAMBDA(const int& cCells) {
      RealArray1D<dim>  cc = {{0.5, 0.5}};
      // rayon interne et externe
      double rb(0.15);
      RealArray1D<dim> v_init;
      if (test->Nom == test->RiderTx) 
	m_cell_velocity_n0(cCells) = {{1., 0.}};
      if (test->Nom == test->RiderTy)  
	m_cell_velocity_n0(cCells)= {{0., 1.}};
      if (test->Nom == test->RiderT45) 
	m_cell_velocity_n0(cCells) = {{1., 1.}};
      if (test->Nom == test->RiderRotation) {
	RealArray1D<dim>  dd = m_cell_coord_n0(cCells) - cc;
	double theta = std::atan2(dd[1], dd[0]);
	double omega = 4. * Pi;
	m_cell_velocity_n0(cCells)[0] = dd[0] * omega * std::sin(omega * 0. +  theta);
	m_cell_velocity_n0(cCells)[1] = - dd[1] * omega * std::cos(omega * 0. +  theta);
      }
      if (test->Nom == test->RiderVortex) {
	RealArray1D<dim>  dd = m_cell_coord_n0(cCells) - cc;
	double Phi = 1. / Pi * sin (Pi *  dd[0]) * sin (Pi *  dd[1]);
	m_cell_velocity_n0(cCells)[0] = Phi;
	m_cell_velocity_n0(cCells)[1] = Phi;
      }
      if (test->Nom == test->RiderDeformation) {	
	RealArray1D<dim>  dd = m_cell_coord_n0(cCells) - cc;
	double Phi = 1. / (4. * Pi) *
	  sin (4. * Pi *  (dd[0] + 0.5)) * sin (4. * Pi *  (dd[1] + 0.5));
	m_cell_velocity_n0(cCells)[0] = Phi;
	m_cell_velocity_n0(cCells)[1] = Phi;
      }
	
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
	double rnode = std::sqrt((m_node_coord_n0(pNodes)[0]- Xb[0]) * (m_node_coord_n0(pNodes)[0]- Xb[0])
				 + (m_node_coord_n0(pNodes)[1]- Xb[1]) * (m_node_coord_n0(pNodes)[1]- Xb[1]));
	rmin = std::min(rmin, rnode);
	if (rmin == rnode) pmin = pNodesOfCellC;
	rmax = std::max(rmax, rnode);
	if (rmax == rnode) pmax = pNodesOfCellC;
      }
      // Air partout
      m_density_env_n0(cCells)[0] = 1;
      m_density_env_n0(cCells)[0] = 0;
      m_density_n0(cCells) = 1;
      
      m_pressure_env_n0(cCells)[0] = 0.0;
      m_pressure_env_n0(cCells)[1] = 0.0;
      m_pressure_n0(cCells) = 0.0;
      
      m_internal_energy_env_n0(cCells)[0] = 1.;
      m_internal_energy_env_n0(cCells)[1] = 1.;
      m_internal_energy_n0(cCells) = 1.;
      
      m_fracvol_env_n0(cCells)[0] = 1.;
      m_fracvol_env_n0(cCells)[1] = 0.;
      
      m_mass_fraction_env_n0(cCells)[0] = 1.;
      m_mass_fraction_env_n0(cCells)[1] = 0.;
      // bulle surchargera l'aire
      // centre de la bulle
      double r = sqrt((m_cell_coord_n0(cCells)[0] - Xb[0]) *
		      (m_cell_coord_n0(cCells)[0] - Xb[0]) +
		      (m_cell_coord_n0(cCells)[1] - Xb[1]) *
		      (m_cell_coord_n0(cCells)[1] - Xb[1]));
      if (rmax < rb) {
	// maille pure de bulle
	m_density_env_n0(cCells)[0] = 0.0;
	m_density_env_n0(cCells)[1] = 1.0;
	m_density_n0(cCells) = 1.0;
	
	m_pressure_env_n0(cCells)[0] = 0.0;
	m_pressure_env_n0(cCells)[1] = 0.0;
	m_pressure_n0(cCells) = 0.0;
	
	m_internal_energy_env_n0(cCells)[0] = 1.;
	m_internal_energy_env_n0(cCells)[1] = 1.;
	m_internal_energy_n0(cCells) = 1.;
	
	m_fracvol_env_n0(cCells)[0] = 0.;
	m_fracvol_env_n0(cCells)[1] = 1.;
	
	m_mass_fraction_env_n0(cCells)[0] = 0.;
	m_mass_fraction_env_n0(cCells)[1] = 1.;
	
      } else if ((rmax >= rb) && (rmin < rb)) {
	double frac_b = (rb - rmin) / (rmax -rmin);
	m_density_env_n0(cCells)[0] = 1.-frac_b;
	m_density_env_n0(cCells)[1] = frac_b;
	m_density_n0(cCells) = 1.0;
	
	m_pressure_env_n0(cCells)[0] = 0.0;
	m_pressure_env_n0(cCells)[1] = 0.0;
	m_pressure_n0(cCells) = 0.0;
	
	m_internal_energy_env_n0(cCells)[0] = 1.;
	m_internal_energy_env_n0(cCells)[1] = 1.;
	m_internal_energy_n0(cCells) = 1.;	
	
	m_fracvol_env_n0(cCells)[0] = 1.-frac_b;
	m_fracvol_env_n0(cCells)[1] = frac_b;
	
	m_mass_fraction_env_n0(cCells)[0] = 1.-frac_b;
	m_mass_fraction_env_n0(cCells)[1] = frac_b;
      }	  	  

    });
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
      RealArray1D<dim>  cc = {{0.5, 0.5}};
      if (test->Nom == test->RiderTx) 
	m_node_velocity_n0(pNodes) = {{1., 0.}};
      if (test->Nom == test->RiderTy)  
	m_node_velocity_n0(pNodes) = {{0., 1.}};
      if (test->Nom == test->RiderT45) 
	m_node_velocity_n0(pNodes) = {{1., 1.}};
      if (test->Nom == test->RiderRotation) {
	RealArray1D<dim>  dd = m_node_coord_n0(pNodes) - cc;
	double theta = std::atan2(dd[1], dd[0]);
	double r = std::sqrt(dd[0] * dd[0] + dd[1] * dd[1]);
	double omega = 4. * Pi;
        m_node_velocity_n0(pNodes)[0] = - r * omega * std::sin(omega * 0. +  theta);
        m_node_velocity_n0(pNodes)[1] = r * omega * std::cos(omega * 0. +  theta);
      }
      if (test->Nom == test->RiderVortex) {
	RealArray1D<dim>  dd = m_node_coord_n0(pNodes) - cc;
	double Phi = 1. / Pi * sin (Pi *  dd[0]) * sin (Pi *  dd[1]);
	m_cell_velocity_n0(cCells)[0] = Phi;
	m_cell_velocity_n0(cCells)[1] = Phi;
      }
       if (test->Nom == test->RiderDeformation) {	
	RealArray1D<dim>  dd = m_cell_coord_n0(cCells) - cc;
	double Phi = 1. / (4. * Pi) *
	  sin (4. * Pi *  (dd[0] + 0.5)) * sin (4. * Pi *  (dd[1] + 0.5));
	m_cell_velocity_n0(cCells)[0] = Phi;
	m_cell_velocity_n0(cCells)[1] = Phi;
      }
    });
}

}  // namespace initlib
