#ifndef SCHEMAPARTICULES_H
#define SCHEMAPARTICULES_H

#include <Kokkos_Core.hpp>                // for KOKKOS_LAMBDA
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>  // for OpenMP::impl_is_initialized
#include "../includes/Constantes.h"
#include "../includes/GestionTemps.h"
#include "../includes/CasTest.h"
#include "../includes/CstMesh.h"
#include "mesh/CartesianMesh2D.h"            // for CartesianMesh2D

using namespace nablalib;

namespace particleslib {
  
class SchemaParticules{
 public:
  struct Particules {
    int DragModel;
    int Kliatchko = 20;
    int Classique = 21;
    int KliatchkoDragModel = 20;

    double Reynolds_min = 1.e-4;
    double Reynolds_max = 1.e3;
    double Drag = 10.;
  };
  Particules* particules;

 private:
  CartesianMesh2D* mesh;
  castestlib::CasTest::Test* test;
  cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh;
  gesttempslib::GestionTempsClass::GestTemps* gt;

  int nbPartMax;
  int nbPart = 0;
  int nbCells;

 public:
  
  Kokkos::View<double*> m_particle_volume;
  Kokkos::View<double*> m_particle_weight;
  Kokkos::View<double*> m_particle_mass;
  Kokkos::View<double*> m_particle_radius;
  Kokkos::View<double*> m_particle_density;
  Kokkos::View<double*> m_particle_drag;
  Kokkos::View<double*> m_particle_mac;
  Kokkos::View<double*> m_particle_reynolds;
  Kokkos::View<double*> m_particle_temperature;
  Kokkos::View<RealArray1D<dim>*> m_particle_coord_n0;
  Kokkos::View<RealArray1D<dim>*> m_particle_coord_n;
  Kokkos::View<RealArray1D<dim>*> m_particle_coord_nplus1;
  Kokkos::View<RealArray1D<dim>*> m_particle_velocity_n0;
  Kokkos::View<RealArray1D<dim>*> m_particle_velocity_n;
  Kokkos::View<RealArray1D<dim>*> m_particle_velocity_nplus1;
  Kokkos::View<int*> m_particle_cell;
  Kokkos::View<int*> m_particle_env;
  Kokkos::View<double*> m_cell_particle_volume_fraction;
  Kokkos::View<vector<int>*> m_cell_particle_list;
  // variables du schéma eucclhyd ou VNR
  Kokkos::View<RealArray1D<nbmatmax>*> m_particlecell_fracvol_env;
  Kokkos::View<RealArray1D<dim>**> m_particlecell_fracvol_gradient_env;
  Kokkos::View<double*> m_particlecell_euler_volume;
  Kokkos::View<RealArray1D<dim>*> m_particlecell_velocity_n;
  Kokkos::View<RealArray1D<dim>*> m_particlecell_velocity_nplus1;
  Kokkos::View<double*> m_particlecell_mass;
  Kokkos::View<double*> m_particlecell_density_n;
  Kokkos::View<RealArray1D<nbmatmax>*> m_particlecell_density_env_n;
  Kokkos::View<RealArray1D<dim>*> m_particlecell_pressure_gradient;
  
  

 public:
 SchemaParticules(
    CartesianMesh2D* aCartesianMesh2D,
    cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* acstmesh,
    gesttempslib::GestionTempsClass::GestTemps* agt,
    castestlib::CasTest::Test* aTest)
   : mesh(aCartesianMesh2D),
     test(aTest),
     gt(agt),
     nbCells(mesh->getNbCells()),
     nbPartMax(1),
    m_particle_volume("particle_volume", nbPartMax),
    m_particle_weight("particle_weight", nbPartMax),
    m_particle_mass("particle_mass", nbPartMax),
    m_particle_radius("particle_radius", nbPartMax),
    m_particle_density("particle_density", nbPartMax),
    m_particle_reynolds("particle_reynolds", nbPartMax),
    m_particle_drag("particle_drag", nbPartMax),
    m_particle_mac("particle_drag", nbPartMax),
    m_cell_particle_list("listepart", nbCells),
    m_cell_particle_volume_fraction("fracPart", nbCells),
    m_particle_coord_n0("particle_coord_n0", nbPartMax),
    m_particle_coord_n("particle_coord_n", nbPartMax),
    m_particle_coord_nplus1("particle_coord_nplus1", nbPartMax),
    m_particle_velocity_n0("particle_velocity_n0", nbPartMax),
    m_particle_velocity_n("particle_velocity_n", nbPartMax),
    m_particle_velocity_nplus1("particle_velocity_nplus1", nbPartMax),
    m_particle_cell("Icellp", nbPartMax),
    m_particle_env("Imatp", nbPartMax),
    m_particlecell_fracvol_env("fracvol_env", nbCells),
    m_particlecell_fracvol_gradient_env("fracvol_gradient_env", nbCells, nbmatmax),
    m_particlecell_euler_volume("euler_volume", nbCells),
    m_particlecell_velocity_n("cell_velocity_n", nbCells),
    m_particlecell_velocity_nplus1("cell_velocity_nplus1", nbCells),
    m_particlecell_mass("cell_mass", nbCells),
    m_particlecell_density_n("density_n", nbCells),
    m_particlecell_density_env_n("density_env_n", nbCells),
    m_particlecell_pressure_gradient("particle_pressure_gradient", nbCells)
      {}
  
  void initPart() noexcept;
  void updateParticlePosition() noexcept;
  void updateParticleCoefficients() noexcept;
  void updateParticleVelocity() noexcept;
  void updateParticleRetroaction() noexcept;

 };
}
#endif  // SCHEMAPARTICULES_H
