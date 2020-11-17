#ifndef INITIALISATIONS_H
#define INITIALISATIONS_H

#include <Kokkos_Core.hpp>                // for KOKKOS_LAMBDA
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>  // for OpenMP::impl_is_initialized

#include "../includes/CasTest.h"
#include "../includes/ConditionsLimites.h"
#include "../includes/Constantes.h"
#include "../includes/CstMesh.h"
#include "../includes/Eos.h"
#include "../includes/Freefunctions.h"
#include "../includes/GestionTemps.h"
#include "../includes/Options.h"
#include "../includes/VariablesLagRemap.h"
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D

using namespace nablalib;

namespace initlib {
class Initialisations {
 public:
  CartesianMesh2D* mesh;
  optionschemalib::OptionsSchema::Options* options;
  castestlib::CasTest::Test* test;
  eoslib::EquationDetat* eos;
  cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh;
  conditionslimiteslib::ConditionsLimites::Cdl* cdl;
  variableslagremaplib::VariablesLagRemap* varlp;
  int nbNodes, nbCells, nbFaces, nbFacesnbCellsOfNode, nbNodesOfCell,
      nbNodesOfFace, nbCellsOfFace, nbCellsOfNode;

  // Variables
  Kokkos::View<RealArray1D<dim>*> m_node_coord_n0;
  Kokkos::View<RealArray1D<dim>*> m_cell_coord_n0;
  Kokkos::View<double*> m_euler_volume_n0;
  Kokkos::View<double*> m_density_n0;
  Kokkos::View<RealArray1D<nbmatmax>*> m_density_env_n0;
  Kokkos::View<double*> m_internal_energy_n0;
  Kokkos::View<RealArray1D<nbmatmax>*> m_internal_energy_env_n0;
  Kokkos::View<RealArray1D<dim>*> m_node_velocity_n0;
  Kokkos::View<RealArray1D<nbmatmax>*> m_mass_fraction_env_n0;
  Kokkos::View<RealArray1D<nbmatmax>*> m_fracvol_env_n0;
  // init EUC
  Kokkos::View<double*> m_cell_perimeter_n0;
  Kokkos::View<RealArray1D<dim>*> m_cell_velocity_n0;
  Kokkos::View<RealArray1D<dim>**> m_node_force_n0;

  // init VNR
  Kokkos::View<double**> m_node_cellvolume_n0;
  Kokkos::View<double*> m_pressure_n0;
  Kokkos::View<RealArray1D<nbmatmax>*> m_pressure_env_n0;
  Kokkos::View<double*> m_pseudo_viscosity_n0;
  Kokkos::View<RealArray1D<nbmatmax>*> m_pseudo_viscosity_env_n0;
  Kokkos::View<double*> m_tau_density_n0;
  Kokkos::View<RealArray1D<nbmatmax>*> m_tau_density_env_n0;
  Kokkos::View<double*> m_divu_n0;
  Kokkos::View<double*> m_speed_velocity_n0;
  Kokkos::View<RealArray1D<nbmatmax>*> m_speed_velocity_env_n0;

 public:
  Initialisations(
      optionschemalib::OptionsSchema::Options* aOptions,
      eoslib::EquationDetat* aEos, CartesianMesh2D* aCartesianMesh2D,
      cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* acstmesh,
      variableslagremaplib::VariablesLagRemap* avarlp,
      conditionslimiteslib::ConditionsLimites::Cdl* aCdl,
      castestlib::CasTest::Test* aTest)
      : options(aOptions),
        eos(aEos),
        mesh(aCartesianMesh2D),
        test(aTest),
        varlp(avarlp),
        cdl(aCdl),
        cstmesh(acstmesh),
        nbCells(mesh->getNbCells()),
        nbNodes(mesh->getNbNodes()),
        nbFaces(mesh->getNbFaces()),
        nbCellsOfNode(CartesianMesh2D::MaxNbCellsOfNode),
        nbNodesOfCell(CartesianMesh2D::MaxNbNodesOfCell),
        nbNodesOfFace(CartesianMesh2D::MaxNbNodesOfFace),
        m_node_coord_n0("node_coord", nbNodes),
        m_cell_coord_n0("cell_coord", nbCells),
        m_euler_volume_n0("euler_volume", nbCells),
        m_node_cellvolume_n0("node_cellvolume_n0", nbCells, nbNodesOfCell),
        m_cell_perimeter_n0("cell_perimeter", nbCells),
        m_node_velocity_n0("node_velocity_n0", nbNodes),
        m_density_n0("density_n0", nbCells),
        m_density_env_n0("density_env_n0", nbCells),
        m_cell_velocity_n0("cell_velocity_n0", nbCells),
        m_internal_energy_n0("internal_energy_n0", nbCells),
        m_internal_energy_env_n0("internal_energy_env_n0", nbCells),
        m_node_force_n0("node_force_n0", nbNodes, nbCellsOfNode),
        m_mass_fraction_env_n0("mass_fraction_env_n0", nbCells),
        m_fracvol_env_n0("fracvol_env_n0", nbCells),
        m_pressure_n0("pressure_n0", nbCells),
        m_pressure_env_n0("pressure_env_n0", nbCells),
        m_tau_density_n0("tau_density_n0", nbCells),
        m_tau_density_env_n0("tau_density_env_n0", nbCells),
        m_divu_n0("divu_n0", nbCells),
        m_pseudo_viscosity_n0("pseudo_viscosity_n", nbCells),
        m_pseudo_viscosity_env_n0("pseudo_viscosity_env_n", nbCells),
        m_speed_velocity_n0("speed_velocity_n0", nbCells),
        m_speed_velocity_env_n0("speed_velocity_n0", nbCells) {
    // Copy node coordinates
    const auto& gNodes = mesh->getGeometry()->getNodes();
    Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const int& rNodes) {
      m_node_coord_n0(rNodes) = gNodes[rNodes];
    });
  }
  // initalisation commune
  void initBoundaryConditions() noexcept;
  void initMeshGeometryForCells() noexcept;
  void initMeshGeometryForFaces() noexcept;
  void initVpAndFpc() noexcept;
  // initialisation des variables EU
  void initCellInternalEnergy() noexcept;
  // initialisation specifique VNR
  void initCellPos() noexcept;
  void initPseudo() noexcept;
  void initSubVol() noexcept;
  // initialisation des variables VNR
  void initVar() noexcept;
  void initInternalEnergy() noexcept;

  void initVarSOD() noexcept;
  void initVarBiSOD() noexcept;
  void initVarImplosion() noexcept;
  void initVarBiImplosion() noexcept;
  void initVarShockBubble() noexcept;
  void initVarTriplePoint() noexcept;
  void initVarBiTriplePoint() noexcept;
  void initVarSEDOV() noexcept;
  void initVarBiSEDOV() noexcept;
  void initVarNOH() noexcept;
  void initVarBiNOH() noexcept;
  void initVarUnitTest() noexcept;
  void initVarBiUnitTest() noexcept;
  void initVarAdvection() noexcept;
  void initVarBiAdvection() noexcept;
  void initVarBiAdvectionVitesse() noexcept;
  void initVarRider(RealArray1D<dim> Xb) noexcept;
};

}  // namespace initlib
#endif  // INITIALISATIONS_H
