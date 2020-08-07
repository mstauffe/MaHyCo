#ifndef VNRREMAP_H
#define VNRREMAP_H

/*---------------------------------------*/
/*---------------------------------------*/

#include <stddef.h>  // for size_t

#include <Kokkos_Core.hpp>                // for KOKKOS_LAMBD
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>  // for OpenMP::impl_is_initialized

/* #include <fstream> */
/* #include <iomanip> */
/* #include <type_traits> */
/* #include <limits> */
/* #include <utility> */
/* #include <cmath> */
/* #include <Kokkos_hwloc.hpp> */

#include "../includes/Constantes.h"
#include "../includes/Options.h"
#include "../includes/CasTest.h"
#include "../includes/ConditionsLimites.h"
#include "../includes/CstMesh.h"
#include "../includes/Eos.h"
#include "../includes/GestionTemps.h"
#include "../includes/Limiteurs.h"
#include "../includes/SchemaParticules.h"

#include "mesh/CartesianMesh2D.h" // for CartesianMesh2D, CartesianM...
#include "mesh/PvdFileWriter2D.h" // for PvdFileWriter2D
#include "utils/kokkos/Parallel.h"

#include "types/Types.h"  // for RealArray1D, RealArray2D
#include "utils/Timer.h"  // for Timer
using namespace nablalib;

/******************** Free functions declarations ********************/

template<size_t x>
KOKKOS_INLINE_FUNCTION
double norm(RealArray1D<x> a);
template<size_t x>
KOKKOS_INLINE_FUNCTION
double dot(RealArray1D<x> a, RealArray1D<x> b);
KOKKOS_INLINE_FUNCTION
double computeLpcPlus(RealArray1D<2> xp, RealArray1D<2> xpPlus);
KOKKOS_INLINE_FUNCTION
RealArray1D<2> computeNpcPlus(RealArray1D<2> xp, RealArray1D<2> xpPlus);
KOKKOS_INLINE_FUNCTION
double crossProduct2d(RealArray1D<2> a, RealArray1D<2> b);
KOKKOS_INLINE_FUNCTION
RealArray1D<2> computeLpcNpc(RealArray1D<2> xp, RealArray1D<2> xpPlus, RealArray1D<2> xpMinus);
template<size_t N>
KOKKOS_INLINE_FUNCTION
RealArray1D<N> symmetricVector(RealArray1D<N> v, RealArray1D<N> sigma);
template<size_t x>
KOKKOS_INLINE_FUNCTION
RealArray1D<x> sumR1(RealArray1D<x> a, RealArray1D<x> b);
KOKKOS_INLINE_FUNCTION
double sumR0(double a, double b);
KOKKOS_INLINE_FUNCTION
double minR0(double a, double b);


/******************** Module declaration ********************/

class VnrRemap
{
public:

private:
	
	// Mesh (can depend on previous definitions)
	CartesianMesh2D* mesh;
        optionschemalib::OptionsSchema::Options* options;
	castestlib::CasTest::Test* test;
	particulelib::SchemaParticules::Particules* particules;
	conditionslimiteslib::ConditionsLimites::Cdl* cdl;
	limiteurslib::LimiteursClass::Limiteurs* limiteurs;
	eoslib::EquationDetat::Eos* eos;
	cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh;
	gesttempslib::GestionTempsClass::GestTemps* gt;
	PvdFileWriter2D writer;
        int nbPartMax;
        int nbPart = 0;
	size_t nbNodes, nbCells, nbFaces, nbInnerNodes, nbNodesOfCell, nbCellsOfNode;
	
	// Global declarations	
	int n, nbCalls;
	//bool x_then_y_n, x_then_y_nplus1;
	double lastDump;
	//double ETOTALE_L, ETOTALE_T, ETOTALE_0;
	//double MASSET_L, MASSET_T, MASSET_0;
	
	Kokkos::View<RealArray1D<dim>*> X_n;
	Kokkos::View<RealArray1D<dim>*> X_nplus1;
	Kokkos::View<RealArray1D<dim>*> X_n0;
	Kokkos::View<double**> SubVol_n;
	Kokkos::View<double**> SubVol_nplus1;
	Kokkos::View<double**> SubVol_n0;
	Kokkos::View<double*> V;
	Kokkos::View<double*> rho_n;
	Kokkos::View<double*> rho_nplus1;
	Kokkos::View<double*> rho_n0;
	Kokkos::View<RealArray1D<nbmatmax>*> rhop_n;
	Kokkos::View<RealArray1D<nbmatmax>*> rhop_nplus1;
	Kokkos::View<RealArray1D<nbmatmax>*> rhop_n0;
	Kokkos::View<double*> p_n;
	Kokkos::View<double*> p_nplus1;
	Kokkos::View<double*> p_n0;
	Kokkos::View<RealArray1D<nbmatmax>*> pp_n;
	Kokkos::View<RealArray1D<nbmatmax>*> pp_nplus1;
	Kokkos::View<RealArray1D<nbmatmax>*> pp_n0;
	Kokkos::View<double*> Q_n0;
	Kokkos::View<double*> Q_n;
	Kokkos::View<double*> Q_nplus1;
	Kokkos::View<RealArray1D<nbmatmax>*> Qp_n0;
	Kokkos::View<RealArray1D<nbmatmax>*> Qp_n;
	Kokkos::View<RealArray1D<nbmatmax>*> Qp_nplus1;
	Kokkos::View<double*> tau_n;
	Kokkos::View<double*> tau_nplus1;
	Kokkos::View<double*> tau_n0;
	Kokkos::View<RealArray1D<nbmatmax>*> taup_n;
	Kokkos::View<RealArray1D<nbmatmax>*> taup_nplus1;
	Kokkos::View<RealArray1D<nbmatmax>*> taup_n0;
	Kokkos::View<double*> divU_n;
	Kokkos::View<double*> divU_nplus1;
	Kokkos::View<double*> divU_n0;
	Kokkos::View<double*> c_n;
	Kokkos::View<double*> c_nplus1;
	Kokkos::View<double*> c_n0;
	Kokkos::View<RealArray1D<nbmatmax>*> cp_n;
	Kokkos::View<RealArray1D<nbmatmax>*> cp_nplus1;
	Kokkos::View<RealArray1D<nbmatmax>*> cp_n0;
	Kokkos::View<double*> e_n;
	Kokkos::View<double*> e_nplus1;
	Kokkos::View<double*> e_n0;
	Kokkos::View<RealArray1D<nbmatmax>*> ep_n;
	Kokkos::View<RealArray1D<nbmatmax>*> ep_nplus1;
	Kokkos::View<RealArray1D<nbmatmax>*> ep_n0;
	Kokkos::View<RealArray1D<dim>*> u_n;
	Kokkos::View<RealArray1D<dim>*> u_nplus1;
	Kokkos::View<RealArray1D<dim>*> u_n0;
	Kokkos::View<double*> m;
	Kokkos::View<double*> cellMass;
	Kokkos::View<RealArray1D<nbmatmax>*> cellMassp;
	Kokkos::View<RealArray1D<dim>*> cellPos_n;
	Kokkos::View<RealArray1D<dim>*> cellPos_nplus1;
	Kokkos::View<RealArray1D<dim>*> cellPos_n0;
	Kokkos::View<RealArray1D<dim>**> C;
	Kokkos::View<RealArray1D<nbmatmax>*> fracmass;
	Kokkos::View<RealArray1D<nbmatmax>*> fracvol;
	Kokkos::View<RealArray1D<nbmatmax>*> fracvolnode;
	
	utils::Timer global_timer;
	utils::Timer cpu_timer;
	utils::Timer io_timer;

	const size_t maxHardThread =
      Kokkos::DefaultExecutionSpace::impl_max_hardware_threads();
	
 public:
  VnrRemap(
      optionschemalib::OptionsSchema::Options* aOptions,
      cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* acstmesh,
      gesttempslib::GestionTempsClass::GestTemps* agt,
      castestlib::CasTest::Test* aTest,
      conditionslimiteslib::ConditionsLimites::Cdl* aCdl,
      limiteurslib::LimiteursClass::Limiteurs* aLimiteurs,
      particulelib::SchemaParticules::Particules* aParticules,
      eoslib::EquationDetat::Eos* aEos, CartesianMesh2D* aCartesianMesh2D,
      string output)
  : options(aOptions),
    cstmesh(acstmesh),
    gt(agt),
    test(aTest),
    cdl(aCdl),
    limiteurs(aLimiteurs),
    particules(aParticules),
    eos(aEos),
    mesh(aCartesianMesh2D),
    nbCalls(0),
    lastDump(0.0),
    writer("VnrRemap", output),
    nbNodes(mesh->getNbNodes()),
    nbCells(mesh->getNbCells()),
    nbNodesOfCell(CartesianMesh2D::MaxNbNodesOfCell),
    nbCellsOfNode(CartesianMesh2D::MaxNbCellsOfNode),
    X_n("X_n", nbNodes),
    X_nplus1("X_nplus1", nbNodes),
    X_n0("X_n0", nbNodes),
    SubVol_n("SubVol_n", nbCells, nbNodesOfCell),
    SubVol_nplus1("SubVol_nplus1", nbCells, nbNodesOfCell),
    SubVol_n0("SubVol_n0", nbCells, nbNodesOfCell),
    V("V", nbNodes),
    rho_n("rho_n", nbCells),
    rho_nplus1("rho_nplus1", nbCells),
    rho_n0("rho_n0", nbCells),
    rhop_n("rhop_n", nbCells),
    rhop_nplus1("rhop_nplus1", nbCells),
    rhop_n0("rhop_n0", nbCells),
    p_n("p_n", nbCells),
    p_nplus1("p_nplus1", nbCells),
    p_n0("p_n0", nbCells),
    pp_n("pp_n", nbCells),
    pp_nplus1("pp_nplus1", nbCells),
    pp_n0("pp_n0", nbCells),
    Q_n0("Q_n", nbCells),
    Q_n("Q_n", nbCells),
    Q_nplus1("Q_nplus1", nbCells),
    Qp_n0("Qp_n", nbCells),
    Qp_n("Qp_n", nbCells),
    Qp_nplus1("Qp_nplus1", nbCells),
    tau_n("tau_n", nbCells),
    tau_nplus1("tau_nplus1", nbCells),
    tau_n0("tau_n0", nbCells),
    taup_n("taup_n", nbCells),
    taup_nplus1("taup_nplus1", nbCells),
    taup_n0("taup_n0", nbCells),
    divU_n("divU_n", nbCells),
    divU_nplus1("divU_nplus1", nbCells),
    divU_n0("divU_n0", nbCells),
    c_n("c_n", nbCells),
    c_nplus1("c_nplus1", nbCells),
    c_n0("c_n0", nbCells),
    cp_n("c_n", nbCells),
    cp_nplus1("c_nplus1", nbCells),
    cp_n0("c_n0", nbCells),
    e_n("e_n", nbCells),
    e_nplus1("e_nplus1", nbCells),
    e_n0("e_n0", nbCells),
    ep_n("epsp_n", nbCells),
    ep_nplus1("epsp_nplus1", nbCells),
    ep_n0("epsp_n0", nbCells),	  
    u_n("u_n", nbNodes),
    u_nplus1("u_nplus1", nbNodes),
    u_n0("u_n0", nbNodes),
    m("m", nbNodes),
    cellMass("cellMass", nbCells),
    cellMassp("cellMassp", nbCells),
    cellPos_n("cellPos_n", nbCells),
    cellPos_nplus1("cellPos_nplus1", nbCells),
    cellPos_n0("cellPos_n0", nbCells),
    fracmass("fracmass", nbCells),
    fracvol("fracvol", nbCells),
    fracvolnode("fracvolnode", nbNodes),
    C("C", nbCells, nbNodesOfCell) {
  // Copy node coordinates
  const auto& gNodes = mesh->getGeometry()->getNodes();
  for (size_t rNodes=0; rNodes<nbNodes; rNodes++) {
    X_n0(rNodes)[0] = gNodes[rNodes][0];
    X_n0(rNodes)[1] = gNodes[rNodes][1];
  }	
 }
 private:
	
	KOKKOS_INLINE_FUNCTION
	void computeDeltaT() noexcept;
	
	KOKKOS_INLINE_FUNCTION
	void computeTime() noexcept;
	
	KOKKOS_INLINE_FUNCTION
	void setUpTimeLoopN() noexcept;

	KOKKOS_INLINE_FUNCTION
	void executeTimeLoopN() noexcept;

	void dumpVariables() noexcept;
	
	// dans Init.cc
	void initBoundaryConditions() noexcept;
	
	void init() noexcept;
	
	void initSubVol() noexcept;
	
	void initCellPos() noexcept;
	
	void initDeltaT() noexcept;
	
	void initInternalEnergy() noexcept;
	
	void initPseudo() noexcept;

	// dans PhaseLagrange.cc
	
	void computeArtificialViscosity() noexcept;
	
	void computeCornerNormal() noexcept;
	
	void updateVelocity() noexcept;
	
	void computeCellMass() noexcept;
		
	void updatePosition() noexcept;
		
	void computeNodeMass() noexcept;
	
	void computeNodeVolume() noexcept;
	
	void computeSubVol() noexcept;	
	
	void updateRho() noexcept;
	
	void computeTau() noexcept;
	
	void updateEnergy() noexcept;
	
	void computeDivU() noexcept;
	
	void computeEOS();
	void computeEOSGP(int imat);
	void computeEOSVoid(int imat);
	void computeEOSSTIFG(int imat);
	void computeEOSMur(int imat);
	void computeEOSSL(int imat);
	void computePressionMoyenne() noexcept;

	void updateVelocityBoundaryConditions() noexcept;

public:
	void simulate();
};

#endif  // VNRREMAP_H
