#ifndef EUCCLHYDREMAP_H
#define EUCCLHYDREMAP_H

/*---------------------------------------*/
/*---------------------------------------*/

#include <stddef.h>  // for size_t

#include <Kokkos_Core.hpp>                // for KOKKOS_LAMBDA
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>  // for OpenMP::impl_is_initialized
#include <algorithm>                      // for copy
#include <array>                          // for array
#include <string>                         // for allocator, string
#include <vector>                         // for vector

#include "../includes/Constantes.h"
#include "../includes/Options.h"
#include "../includes/CasTest.h"
#include "../includes/ConditionsLimites.h"
#include "../includes/CstMesh.h"
#include "../includes/Eos.h"
#include "../includes/GestionTemps.h"
#include "../includes/Limiteurs.h"
#include "../includes/SchemaParticules.h"
#include "../includes/VariablesLagRemap.h"
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D, CartesianM...
#include "mesh/MeshGeometry.h"     // for MeshGeometry
#include "mesh/PvdFileWriter2D.h"  // for PvdFileWriter2D

#include "types/Types.h"  // for RealArray1D, RealArray2D
#include "utils/Timer.h"  // for Timer

#include "../includes/Freefunctions.h"
#include "../remap/Remap.h"

/*---------------------------------------*/
/*---------------------------------------*/
using namespace nablalib;

class Eucclhyd {
 public:
  struct interval {
    double inf, sup;
  };

  
 private:
  CartesianMesh2D* mesh;
  optionschemalib::OptionsSchema::Options* options;
  castestlib::CasTest::Test* test;
  particulelib::SchemaParticules::Particules* particules;
  conditionslimiteslib::ConditionsLimites::Cdl* cdl;
  limiteurslib::LimiteursClass::Limiteurs* limiteurs;
  eoslib::EquationDetat::Eos* eos;
  cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh;
  gesttempslib::GestionTempsClass::GestTemps* gt;
  variableslagremaplib::VariablesLagRemap* varlp;
  Remap* remap;
  PvdFileWriter2D writer;
  PvdFileWriter2D writerpart;
  int nbPartMax;
  int nbPart = 0;
  int nbNodes, nbCells, nbFaces, nbCellsOfNode, nbNodesOfCell,
      nbNodesOfFace, nbCellsOfFace;

  // Global Variables
  int n, nbCalls;
  double lastDump;
  double m_global_total_energy_L, m_global_total_energy_T, m_global_total_energy_0;
  double m_total_masse_L, m_total_masse_T, m_total_masse_0;

  // cells a debuguer
  int dbgcell1 = -389;
  int dbgcell2 = -126;
  int dbgcell3 = -156;
  int face_debug1 = -2033;
  int face_debug2 = -410;
  int test_debug = 1;

  // Connectivity Variables
  Kokkos::View<RealArray1D<dim>*> m_node_coord;
  Kokkos::View<RealArray1D<dim>*> m_cell_coord;
  Kokkos::View<double*> m_cell_coord_x;
  Kokkos::View<double*> m_cell_coord_y;
  Kokkos::View<RealArray1D<dim>**> m_lpc_n;
  Kokkos::View<RealArray1D<dim>**> nplus;
  Kokkos::View<RealArray1D<dim>**> nminus;
  Kokkos::View<double**> lplus;
  Kokkos::View<double**> lminus;
  Kokkos::View<double*> m_pressure;
  Kokkos::View<RealArray1D<nbmatmax>*> pp;
  Kokkos::View<double*> m;
  Kokkos::View<RealArray1D<nbmatmax>*> mp;
  Kokkos::View<double*> m_euler_volume;
  Kokkos::View<RealArray1D<nbmatmax>*> vpLagrange;
  Kokkos::View<double*> perim;
  Kokkos::View<double*> vitson;
  Kokkos::View<RealArray1D<nbmatmax>*> vitsonp;
  Kokkos::View<double*> deltatc;
  Kokkos::View<RealArray1D<dim>*> Vnode_n;
  Kokkos::View<RealArray1D<dim>*> Vnode_nplus1;
  Kokkos::View<RealArray1D<dim>*> Vnode_n0;
  Kokkos::View<double*> m_total_energy_0;
  Kokkos::View<double*> m_total_energy_T;
  Kokkos::View<double*> m_total_energy_L;
  Kokkos::View<double*> m_global_masse_0;
  Kokkos::View<double*> m_global_masse_T;
  Kokkos::View<double*> m_global_masse_L;
  Kokkos::View<double*> rho_n;
  Kokkos::View<double*> rho_nplus1;
  Kokkos::View<double*> rho_n0;
  Kokkos::View<RealArray1D<nbmatmax>*> rhop_n;
  Kokkos::View<RealArray1D<nbmatmax>*> rhop_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> rhop_n0;
  Kokkos::View<RealArray1D<dim>*> V_n;
  Kokkos::View<RealArray1D<dim>*> V_nplus1;
  Kokkos::View<RealArray1D<dim>*> V_n0;
  Kokkos::View<double*> Vxc;
  Kokkos::View<double*> Vyc;
  Kokkos::View<double*> e_n;
  Kokkos::View<double*> e_nplus1;
  Kokkos::View<double*> e_n0;
  Kokkos::View<RealArray1D<nbmatmax>*> ep_n;
  Kokkos::View<RealArray1D<nbmatmax>*> ep_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> ep_n0;
  Kokkos::View<double**> p_extrap;
  Kokkos::View<RealArray1D<nbmatmax>**> pp_extrap;
  Kokkos::View<RealArray1D<dim>**> V_extrap;
  Kokkos::View<RealArray1D<dim>*> gradp;
  Kokkos::View<RealArray1D<dim>*> gradp1;
  Kokkos::View<RealArray1D<dim>*> gradp2;
  Kokkos::View<RealArray1D<dim>*> gradp3;
  Kokkos::View<RealArray1D<dim>*> gradf1;
  Kokkos::View<RealArray1D<dim>*> gradf2;
  Kokkos::View<RealArray1D<dim>*> gradf3;
  Kokkos::View<RealArray2D<dim, dim>*> gradV;
  Kokkos::View<RealArray1D<dim>**> F_n;
  Kokkos::View<RealArray1D<dim>**> F_nplus1;
  Kokkos::View<RealArray1D<dim>**> F_n0;
  Kokkos::View<RealArray1D<dim>**> F1_n;
  Kokkos::View<RealArray1D<dim>**> F1_nplus1;
  Kokkos::View<RealArray1D<dim>**> F2_n;
  Kokkos::View<RealArray1D<dim>**> F2_nplus1;
  Kokkos::View<RealArray1D<dim>**> F3_n;
  Kokkos::View<RealArray1D<dim>**> F3_nplus1;
  Kokkos::View<RealArray1D<dim>*> G;
  Kokkos::View<RealArray2D<dim, dim>**> M;
  Kokkos::View<RealArray2D<dim, dim>**> M1;
  Kokkos::View<RealArray2D<dim, dim>**> M2;
  Kokkos::View<RealArray2D<dim, dim>**> M3;
  Kokkos::View<RealArray2D<dim, dim>*> Mnode;
  Kokkos::View<RealArray1D<nbmatmax>*> fracmass;
  Kokkos::View<RealArray1D<nbmatmax>*> m_fracvol_env;
  Kokkos::View<RealArray1D<nbmatmax>*> m_node_fracvol;
  Kokkos::View<double*> m_fracvol_env1;
  Kokkos::View<double*> m_fracvol_env2;
  Kokkos::View<double*> m_fracvol_env3;
  Kokkos::View<double*> p1;
  Kokkos::View<double*> p2;
  Kokkos::View<double*> p3;

  Kokkos::View<double*> vpart;
  Kokkos::View<double*> wpart;
  Kokkos::View<double*> mpart;
  Kokkos::View<double*> rpart;
  Kokkos::View<double*> rhopart;
  Kokkos::View<double*> Cdpart;
  Kokkos::View<double*> Mcpart;
  Kokkos::View<double*> Repart;
  Kokkos::View<double*> Temppart;
  Kokkos::View<RealArray1D<dim>*> m_particle_coord_n0;
  Kokkos::View<RealArray1D<dim>*> m_particle_coord_n;
  Kokkos::View<RealArray1D<dim>*> m_particle_coord_nplus1;
  Kokkos::View<RealArray1D<dim>*> ForceGradp;
  Kokkos::View<RealArray1D<dim>*> Vpart_n0;
  Kokkos::View<RealArray1D<dim>*> Vpart_n;
  Kokkos::View<RealArray1D<dim>*> Vpart_nplus1;
  Kokkos::View<int*> ICellp;
  Kokkos::View<int*> IMatp;
  Kokkos::View<double*> fracpart;
  Kokkos::View<vector<int>*> listpart;

  utils::Timer global_timer;
  utils::Timer cpu_timer;
  utils::Timer io_timer;
  const size_t maxHardThread =
      Kokkos::DefaultExecutionSpace::impl_max_hardware_threads();

 public:
  Eucclhyd(
      optionschemalib::OptionsSchema::Options* aOptions,
      cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* acstmesh,
      gesttempslib::GestionTempsClass::GestTemps* agt,
      castestlib::CasTest::Test* aTest,
      conditionslimiteslib::ConditionsLimites::Cdl* aCdl,
      limiteurslib::LimiteursClass::Limiteurs* aLimiteurs,
      particulelib::SchemaParticules::Particules* aParticules,
      eoslib::EquationDetat::Eos* aEos, CartesianMesh2D* aCartesianMesh2D,
      variableslagremaplib::VariablesLagRemap* avarlp,
      Remap* aremap,
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
        varlp(avarlp),
        remap(aremap),
        writer("EucclhydRemap", output),
        writerpart("Particules", output),
        nbNodes(mesh->getNbNodes()),
        nbPartMax(1),
        nbCells(mesh->getNbCells()),
        nbFaces(mesh->getNbFaces()),
        nbCellsOfNode(CartesianMesh2D::MaxNbCellsOfNode),
        nbNodesOfCell(CartesianMesh2D::MaxNbNodesOfCell),
        nbNodesOfFace(CartesianMesh2D::MaxNbNodesOfFace),
        nbCalls(0),
        lastDump(0.0),
        vpart("VolumePart", nbPartMax),
        wpart("WeightPart", nbPartMax),
        mpart("MassePart", nbPartMax),
        rpart("RayonPart", nbPartMax),
        rhopart("RhoPart", nbPartMax),
        Repart("RePart", nbPartMax),
        Cdpart("Cdpart", nbPartMax),
        Mcpart("Cdpart", nbPartMax),
        ForceGradp("ForceGradp", nbCells),
        listpart("listepart", nbCells),
        fracpart("fracPart", nbCells),
        m_particle_coord_n0("m_particle_coord_n0", nbPartMax),
        m_particle_coord_n("m_particle_coord_n", nbPartMax),
        m_particle_coord_nplus1("m_particle_coord_nplus1", nbPartMax),
        Vpart_n0("Vpart_n0", nbPartMax),
        Vpart_n("Vpart_n", nbPartMax),
        Vpart_nplus1("Vpart_nplus1", nbPartMax),
        ICellp("Icellp", nbPartMax),
        IMatp("Imatp", nbPartMax),
        m_node_coord("m_node_coord", nbNodes),
        m_cell_coord("m_cell_coord", nbCells),
        m_cell_coord_x("m_cell_coord_x", nbCells),
        m_cell_coord_y("m_cell_coord_y", nbCells),
        m_lpc_n("m_lpc_n", nbNodes, nbCellsOfNode),
        nplus("nplus", nbNodes, nbCellsOfNode),
        nminus("nminus", nbNodes, nbCellsOfNode),
        lplus("lplus", nbNodes, nbCellsOfNode),
        lminus("lminus", nbNodes, nbCellsOfNode),
        m_pressure("p", nbCells),
        p1("p1", nbCells),
        p2("p2", nbCells),
        p3("p3", nbCells),
        pp("pp", nbCells),
        m("m", nbCells),
        mp("mp", nbCells),
        m_euler_volume("m_euler_volume", nbCells),
        fracmass("fracmass", nbCells),
        m_fracvol_env("m_fracvol_env", nbCells),
        m_node_fracvol("m_node_fracvol", nbNodes),
        m_fracvol_env1("m_fracvol_env1", nbCells),
        m_fracvol_env2("m_fracvol_env2", nbCells),
        m_fracvol_env3("m_fracvol_env3", nbCells),
        vpLagrange("vpLagrange", nbCells),
        perim("perim", nbCells),
        vitson("vitson", nbCells),
        vitsonp("vitsonp", nbCells),
        deltatc("deltatc", nbCells),
        Vnode_n("Vnode_n", nbNodes),
        Vnode_nplus1("Vnode_nplus1", nbNodes),
        Vnode_n0("Vnode_n0", nbNodes),
        m_total_energy_0("m_total_energy_0", nbCells),
        m_total_energy_T("m_total_energy_T", nbCells),
        m_total_energy_L("m_total_energy_L", nbCells),
        m_global_masse_0("m_global_masse_0", nbCells),
        m_global_masse_T("m_global_masse_T", nbCells),
        m_global_masse_L("m_global_masse_L", nbCells),
        rho_n("rho_n", nbCells),
        rho_nplus1("rho_nplus1", nbCells),
        rho_n0("rho_n0", nbCells),
        rhop_n("rhop_n", nbCells),
        rhop_nplus1("rhop_nplus1", nbCells),
        rhop_n0("rhop_n0", nbCells),
        V_n("V_n", nbCells),
        V_nplus1("V_nplus1", nbCells),
        V_n0("V_n0", nbCells),
        Vxc("Vxc", nbCells),
        Vyc("Vyc", nbCells),
        e_n("e_n", nbCells),
        e_nplus1("e_nplus1", nbCells),
        e_n0("e_n0", nbCells),
        ep_n("ep_n", nbCells),
        ep_nplus1("ep_nplus1", nbCells),
        ep_n0("ep_n0", nbCells),
        p_extrap("p_extrap", nbCells, nbNodesOfCell),
        pp_extrap("pp_extrap", nbCells, nbNodesOfCell),
        V_extrap("V_extrap", nbCells, nbNodesOfCell),
        gradp("gradp", nbCells),
        gradp1("gradp1", nbCells),
        gradp2("gradp2", nbCells),
        gradp3("gradp3", nbCells),
        gradf1("gradf1", nbCells),
        gradf2("gradf2", nbCells),
        gradf3("gradf3", nbCells),
        gradV("gradV", nbCells),
        F_n("F_n", nbNodes, nbCellsOfNode),
        F_nplus1("F_nplus1", nbNodes, nbCellsOfNode),
        F_n0("F_n0", nbNodes, nbCellsOfNode),
        F1_n("F1_n", nbNodes, nbCellsOfNode),
        F1_nplus1("F1_nplus1", nbNodes, nbCellsOfNode),
        F2_n("F2_n", nbNodes, nbCellsOfNode),
        F2_nplus1("F2_nplus1", nbNodes, nbCellsOfNode),
        F3_n("F3_n", nbNodes, nbCellsOfNode),
        F3_nplus1("F3_nplus1", nbNodes, nbCellsOfNode),
        G("G", nbNodes),
        M("M", nbNodes, nbCellsOfNode),
        M1("M1", nbNodes, nbCellsOfNode),
        M2("M2", nbNodes, nbCellsOfNode),
        M3("M3", nbNodes, nbCellsOfNode),
        Mnode("Mnode", nbNodes) {
    // Copy node coordinates
    const auto& gNodes = mesh->getGeometry()->getNodes();
    Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const int& rNodes) {
      m_node_coord(rNodes) = gNodes[rNodes];
      });
  }

 private:
  void computeBoundaryNodeVelocities() noexcept;
  RealArray1D<dim> nodeVelocityBoundaryCondition(int BC,
                                                 RealArray1D<dim> BCValue,
                                                 RealArray2D<dim, dim> Mp,
                                                 RealArray1D<dim> Gp);
  RealArray1D<dim> nodeVelocityBoundaryConditionCorner(
      int BC1, RealArray1D<dim> BCValue1, int BC2, RealArray1D<dim> BCValue2,
      RealArray2D<dim, dim> Mp, RealArray1D<dim> Gp);

  void initBoundaryConditions() noexcept;
  void initMeshGeometryForCells() noexcept;
  void initVpAndFpc() noexcept;
  void initCellInternalEnergy() noexcept;
  void initCellVelocity() noexcept;
  void initDensity() noexcept;
  void initMeshGeometryForFaces() noexcept;
  void initPart() noexcept;
  void setUpTimeLoopN() noexcept;

  void computeCornerNormal() noexcept;
  void computeEOS();
  void computeEOSGP(int imat);
  void computeEOSVoid(int imat);
  void computeEOSSTIFG(int imat);
  void computeEOSMur(int imat);
  void computeEOSSL(int imat);
  void computePressionMoyenne() noexcept;
  void computeGradients() noexcept;
  void computeMass() noexcept;
  void computeDissipationMatrix() noexcept;
  void computedeltatc() noexcept;
  void extrapolateValue() noexcept;
  void computeG() noexcept;
  void computeNodeDissipationMatrixAndG() noexcept;
  void computeNodeVelocity() noexcept;
  void computeFaceVelocity() noexcept;
  void computeLagrangePosition() noexcept;
  void computeSubCellForce() noexcept;
  void computeLagrangeVolumeAndCenterOfGravity() noexcept;
  void computeFacedeltaxLagrange() noexcept;
  void updateCellCenteredLagrangeVariables() noexcept;

  void remapCellcenteredVariable() noexcept;

  void updateParticlePosition() noexcept;
  void updateParticleCoefficients() noexcept;
  void updateParticleVelocity() noexcept;
  void updateParticleRetroaction() noexcept;

  void switchalpharho_rho() noexcept;
  void switchrho_alpharho() noexcept;

  RealArray2D<2, 2> inverse(RealArray2D<2, 2> a);
  template <size_t N, size_t M>
  RealArray2D<N, M> tensProduct(RealArray1D<N> a, RealArray1D<M> b);
  double crossProduct2d(RealArray1D<2> a, RealArray1D<2> b);
  
  /**
   * Job dumpVariables called @2.0 in executeTimeLoopN method.
   * In variables: m_cell_coord_x, m_cell_coord_y, e_n, m, p, rho_n, t_n, v
   * Out variables:
   */
  void dumpVariables() noexcept;

  /**
   * Job executeTimeLoopN called @4.0 in simulate method.
   * In variables: F_n, F_nplus1, G, M, Mnode, ULagrange, Uremap1, Uremap2,
   * V_extrap, V_n, Vnode_n, Vnode_nplus1, X, XLagrange, m_cell_coord, m_cell_coordLagrange, m_cell_coord_x,
   * m_cell_coord_y, Xf, bottomBC, bottomBCValue, c, cfl, deltat_n, deltat_nplus1,
   * deltatc, deltaxLagrange, eos, eosPerfectGas, e_n, faceLength, faceNormal,
   * faceNormalVelocity, gamma, gradPhi1, gradPhi2, gradPhiFace1, gradPhiFace2,
   * gradV, gradp, leftBC, leftBCValue, lminus, m_lpc_n, lplus, m, nminus, nplus,
   * outerFaceNormal, p, p_extrap, perim, phiFace1, phiFace2,
   * projectionLimiterId, projectionOrder, rho_n, rightBC, rightBCValue,
   * spaceOrder, t_n, topBC, topBCValue, v, vLagrange, x_then_y_n Out variables:
   * F_nplus1, G, M, Mnode, ULagrange, Uremap1, Uremap2, V_extrap, V_nplus1,
   * Vnode_nplus1, XLagrange, m_cell_coordLagrange, c, deltat_nplus1, deltatc,
   * deltaxLagrange, e_nplus1, faceNormalVelocity, gradPhi1, gradPhi2,
   * gradPhiFace1, gradPhiFace2, gradV, gradp, m, p, p_extrap, phiFace1,
   * phiFace2, rho_nplus1, t_nplus1, vLagrange, x_then_y_nplus1
   */
  void executeTimeLoopN() noexcept;

  /**
   * Job computedeltat called @3.0 in executeTimeLoopN method.
   * In variables: cfl, deltat_n, deltatc
   * Out variables: deltat_nplus1
   */
  void computedeltat() noexcept;

  /**
   * Job updateTime called @4.0 in executeTimeLoopN method.
   * In variables: deltat_nplus1, t_n
   * Out variables: t_nplus1
   */
  void updateTime() noexcept;

 public:
  void simulate();
};

#include "Utiles-Impl.h"

#endif  // EUCCLHYDREMAP_H
