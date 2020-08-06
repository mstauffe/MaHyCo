#ifndef REMAP_H
#define REMAP_H

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

/*---------------------------------------*/
/*---------------------------------------*/
using namespace nablalib;

class Remap {
 public:
  struct interval {
    double inf, sup;
  };

  
 private:
  CartesianMesh2D* mesh;
  optionschemalib::OptionsSchema::Options* options;
  cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh;
  conditionslimiteslib::ConditionsLimites::Cdl* cdl;
  limiteurslib::LimiteursClass::Limiteurs* limiteurs;
  gesttempslib::GestionTempsClass::GestTemps* gt;
  variableslagremaplib::VariablesLagRemap* varlp;
  int nbPartMax;
  int nbPart = 0;
  int nbNodes, nbCells, nbFaces, nbCellsOfNode, nbNodesOfCell,
      nbNodesOfFace, nbCellsOfFace;


  // cells a debuguer
  int dbgcell1 = -389;
  int dbgcell2 = -126;
  int dbgcell3 = -156;
  int face_debug1 = -2033;
  int face_debug2 = -410;
  int test_debug = 1;

  // Connectivity Variables
  Kokkos::View<double*> LfLagrange;
  Kokkos::View<double*> HvLagrange;
  Kokkos::View<RealArray1D<nbequamax>*> Uremap1;
  Kokkos::View<RealArray1D<nbequamax>*> gradPhiFace1;
  Kokkos::View<RealArray1D<nbequamax>*> gradPhiFace2;
  Kokkos::View<RealArray1D<nbequamax>*> gradPhi1;
  Kokkos::View<RealArray1D<nbequamax>*> gradPhi2;
  Kokkos::View<RealArray1D<nbequamax>*> phiFace1;
  Kokkos::View<RealArray1D<nbequamax>*> phiFace2;
  Kokkos::View<RealArray1D<nbequamax>*> deltaPhiFaceAv;
  Kokkos::View<RealArray1D<nbequamax>*> deltaPhiFaceAr;
 public:
  Remap(
      optionschemalib::OptionsSchema::Options* aOptions,
      cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* acstmesh,
      gesttempslib::GestionTempsClass::GestTemps* agt,
      conditionslimiteslib::ConditionsLimites::Cdl* aCdl,
      limiteurslib::LimiteursClass::Limiteurs* aLimiteurs,
      CartesianMesh2D* aCartesianMesh2D,
      variableslagremaplib::VariablesLagRemap* avarlp)
      : options(aOptions),
        cstmesh(acstmesh),
        gt(agt),
        cdl(aCdl),
        limiteurs(aLimiteurs),
        mesh(aCartesianMesh2D),
        varlp(avarlp),
        nbNodes(mesh->getNbNodes()),
        nbPartMax(1),
        nbCells(mesh->getNbCells()),
        nbFaces(mesh->getNbFaces()),
        nbCellsOfNode(CartesianMesh2D::MaxNbCellsOfNode),
        nbNodesOfCell(CartesianMesh2D::MaxNbNodesOfCell),
        nbNodesOfFace(CartesianMesh2D::MaxNbNodesOfFace),
        LfLagrange("LfLagrange", nbFaces),
        HvLagrange("HvLagrange", nbCells),
        Uremap1("Uremap1", nbCells),
        gradPhiFace1("gradPhiFace1", nbFaces),
        gradPhiFace2("gradPhiFace2", nbFaces),
        gradPhi1("gradPhi1", nbCells),
        gradPhi2("gradPhi2", nbCells),
        phiFace1("phiFace1", nbFaces),
        phiFace2("phiFace2", nbFaces),
        deltaPhiFaceAv("deltaPhiFaceAv", nbCells),
        deltaPhiFaceAr("deltaPhiFaceAr", nbCells){}
  
 

  void computeGradPhiFace1() noexcept;
  void computeGradPhi1() noexcept;
  void computeUpwindFaceQuantitiesForProjection1() noexcept;
  void computeUremap1() noexcept;

  void computeGradPhiFace2() noexcept;
  void computeGradPhi2() noexcept;
  void computeUpwindFaceQuantitiesForProjection2() noexcept;
  void computeUremap2() noexcept;
  
  template <size_t d>
  RealArray1D<d> computeRemapFlux(int projectionOrder,
                                  int projectionAvecPlateauPente,
                                  double face_normal_velocity,
                                  RealArray1D<dim> face_normal,
                                  double face_length, RealArray1D<d> phi_face,
                                  RealArray1D<dim> outer_face_normal,
                                  RealArray1D<dim> exy, double deltat_n);
  
  RealArray1D<nbequamax> computeBoundaryFluxes(int proj, int cCells,
                                               RealArray1D<dim> exy);

 private:

  int getLeftCells(const int cells);
  int getRightCells(const int cells);
  int getBottomCells(const int cells);
  int getTopCells(const int cells);
  
  double fluxLimiter(int projectionLimiterId, double r);
  double fluxLimiterPP(int projectionLimiterId, double gradplus,
                       double gradmoins, double y0, double yplus, double ymoins,
                       double h0, double hplus, double hmoins);
  double computeY0(int projectionLimiterId, double y0, double yplus,
                   double ymoins, double h0, double hplus, double hmoins,
                   int type);
  double computexgxd(double y0, double yplus, double ymoins, double h0,
                     double y0plus, double y0moins, int type);
  double computeygyd(double y0, double yplus, double ymoins, double h0,
                     double y0plus, double y0moins, double grady, int type);
  double INTY(double X, double x0, double y0, double x1, double y1);
  double INT2Y(double X, double x0, double y0, double x1, double y1);
  template <size_t d>
  RealArray1D<d> computeAndLimitGradPhi(
      int projectionLimiterId, RealArray1D<d> gradphiplus,
      RealArray1D<d> gradphimoins, RealArray1D<d> phi, RealArray1D<d> phiplus,
      RealArray1D<d> phimoins, double h0, double hplus, double hmoins);
  template <size_t d>
  RealArray1D<d> computeFluxPP(RealArray1D<d> gradphi, RealArray1D<d> phi,
                               RealArray1D<d> phiplus, RealArray1D<d> phimoins,
                               double h0, double hplus, double hmoins,
                               double face_normal_velocity, double deltat_n,
                               int type, int cell, double flux_threhold);
  template <size_t d>
  RealArray1D<d> computeFluxPPPure(RealArray1D<d> gradphi, RealArray1D<d> phi,
                                   RealArray1D<d> phiplus,
                                   RealArray1D<d> phimoins, double h0,
                                   double hplus, double hmoins,
                                   double face_normal_velocity, double deltat_n,
                                   int type, int cell, double flux_threhold);
  template <size_t d>
  RealArray1D<d> computeUpwindFaceQuantities(
      RealArray1D<dim> face_normal, double face_normal_velocity, double delta_x,
      RealArray1D<dim> x_f, RealArray1D<d> phi_cb, RealArray1D<d> grad_phi_cb,
      RealArray1D<dim> x_cb, RealArray1D<d> phi_cf, RealArray1D<d> grad_phi_cf,
      RealArray1D<dim> x_cf);
  RealArray1D<dim> xThenYToDirection(bool x_then_y_);

  template <size_t d>
  RealArray1D<d> computeVecFluxOrdre3(
      RealArray1D<d> phimmm, RealArray1D<d> phimm, RealArray1D<d> phim,
      RealArray1D<d> phip, RealArray1D<d> phipp, RealArray1D<d> phippp,
      double hmmm, double hmm, double hm, double hp, double hpp, double hppp,
      double face_normal_velocity, double deltat_n);
  interval define_interval(double a, double b);
  interval intersection(interval I1, interval I2);
  double evaluate_grad(double hm, double h0, double hp, double ym, double y0,
                       double yp);
  double evaluate_ystar(double hmm, double hm, double hp, double hpp,
                        double ymm, double ym, double yp, double ypp,
                        double gradm, double gradp);
  double evaluate_fm(double x, double dx, double up, double du, double u6);
  double evaluate_fp(double x, double dx, double um, double du, double u6);
  double ComputeFluxOrdre3(double ymmm, double ymm, double ym, double yp,
                           double ypp, double yppp, double hmmm, double hmm,
                           double hm, double hp, double hpp, double hppp,
                           double v_dt);

 public:
};

#include "UtilesRemap-Impl.h"

#endif  // EUCCLHYDREMAP_H
