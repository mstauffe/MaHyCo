#include <math.h>  // for sqrt

#include <Kokkos_Core.hpp>
#include <algorithm>  // for copy
#include <array>      // for array
#include <iostream>   // for operator<<, basic_ostream::operat...
#include <vector>     // for allocator, vector

#include "EucclhydRemap.h"          // for EucclhydRemap, EucclhydRemap::Opt...
#include "UtilesRemap-Impl.h"       // for EucclhydRemap::computeFluxPP
#include "mesh/CartesianMesh2D.h"   // for CartesianMesh2D
#include "types/ArrayOperations.h"  // for divide, minus, plus
#include "types/MathFunctions.h"    // for dot
#include "types/MultiArray.h"       // for operator<<
#include "utils/Utils.h"            // for indexOf

/**
 * Job computeGradPhiFace1 called @8.0 in executeTimeLoopN method.
 * In variables: ULagrange, deltaxLagrange, projectionOrder, vLagrange,
 * x_then_y_n Out variables: gradPhiFace1
 */
void EucclhydRemap::computeGradPhiFace1() noexcept {
  if (options->projectionOrder > 1) {
    if (x_then_y_n) {
      auto innerVerticalFaces(mesh->getInnerVerticalFaces());
      int nbInnerVerticalFaces(mesh->getNbInnerVerticalFaces());
      Kokkos::parallel_for(
          "computeGradPhiFace1", nbInnerVerticalFaces,
          KOKKOS_LAMBDA(const int& fInnerVerticalFaces) {
            int fId(innerVerticalFaces[fInnerVerticalFaces]);
            int fFaces(utils::indexOf(mesh->getFaces(), fId));
            int cfFrontCellF(mesh->getFrontCell(fId));
            int cfId(cfFrontCellF);
            int cfCells(cfId);
            int cbBackCellF(mesh->getBackCell(fId));
            int cbId(cbBackCellF);
            int cbCells(cbId);
            //
            gradPhiFace1(fFaces) = ArrayOperations::divide(
                ArrayOperations::minus(Phi(cfCells), Phi(cbCells)),
                deltaxLagrange(fFaces));
            //
            int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
            int n1Id(n1FirstNodeOfFaceF);
            int n1Nodes(n1Id);
            int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
            int n2Id(n2SecondNodeOfFaceF);
            int n2Nodes(n2Id);
            LfLagrange(fFaces) =
                sqrt((XLagrange(n1Nodes)[0] - XLagrange(n2Nodes)[0]) *
                         (XLagrange(n1Nodes)[0] - XLagrange(n2Nodes)[0]) +
                     (XLagrange(n1Nodes)[1] - XLagrange(n2Nodes)[1]) *
                         (XLagrange(n1Nodes)[1] - XLagrange(n2Nodes)[1]));
          });

      Kokkos::parallel_for(
          "computeGradPhiFace1", nbCells, KOKKOS_LAMBDA(const int& cCells) {
            int cId(cCells);
            HvLagrange(cCells) = 0.;

            int fbBottomFaceOfCellC(mesh->getBottomFaceOfCell(cId));
            int fbId(fbBottomFaceOfCellC);
            int fbFaces(utils::indexOf(mesh->getFaces(), fbId));
            HvLagrange(cCells) += 0.5 * faceLengthLagrange(fbFaces);

            int ftTopFaceOfCellC(mesh->getTopFaceOfCell(cId));
            int ftId(ftTopFaceOfCellC);
            int ftFaces(utils::indexOf(mesh->getFaces(), ftId));
            HvLagrange(cCells) += 0.5 * faceLengthLagrange(ftFaces);
          });
    } else {
      auto innerHorizontalFaces(mesh->getInnerHorizontalFaces());
      int nbInnerHorizontalFaces(mesh->getNbInnerHorizontalFaces());
      Kokkos::parallel_for(
          "computeGradPhiFace1", nbInnerHorizontalFaces,
          KOKKOS_LAMBDA(const int& fInnerHorizontalFaces) {
            int fId(innerHorizontalFaces[fInnerHorizontalFaces]);
            int fFaces(utils::indexOf(mesh->getFaces(), fId));
            int cfFrontCellF(mesh->getFrontCell(fId));
            int cfId(cfFrontCellF);
            int cfCells(cfId);
            int cbBackCellF(mesh->getBackCell(fId));
            int cbId(cbBackCellF);
            int cbCells(cbId);
            //
            gradPhiFace1(fFaces) = ArrayOperations::divide(
                ArrayOperations::minus(Phi(cfCells), Phi(cbCells)),
                deltaxLagrange(fFaces));
            int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
            int n1Id(n1FirstNodeOfFaceF);
            int n1Nodes(n1Id);
            int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
            int n2Id(n2SecondNodeOfFaceF);
            int n2Nodes(n2Id);
            //
            LfLagrange(fFaces) =
                sqrt((XLagrange(n1Nodes)[0] - XLagrange(n2Nodes)[0]) *
                         (XLagrange(n1Nodes)[0] - XLagrange(n2Nodes)[0]) +
                     (XLagrange(n1Nodes)[1] - XLagrange(n2Nodes)[1]) *
                         (XLagrange(n1Nodes)[1] - XLagrange(n2Nodes)[1]));
          });
      Kokkos::parallel_for(
          "computeGradPhiFace1", nbCells, KOKKOS_LAMBDA(const int& cCells) {
            int cId(cCells);

            // seconde methode
            HvLagrange(cCells) = 0.;
            int frRightFaceOfCellC(mesh->getRightFaceOfCell(cId));
            int frId(frRightFaceOfCellC);
            int frFaces(utils::indexOf(mesh->getFaces(), frId));
            HvLagrange(cCells) += 0.5 * faceLengthLagrange(frFaces);

            int flLeftFaceOfCellC(mesh->getLeftFaceOfCell(cId));
            int flId(flLeftFaceOfCellC);
            int flFaces(utils::indexOf(mesh->getFaces(), flId));
            HvLagrange(cCells) += 0.5 * faceLengthLagrange(flFaces);
          });
    }
  }
}
/**
 * Job computeGradPhi1 called @9.0 in executeTimeLoopN method.
 * In variables: gradPhiFace1, projectionLimiterId, projectionOrder, x_then_y_n
 * Out variables: gradPhi1
 */
void EucclhydRemap::computeGradPhi1() noexcept {
  if (options->projectionOrder > 1) {
    if (x_then_y_n) {
      // std::cout << " Phase 1 Horizontale computeGradPhi1 " << std::endl;
      Kokkos::parallel_for(
          "computeGradPhi1", nbCells, KOKKOS_LAMBDA(const int& cCells) {
            int cId(cCells);
            int frRightFaceOfCellC(mesh->getRightFaceOfCell(cId));
            int frId(frRightFaceOfCellC);
            int frFaces(utils::indexOf(mesh->getFaces(), frId));

            int flLeftFaceOfCellC(mesh->getLeftFaceOfCell(cId));
            int flId(flLeftFaceOfCellC);
            int flFaces(utils::indexOf(mesh->getFaces(), flId));
            // maille devant
            int cfFrontCellF(mesh->getFrontCell(frId));
            int cfId(cfFrontCellF);
            int cfCells(cfId);
            int frFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), frId));
            // maille deriere
            int cbBackCellF(mesh->getBackCell(flId));
            int cbId(cbBackCellF);
            int cbCells(cbId);
            int flFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), flId));

            // std::cout << " Phase 1 Horizontale " << std::endl;
            RealArray1D<dim> exy = {{1.0, 0.0}};

            if (cbCells == -1) cbCells = cCells;
            if (cfCells == -1) cfCells = cCells;
            bool voisinage_pure =
                (limiteurs->projectionLimiteurMixte == 1) &&
                (mixte(cCells) == 0 && mixte(cfCells) == 0 &&
                 mixte(cbCells) == 0 && pure(cCells) == pure(cfCells) &&
                 pure(cCells) == pure(cbCells));

            int limiter = limiteurs->projectionLimiterId;
            if ((limiteurs->projectionAvecPlateauPente == 1) && voisinage_pure)
              limiter = limiteurs->projectionLimiterIdPure;

            gradPhi1(cCells) = computeAndLimitGradPhi(
                limiter, gradPhiFace1(frFaces), gradPhiFace1(flFaces),
                Phi(cCells), Phi(cfCells), Phi(cbCells), HvLagrange(cCells),
                HvLagrange(cfCells), HvLagrange(cbCells));

            if (limiteurs->projectionAvecPlateauPente == 1) {
              double Flux_sortant_ar =
                  MathFunctions::dot(outerFaceNormal(cCells, flFacesOfCellC),
                                     exy) *
                  faceNormalVelocity(flFaces);

              if (voisinage_pure)
                deltaPhiFaceAr(cCells) = computeFluxPPPure(
                    gradPhi1(cCells), Phi(cCells), Phi(cfCells), Phi(cbCells),
                    HvLagrange(cCells), HvLagrange(cfCells),
                    HvLagrange(cbCells), Flux_sortant_ar, gt->deltat_n, 0,
                    cCells, options->threshold);
              else
                deltaPhiFaceAr(cCells) = computeFluxPP(
                    gradPhi1(cCells), Phi(cCells), Phi(cfCells), Phi(cbCells),
                    HvLagrange(cCells), HvLagrange(cfCells),
                    HvLagrange(cbCells), Flux_sortant_ar, gt->deltat_n, 0,
                    cCells, options->threshold);

              double Flux_sortant_av =
                  MathFunctions::dot(outerFaceNormal(cCells, frFacesOfCellC),
                                     exy) *
                  faceNormalVelocity(frFaces);
              if (voisinage_pure)
                deltaPhiFaceAv(cCells) = computeFluxPPPure(
                    gradPhi1(cCells), Phi(cCells), Phi(cfCells), Phi(cbCells),
                    HvLagrange(cCells), HvLagrange(cfCells),
                    HvLagrange(cbCells), Flux_sortant_av, gt->deltat_n, 1,
                    cCells, options->threshold);
              else
                deltaPhiFaceAv(cCells) = computeFluxPP(
                    gradPhi1(cCells), Phi(cCells), Phi(cfCells), Phi(cbCells),
                    HvLagrange(cCells), HvLagrange(cfCells),
                    HvLagrange(cbCells), Flux_sortant_av, gt->deltat_n, 1,
                    cCells, options->threshold);
            }
          });
    } else {
      // std::cout << " Phase 1 Verticale computeGradPhi1 " << std::endl;
      Kokkos::parallel_for(
          "computeGradPhi1", nbCells, KOKKOS_LAMBDA(const int& cCells) {
            int cId(cCells);
            int fbBottomFaceOfCellC(mesh->getBottomFaceOfCell(cId));
            int fbId(fbBottomFaceOfCellC);
            int fbFaces(utils::indexOf(mesh->getFaces(), fbId));
            int ftTopFaceOfCellC(mesh->getTopFaceOfCell(cId));
            int ftId(ftTopFaceOfCellC);
            int ftFaces(utils::indexOf(mesh->getFaces(), ftId));
            // maille dessous
            int cfFrontCellF(mesh->getFrontCell(fbId));
            int cfId(cfFrontCellF);
            int cfCells(cfId);
            int fbFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), fbId));
            // maille dessus
            int cbBackCellF(mesh->getBackCell(ftId));
            int cbId(cbBackCellF);
            int cbCells(cbId);
            int ftFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), ftId));

            if (cbCells == -1) cbCells = cCells;
            if (cfCells == -1) cfCells = cCells;
            bool voisinage_pure =
                (limiteurs->projectionLimiteurMixte == 1) &&
                (mixte(cCells) == 0 && mixte(cfCells) == 0 &&
                 mixte(cbCells) == 0 && pure(cCells) == pure(cfCells) &&
                 pure(cCells) == pure(cbCells));

            int limiter = limiteurs->projectionLimiterId;
            if ((limiteurs->projectionAvecPlateauPente == 1) && voisinage_pure)
              limiter = limiteurs->projectionLimiterIdPure;

            gradPhi1(cCells) = computeAndLimitGradPhi(
                limiter, gradPhiFace1(fbFaces), gradPhiFace1(ftFaces),
                Phi(cCells), Phi(cbCells), Phi(cfCells), HvLagrange(cCells),
                HvLagrange(cbCells), HvLagrange(cfCells));

            if (limiteurs->projectionAvecPlateauPente == 1) {
              RealArray1D<dim> exy = {{0.0, 1.0}};

              double Flux_sortant_av =
                  MathFunctions::dot(outerFaceNormal(cCells, fbFacesOfCellC),
                                     exy) *
                  faceNormalVelocity(fbFaces);
              if (voisinage_pure)
                deltaPhiFaceAr(cCells) = computeFluxPPPure(
                    gradPhi1(cCells), Phi(cCells), Phi(cbCells), Phi(cfCells),
                    HvLagrange(cCells), HvLagrange(cbCells),
                    HvLagrange(cfCells), Flux_sortant_av, gt->deltat_n, 0,
                    cCells, options->threshold);
              else
                deltaPhiFaceAr(cCells) = computeFluxPP(
                    gradPhi1(cCells), Phi(cCells), Phi(cbCells), Phi(cfCells),
                    HvLagrange(cCells), HvLagrange(cbCells),
                    HvLagrange(cfCells), Flux_sortant_av, gt->deltat_n, 0,
                    cCells, options->threshold);

              double Flux_sortant_ar =
                  MathFunctions::dot(outerFaceNormal(cCells, ftFacesOfCellC),
                                     exy) *
                  faceNormalVelocity(ftFaces);

              if (voisinage_pure)
                deltaPhiFaceAv(cCells) = computeFluxPPPure(
                    gradPhi1(cCells), Phi(cCells), Phi(cbCells), Phi(cfCells),
                    HvLagrange(cCells), HvLagrange(cbCells),
                    HvLagrange(cfCells), Flux_sortant_ar, gt->deltat_n, 1,
                    cCells, options->threshold);
              else
                deltaPhiFaceAv(cCells) = computeFluxPP(
                    gradPhi1(cCells), Phi(cCells), Phi(cbCells), Phi(cfCells),
                    HvLagrange(cCells), HvLagrange(cbCells),
                    HvLagrange(cfCells), Flux_sortant_ar, gt->deltat_n, 1,
                    cCells, options->threshold);
            }
          });
    }
  }
}

/**
 * Job computeUpwindFaceQuantitiesForProjection1 called @10.0 in
 * executeTimeLoopN method. In variables: ULagrange, XcLagrange, Xf,
 * deltaxLagrange, faceNormal, faceNormalVelocity, gradPhi1, vLagrange,
 * x_then_y_n Out variables: phiFace1
 */
void EucclhydRemap::computeUpwindFaceQuantitiesForProjection1() noexcept {
  if (x_then_y_n) {
    // std::cout << " Phase Projection 1 Horizontale " << std::endl;
    auto innerVerticalFaces(mesh->getInnerVerticalFaces());
    int nbInnerVerticalFaces(mesh->getNbInnerVerticalFaces());
    Kokkos::parallel_for(
        "computeUpwindFaceQuantitiesForProjection1", nbInnerVerticalFaces,
        KOKKOS_LAMBDA(const int& fInnerVerticalFaces) {
          int fId(innerVerticalFaces[fInnerVerticalFaces]);
          int fFaces(utils::indexOf(mesh->getFaces(), fId));
          int cfFrontCellF(mesh->getFrontCell(fId));
          int cfId(cfFrontCellF);
          int cfCells(cfId);
          int cbBackCellF(mesh->getBackCell(fId));
          int cbId(cbBackCellF);
          int cbCells(cbId);
          // phiFace1 correspond
          // à la valeur de phi(x) à la face pour l'ordre 2 sans plateau pente
          // à la valeur du flux (integration de phi(x)) pour l'ordre 2 avec
          // Plateau-Pente à la valeur du flux (integration de phi(x)) pour
          // l'ordre 3
          if (options->projectionOrder == 2) {
            if (limiteurs->projectionAvecPlateauPente == 0) {
              phiFace1(fFaces) = computeUpwindFaceQuantities(
                  faceNormal(fFaces), faceNormalVelocity(fFaces),
                  deltaxLagrange(fFaces), Xf(fFaces),
                  ArrayOperations::divide(ULagrange(cbCells),
                                          vLagrange(cbCells)),
                  gradPhi1(cbCells), XcLagrange(cbCells),
                  ArrayOperations::divide(ULagrange(cfCells),
                                          vLagrange(cfCells)),
                  gradPhi1(cfCells), XcLagrange(cfCells));
            } else {
              phiFace1(fFaces) = ArrayOperations::minus(
                  deltaPhiFaceAv(cbCells), deltaPhiFaceAr(cfCells));
            }
          } else if (options->projectionOrder == 3) {
            // cfCells est à droite de cCells
            // cffcells est à droite de cfcells
            // cffcells est à droite de cffcells
            int cffCells(getRightCells(cfCells));
            int cfffCells(getRightCells(cffCells));

            // cbCells est à gauche de cCells
            // cbbcells est à gauche de cbcells
            // cbbbcells est à gauche de cbbcells
            int cbbCells(getLeftCells(cbCells));
            int cbbbCells(getLeftCells(cbbCells));

            phiFace1(fFaces) = computeVecFluxOrdre3(
                ArrayOperations::divide(ULagrange(cbbbCells),
                                        vLagrange(cbbbCells)),
                ArrayOperations::divide(ULagrange(cbbCells),
                                        vLagrange(cbbCells)),
                ArrayOperations::divide(ULagrange(cbCells), vLagrange(cbCells)),
                ArrayOperations::divide(ULagrange(cfCells), vLagrange(cfCells)),
                ArrayOperations::divide(ULagrange(cffCells),
                                        vLagrange(cffCells)),
                ArrayOperations::divide(ULagrange(cfffCells),
                                        vLagrange(cfffCells)),
                HvLagrange(cbbbCells), HvLagrange(cbbCells),
                HvLagrange(cbCells), HvLagrange(cfCells), HvLagrange(cffCells),
                HvLagrange(cfffCells), faceNormalVelocity(fFaces),
                gt->deltat_n);
          }
        });
  } else {
    // std::cout << " Phase Projection 1 Verticale " << std::endl;
    auto innerHorizontalFaces(mesh->getInnerHorizontalFaces());
    int nbInnerHorizontalFaces(mesh->getNbInnerHorizontalFaces());
    Kokkos::parallel_for(
        "computeUpwindFaceQuantitiesForProjection1", nbInnerHorizontalFaces,
        KOKKOS_LAMBDA(const int& fInnerHorizontalFaces) {
          int fId(innerHorizontalFaces[fInnerHorizontalFaces]);
          int fFaces(utils::indexOf(mesh->getFaces(), fId));
          int cfFrontCellF(mesh->getFrontCell(fId));
          int cfId(cfFrontCellF);
          int cfCells(cfId);
          int cbBackCellF(mesh->getBackCell(fId));
          int cbId(cbBackCellF);
          int cbCells(cbId);
          // phiFace1 correspond
          // à la valeur de phi(x) à la face pour l'ordre 2 sans plateau pente
          // à la valeur du flux (integration de phi(x)) pour l'ordre 2 avec
          // Plateau-Pente à la valeur du flux (integration de phi(x)) pour
          // l'ordre 3
          if (options->projectionOrder == 2) {
            if (limiteurs->projectionAvecPlateauPente == 0) {
              phiFace1(fFaces) = computeUpwindFaceQuantities(
                  faceNormal(fFaces), faceNormalVelocity(fFaces),
                  deltaxLagrange(fFaces), Xf(fFaces),
                  ArrayOperations::divide(ULagrange(cbCells),
                                          vLagrange(cbCells)),
                  gradPhi1(cbCells), XcLagrange(cbCells),
                  ArrayOperations::divide(ULagrange(cfCells),
                                          vLagrange(cfCells)),
                  gradPhi1(cfCells), XcLagrange(cfCells));
            } else {
              phiFace1(fFaces) = ArrayOperations::minus(
                  deltaPhiFaceAv(cfCells), deltaPhiFaceAr(cbCells));
            }
          } else if (options->projectionOrder == 3) {
            // cfCells est en dessous de cCells
            // cffCells est en dessous de cfcells
            // cfffCells est en dessous de cffcells
            int cffCells(getBottomCells(cfCells));
            int cfffCells(getBottomCells(cffCells));

            // cbCells est au dessus de cCells
            // cbbCells est au dessus de cbcells
            // cbbbCells est au dessus de cbbcells
            int cbbCells(getTopCells(cbCells));
            int cbbbCells(getTopCells(cbbCells));

            phiFace1(fFaces) = computeVecFluxOrdre3(
                ArrayOperations::divide(ULagrange(cfffCells),
                                        vLagrange(cfffCells)),
                ArrayOperations::divide(ULagrange(cffCells),
                                        vLagrange(cffCells)),
                ArrayOperations::divide(ULagrange(cfCells), vLagrange(cfCells)),
                ArrayOperations::divide(ULagrange(cbCells), vLagrange(cbCells)),
                ArrayOperations::divide(ULagrange(cbbCells),
                                        vLagrange(cbbCells)),
                ArrayOperations::divide(ULagrange(cbbbCells),
                                        vLagrange(cbbbCells)),
                HvLagrange(cfffCells), HvLagrange(cffCells),
                HvLagrange(cfCells), HvLagrange(cbCells), HvLagrange(cbbCells),
                HvLagrange(cbbbCells), faceNormalVelocity(fFaces),
                gt->deltat_n);
          }
        });
  }
}
/**
 * Job computeUremap1 called @11.0 in executeTimeLoopN method.
 * In variables: ULagrange, deltat_n, faceLength, faceNormal,
 * faceNormalVelocity, outerFaceNormal, phiFace1, x_then_y_n Out variables:
 * Uremap1
 */
void EucclhydRemap::computeUremap1() noexcept {
  int nbmat = options->nbmat;
  RealArray1D<dim> exy = xThenYToDirection(x_then_y_n);
  Kokkos::parallel_for(
      "computeUremap1", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        int cId(cCells);

        // std::cout << " cCells " << cCells << std::endl;
        // std::cout << "ULagrange " << ULagrange(cCells) << std::endl;
        RealArray1D<nbequamax> reduction8 = Uzero;
        {
          auto neighbourCellsC(mesh->getNeighbourCells(cId));
          for (int dNeighbourCellsC = 0;
               dNeighbourCellsC < neighbourCellsC.size(); dNeighbourCellsC++) {
            int dId(neighbourCellsC[dNeighbourCellsC]);
            int fCommonFaceCD(mesh->getCommonFace(cId, dId));
            int fId(fCommonFaceCD);
            int fFaces(utils::indexOf(mesh->getFaces(), fId));
            int fFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), fId));
            reduction8 = ArrayOperations::plus(
                reduction8, (computeRemapFlux(
                                options->projectionOrder,
                                limiteurs->projectionAvecPlateauPente,
                                faceNormalVelocity(fFaces), faceNormal(fFaces),
                                faceLength(fFaces), phiFace1(fFaces),
                                outerFaceNormal(cCells, fFacesOfCellC), exy,
                                gt->deltat_n)));
          }
          if (cdl->FluxBC > 0) {
            // flux exterieur eventuel
            if ((cCells == dbgcell3 || cCells == dbgcell2 ||
                 cCells == dbgcell1)) {
              std::cout << " --------- flux face exterieur "
                           "----------------------------------"
                        << std::endl;
              std::cout << exy << " AV cell   " << cCells << "reduction8 "
                        << reduction8 << std::endl;
            }
            //
            reduction8 = ArrayOperations::plus(
                reduction8, (computeBoundaryFluxes(1, cCells, exy)));
            //
            if ((cCells == dbgcell3 || cCells == dbgcell2 ||
                 cCells == dbgcell1)) {
              std::cout << exy << " AP cell   " << cCells << "reduction8 "
                        << reduction8 << std::endl;
              std::cout << computeBoundaryFluxes(1, cCells, exy) << std::endl;
              std::cout << " --------- ----------------------------------"
                        << std::endl;
            }
          }
        }

        Uremap1(cCells) = ArrayOperations::minus(ULagrange(cCells), reduction8);

        if (limiteurs->projectionAvecPlateauPente == 1) {
          // option ou on ne regarde pas la variation de rho, V et e
          // phi = (f1, f2, rho1*f1, rho2*f2, Vx, Vy, e1, e2
          // ce qui permet d'ecrire le flux telque
          // Flux = (dv1 = f1dv, dv2=f2*dv, dm1=rho1*df1, dm2=rho2*df2, d(mVx) =
          // Vx*(dm1+dm2), d(mVy) = Vy*(dm1+dm2), d(m1e1) = e1*dm1,  d(m2e2) =
          // e2*dm2 dans computeFluxPP

          double somme_volume = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            somme_volume += Uremap1(cCells)[imat];
          }
          // Phi volume
          double somme_masse = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            Phi(cCells)[imat] = Uremap1(cCells)[imat] / somme_volume;
            // Phi masse
            if (Uremap1(cCells)[imat] != 0.)
              Phi(cCells)[nbmat + imat] =
                  Uremap1(cCells)[nbmat + imat] / (Uremap1(cCells)[imat]);
            else
              Phi(cCells)[nbmat + imat] = 0.;
            somme_masse += Uremap1(cCells)[nbmat + imat];
          }
          // Phi Vitesse
          Phi(cCells)[3 * nbmat] = Uremap1(cCells)[3 * nbmat] / somme_masse;
          Phi(cCells)[3 * nbmat + 1] =
              Uremap1(cCells)[3 * nbmat + 1] / somme_masse;
          // Phi energie
          for (int imat = 0; imat < nbmat; imat++) {
            if (Uremap1(cCells)[nbmat + imat] != 0.)
              Phi(cCells)[2 * nbmat + imat] =
                  Uremap1(cCells)[2 * nbmat + imat] /
                  Uremap1(cCells)[nbmat + imat];
            else
              Phi(cCells)[2 * nbmat + imat] = 0.;
          }
          // Phi energie cinétique
          if (options->projectionConservative == 1)
            Phi(cCells)[3 * nbmat + 2] =
                Uremap1(cCells)[3 * nbmat + 2] / somme_masse;

        } else {
          Phi(cCells) =
              ArrayOperations::divide(Uremap1(cCells), vLagrange(cCells));
        }

        if ((cCells == dbgcell3 || cCells == dbgcell2 || cCells == dbgcell1)) {
          std::cout << " cell   " << cCells << "Phi apres Uremap1"
                    << Phi(cCells) << std::endl;
        }
        // Mises à jour de l'indicateur mailles mixtes
        int matcell(0);
        int imatpure(-1);
        for (int imat = 0; imat < nbmat; imat++)
          if (Phi(cCells)[imat] > 0.) {
            matcell++;
            imatpure = imat;
          }
        if (matcell > 1) {
          mixte(cCells) = 1;
          pure(cCells) = -1;
        } else {
          mixte(cCells) = 0;
          pure(cCells) = imatpure;
        }
      });
}
