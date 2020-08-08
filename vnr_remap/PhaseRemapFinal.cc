#include <stdlib.h>  // for exit

#include <Kokkos_Core.hpp>  // for KOKKOS_LAMBDA
#include <array>            // for array
#include <iostream>         // for operator<<, basic_ostream, endl
#include <memory>           // for allocator

#include "Vnr.h"        // for Eucclhyd, Eucclhyd::...
#include "types/MathFunctions.h"  // for max, min
#include "types/MultiArray.h"     // for operator<<

/**
 * Job remapCellcenteredVariable called @16.0 in executeTimeLoopN method.
 * In variables: Uremap2, v, x_then_y_n
 * Out variables: V_nplus1, e_nplus1, rho_nplus1, x_then_y_nplus1
 */
void Vnr::remapVariables() noexcept {
  ETOTALE_T = 0.;
  varlp->x_then_y_nplus1 = !(varlp->x_then_y_n);
  int nbmat = options->nbmat;
  Kokkos::parallel_for(
      "remapCellcenteredVariable", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        double vol = volE(cCells);  // volume euler
        double volt = 0.;
        double masset = 0.;
        RealArray1D<nbmatmax> vol_np1;
        for (int imat = 0; imat < nbmat; imat++) {
          vol_np1[imat] = varlp->Uremap2(cCells)[imat];
          volt += vol_np1[imat];
          // somme des masses
          masset += varlp->Uremap2(cCells)[nbmat + imat];
        }

        double volt_normalise = 0.;
        // normalisation des volumes + somme
        for (int imat = 0; imat < nbmat; imat++) {
          vol_np1[imat] *= vol / volt;
          volt_normalise += vol_np1[imat];
        }
        double somme_frac = 0.;
        for (int imat = 0; imat < nbmat; imat++) {
          fracvol(cCells)[imat] = vol_np1[imat] / volt_normalise;
          if (fracvol(cCells)[imat] < options->threshold)
            fracvol(cCells)[imat] = 0.;
          somme_frac += fracvol(cCells)[imat];
        }
        for (int imat = 0; imat < nbmat; imat++)
          fracvol(cCells)[imat] = fracvol(cCells)[imat] / somme_frac;

        int matcell(0);
        int imatpure(-1);
        for (int imat = 0; imat < nbmat; imat++)
          if (fracvol(cCells)[imat] > 0.) {
            matcell++;
            imatpure = imat;
          }
        if (matcell > 1) {
          varlp->mixte(cCells) = 1;
          varlp->pure(cCells) = -1;
        } else {
          varlp->mixte(cCells) = 0;
          varlp->pure(cCells) = imatpure;
        }
        // -----
        for (int imat = 0; imat < nbmat; imat++)
          fracmass(cCells)[imat] = varlp->Uremap2(cCells)[nbmat + imat] / masset;

        // on enleve les petits fractions de volume aussi sur la fraction
        // massique et on normalise
        double fmasset = 0.;
        for (int imat = 0; imat < nbmat; imat++) {
          if (fracvol(cCells)[imat] < options->threshold) {
            fracmass(cCells)[imat] = 0.;
          }
          fmasset += fracmass(cCells)[imat];
        }
        for (int imat = 0; imat < nbmat; imat++)
          fracmass(cCells)[imat] /= fmasset;

        RealArray1D<nbmatmax> rhop_np1 = zeroVectmat;
        double rho_np1 = 0.;
        // std::cout << " cell--m   " << cCells << " " <<  volt << " " <<
        // vol_np1[0] << " " << vol_np1[1] << std::endl;
        for (int imat = 0; imat < nbmat; imat++) {
          if (fracvol(cCells)[imat] > options->threshold)
            rhop_np1[imat] = varlp->Uremap2(cCells)[nbmat + imat] / vol_np1[imat];
          // 1/rho_np1 += fracmass(cCells)[imat] / rhop_np1[imat];
          rho_np1 += fracvol(cCells)[imat] * rhop_np1[imat];
        }

        // Vitesse aux noeuds
	// RealArray1D<dim> V_np1 = {
        //    {varlp->Uremap2(cCells)[3 * nbmat] / (rho_np1 * vol),
        //     varlp->Uremap2(cCells)[3 * nbmat + 1] / (rho_np1 * vol)}};

        RealArray1D<nbmatmax> pesp_np1 = zeroVectmat;
        for (int imat = 0; imat < nbmat; imat++) {
          if ((fracvol(cCells)[imat] > options->threshold) &&
              (varlp->Uremap2(cCells)[nbmat + imat] != 0.))
            pesp_np1[imat] = varlp->Uremap2(cCells)[2 * nbmat + imat] /
                             varlp->Uremap2(cCells)[nbmat + imat];
        }
        rho_nplus1(cCells) = rho_np1;
        // vitesse
        //V_nplus1(cCells) = V_np1;
        // energie
        e_nplus1(cCells) = 0.;

        // conservation energie totale avec (rho_np1 * vol) au lieu de masset
        // idem
        // double delta_ec(0.);
        //if (options->projectionConservative == 1)
        //  delta_ec = varlp->Uremap2(cCells)[3 * nbmat + 2] / masset -
        //                     0.5 * (V_np1[0] * V_np1[0] + V_np1[1] * V_np1[1]);

        for (int imat = 0; imat < nbmat; imat++) {
          // densité
          rhop_nplus1(cCells)[imat] = rhop_np1[imat];
          // energies
          ep_nplus1(cCells)[imat] = pesp_np1[imat];
          // conservation energie totale
          // delta_ec : energie specifique
          //ep_nplus1(cCells)[imat] += delta_ec(cCells);
          // energie interne totale
          e_nplus1(cCells) +=
              fracmass(cCells)[imat] * ep_nplus1(cCells)[imat];
        }

	// ETOT_T(cCells) =
        //    (rho_np1 * vol) * e_nplus1(cCells) +
        //    0.5 * (rho_np1 * vol) * (V_np1[0] * V_np1[0] + V_np1[1] * V_np1[1]);
        MTOT_T(cCells) = 0.;
        for (int imat = 0; imat < nbmat; imat++)
          MTOT_T(cCells) +=
              rhop_nplus1(cCells)[imat] *
              vol_np1[imat];  // fracmass(cCells)[imat] * (rho_np1 * vol) ; //
                              // rhop_nplus1(cCells)[imat] * vol_np1[imat];

        for (int imat = 0; imat < nbmat; imat++) {
          if (pesp_np1[imat] < 0. || rhop_np1[imat] < 0.) {
            std::cout << " cell " << cCells << " --energy ou masse negative   "
                      << imat << std::endl;
            std::cout << " energies   " << ep_nplus1(cCells)[0] << " "
                      << ep_nplus1(cCells)[1] << " " << ep_nplus1(cCells)[2]
                      << std::endl;
            std::cout << " pesp_np1   " << pesp_np1[0] << " " << pesp_np1[1]
                      << " " << pesp_np1[2] << std::endl;
            std::cout << " rhop_np1   " << rhop_np1[0] << " " << rhop_np1[1]
                      << " " << rhop_np1[2] << std::endl;
            std::cout << " fractionvol   " << fracvol(cCells)[0] << " "
                      << fracvol(cCells)[1] << " " << fracvol(cCells)[2]
                      << std::endl;
            std::cout << " concentrations   " << fracmass(cCells)[0] << " "
                      << fracmass(cCells)[1] << " " << fracmass(cCells)[2]
                      << std::endl;
            std::cout << "varlp->ULagrange " << varlp->ULagrange(cCells) << std::endl;
            std::cout << "varlp->Uremap2 " << varlp->Uremap2(cCells) << std::endl;
            rhop_nplus1(cCells)[imat] = 0.;
            ep_nplus1(cCells)[imat] = 0.;
            fracmass(cCells)[imat] = 0.;
            fracvol(cCells)[imat] = 0.;
            // exit(1);
          }
        }
        if (e_nplus1(cCells) != e_nplus1(cCells) ||
            (rho_nplus1(cCells) != rho_nplus1(cCells))) {
          std::cout << " cell--Nan   " << cCells << std::endl;
          std::cout << " densites  " << rhop_np1[0] << " " << rhop_np1[1] << " "
                    << rhop_np1[0] << std::endl;
          std::cout << " concentrations   " << fracmass(cCells)[0] << " "
                    << fracmass(cCells)[1] << " " << fracmass(cCells)[2]
                    << std::endl;
          std::cout << " fractionvol   " << fracvol(cCells)[0] << " "
                    << fracvol(cCells)[1] << " " << fracvol(cCells)[2]
                    << std::endl;
          std::cout << " energies   " << ep_nplus1(cCells)[0] << " "
                    << ep_nplus1(cCells)[1] << " " << ep_nplus1(cCells)[2]
                    << std::endl;
          std::cout << "varlp->ULagrange " << varlp->ULagrange(cCells) << std::endl;
          std::cout << "varlp->Uremap2 " << varlp->Uremap2(cCells) << std::endl;
          exit(1);
        }
        // pour les sorties :
        fracvol1(cCells) = fracvol(cCells)[0];
        fracvol2(cCells) = fracvol(cCells)[1];
        fracvol3(cCells) = fracvol(cCells)[2];
        // pression
        p1(cCells) = pp_nplus1(cCells)[0];
        p2(cCells) = pp_nplus1(cCells)[1];
        p3(cCells) = pp_nplus1(cCells)[2];
        
      });
  double reductionE(0.), reductionM(0.);
  {
    Kokkos::Sum<double> reducerE(reductionE);
    Kokkos::parallel_reduce("reductionE", nbCells,
                            KOKKOS_LAMBDA(const int& cCells, double& x) {
                              reducerE.join(x, ETOT_T(cCells));
                            },
                            reducerE);
    Kokkos::Sum<double> reducerM(reductionM);
    Kokkos::parallel_reduce("reductionM", nbCells,
                            KOKKOS_LAMBDA(const int& cCells, double& x) {
                              reducerM.join(x, MTOT_0(cCells));
                            },
                            reducerM);
  }
  ETOTALE_T = reductionE;
  MASSET_T = reductionM;
}
