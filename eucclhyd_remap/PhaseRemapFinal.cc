#include <stdlib.h>  // for exit

#include <Kokkos_Core.hpp>  // for KOKKOS_LAMBDA
#include <array>            // for array
#include <iostream>         // for operator<<, basic_ostream, endl
#include <memory>           // for allocator

#include "Eucclhyd.h"             // for Eucclhyd, Eucclhyd::...
#include "types/MathFunctions.h"  // for max, min
#include "types/MultiArray.h"     // for operator<<

/**
 * Job remapCellcenteredVariable called @16.0 in executeTimeLoopN method.
 * In variables: Uremap2, v, x_then_y_n
 * Out variables: m_cell_velocity_nplus1, m_internal_energy_nplus1,
 * m_density_nplus1, x_then_y_nplus1
 */
void Eucclhyd::remapCellcenteredVariable() noexcept {
  m_global_total_energy_T = 0.;
  varlp->x_then_y_nplus1 = !(varlp->x_then_y_n);
  int nbmat = options->nbmat;
  Kokkos::parallel_for(
      "remapCellcenteredVariable", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        double vol = m_euler_volume(cCells);  // volume euler
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
          m_fracvol_env(cCells)[imat] = vol_np1[imat] / volt_normalise;
          if (m_fracvol_env(cCells)[imat] < options->threshold)
            m_fracvol_env(cCells)[imat] = 0.;
          somme_frac += m_fracvol_env(cCells)[imat];
        }
        for (int imat = 0; imat < nbmat; imat++)
          m_fracvol_env(cCells)[imat] =
              m_fracvol_env(cCells)[imat] / somme_frac;

        int matcell(0);
        int imatpure(-1);
        for (int imat = 0; imat < nbmat; imat++)
          if (m_fracvol_env(cCells)[imat] > 0.) {
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
          m_mass_fraction_env(cCells)[imat] =
              varlp->Uremap2(cCells)[nbmat + imat] / masset;

        // on enleve les petits fractions de volume aussi sur la fraction
        // massique et on normalise
        double fmasset = 0.;
        for (int imat = 0; imat < nbmat; imat++) {
          if (m_fracvol_env(cCells)[imat] < options->threshold) {
            m_mass_fraction_env(cCells)[imat] = 0.;
          }
          fmasset += m_mass_fraction_env(cCells)[imat];
        }
        for (int imat = 0; imat < nbmat; imat++)
          m_mass_fraction_env(cCells)[imat] /= fmasset;

        RealArray1D<nbmatmax> m_density_env_np1 = zeroVectmat;
        double m_density_np1 = 0.;
        // std::cout << " cell--m   " << cCells << " " <<  volt << " " <<
        // vol_np1[0] << " " << vol_np1[1] << std::endl;
        for (int imat = 0; imat < nbmat; imat++) {
          if (m_fracvol_env(cCells)[imat] > options->threshold)
            m_density_env_np1[imat] =
                varlp->Uremap2(cCells)[nbmat + imat] / vol_np1[imat];
          // 1/m_density_np1 += m_mass_fraction_env(cCells)[imat] /
          // m_density_env_np1[imat];
          m_density_np1 +=
              m_fracvol_env(cCells)[imat] * m_density_env_np1[imat];
        }

        RealArray1D<dim> m_cell_velocity_np1 = {
            {varlp->Uremap2(cCells)[3 * nbmat] / (m_density_np1 * vol),
             varlp->Uremap2(cCells)[3 * nbmat + 1] / (m_density_np1 * vol)}};

        // double m_internal_energy_np1 = Uremap2(cCells)[6] / (m_density_np1 *
        // vol);
        RealArray1D<nbmatmax> pesp_np1 = zeroVectmat;
        for (int imat = 0; imat < nbmat; imat++) {
          if ((m_fracvol_env(cCells)[imat] > options->threshold) &&
              (varlp->Uremap2(cCells)[nbmat + imat] != 0.))
            pesp_np1[imat] = varlp->Uremap2(cCells)[2 * nbmat + imat] /
                             varlp->Uremap2(cCells)[nbmat + imat];
        }
        m_density_nplus1(cCells) = m_density_np1;
        // vitesse
        m_cell_velocity_nplus1(cCells) = m_cell_velocity_np1;
        // energie
        m_internal_energy_nplus1(cCells) = 0.;

        // conservation energie totale avec (m_density_np1 * vol) au lieu de
        // masset idem
        double delta_ec(0.);
        if (options->projectionConservative == 1)
          delta_ec = varlp->Uremap2(cCells)[3 * nbmat + 2] / masset -
                     0.5 * (m_cell_velocity_np1[0] * m_cell_velocity_np1[0] +
                            m_cell_velocity_np1[1] * m_cell_velocity_np1[1]);

        for (int imat = 0; imat < nbmat; imat++) {
          // densité
          m_density_env_nplus1(cCells)[imat] = m_density_env_np1[imat];
          // energies
          m_internal_energy_env_nplus1(cCells)[imat] = pesp_np1[imat];
          // conservation energie totale
          // delta_ec : energie specifique
          m_internal_energy_env_nplus1(cCells)[imat] += delta_ec;
          // energie interne totale
          m_internal_energy_nplus1(cCells) +=
              m_mass_fraction_env(cCells)[imat] *
              m_internal_energy_env_nplus1(cCells)[imat];
        }

        m_total_energy_T(cCells) =
            (m_density_np1 * vol) * m_internal_energy_nplus1(cCells) +
            0.5 * (m_density_np1 * vol) *
                (m_cell_velocity_np1[0] * m_cell_velocity_np1[0] +
                 m_cell_velocity_np1[1] * m_cell_velocity_np1[1]);
        m_global_masse_T(cCells) = 0.;
        for (int imat = 0; imat < nbmat; imat++)
          m_global_masse_T(cCells) +=
              m_density_env_nplus1(cCells)[imat] *
              vol_np1[imat];  // m_mass_fraction_env(cCells)[imat] *
                              // (m_density_np1 * vol) ; //
                              // m_density_env_nplus1(cCells)[imat] *
                              // vol_np1[imat];

        for (int imat = 0; imat < nbmat; imat++) {
          if (pesp_np1[imat] < 0. || m_density_env_np1[imat] < 0.) {
            std::cout << " cell " << cCells << " --energy ou masse negative   "
                      << imat << std::endl;
            std::cout << " energies   "
                      << m_internal_energy_env_nplus1(cCells)[0] << " "
                      << m_internal_energy_env_nplus1(cCells)[1] << " "
                      << m_internal_energy_env_nplus1(cCells)[2] << std::endl;
            std::cout << " pesp_np1   " << pesp_np1[0] << " " << pesp_np1[1]
                      << " " << pesp_np1[2] << std::endl;
            std::cout << " m_density_env_np1   " << m_density_env_np1[0] << " "
                      << m_density_env_np1[1] << " " << m_density_env_np1[2]
                      << std::endl;
            std::cout << " fractionvol   " << m_fracvol_env(cCells)[0] << " "
                      << m_fracvol_env(cCells)[1] << " "
                      << m_fracvol_env(cCells)[2] << std::endl;
            std::cout << " concentrations   " << m_mass_fraction_env(cCells)[0]
                      << " " << m_mass_fraction_env(cCells)[1] << " "
                      << m_mass_fraction_env(cCells)[2] << std::endl;
#ifdef TEST
            std::cout << "varlp->ULagrange " << varlp->ULagrange(cCells)
                      << std::endl;
            std::cout << "varlp->Uremap2 " << varlp->Uremap2(cCells)
                      << std::endl;
#endif
            m_density_env_nplus1(cCells)[imat] = 0.;
            m_internal_energy_env_nplus1(cCells)[imat] = 0.;
            m_mass_fraction_env(cCells)[imat] = 0.;
            m_fracvol_env(cCells)[imat] = 0.;
            // exit(1);
          }
        }
        if (m_internal_energy_nplus1(cCells) !=
                m_internal_energy_nplus1(cCells) ||
            (m_density_nplus1(cCells) != m_density_nplus1(cCells))) {
          std::cout << " cell--Nan   " << cCells << std::endl;
          std::cout << " densites  " << m_density_env_np1[0] << " "
                    << m_density_env_np1[1] << " " << m_density_env_np1[0]
                    << std::endl;
          std::cout << " concentrations   " << m_mass_fraction_env(cCells)[0]
                    << " " << m_mass_fraction_env(cCells)[1] << " "
                    << m_mass_fraction_env(cCells)[2] << std::endl;
          std::cout << " fractionvol   " << m_fracvol_env(cCells)[0] << " "
                    << m_fracvol_env(cCells)[1] << " "
                    << m_fracvol_env(cCells)[2] << std::endl;
          std::cout << " energies   " << m_internal_energy_env_nplus1(cCells)[0]
                    << " " << m_internal_energy_env_nplus1(cCells)[1] << " "
                    << m_internal_energy_env_nplus1(cCells)[2] << std::endl;
#ifdef TEST
          std::cout << "varlp->ULagrange " << varlp->ULagrange(cCells)
                    << std::endl;
          std::cout << "varlp->Uremap2 " << varlp->Uremap2(cCells) << std::endl;
#endif
          exit(1);
        }
        // pour les sorties :
        m_fracvol_env1(cCells) = m_fracvol_env(cCells)[0];
        m_fracvol_env2(cCells) = m_fracvol_env(cCells)[1];
        m_fracvol_env3(cCells) = m_fracvol_env(cCells)[2];
        // pression
        m_pressure_env1(cCells) = m_pressure_env(cCells)[0];
        m_pressure_env2(cCells) = m_pressure_env(cCells)[1];
        m_pressure_env3(cCells) = m_pressure_env(cCells)[2];
        // sorties paraview limitées
        if (m_cell_velocity_nplus1(cCells)[0] > 0.)
          m_x_cell_velocity(cCells) = MathFunctions::max(
              m_cell_velocity_nplus1(cCells)[0], options->threshold);
        if (m_cell_velocity_nplus1(cCells)[0] < 0.)
          m_x_cell_velocity(cCells) = MathFunctions::min(
              m_cell_velocity_nplus1(cCells)[0], -options->threshold);

        if (m_cell_velocity_nplus1(cCells)[1] > 0.)
          m_y_cell_velocity(cCells) = MathFunctions::max(
              m_cell_velocity_nplus1(cCells)[1], options->threshold);
        if (m_cell_velocity_nplus1(cCells)[1] < 0.)
          m_y_cell_velocity(cCells) = MathFunctions::min(
              m_cell_velocity_nplus1(cCells)[1], -options->threshold);
      });
  double reductionE(0.), reductionM(0.);
  {
    Kokkos::Sum<double> reducerE(reductionE);
    Kokkos::parallel_reduce(
        "reductionE", nbCells,
        KOKKOS_LAMBDA(const int& cCells, double& x) {
          reducerE.join(x, m_total_energy_T(cCells));
        },
        reducerE);
    Kokkos::Sum<double> reducerM(reductionM);
    Kokkos::parallel_reduce(
        "reductionM", nbCells,
        KOKKOS_LAMBDA(const int& cCells, double& x) {
          reducerM.join(x, m_global_masse_0(cCells));
        },
        reducerM);
  }
  m_global_total_energy_T = reductionE;
  m_total_masse_T = reductionM;
}
