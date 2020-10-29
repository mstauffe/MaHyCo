#include <math.h>    // for pow, floor, sqrt
#include <stdlib.h>  // for exit

#include <Kokkos_Core.hpp>
#include <algorithm>  // for max, min
#include <array>      // for array
#include <iostream>   // for operator<<, basic_ostream, basic_...
#include <vector>     // for allocator, vector

#include "../includes/SchemaParticules.h"  // for SchemaParticules, SchemaParticules::Particules
#include "Eucclhyd.h"                      // for Eucclhyd, Eucclhyd::Opt...
#include "types/MathFunctions.h"  // for max

void Eucclhyd::updateParticlePosition() noexcept {
  Kokkos::parallel_for(
      "updateParticleCoefficient", mesh->getNbCells(),
      KOKKOS_LAMBDA(const int& cCells) { m_cell_particle_list(cCells).clear(); });
  Kokkos::parallel_for("initPart", nbPart, KOKKOS_LAMBDA(const int& ipart) {
    m_particle_coord_nplus1(ipart) =
        m_particle_coord_n(ipart) + m_particle_velocity_n(ipart) * gt->deltat_n;
    if (m_particle_coord_nplus1(ipart)[1] < 0.) {
      m_particle_coord_nplus1(ipart)[1] = -m_particle_coord_nplus1(ipart)[1];
      m_particle_velocity_n(ipart)[1] = -m_particle_velocity_n(ipart)[1];
    }
    int icell = MathFunctions::max(
        floor(m_particle_coord_nplus1(ipart)[0] / cstmesh->X_EDGE_LENGTH), 0);
    int jcell = MathFunctions::max(
        floor(m_particle_coord_nplus1(ipart)[1] / cstmesh->Y_EDGE_LENGTH), 0);
    m_particle_cell(ipart) = jcell * cstmesh->X_EDGE_ELEMS + icell;

    // conditions limites
    if (m_fracvol_env(m_particle_cell(ipart))[m_particle_env(ipart)] < 0.25) {
      RealArray1D<dim> gradf = zeroVect;
      if (m_particle_env(ipart) == 0) gradf = m_fracvol_gradient_env(m_particle_cell(ipart), 0);
      if (m_particle_env(ipart) == 1) gradf = m_fracvol_gradient_env(m_particle_cell(ipart), 1);
      if (m_particle_env(ipart) == 2) gradf = m_fracvol_gradient_env(m_particle_cell(ipart), 2);

      // std::cout << " CL : Part  " << ipart << " vit " << m_particle_velocity_n(ipart)
      //	    << "  xp= " << m_particle_coord_nplus1(ipart)[0] << "  yp= "
      //<<
      // m_particle_coord_nplus1(ipart)[1] << "  " << m_particle_cell(ipart) <<
      // std::endl;

      // particule sort du materiau m_particle_env(ipart)
      // changement de la vitesse
      double module_vit = sqrt(m_particle_velocity_n(ipart)[0] * m_particle_velocity_n(ipart)[0] +
                               m_particle_velocity_n(ipart)[1] * m_particle_velocity_n(ipart)[1]);
      double module_gradf = sqrt(gradf[0] * gradf[0] + gradf[1] * gradf[1]);
      m_particle_velocity_n(ipart) = gradf * 1.1 * module_vit / module_gradf;
      m_particle_coord_nplus1(ipart) =
          m_particle_coord_n(ipart) + m_particle_velocity_n(ipart) * gt->deltat_n;
      if (m_particle_coord_nplus1(ipart)[1] < 0.) {
        m_particle_coord_nplus1(ipart)[1] = -m_particle_coord_nplus1(ipart)[1];
        m_particle_velocity_n(ipart)[1] = -m_particle_velocity_n(ipart)[1];
      }
      int icell = MathFunctions::max(
          floor(m_particle_coord_nplus1(ipart)[0] / cstmesh->X_EDGE_LENGTH), 0);
      int jcell = MathFunctions::max(
          floor(m_particle_coord_nplus1(ipart)[1] / cstmesh->Y_EDGE_LENGTH), 0);
      m_particle_cell(ipart) = jcell * cstmesh->X_EDGE_ELEMS + icell;

      // std::cout << " AP : Part  " << ipart << " vit " << m_particle_velocity_n(ipart)
      //	    << "  xp= " << m_particle_coord_nplus1(ipart)[0] << "  yp= "
      //<<
      // m_particle_coord_nplus1(ipart)[1] << "  " << m_particle_cell(ipart) <<
      // std::endl; std::cout << " AP : Part  " << ipart << " gradf " << gradf
      // << " module_vit " << module_vit << std::endl;

      if (m_fracvol_env(m_particle_cell(ipart))[m_particle_env(ipart)] < 0.05)
        std::cout << " Part  " << ipart << " non recuperee " << std::endl;
    }

    m_cell_particle_list(m_particle_cell(ipart)).push_back(ipart);
    // std::cout << " Part  " << ipart << " vit " << m_particle_velocity_nplus1(ipart) <<
    // "  xp= " << m_particle_coord_nplus1(ipart)[0] << "  yp= " <<
    // m_particle_coord_nplus1(ipart)[1] << "  " << m_particle_cell(ipart)
    //	  << " " << m_cell_particle_list(m_particle_cell(ipart)).size() << std::endl;
  });
}

void Eucclhyd::updateParticleCoefficients() noexcept {
  Kokkos::parallel_for(
      "updateParticleCoefficient", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        m_cell_particle_volume_fraction(cCells) = 1.;
        for (int ipart = 0; ipart < m_cell_particle_list(cCells).size(); ipart++) {
          m_cell_particle_volume_fraction(cCells) -=
              m_particle_weight(ipart) * m_particle_volume(ipart) / m_euler_volume(cCells);
          m_cell_particle_volume_fraction(cCells) = max(m_cell_particle_volume_fraction(cCells), 0.);
          if (m_cell_particle_volume_fraction(cCells) == 0.) {
            std::cout << cCells << "  " << m_cell_particle_list(cCells).size() << " v "
                      << m_euler_volume(cCells) << " " << m_cell_particle_volume_fraction(cCells)
                      << std::endl;
            std::cout << " Plus de gaz dans la maille -> fin du calcul "
                      << std::endl;
            exit(1);
          }
        }
        // if (m_cell_particle_list(cCells).size() > 0) std::cout << cCells << "  " <<
        // m_cell_particle_list(cCells).size() << " v " << m_euler_volume(cCells) << " " <<
        // m_cell_particle_volume_fraction(cCells) << std::endl;
      });

  Kokkos::parallel_for(
      "updateParticleCoefficient", nbPart, KOKKOS_LAMBDA(const int& ipart) {
        int icells = m_particle_cell(ipart);
        RealArray1D<dim> cell_velocity = m_cell_velocity_n(icells);
        double normvpvg = (m_particle_velocity_n(ipart)[0] - cell_velocity[0]) *
                              (m_particle_velocity_n(ipart)[0] - cell_velocity[0]) +
                          (m_particle_velocity_n(ipart)[1] - cell_velocity[1]) *
                              (m_particle_velocity_n(ipart)[1] - cell_velocity[1]);
        m_particle_reynolds(ipart) = max(2. * m_density_n(icells) * m_particle_radius(ipart) *
                                MathFunctions::sqrt(normvpvg) / viscosity,
                            particules->Reynolds_min);
        if (particules->DragModel == particules->Kliatchko) {
          if (m_particle_reynolds(ipart) > 1000)
            m_particle_drag(ipart) = 0.02 * pow(m_cell_particle_volume_fraction(icells), (-2.55)) +
                            0.4 * pow(m_cell_particle_volume_fraction(icells), (-1.78));
          else
            m_particle_drag(ipart) =
                24. / (m_particle_reynolds(ipart) * pow(m_cell_particle_volume_fraction(icells), (2.65))) +
                4. / (pow(m_particle_reynolds(ipart), (1. / 3.)) *
                      pow(m_cell_particle_volume_fraction(icells), (1.78)));
        } else {
          double fMac_part;
          if (m_particle_mac(ipart) < 0.5)
            fMac_part = 1.;
          else if (m_particle_mac(ipart) >= 0.5 && m_particle_mac(ipart) < 1.0)
            fMac_part = 2.44 * m_particle_mac(ipart) - 0.2222;
          else if (m_particle_mac(ipart) >= 1.0)
            fMac_part = 0.2222;
          m_particle_drag(ipart) =
              fMac_part *
              pow((24. + 4. * min(m_particle_reynolds(ipart), particules->Reynolds_max)),
                  2.13) /
              (min(m_particle_reynolds(ipart), particules->Reynolds_min));
        }
        // std::cout << " Part  " << ipart << " cd " << m_particle_drag(ipart)  << "  Re
        // " << m_particle_reynolds(ipart) << " " << m_cell_particle_volume_fraction(icells) << std::endl;
      });
}

void Eucclhyd::updateParticleVelocity() noexcept {
  Kokkos::parallel_for("initPart", nbPart, KOKKOS_LAMBDA(const int& ipart) {
    int icells = m_particle_cell(ipart);
    m_particle_velocity_nplus1(ipart) =
        m_particle_velocity_n(ipart) - m_particle_pressure_gradient(icells) * gt->deltat_n / m_particle_density(ipart);

    // std::cout << " Part  " << ipart << " vit " << m_particle_velocity_nplus1(ipart)  <<
    // std::endl;

    RealArray1D<dim> cell_velocity = m_cell_velocity_n(icells);
    double normvpvg =
        (m_particle_velocity_n(ipart)[0] - cell_velocity[0]) *
            (m_particle_velocity_n(ipart)[0] - cell_velocity[0]) +
        (m_particle_velocity_n(ipart)[1] - cell_velocity[1]) * (m_particle_velocity_n(ipart)[1] - cell_velocity[1]);
    double drag = (3 * m_density_n(icells) * m_particle_drag(ipart)) /
                  (8 * m_particle_density(ipart) * m_particle_radius(ipart)) *
                  MathFunctions::sqrt(normvpvg);
    // drag = 2000.;

    // m_particle_velocity_nplus1(ipart) = ArrayOperations::minus(m_particle_velocity_nplus1(ipart),
    // ArrayOperations::multiply(options->Drag,
    // (ArrayOperations::minus(m_particle_velocity_n(ipart) - cell_velocity))));
    m_particle_velocity_nplus1(ipart)[0] =
        m_particle_velocity_nplus1(ipart)[0] -
        drag * (m_particle_velocity_n(ipart)[0] - cell_velocity[0]) * gt->deltat_n;
    m_particle_velocity_nplus1(ipart)[1] =
        m_particle_velocity_nplus1(ipart)[1] -
        drag * (m_particle_velocity_n(ipart)[1] - cell_velocity[1]) * gt->deltat_n;
    // if (abs(m_particle_velocity_nplus1(ipart)[0]) > options->threshold) std::cout << "
    // Part  " << ipart << " vit " << m_particle_velocity_nplus1(ipart)  << std::endl;
  });
}

void Eucclhyd::updateParticleRetroaction() noexcept {
  Kokkos::parallel_for("initPart", nbPart, KOKKOS_LAMBDA(const int& ipart) {
    int icells = m_particle_cell(ipart);
    RealArray1D<dim> AccelerationP = m_particle_velocity_n(ipart) - m_particle_velocity_nplus1(ipart);
    m_cell_velocity_nplus1(icells)[0] += m_particle_mass(ipart) * AccelerationP[0] / m_cell_mass(icells);
    m_cell_velocity_nplus1(icells)[1] += m_particle_mass(ipart) * AccelerationP[1] / m_cell_mass(icells);
  });
}

void Eucclhyd::switchalpham_density_rho() noexcept {
  Kokkos::parallel_for("updateParticleCoefficient", nbCells,
                       KOKKOS_LAMBDA(const int& cCells) {
                         m_density_n(cCells) /= m_cell_particle_volume_fraction(cCells);
                         for (int imat = 0; imat < nbmatmax; imat++)
                           m_density_env_n(cCells)[imat] /= m_cell_particle_volume_fraction(cCells);
                       });
}

void Eucclhyd::switchm_density_alpharho() noexcept {
  Kokkos::parallel_for("updateParticleCoefficient", nbCells,
                       KOKKOS_LAMBDA(const int& cCells) {
                         m_density_n(cCells) *= m_cell_particle_volume_fraction(cCells);
                         for (int imat = 0; imat < nbmatmax; imat++)
                           m_density_env_n(cCells)[imat] *= m_cell_particle_volume_fraction(cCells);
                       });
}
