#include "Vnr.h"

using namespace nablalib;

#include "../includes/Freefunctions.h"
#include "types/MathFunctions.h"  // for max, min, dot, matVectProduct
#include "utils/Utils.h"          // for Indexof

/**
 * Job computeFacedeltaxLagrange called @7.0 in executeTimeLoopN method.
 * In variables: XcLagrange, faceNormal
 * Out variables: deltaxLagrange
 */
void Vnr::computeFaceQuantitesForRemap() noexcept {
  auto Innerfaces(mesh->getInnerFaces());
  int nbInnerFaces(mesh->getNbInnerFaces());
  // auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "computeFacedeltaxLagrange", nbInnerFaces,
      KOKKOS_LAMBDA(const int& fFaces) {
        size_t fId(Innerfaces[fFaces]);
        int cfFrontCellF(mesh->getFrontCell(fId));
        int cfId(cfFrontCellF);
        int cfCells(cfId);
        int cbBackCellF(mesh->getBackCell(fId));
        int cbId(cbBackCellF);
        int cbCells(cbId);
        varlp->deltaxLagrange(fId) =
            dot((varlp->XcLagrange(cfCells) - varlp->XcLagrange(cbCells)),
                varlp->faceNormal(fId));
      });
  auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "computeFaceQuantities", nbFaces, KOKKOS_LAMBDA(const int& fFaces) {
        int fId(faces[fFaces]);
        int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
        int n1Id(n1FirstNodeOfFaceF);
        int n1Nodes(n1Id);
        int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
        int n2Id(n2SecondNodeOfFaceF);
        int n2Nodes(n2Id);
        RealArray1D<dim> X_face =
            0.5 * (varlp->XLagrange(n1Nodes) + varlp->XLagrange(n2Nodes));
        RealArray1D<dim> face_vec =
            varlp->XLagrange(n2Nodes) - varlp->XLagrange(n1Nodes);

        varlp->XfLagrange(fFaces) = X_face;
        varlp->faceLengthLagrange(fFaces) = MathFunctions::norm(face_vec);

        RealArray1D<dim> reduction5 = zeroVect;
        {
          auto nodesOfFaceF(mesh->getNodesOfFace(fId));
          for (int pNodesOfFaceF = 0; pNodesOfFaceF < nodesOfFaceF.size();
               pNodesOfFaceF++) {
            int pId(nodesOfFaceF[pNodesOfFaceF]);
            int pNodes(pId);
            reduction5 = reduction5 + (m_node_velocity_nplus1(pNodes));
          }
        }

        varlp->faceNormalVelocity(fFaces) =
            dot((0.5 * reduction5), varlp->faceNormal(fFaces));
      });
}
void Vnr::computeCellQuantitesForRemap() noexcept {
  Kokkos::parallel_for(
      "computeLagrangeVolumeAndCenterOfGravity", nbCells,
      KOKKOS_LAMBDA(const int& cCells) {
        int cId(cCells);
        RealArray1D<dim> reduction7 = zeroVect;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            reduction7 =
                reduction7 +
                (crossProduct2d(varlp->XLagrange(pNodes),
                                varlp->XLagrange(pPlus1Nodes)) *
                 (varlp->XLagrange(pNodes) + varlp->XLagrange(pPlus1Nodes)));
          }
        }
        varlp->XcLagrange(cCells) =
            (1.0 / (6.0 * varlp->vLagrange(cCells)) * reduction7);
      });
}
/**
 * Job computeVariablesForRemap called in executeTimeLoopN method.
 * In variables: m_internal_energy_nplus1, m_density_nplus1,
 * Out variables: c, p
 */
void Vnr::computeVariablesForRemap() noexcept {
  Kokkos::parallel_for(
      "computeVariablesForRemap", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        int nbmat = options->nbmat;
        for (int imat = 0; imat < nbmat; imat++) {
          // volumes matériels (partiels)
          varlp->ULagrange(cCells)[imat] =
              m_fracvol_env(cCells)[imat] * varlp->vLagrange(cCells);

          // masses matériels (partiels)
          varlp->ULagrange(cCells)[nbmat + imat] =
              m_mass_fraction_env(cCells)[imat] * varlp->vLagrange(cCells) *
              m_density_nplus1(cCells);

          // energies matériels (partiels)
          varlp->ULagrange(cCells)[2 * nbmat + imat] =
              m_mass_fraction_env(cCells)[imat] * varlp->vLagrange(cCells) *
              m_density_nplus1(cCells) * m_internal_energy_nplus1(cCells);
        }
        // les vitesses et énergies cinétiques am_x_velocity cCells
        // correspondent à  varlp->ULagrange(cCells)[3 * nbmat + 0] pour VitX à
        // varlp->ULagrange(cCells)[3 * nbmat + 1] pour VitY à
        // varlp->ULagrange(cCells)[3 * nbmat + 2] pour Ec n'existe pas en VnR
        // --> tableau vide Ils sont am_x_velocity noeuds voir à la fin de la
        // routine
        varlp->ULagrange(cCells)[3 * nbmat + 0] = 0.;
        varlp->ULagrange(cCells)[3 * nbmat + 1] = 0.;
        varlp->ULagrange(cCells)[3 * nbmat + 2] = 0.;

        // Pseudo moyenne pour la quantité de mouvement
        varlp->ULagrange(cCells)[3 * nbmat + 3] =
            m_pseudo_viscosity_nplus1(cCells) * varlp->vLagrange(cCells);

        if (limiteurs->projectionAvecPlateauPente == 1) {
          // option ou on ne regarde pas la variation de rho, V et e
          // phi = (f1, f2, ..., rho1, rho2, ... e1, e2, ... Vx, Vy,
          // ce qui permet d'ecrire le flm_x_velocity telque
          // Flm_x_velocity = (dv1 = f1dv, dv2=f2*dv, dm1=rho1*df1,
          // dm2=rho2*df2, d(m1e1) = e1*dm1,  d(m2e2) = e2*dm2 d(mV) =
          // V*(dm1+dm2) dans computeFlm_x_velocityPP

          double somme_volume = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            somme_volume += varlp->ULagrange(cCells)[imat];
          }
          // Phi volume
          double somme_masse = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            varlp->Phi(cCells)[imat] =
                varlp->ULagrange(cCells)[imat] / somme_volume;

            // Phi masse
            if (varlp->ULagrange(cCells)[imat] != 0.)
              varlp->Phi(cCells)[nbmat + imat] =
                  varlp->ULagrange(cCells)[nbmat + imat] /
                  varlp->ULagrange(cCells)[imat];
            else
              varlp->Phi(cCells)[nbmat + imat] = 0.;
            somme_masse += varlp->ULagrange(cCells)[nbmat + imat];
          }
          // Phi energie
          for (int imat = 0; imat < nbmat; imat++) {
            if (varlp->ULagrange(cCells)[nbmat + imat] != 0.)
              varlp->Phi(cCells)[2 * nbmat + imat] =
                  varlp->ULagrange(cCells)[2 * nbmat + imat] /
                  varlp->ULagrange(cCells)[nbmat + imat];
            else
              varlp->Phi(cCells)[2 * nbmat + imat] = 0.;
          }
          // les phi sur la vitesse et energie cinétique n'existent pas en VnR
          varlp->Phi(cCells)[3 * nbmat + 0] = 0.;
          varlp->Phi(cCells)[3 * nbmat + 1] = 0.;
          varlp->Phi(cCells)[3 * nbmat + 2] = 0.;
          // Phi Pseudo
          varlp->Phi(cCells)[3 * nbmat + 3] = m_pseudo_viscosity_nplus1(cCells);
        } else {
          varlp->Phi(cCells) =
              varlp->ULagrange(cCells) / varlp->vLagrange(cCells);
        }
        varlp->rLagrange(cCells) = m_density_nplus1(cCells);
      });
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    // Position fin de phase Lagrange
    varlp->XLagrange(pNodes) = m_node_coord_nplus1(pNodes);
    // quantité de mouvement
    varlp->UDualLagrange(pNodes)[0] =
        m_node_mass(pNodes) * m_node_velocity_nplus1(pNodes)[0];
    varlp->UDualLagrange(pNodes)[1] =
        m_node_mass(pNodes) * m_node_velocity_nplus1(pNodes)[1];
    // masse nodale
    varlp->UDualLagrange(pNodes)[2] = m_node_mass(pNodes);
    // projection de l'energie cinétique
    if (options->projectionConservative == 1)
      varlp->UDualLagrange(pNodes)[3] =
          m_node_mass(pNodes) * (m_node_velocity_nplus1(pNodes)[0] *
                                     m_node_velocity_nplus1(pNodes)[0] +
                                 m_node_velocity_nplus1(pNodes)[1] *
                                     m_node_velocity_nplus1(pNodes)[1]);

    varlp->DualPhi(pNodes)[0] = m_node_velocity_nplus1(pNodes)[0];
    varlp->DualPhi(pNodes)[1] = m_node_velocity_nplus1(pNodes)[1];
    // masse nodale
    varlp->DualPhi(pNodes)[2] = m_node_mass(pNodes);
    // Phi energie cinétique
    if (options->projectionConservative == 1)
      varlp->DualPhi(pNodes)[3] = (m_node_velocity_nplus1(pNodes)[0] *
                                       m_node_velocity_nplus1(pNodes)[0] +
                                   m_node_velocity_nplus1(pNodes)[1] *
                                       m_node_velocity_nplus1(pNodes)[1]);

    // if ((pNodes == 600) || (pNodes == 601) || (pNodes == 602))
    //    std::cout << " pNodes " <<  pNodes << " sortie Lagrange "
    //  	    << varlp->UDualLagrange(pNodes)[0]
    //  	    << "  " << varlp->UDualLagrange(pNodes)[1]
    // 	     << "  " << varlp->UDualLagrange(pNodes)[2]
    // 	     << " vit " << m_node_velocity_nplus1(pNodes) << std::endl;
  });
}
