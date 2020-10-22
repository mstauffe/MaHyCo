#include <stdlib.h>  // for exit

#include <Kokkos_Core.hpp>  // for KOKKOS_LAMBDA
#include <algorithm>        // for equal, copy
#include <array>            // for array, operator!=
#include <iostream>         // for operator<<, basic_ostream::ope...
#include <limits>           // for numeric_limits
#include <vector>           // for vector, allocator

#include "Eucclhyd.h"          // for Eucclhyd, Eucclhyd::...
#include "Utiles-Impl.h"            // for Eucclhyd::tensProduct
#include "mesh/CartesianMesh2D.h"   // for CartesianMesh2D
#include "types/MathFunctions.h"    // for max, min, dot, matVectProduct
#include "types/MultiArray.h"       // for operator<<
#include "utils/Utils.h"            // for indexOf

#include <thread>

#include "../includes/VariablesLagRemap.h"
using namespace variableslagremaplib;

/**
 * Job computeCornerNormal called @1.0 in simulate method.
 * In variables: X
 * Out variables: lminus, lpc_n, lplus, nminus, nplus
 */
void Eucclhyd::computeCornerNormal() noexcept {
  Kokkos::parallel_for(
      "computeCornerNormal", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        size_t cId(cCells);
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pMinus1Id(nodesOfCellC[(pNodesOfCellC - 1 + nbNodesOfCell) %
                                       nbNodesOfCell]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
            int pMinus1Nodes(pMinus1Id);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            RealArray1D<dim> xp = X(pNodes);
            RealArray1D<dim> xpPlus = X(pPlus1Nodes);
            RealArray1D<dim> xpMinus = X(pMinus1Nodes);
            RealArray1D<dim> npc_plus;
            npc_plus[0] = 0.5 * (xpPlus[1] - xp[1]);
            npc_plus[1] = 0.5 * (xp[0] - xpPlus[0]);
            double lpc_plus = MathFunctions::norm(npc_plus);
            npc_plus = npc_plus / lpc_plus;
            nplus(pNodes, cCellsOfNodeP) = npc_plus;
            lplus(pNodes, cCellsOfNodeP) = lpc_plus;
            RealArray1D<dim> npc_minus;
            npc_minus[0] = 0.5 * (xp[1] - xpMinus[1]);
            npc_minus[1] = 0.5 * (xpMinus[0] - xp[0]);
            double lpc_minus = MathFunctions::norm(npc_minus);
            npc_minus = npc_minus / lpc_minus;
            nminus(pNodes, cCellsOfNodeP) = npc_minus;
            lminus(pNodes, cCellsOfNodeP) = lpc_minus;
            lpc_n(pNodes, cCellsOfNodeP) = (lpc_plus * npc_plus) + (lpc_minus * npc_minus);
          }
        }
      });
}
/**
 * Job computeEOS called in executeTimeLoopN method.
 */
void Eucclhyd::computeEOS() {
  for (int imat = 0; imat < options->nbmat; ++imat) {
    if (eos->Nom[imat] == eos->PerfectGas) computeEOSGP(imat);
    if (eos->Nom[imat] == eos->Void) computeEOSVoid(imat);
    if (eos->Nom[imat] == eos->StiffenedGas) computeEOSSTIFG(imat);
    if (eos->Nom[imat] == eos->Murnhagan) computeEOSMur(imat);
    if (eos->Nom[imat] == eos->SolidLinear) computeEOSSL(imat);
  }
}
/**
 * Job computeEOSGP called @1.0 in executeTimeLoopN method.
 * In variables: eos, eosPerfectGas, e_n, gammap, rho_n
 * Out variables: c, p
 */
void Eucclhyd::computeEOSGP(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    pp(cCells)[imat] =
        (eos->gammap[imat] - 1.0) * rhop_n(cCells)[imat] * ep_n(cCells)[imat];
    if (rhop_n(cCells)[imat] > 0.) {
      vitsonp(cCells)[imat] = MathFunctions::sqrt(
          eos->gammap[imat] * pp(cCells)[imat] / rhop_n(cCells)[imat]);
    } else
      vitsonp(cCells)[imat] = 1.e-20;
  });
}
/**
 * Job computeEOSVoid called in executeTimeLoopN method.
 * In variables: eos, eosPerfectGas, e_n, gammap, rho_n
 * Out variables: c, p
 */
void Eucclhyd::computeEOSVoid(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    pp(cCells)[imat] = 0.;
    vitsonp(cCells)[imat] = 1.e-20;
  });
}
/**
 * Job computeEOSSTIFG
 * In variables: e_n, rho_n
 * Out variables: c, p
 */
void Eucclhyd::computeEOSSTIFG(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    std::cout << " Pas encore programmée" << std::endl;
  });
}
/**
 * Job computeEOSMur called @1.0 in executeTimeLoopN method.
 * In variables: e_n, rho_n
 * Out variables: c, p
 */
void Eucclhyd::computeEOSMur(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    std::cout << " Pas encore programmée" << std::endl;
  });
}
/**
 * Job computeEOSSL called @1.0 in executeTimeLoopN method.
 * In variables: e_n, rho_n
 * Out variables: c, p
 */
void Eucclhyd::computeEOSSL(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    std::cout << " Pas encore programmée" << std::endl;
  });
}
/**
 * Job computeEOS called in executeTimeLoopN method.
 */
void Eucclhyd::computePressionMoyenne() noexcept {
  for (int cCells = 0; cCells < nbCells; cCells++) {
    int nbmat = options->nbmat;
    p(cCells) = 0.;
    for (int imat = 0; imat < nbmat; ++imat) {
      p(cCells) += fracvol(cCells)[imat] * pp(cCells)[imat];
      vitson(cCells) =
          MathFunctions::max(vitson(cCells), vitsonp(cCells)[imat]);
    }
    // NONREG GP A SUPPRIMER
    if (rho_n(cCells) > 0.) {
      vitson(cCells) =
          MathFunctions::sqrt(eos->gammap[0] * p(cCells) / rho_n(cCells));
    }
  }
}
/**
 * Job computeGradients called @1.0 in executeTimeLoopN method.
 * In variables: F_n, Vnode_n, lpc_n, spaceOrder, v
 * Out variables: gradV, gradp, gradp1, gradp2, gradp3
 */
void Eucclhyd::computeGradients() noexcept {
  Kokkos::parallel_for(
      "computeDissipationMatrix", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
        int pId(pNodes);
        {
          int nbmat = options->nbmat;
          for (int imat = 0; imat < nbmat; imat++)
            fracvolnode(pNodes)[imat] = 0.;
          auto cellsOfNodeP(mesh->getCellsOfNode(pId));
          for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
               cCellsOfNodeP++) {
            int cId(cellsOfNodeP[cCellsOfNodeP]);
            int cCells(cId);
            for (int imat = 0; imat < nbmat; imat++)
              fracvolnode(pNodes)[imat] += fracvol(cCells)[imat] * 0.25;
          }
        }
      });
  Kokkos::parallel_for(
      "computeGradients", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        size_t cId(cCells);
        RealArray1D<dim> reductionF1 = zeroVect;
        RealArray1D<dim> reductionF2 = zeroVect;
        RealArray1D<dim> reductionF3 = zeroVect;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
            int pNodes(pId);
            reductionF1 = 
                reductionF1 +(fracvolnode(pNodes)[0] * lpc_n(pNodes, cCellsOfNodeP));
            reductionF2 = 
                reductionF2 +(fracvolnode(pNodes)[1] * lpc_n(pNodes, cCellsOfNodeP));
            reductionF3 = 
                reductionF3 +(fracvolnode(pNodes)[2] * lpc_n(pNodes, cCellsOfNodeP));
          }
        }
        gradf1(cCells) = reductionF1 / volE(cCells);
        gradf2(cCells) = reductionF2 / volE(cCells);
        gradf3(cCells) = reductionF3 / volE(cCells);
      });
  if (options->spaceOrder == 2)
    Kokkos::parallel_for(
        "computeGradients", nbCells, KOKKOS_LAMBDA(const int& cCells) {
          size_t cId(cCells);
          RealArray2D<dim, dim> reduction14 = zeroMat;
          {
            auto nodesOfCellC(mesh->getNodesOfCell(cId));
            for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
                 pNodesOfCellC++) {
              int pId(nodesOfCellC[pNodesOfCellC]);
              int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
              int pNodes(pId);
              reduction14 = 
                  reduction14 +
                  (tensProduct(Vnode_n(pNodes), lpc_n(pNodes, cCellsOfNodeP)));
            }
          }
          gradV(cCells) = reduction14 / volE(cCells);
          RealArray1D<dim> reduction15 = zeroVect;
          RealArray1D<dim> reduction15a = zeroVect;
          RealArray1D<dim> reduction15b = zeroVect;
          RealArray1D<dim> reduction15c = zeroVect;
          {
            auto nodesOfCellC(mesh->getNodesOfCell(cId));
            for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
                 pNodesOfCellC++) {
              int pId(nodesOfCellC[pNodesOfCellC]);
              int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
              int pNodes(pId);
              reduction15 = reduction15 + F_n(pNodes, cCellsOfNodeP);

              reduction15a = reduction15a + F1_n(pNodes, cCellsOfNodeP);
              reduction15b = reduction15b + F2_n(pNodes, cCellsOfNodeP);

              reduction15c = reduction15c + F3_n(pNodes, cCellsOfNodeP);
            }
          }
          gradp(cCells)  = reduction15  / volE(cCells);
          gradp1(cCells) = reduction15a / volE(cCells);
          gradp2(cCells) = reduction15b / volE(cCells);
          gradp3(cCells) = reduction15c / volE(cCells);
        });
}

/**
 * Job computeMass called @1.0 in executeTimeLoopN method.
 * In variables: rho_n, v
 * Out variables: m
 */
void Eucclhyd::computeMass() noexcept {
  int nbmat = options->nbmat;
  Kokkos::parallel_for(
      "computeMass", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        int nbmat = options->nbmat;
        m(cCells) = rho_n(cCells) * volE(cCells);
        for (int imat = 0; imat < nbmat; imat++)
          mp(cCells)[imat] = fracmass(cCells)[imat] * m(cCells);
      });
}
/**
 * Job computeDissipationMatrix called @2.0 in executeTimeLoopN method.
 * In variables: c, lminus, lplus, nminus, nplus, rho_n
 * Out variables: M
 */
void Eucclhyd::computeDissipationMatrix() noexcept {
  Kokkos::parallel_for(
      "computeDissipationMatrix", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
        int pId(pNodes);
        {
          auto cellsOfNodeP(mesh->getCellsOfNode(pId));
          for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
               cCellsOfNodeP++) {
            int cId(cellsOfNodeP[cCellsOfNodeP]);
            int cCells(cId);
            RealArray2D<dim, dim> cornerMatrix =          
	      (lplus(pNodes, cCellsOfNodeP) *
                    tensProduct(nplus(pNodes, cCellsOfNodeP),
                                nplus(pNodes, cCellsOfNodeP)))
	       +
              (lminus(pNodes, cCellsOfNodeP) *
                    tensProduct(nminus(pNodes, cCellsOfNodeP),
                                nminus(pNodes, cCellsOfNodeP)));
	    
            M(pNodes, cCellsOfNodeP) = rho_n(cCells) * vitson(cCells) * cornerMatrix;

            M1(pNodes, cCellsOfNodeP) = rhop_n(cCells)[0] * vitson(cCells) * cornerMatrix;
            M2(pNodes, cCellsOfNodeP) = rhop_n(cCells)[1] * vitson(cCells) * cornerMatrix;
            M3(pNodes, cCellsOfNodeP) = rhop_n(cCells)[2] * vitson(cCells) * cornerMatrix;
          }
        }
      });
}
/**
 * Job computedeltatc called @2.0 in executeTimeLoopN method.
 * In variables: V_n, c, perim, v
 * Out variables: deltatc
 */
void Eucclhyd::computedeltatc() noexcept {
  Kokkos::parallel_for(
      "computedeltatc", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        if (options->AvecProjection == 1) {
          // cfl euler
          deltatc(cCells) =
              volE(cCells) / (perim(cCells) *
                           (MathFunctions::norm(V_n(cCells)) + vitson(cCells)));
        } else {
          // cfl lagrange
          deltatc(cCells) = volE(cCells) / (perim(cCells) * vitson(cCells));
        }
      });
}
/**
 * Job extrapolateValue called @2.0 in executeTimeLoopN method.
 * In variables: V_n, X, Xc, gradV, gradp, p, spaceOrder
 * Out variables: V_extrap, p_extrap, pp_extrap
 */
void Eucclhyd::extrapolateValue() noexcept {
  if (options->spaceOrder == 1) {
    Kokkos::parallel_for(
        "extrapolateValue", nbCells, KOKKOS_LAMBDA(const int& cCells) {
          int cId(cCells);
          {
            auto nodesOfCellC(mesh->getNodesOfCell(cId));
            for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
                 pNodesOfCellC++) {
              V_extrap(cCells, pNodesOfCellC) = V_n(cCells);
              p_extrap(cCells, pNodesOfCellC) = p(cCells);
              int nbmat = options->nbmat;
              for (int imat = 0; imat < nbmat; imat++)
                pp_extrap(cCells, pNodesOfCellC)[imat] = pp(cCells)[imat];
            }
          }
        });
  } else {
    Kokkos::parallel_for(
        "extrapolateValue", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
          size_t pId(pNodes);
          double reduction16 = numeric_limits<double>::max();
          double reduction16a = numeric_limits<double>::max();
          double reduction16b = numeric_limits<double>::max();
          double reduction16c = numeric_limits<double>::max();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction16 = MathFunctions::min(reduction16, p(dCells));
              reduction16a = MathFunctions::min(reduction16a, pp(dCells)[0]);
              reduction16b = MathFunctions::min(reduction16b, pp(dCells)[1]);
              reduction16c = MathFunctions::min(reduction16c, pp(dCells)[2]);
            }
          }
          double minP = reduction16;
          double minP1 = reduction16a;
          double minP2 = reduction16b;
          double minP3 = reduction16c;
          double reduction17 = numeric_limits<double>::min();
          double reduction17a = numeric_limits<double>::min();
          double reduction17b = numeric_limits<double>::min();
          double reduction17c = numeric_limits<double>::min();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction17 = MathFunctions::max(reduction17, p(dCells));
              reduction17a = MathFunctions::max(reduction17a, pp(dCells)[0]);
              reduction17b = MathFunctions::max(reduction17b, pp(dCells)[1]);
              reduction17c = MathFunctions::max(reduction17c, pp(dCells)[2]);
            }
          }
          double maxP = reduction17;
          double maxP1 = reduction17a;
          double maxP2 = reduction17b;
          double maxP3 = reduction17c;
          double reduction18 = numeric_limits<double>::max();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction18 = MathFunctions::min(reduction18, V_n(dCells)[0]);
            }
          }
          double minVx = reduction18;
          double reduction19 = numeric_limits<double>::min();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction19 = MathFunctions::max(reduction19, V_n(dCells)[0]);
            }
          }
          double maxVx = reduction19;
          double reduction20 = numeric_limits<double>::max();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction20 = MathFunctions::min(reduction20, V_n(dCells)[1]);
            }
          }
          double minVy = reduction20;
          double reduction21 = numeric_limits<double>::min();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction21 = MathFunctions::max(reduction21, V_n(dCells)[1]);
            }
          }
          double maxVy = reduction21;
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
                 cCellsOfNodeP++) {
              int cId(cellsOfNodeP[cCellsOfNodeP]);
              int cCells(cId);
              int pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));

              // double ptmp = p(cCells) + MathFunctions::dot(gradp(cCells),
              // ArrayOperations::minus(X(pNodes), Xc(cCells)));
              // p_extrap(cCells,pNodesOfCellC) =
              // MathFunctions::max(MathFunctions::min(maxP, ptmp), minP);

              // pour chaque matériau,
              double ptmp1 = pp(cCells)[0] +
                             dot(gradp1(cCells),
				 (X(pNodes)- Xc(cCells)));
              pp_extrap(cCells, pNodesOfCellC)[0] =
		  MathFunctions::max(MathFunctions::min(maxP1, ptmp1), minP1);
              double ptmp2 = pp(cCells)[1] +
		             dot(gradp2(cCells),
                                 (X(pNodes) - Xc(cCells)));
              pp_extrap(cCells, pNodesOfCellC)[1] =
                  MathFunctions::max(MathFunctions::min(maxP2, ptmp2), minP2);
              double ptmp3 = pp(cCells)[2] +
                             dot(gradp3(cCells),
                                 (X(pNodes) - Xc(cCells)));
              pp_extrap(cCells, pNodesOfCellC)[2] =
                  MathFunctions::max(MathFunctions::min(maxP3, ptmp3), minP3);

              p_extrap(cCells, pNodesOfCellC) = 0.;
              int nbmat = options->nbmat;
              // et on recalcule la moyenne
              for (int imat = 0; imat < nbmat; imat++)
                p_extrap(cCells, pNodesOfCellC) +=
                    fracvol(cCells)[imat] *
                    pp_extrap(cCells, pNodesOfCellC)[imat];

              RealArray1D<dim> Vtmp = 
                  V_n(cCells) +
		MathFunctions::matVectProduct(gradV(cCells), (X(pNodes) - Xc(cCells)));
              V_extrap(cCells, pNodesOfCellC)[0] =
                  std::max(MathFunctions::min(maxVx, Vtmp[0]), minVx);
              V_extrap(cCells, pNodesOfCellC)[1] =
                  std::max(MathFunctions::min(maxVy, Vtmp[1]), minVy);
            }
          }
        });
  }
}
/**
 * Job computeG called @3.0 in executeTimeLoopN method.
 * In variables: M, V_extrap, lpc_n, p_extrap
 * Out variables: G
 */
void Eucclhyd::computeG() noexcept {
  Kokkos::parallel_for("computeG", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
    size_t pId(pNodes);
    RealArray1D<dim> reduction1 = zeroVect;
    {
      auto cellsOfNodeP(mesh->getCellsOfNode(pId));
      for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
           cCellsOfNodeP++) {
        int cId(cellsOfNodeP[cCellsOfNodeP]);
        int cCells(cId);
        int pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
        reduction1 = 
            reduction1 +
            (MathFunctions::matVectProduct(M(pNodes, cCellsOfNodeP),
                                              V_extrap(cCells, pNodesOfCellC))
		+
	     (p_extrap(cCells, pNodesOfCellC) * lpc_n(pNodes, cCellsOfNodeP)));
      }
    }
    G(pNodes) = reduction1;
  });
}

/**
 * Job computeNodeDissipationMatrixAndG called @3.0 in executeTimeLoopN method.
 * In variables: M
 * Out variables: Mnode
 */
void Eucclhyd::computeNodeDissipationMatrixAndG() noexcept {
  Kokkos::parallel_for(
      "computeNodeDissipationMatrixAndG", nbNodes,
      KOKKOS_LAMBDA(const int& pNodes) {
        int pId(pNodes);
        RealArray2D<dim, dim> reduction0 = zeroMat;
        {
          auto cellsOfNodeP(mesh->getCellsOfNode(pId));
          for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
               cCellsOfNodeP++) {
            reduction0 = reduction0 + (M(pNodes, cCellsOfNodeP));
          }
        }
        Mnode(pNodes) = reduction0;
      });
}
/**
 * Job computeNodeVelocity called @4.0 in executeTimeLoopN method.
 * In variables: G, Mnode
 * Out variables: Vnode_nplus1
 */
void Eucclhyd::computeNodeVelocity() noexcept {
  auto innerNodes(mesh->getInnerNodes());
  int nbInnerNodes(mesh->getNbInnerNodes());
  Kokkos::parallel_for("computeNodeVelocity", nbInnerNodes,
                       KOKKOS_LAMBDA(const int& pInnerNodes) {
                         int pId(innerNodes[pInnerNodes]);
                         int pNodes(pId);
                         Vnode_nplus1(pNodes) = MathFunctions::matVectProduct(
                             inverse(Mnode(pNodes)), G(pNodes));
                       });
}
/**
 * Job computeFaceVelocity called @5.0 in executeTimeLoopN method.
 * In variables: Vnode_nplus1, faceNormal
 * Out variables: faceNormalVelocity
 */
void Eucclhyd::computeFaceVelocity() noexcept {
  auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "computeFaceVelocity", nbFaces, KOKKOS_LAMBDA(const int& fFaces) {
        int fId(faces[fFaces]);
        RealArray1D<dim> reduction5 = zeroVect;
        {
          auto nodesOfFaceF(mesh->getNodesOfFace(fId));
          for (int pNodesOfFaceF = 0; pNodesOfFaceF < nodesOfFaceF.size();
               pNodesOfFaceF++) {
            int pId(nodesOfFaceF[pNodesOfFaceF]);
            int pNodes(pId);
            reduction5 = reduction5 + (Vnode_nplus1(pNodes));
          }
        }
        varlp->faceNormalVelocity(fFaces) = dot((0.5 * reduction5), varlp->faceNormal(fFaces));
      });
}

/**
 * Job computeLagrangePosition called @5.0 in executeTimeLoopN method.
 * In variables: Vnode_nplus1, X, deltat_n
 * Out variables: XLagrange
 */
void Eucclhyd::computeLagrangePosition() noexcept {
  Kokkos::parallel_for(
      "computeLagrangePosition", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
        varlp->XLagrange(pNodes) = 
            X(pNodes) + Vnode_nplus1(pNodes) * gt->deltat_n;
      });
  auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "computeLagrangePosition", nbFaces, KOKKOS_LAMBDA(const int& fFaces) {
        int fId(faces[fFaces]);
        int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
        int n1Id(n1FirstNodeOfFaceF);
        int n1Nodes(n1Id);
        int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
        int n2Id(n2SecondNodeOfFaceF);
        int n2Nodes(n2Id);
        RealArray1D<dim> X_face = 0.5 * (varlp->XLagrange(n1Nodes) + varlp->XLagrange(n2Nodes));
        RealArray1D<dim> face_vec = varlp->XLagrange(n2Nodes) - varlp->XLagrange(n1Nodes);
        varlp->XfLagrange(fFaces) = X_face;
        varlp->faceLengthLagrange(fFaces) = MathFunctions::norm(face_vec);
      });
  if (options->AvecProjection == 0) {
    Kokkos::parallel_for(
        "computeLagrangePosition", nbCells, KOKKOS_LAMBDA(const int& cCells) {
          int cId(cCells);
          double reduction13 = 0.0;
          {
            auto nodesOfCellC(mesh->getNodesOfCell(cId));
            for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
                 pNodesOfCellC++) {
              int pId(nodesOfCellC[pNodesOfCellC]);
              int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                        nbNodesOfCell]);
              int pNodes(pId);
              int pPlus1Nodes(pPlus1Id);
              reduction13 =
		reduction13 + (MathFunctions::norm(X(pNodes) - X(pPlus1Nodes)));
            }
          }
          perim(cCells) = reduction13;
        });
  }
}

/**
 * Job computeSubCellForce called @5.0 in executeTimeLoopN method.
 * In variables: M, V_extrap, Vnode_nplus1, lpc_n, p_extrap
 * Out variables: F_nplus1
 */
void Eucclhyd::computeSubCellForce() noexcept {
  Kokkos::parallel_for(
      "computeSubCellForce", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
        size_t pId(pNodes);
        {
          auto cellsOfNodeP(mesh->getCellsOfNode(pId));
          for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
               cCellsOfNodeP++) {
            int cId(cellsOfNodeP[cCellsOfNodeP]);
            int cCells(cId);
            int pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
            F_nplus1(pNodes, cCellsOfNodeP) = 
                (-p_extrap(cCells, pNodesOfCellC) * lpc_n(pNodes, cCellsOfNodeP))
	      +
                MathFunctions::matVectProduct( M(pNodes, cCellsOfNodeP),
                              (Vnode_nplus1(pNodes) - V_extrap(cCells, pNodesOfCellC)));

            F1_nplus1(pNodes, cCellsOfNodeP) = 
                (-pp_extrap(cCells, pNodesOfCellC)[0] * lpc_n(pNodes, cCellsOfNodeP))
	      +
                MathFunctions::matVectProduct(
                    M1(pNodes, cCellsOfNodeP),
		    Vnode_nplus1(pNodes) - V_extrap(cCells, pNodesOfCellC));
	    
	    F2_nplus1(pNodes, cCellsOfNodeP) = 
                (-pp_extrap(cCells, pNodesOfCellC)[1] * lpc_n(pNodes, cCellsOfNodeP))
	      +
                MathFunctions::matVectProduct(
                    M2(pNodes, cCellsOfNodeP),
		    Vnode_nplus1(pNodes) - V_extrap(cCells, pNodesOfCellC));

	    F3_nplus1(pNodes, cCellsOfNodeP) = 
                (-pp_extrap(cCells, pNodesOfCellC)[2] * lpc_n(pNodes, cCellsOfNodeP))
	      +
                MathFunctions::matVectProduct(
                    M3(pNodes, cCellsOfNodeP),
		    Vnode_nplus1(pNodes) - V_extrap(cCells, pNodesOfCellC));
          }
        }
      });
}
/**
 * Job computeLagrangeVolumeAndCenterOfGravity called @6.0 in executeTimeLoopN
 * method. In variables: XLagrange Out variables: XcLagrange, vLagrange
 */
void Eucclhyd::computeLagrangeVolumeAndCenterOfGravity() noexcept {
  Kokkos::parallel_for(
      "computeLagrangeVolumeAndCenterOfGravity", nbCells,
      KOKKOS_LAMBDA(const int& cCells) {
        int cId(cCells);
        double reduction6 = 0.0;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            reduction6 = reduction6 + (crossProduct2d(varlp->XLagrange(pNodes),
                                                      varlp->XLagrange(pPlus1Nodes)));
          }
        }
        double vol = 0.5 * reduction6;
        varlp->vLagrange(cCells) = vol;
        int nbmat = options->nbmat;
        for (int imat = 0; imat < nbmat; imat++)
          vpLagrange(cCells)[imat] = fracvol(cCells)[imat] * vol;
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
        varlp->XcLagrange(cCells) = (1.0 / (6.0 * vol) * reduction7);
      });
}
/**
 * Job computeFacedeltaxLagrange called @7.0 in executeTimeLoopN method.
 * In variables: XcLagrange, faceNormal
 * Out variables: deltaxLagrange
 */
void Eucclhyd::computeFacedeltaxLagrange() noexcept {
  auto faces(mesh->getInnerFaces());
  int nbInnerFaces(mesh->getNbInnerFaces());
  //auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "computeFacedeltaxLagrange", nbInnerFaces, KOKKOS_LAMBDA(const int& fFaces) {
        size_t fId(faces[fFaces]);
        int cfFrontCellF(mesh->getFrontCell(fId));
        int cfId(cfFrontCellF);
        int cfCells(cfId);
        int cbBackCellF(mesh->getBackCell(fId));
        int cbId(cbBackCellF);
        int cbCells(cbId);
	varlp->deltaxLagrange(fId) = dot(
	 (varlp->XcLagrange(cfCells) - varlp->XcLagrange(cbCells)), varlp->faceNormal(fId));
      });
}

/**
 * Job updateCellCenteredLagrangeVariables called @7.0 in executeTimeLoopN
 * method. In variables: F_nplus1, V_n, Vnode_nplus1, deltat_n, e_n, lpc_n, m,
 * rho_n, vLagrange Out variables: ULagrange
 */
void Eucclhyd::updateCellCenteredLagrangeVariables() noexcept {
  Kokkos::parallel_for(
      "updateCellCenteredLagrangeVariables", nbCells,
      KOKKOS_LAMBDA(const int& cCells) {
        size_t cId(cCells);
        double reduction2 = 0.0;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
            int pNodes(pId);
            reduction2 =
                reduction2 + (dot(lpc_n(pNodes, cCellsOfNodeP),
                                                 Vnode_nplus1(pNodes)));
          }
        }
        double rhoLagrange =
            1 / (1 / rho_n(cCells) + gt->deltat_n / m(cCells) * reduction2);

        RealArray1D<dim> reduction3 = zeroVect;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
            int pNodes(pId);
            reduction3 = reduction3 + F_nplus1(pNodes, cCellsOfNodeP);
          }
        }
        ForceGradp(cCells) = reduction3 / volE(cCells);
        RealArray1D<dim> VLagrange =
            V_n(cCells) + reduction3 * gt->deltat_n / m(cCells);

        double reduction4 = 0.0;
        RealArray1D<nbmatmax> preduction4 = zeroVectmat;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
            int pNodes(pId);
            reduction4 =
                reduction4 + (dot(F_nplus1(pNodes, cCellsOfNodeP),
				  (Vnode_nplus1(pNodes) -
				   (0.5 * (V_n(cCells) + VLagrange)))));
            preduction4[0] =
                preduction4[0] + (dot(F1_nplus1(pNodes, cCellsOfNodeP),
				  (Vnode_nplus1(pNodes) -
				   (0.5 * (V_n(cCells) + VLagrange)))));
            preduction4[1] = 
                preduction4[1] + (dot(F2_nplus1(pNodes, cCellsOfNodeP),
				  (Vnode_nplus1(pNodes) -
				   (0.5 * (V_n(cCells) + VLagrange)))));
            preduction4[2] = 
                preduction4[2] + (dot(F3_nplus1(pNodes, cCellsOfNodeP),
				  (Vnode_nplus1(pNodes) -
				   (0.5 * (V_n(cCells) + VLagrange)))));
          }
        }

        int nbmat = options->nbmat;
        double eLagrange =
            e_n(cCells) + gt->deltat_n / m(cCells) * reduction4;

        RealArray1D<nbmatmax> peLagrange;
        RealArray1D<nbmatmax> peLagrangec;
        for (int imat = 0; imat < nbmat; imat++) {
          peLagrange[imat] = 0.;
          peLagrangec[imat] = 0.;
          if (fracvol(cCells)[imat] > options->threshold &&
              mp(cCells)[imat] != 0.)
            peLagrange[imat] =
                ep_n(cCells)[imat] + fracvol(cCells)[imat] * gt->deltat_n /
                                           mp(cCells)[imat] * preduction4[imat];
        }
        for (int imat = 0; imat < nbmat; imat++) {
          varlp->ULagrange(cCells)[imat] = vpLagrange(cCells)[imat];

          varlp->ULagrange(cCells)[nbmat + imat] =
              fracmass(cCells)[imat] * varlp->vLagrange(cCells) * rhoLagrange;

          varlp->ULagrange(cCells)[2 * nbmat + imat] =
              fracmass(cCells)[imat] * varlp->vLagrange(cCells) * rhoLagrange *
              peLagrange[imat];
        }

        varlp->ULagrange(cCells)[3 * nbmat] =
            varlp->vLagrange(cCells) * rhoLagrange * VLagrange[0];
        varlp->ULagrange(cCells)[3 * nbmat + 1] =
            varlp->vLagrange(cCells) * rhoLagrange * VLagrange[1];
        // projection de l'energie cinétique
        if (options->projectionConservative == 1)
          varlp->ULagrange(cCells)[3 * nbmat + 2] =
              0.5 * varlp->vLagrange(cCells) * rhoLagrange *
              (VLagrange[0] * VLagrange[0] + VLagrange[1] * VLagrange[1]);
	
        if (options->AvecProjection == 0) {
          // Calcul des valeurs en n+1 si on ne fait pas de projection
          // Vnode_nplus1
          V_nplus1(cCells) = VLagrange;
          // densites et energies
          rho_nplus1(cCells) = 0.;
          e_nplus1(cCells) = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            // densités
            rho_nplus1(cCells) += fracmass(cCells)[imat] * rhoLagrange;
            if (fracvol(cCells)[imat] > options->threshold) {
              rhop_nplus1(cCells)[imat] =
                  fracmass(cCells)[imat] * rhoLagrange / fracvol(cCells)[imat];
            } else {
              rhop_nplus1(cCells)[imat] = 0.;
            }
            // energies
            e_nplus1(cCells) += fracmass(cCells)[imat] * peLagrange[imat];
            if (fracvol(cCells)[imat] > options->threshold) {
              ep_nplus1(cCells)[imat] = peLagrange[imat];
            } else {
              ep_nplus1(cCells)[imat] = 0.;
            }
          }
          // variables pour les sorties du code
          fracvol1(cCells) = fracvol(cCells)[0];
          fracvol2(cCells) = fracvol(cCells)[1];
          fracvol3(cCells) = fracvol(cCells)[2];
          // sorties paraview limitées
          if (V_nplus1(cCells)[0] > 0.)
            Vxc(cCells) =
                MathFunctions::max(V_nplus1(cCells)[0], options->threshold);
          if (V_nplus1(cCells)[0] < 0.)
            Vxc(cCells) =
                MathFunctions::min(V_nplus1(cCells)[0], -options->threshold);

          if (V_nplus1(cCells)[1] > 0.)
            Vyc(cCells) =
                MathFunctions::max(V_nplus1(cCells)[1], options->threshold);
          if (V_nplus1(cCells)[1] < 0.)
            Vyc(cCells) =
                MathFunctions::min(V_nplus1(cCells)[1], -options->threshold);
          // pression
          p1(cCells) = pp(cCells)[0];
          p2(cCells) = pp(cCells)[1];
          p3(cCells) = pp(cCells)[2];
        }

        if (limiteurs->projectionAvecPlateauPente == 1) {
          // option ou on ne regarde pas la variation de rho, V et e
          // phi = (f1, f2, rho1, rho2,  e1, e2, Vx, Vy,
          // ce qui permet d'ecrire le flux telque
          // Flux = (dv1 = f1dv, dv2=f2*dv, dm1=rho1*df1, dm2=rho2*df2d(m1e1) = e1*dm1,  d(m2e2) = e2*dm2,
	  // d(mV) = V*(dm1+dm2), dans computeFluxPP

          double somme_volume = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            somme_volume += varlp->ULagrange(cCells)[imat];
          }
          // Phi volume
          double somme_masse = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            varlp->Phi(cCells)[imat] = varlp->ULagrange(cCells)[imat] / somme_volume;

            // Phi masse
            if (varlp->ULagrange(cCells)[imat] != 0.)
              varlp->Phi(cCells)[nbmat + imat] =
                  varlp->ULagrange(cCells)[nbmat + imat] /
		  varlp->ULagrange(cCells)[imat];
            else
              varlp->Phi(cCells)[nbmat + imat] = 0.;
            somme_masse += varlp->ULagrange(cCells)[nbmat + imat];
          }
          // Phi Vitesse
          varlp->Phi(cCells)[3 * nbmat] =
	    varlp->ULagrange(cCells)[3 * nbmat] / somme_masse;
          varlp->Phi(cCells)[3 * nbmat + 1] =
	    varlp->ULagrange(cCells)[3 * nbmat + 1] / somme_masse;
          // Phi energie
          for (int imat = 0; imat < nbmat; imat++) {
            if (varlp->ULagrange(cCells)[nbmat + imat] != 0.)
              varlp->Phi(cCells)[2 * nbmat + imat] =
                  varlp->ULagrange(cCells)[2 * nbmat + imat] /
                  varlp->ULagrange(cCells)[nbmat + imat];
            else
              varlp->Phi(cCells)[2 * nbmat + imat] = 0.;
          }
          // Phi energie cinétique
          if (options->projectionConservative == 1)
            varlp->Phi(cCells)[3 * nbmat + 2] =
                varlp->ULagrange(cCells)[3 * nbmat + 2] / somme_masse;

        } else {
          varlp->Phi(cCells) = varlp->ULagrange(cCells) /
	    varlp->vLagrange(cCells);
        }

#ifdef TEST 	
        if ((cCells == dbgcell3 || cCells == dbgcell2 || cCells == dbgcell1) &&
            test_debug == 1) {
          std::cout << " Apres Phase Lagrange cell   " << cCells << "Phi"
                    << varlp->Phi(cCells) << std::endl;
          std::cout << " cell   " << cCells << "varlp->ULagrange "
		    << varlp->ULagrange(cCells)
                    << std::endl;
	}
        if (varlp->ULagrange(cCells) != varlp->ULagrange(cCells)) {
          std::cout << " cell   " << cCells << " varlp->Ulagrange "
                    << varlp->ULagrange(cCells) << std::endl;
          std::cout << " cell   " << cCells << " f1 " << fracvol(cCells)[0]
                    << " f2 " << fracvol(cCells)[1] << " f3 "
                    << fracvol(cCells)[2] << std::endl;
          std::cout << " cell   " << cCells << " c1 " << fracmass(cCells)[0]
                    << " c2 " << fracmass(cCells)[1] << " c3 "
                    << fracmass(cCells)[2] << std::endl;
          std::cout << " cell   " << cCells << " m1 " << mp(cCells)[0] << " m2 "
                    << mp(cCells)[1] << " m3 " << mp(cCells)[2] << std::endl;
          std::cout << " cell   " << cCells << " peLagrange[0] "
                    << peLagrange[0] << " preduction4[0] " << preduction4[0]
                    << " ep_n[0] " << ep_n(cCells)[0] << std::endl;
          std::cout << " cell   " << cCells << " peLagrange[1] "
                    << peLagrange[1] << " preduction4[1] " << preduction4[1]
                    << " ep_n[1] " << ep_n(cCells)[1] << std::endl;
          std::cout << " cell   " << cCells << " peLagrange[2] "
                    << peLagrange[2] << " preduction4[2] " << preduction4[2]
                    << " ep_n[2] " << ep_n(cCells)[2] << std::endl;
          std::cout << " densites  1 " << rhop_n(cCells)[0] << " 2 "
                    << rhop_n(cCells)[1] << " 3 " << rhop_n(cCells)[2]
                    << std::endl;
          exit(1);
        }
#endif

        // ETOT_L(cCells) = (rhoLagrange * vLagrange(cCells)) * (eLagrange +
        // 0.5 * (VLagrange[0] * VLagrange[0] + VLagrange[1] * VLagrange[1]));
        MTOT_L(cCells) = 0.;
        ETOT_L(cCells) =
            (rhoLagrange * varlp->vLagrange(cCells)) *
            (fracmass(cCells)[0] * peLagrange[0] +
             fracmass(cCells)[1] * peLagrange[1] +
             fracmass(cCells)[2] * peLagrange[2] +
             0.5 * (VLagrange[0] * VLagrange[0] + VLagrange[1] * VLagrange[1]));
        for (int imat = 0; imat < nbmat; imat++) {
          MTOT_L(cCells) +=
              fracmass(cCells)[imat] * (rhoLagrange * varlp->vLagrange(cCells));
        }
      });
  double reductionE(0.), reductionM(0.);
  {
    Kokkos::Sum<double> reducerE(reductionE);
    Kokkos::parallel_reduce("reductionE", nbCells,
                            KOKKOS_LAMBDA(const int& cCells, double& x) {
                              reducerE.join(x, ETOT_L(cCells));
                            },
                            reducerE);
    Kokkos::Sum<double> reducerM(reductionM);
    Kokkos::parallel_reduce("reductionM", nbCells,
                            KOKKOS_LAMBDA(const int& cCells, double& x) {
                              reducerM.join(x, MTOT_L(cCells));
                            },
                            reducerM);
  }
  ETOTALE_L = reductionE;
  MASSET_L = reductionM;
}
