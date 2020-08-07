#include "Vnr.h"

using namespace nablalib;

#include "../includes/Freefunctions.h"
#include "types/MathFunctions.h"    // for max, min, dot, matVectProduct
#include "utils/Utils.h"  // for Indexof

/**
 * Job computeCellMass called @3.0 in simulate method.
 * In variables: X_EDGE_LENGTH, Y_EDGE_LENGTH, rho_n0
 * Out variables: cellMass
 */
void Vnr::computeCellMass() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  int nbmat = options->nbmat;
	  cellMass(cCells) = cstmesh->X_EDGE_LENGTH * cstmesh->Y_EDGE_LENGTH * rho_n0(cCells);
	  for (int imat = 0; imat < nbmat; ++imat) {
	    cellMassp(cCells)[imat] = fracmass(cCells)[imat] * cellMass(cCells);
	  }
	});
}

/**
 * Job computeNodeMass called @4.0 in simulate method.
 * In variables: cellMass
 * Out variables: m
 */
void Vnr::computeNodeMass() noexcept
{
	Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes)
	{
		const Id pId(pNodes);
		const auto cells_of_node(mesh->getCellsOfNode(pId));
		double reduction0(0.0);
		{
			const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
			const size_t nbCellsOfNodeP(cellsOfNodeP.size());
			for (size_t cCellsOfNodeP=0; cCellsOfNodeP<nbCellsOfNodeP; cCellsOfNodeP++)
			{
				const Id cId(cellsOfNodeP[cCellsOfNodeP]);
				const size_t cCells(cId);
				reduction0 = sumR0(reduction0, cellMass(cCells));
			}
		}
		m(pNodes) = reduction0 / cells_of_node.size();
	});
}
/**
 * Job computeArtificialViscosity called @1.0 in executeTimeLoopN method.
 * In variables: SubVol_n, c_n, divU_n, gamma, tau_n
 * Out variables: Q_nplus1
 */
void Vnr::computeArtificialViscosity() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		const Id cId(cCells);
		if (divU_n(cCells) < 0.0) 
		{
			double reduction0(0.0);
			{
				const auto nodesOfCellC(mesh->getNodesOfCell(cId));
				const size_t nbNodesOfCellC(nodesOfCellC.size());
				for (size_t pNodesOfCellC=0; pNodesOfCellC<nbNodesOfCellC; pNodesOfCellC++)
				{
					reduction0 = sumR0(reduction0, SubVol_n(cCells,pNodesOfCellC));
				}
			}
			double reduction1(0.0);
			{
				const auto nodesOfCellC(mesh->getNodesOfCell(cId));
				const size_t nbNodesOfCellC(nodesOfCellC.size());
				for (size_t pNodesOfCellC=0; pNodesOfCellC<nbNodesOfCellC; pNodesOfCellC++)
				{
					reduction1 = sumR0(reduction1, SubVol_n(cCells,pNodesOfCellC));
				}
			}
			Q_nplus1(cCells) = 1.0 / tau_nplus1(cCells) *
			  (-0.5 * std::sqrt(reduction0) * c_n(cCells) * divU_nplus1(cCells)
			   + (eos->gamma + 1) / 2.0 * reduction1 * divU_nplus1(cCells) * divU_nplus1(cCells));
		}
		else
			Q_nplus1(cCells) = 0.0;
		//
		// pour chaque matériau
		for (int imat = 0; imat < options->nbmat; ++imat)
		  Qp_nplus1(cCells)[imat] = fracvol(cCells)[imat] * Q_nplus1(cCells);		  
		
	});
}

/**
 * Job computeCornerNormal called @1.0 in executeTimeLoopN method.
 * In variables: X_n
 * Out variables: C
 */
void Vnr::computeCornerNormal() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		const Id cId(cCells);
		{
			const auto nodesOfCellC(mesh->getNodesOfCell(cId));
			const size_t nbNodesOfCellC(nodesOfCellC.size());
			for (size_t pNodesOfCellC=0; pNodesOfCellC<nbNodesOfCellC; pNodesOfCellC++)
			{
				const Id pId(nodesOfCellC[pNodesOfCellC]);
				const Id pPlus1Id(nodesOfCellC[(pNodesOfCellC+1+nbNodesOfCell)%nbNodesOfCell]);
				const Id pMinus1Id(nodesOfCellC[(pNodesOfCellC-1+nbNodesOfCell)%nbNodesOfCell]);
				const size_t pNodes(pId);
				const size_t pPlus1Nodes(pPlus1Id);
				const size_t pMinus1Nodes(pMinus1Id);
				C(cCells,pNodesOfCellC) = computeLpcNpc(X_n(pNodes), X_n(pPlus1Nodes), X_n(pMinus1Nodes));
			}
		}
	});
}
/**
 * Job computeNodeVolume called @1.0 in executeTimeLoopN method.
 * In variables: SubVol_n
 * Out variables: V
 */
void Vnr::computeNodeVolume() noexcept
{
	Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes)
	{
		const Id pId(pNodes);
		double reduction0(0.0);
		{
			const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
			const size_t nbCellsOfNodeP(cellsOfNodeP.size());
			for (size_t cCellsOfNodeP=0; cCellsOfNodeP<nbCellsOfNodeP; cCellsOfNodeP++)
			{
				const Id cId(cellsOfNodeP[cCellsOfNodeP]);
				const size_t cCells(cId);
				const size_t pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
				reduction0 = sumR0(reduction0, SubVol_n(cCells,pNodesOfCellC));
			}
		}
		V(pNodes) = reduction0;
	});
}
/**
 * Job updateVelocity called @2.0 in executeTimeLoopN method.
 * In variables: C, Q_nplus1, deltat_n, deltat_nplus1, m, p_n, u_n
 * Out variables: u_nplus1
 */
void Vnr::updateVelocity() noexcept
{
  const double dt(0.5 * (gt->deltat_nplus1 + gt->deltat_n));
  {
    const auto innerNodes(mesh->getInnerNodes());
    const size_t nbInnerNodes(mesh->getNbInnerNodes());
    Kokkos::parallel_for(nbInnerNodes, KOKKOS_LAMBDA(const size_t& pInnerNodes)
	  {
	    const Id pId(innerNodes[pInnerNodes]);
	    const size_t pNodes(pId);
	    RealArray1D<2> reduction0({0.0, 0.0});
	    {
	      const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
	      const size_t nbCellsOfNodeP(cellsOfNodeP.size());
	      for (size_t cCellsOfNodeP=0; cCellsOfNodeP<nbCellsOfNodeP; cCellsOfNodeP++)
		{
		  const Id cId(cellsOfNodeP[cCellsOfNodeP]);
		  const size_t cCells(cId);
		  const size_t pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
		  reduction0 = sumR1(reduction0, (p_n(cCells) + Q_n(cCells)) * C(cCells,pNodesOfCellC));
		}
	    }
	    u_nplus1(pNodes) = u_n(pNodes) + dt / m(pNodes) * reduction0;
	  });
  }
}
/**
 * Job updatePosition called @3.0 in executeTimeLoopN method.
 * In variables: X_n, deltat_nplus1, u_nplus1
 * Out variables: X_nplus1
 */
void Vnr::updatePosition() noexcept
{
	Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes)
	{
		X_nplus1(pNodes) = X_n(pNodes) + gt->deltat_nplus1 * u_nplus1(pNodes);
	});
}
/**
 * Job computeSubVol called @4.0 in executeTimeLoopN method.
 * In variables: X_nplus1, cellPos_nplus1
 * Out variables: SubVol_nplus1
 */
void Vnr::computeSubVol() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		const Id cId(cCells);
		{
			const auto nodesOfCellC(mesh->getNodesOfCell(cId));
			const size_t nbNodesOfCellC(nodesOfCellC.size());
			for (size_t pNodesOfCellC=0; pNodesOfCellC<nbNodesOfCellC; pNodesOfCellC++)
			{
				const Id pMinus1Id(nodesOfCellC[(pNodesOfCellC-1+nbNodesOfCell)%nbNodesOfCell]);
				const Id pId(nodesOfCellC[pNodesOfCellC]);
				const Id pPlus1Id(nodesOfCellC[(pNodesOfCellC+1+nbNodesOfCell)%nbNodesOfCell]);
				const size_t pMinus1Nodes(pMinus1Id);
				const size_t pNodes(pId);
				const size_t pPlus1Nodes(pPlus1Id);
				const RealArray1D<2> x1(cellPos_nplus1(cCells));
				const RealArray1D<2> x2(0.5 * (X_nplus1(pMinus1Nodes) + X_nplus1(pNodes)));
				const RealArray1D<2> x3(X_nplus1(pNodes));
				const RealArray1D<2> x4(0.5 * (X_nplus1(pPlus1Nodes) + X_nplus1(pNodes)));
				SubVol_nplus1(cCells,pNodesOfCellC) = 0.5 * (crossProduct2d(x1, x2) + crossProduct2d(x2, x3) + crossProduct2d(x3, x4) + crossProduct2d(x4, x1));
			}
		}
	});
}
/**
 * Job updateRho called @5.0 in executeTimeLoopN method.
 * In variables: SubVol_nplus1, cellMass
 * Out variables: rho_nplus1
 */
void Vnr::updateRho() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		const Id cId(cCells);
		double reduction0(0.0);
		{
			const auto nodesOfCellC(mesh->getNodesOfCell(cId));
			const size_t nbNodesOfCellC(nodesOfCellC.size());
			for (size_t pNodesOfCellC=0; pNodesOfCellC<nbNodesOfCellC; pNodesOfCellC++)
			{
				reduction0 = sumR0(reduction0, SubVol_nplus1(cCells,pNodesOfCellC));
			}
		}
		rho_nplus1(cCells) = 0.;
		for (int imat = 0; imat < options->nbmat; ++imat) {
		  if (fracvol(cCells)[imat] > options->threshold) 
		    rhop_nplus1(cCells)[imat] = cellMassp(cCells)[imat] / (fracvol(cCells)[imat] * reduction0);
		  // ou 1/rhon_nplus1 += fracmass(cCells)[imat] / rhop_nplus1[imat];
		  rho_nplus1(cCells) += fracvol(cCells)[imat] * rhop_nplus1(cCells)[imat];
		  std::cout << "rho" << imat << " " << cCells << " " << rhop_nplus1(cCells)[imat] << std::endl;
		}
	});
}

/**
 * Job computeTau called @6.0 in executeTimeLoopN method.
 * In variables: rho_n, rho_nplus1
 * Out variables: tau_nplus1
 */
void Vnr::computeTau() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  tau_nplus1(cCells) = 0.5 * (1.0 / rho_nplus1(cCells) + 1.0 / rho_n(cCells));
	  for (int imat = 0; imat < options->nbmat; ++imat) {
	    taup_nplus1(cCells)[imat] = 0.;
	    if ((rhop_nplus1(cCells)[imat] > options->threshold) && (rhop_n(cCells)[imat] > options->threshold))
	    taup_nplus1(cCells)[imat] = 0.5 * (1.0 / rhop_nplus1(cCells)[imat] + 1.0 / rhop_n(cCells)[imat]);
	  }
	});
}

/**
 * Job updateEnergy called @6.0 in executeTimeLoopN method.
 * In variables: Q_nplus1, e_n, gamma, p_n, rho_n, rho_nplus1
 * Out variables: e_nplus1
 */
void Vnr::updateEnergy() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  e_nplus1(cCells) = 0.;
	  for (int imat = 0; imat < options->nbmat; ++imat) {
	    ep_nplus1(cCells)[imat] =0.;
	    if ((rhop_nplus1(cCells)[imat] > options->threshold) && (rhop_n(cCells)[imat] > options->threshold)) {
	      // calcul du DV a changer utiliser divU
	      double pseudo(0.);
	      if ( (options->pseudo_centree == 1) &&
		   ( (Qp_nplus1(cCells)[imat] + Qp_n(cCells)[imat]) *
		     (1.0 / rhop_nplus1(cCells)[imat] - 1.0 / rhop_n(cCells)[imat]) > 0.) )
		{
		  pseudo = 0.5 * (Qp_nplus1(cCells)[imat] + Qp_n(cCells)[imat]);
		}
	      if (options->pseudo_centree == 0 ) { // test sur la positivité du travail dans le calcul de Q_nplus1(cCells)
		pseudo = Qp_nplus1(cCells)[imat];
	      }
	      const double den(1 + 0.5 * (eos->gammap[imat] - 1.0) * rhop_nplus1(cCells)[imat] *
			       (1.0 / rhop_nplus1(cCells)[imat] - 1.0 / rhop_n(cCells)[imat]));
	      const double num(ep_n(cCells)[imat] - (0.5 * pp_n(cCells)[imat] + pseudo) *
			       (1.0 / rhop_nplus1(cCells)[imat] - 1.0 / rhop_n(cCells)[imat]));
	      ep_nplus1(cCells)[imat] = num / den;
	      e_nplus1(cCells) += fracmass(cCells)[imat] * ep_nplus1(cCells)[imat];
	    }
	  }
	});
}

/**
 * Job computeDivU called @7.0 in executeTimeLoopN method.
 * In variables: deltat_nplus1, rho_n, rho_nplus1, tau_nplus1
 * Out variables: divU_nplus1
 */
void Vnr::computeDivU() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  divU_nplus1(cCells) = 1.0 / gt->deltat_nplus1 * (1.0 / rho_nplus1(cCells) - 1.0 / rho_n(cCells)) / tau_nplus1(cCells);
	  // a changer comme le calcul du DV, utiliser les C(cCells,pNodesOfCellC)
	});
}
/**
 * Job computeEOS called in executeTimeLoopN method.
 */
void Vnr::computeEOS() {
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
 * In variables: eos, eosPerfectGas, eps_n, gammap, rho_n
 * Out variables: c, p
 */
void Vnr::computeEOSGP(int imat)  {
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	 pp_nplus1(cCells)[imat] = (eos->gammap[imat] - 1.0) * rhop_nplus1(cCells)[imat] * ep_nplus1(cCells)[imat];
	 cp_nplus1(cCells)[imat] = std::sqrt(eos->gammap[imat] * (eos->gammap[imat] - 1.0) * ep_nplus1(cCells)[imat]);
	});
}
/**
 * Job computeEOSVoid called in executeTimeLoopN method.
 * In variables: eos, eosPerfectGas, eps_n, gammap, rho_n
 * Out variables: c, p
 */
void Vnr::computeEOSVoid(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    pp_nplus1(cCells)[imat] = 0.;
    cp_nplus1(cCells)[imat] = 1.e-20;
  });
}
/**
 * Job computeEOSSTIFG
 * In variables: eps_n, rho_n
 * Out variables: c, p
 */
void Vnr::computeEOSSTIFG(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    std::cout << " Pas encore programmée" << std::endl;
  });
}
/**
 * Job computeEOSMur called @1.0 in executeTimeLoopN method.
 * In variables: eps_n, rho_n
 * Out variables: c, p
 */
void Vnr::computeEOSMur(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    std::cout << " Pas encore programmée" << std::endl;
  });
}
/**
 * Job computeEOSSL called @1.0 in executeTimeLoopN method.
 * In variables: eps_n, rho_n
 * Out variables: c, p
 */
void Vnr::computeEOSSL(int imat) {
  Kokkos::parallel_for("computeEOS", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    std::cout << " Pas encore programmée" << std::endl;
  });
}
/**
 * Job computeEOS called in executeTimeLoopN method.
 */
void Vnr::computePressionMoyenne() noexcept {
  for (int cCells = 0; cCells < nbCells; cCells++) {
    p_nplus1(cCells) = 0.;
    for (int imat = 0; imat < options->nbmat; ++imat) {
      p_nplus1(cCells) += fracvol(cCells)[imat] * pp_nplus1(cCells)[imat];
      c_nplus1(cCells) =
          MathFunctions::max(c_nplus1(cCells), cp_nplus1(cCells)[imat]);
    }
    // NONREG GP A SUPPRIMER
    if (rho_nplus1(cCells) > 0.) {
      c_nplus1(cCells) =
	std::sqrt(eos->gammap[0] * (eos->gammap[0] - 1.0) * e_nplus1(cCells));
    }
  }
}
