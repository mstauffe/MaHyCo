#include "VnrRemap.h"

using namespace nablalib;

#include "../includes/Freefunctions.h"

void VnrRemap::initBoundaryConditions() noexcept {
  if (test->Nom == test->SodCaseX || test->Nom == test->BiSodCaseX) {
    // maillage 200 5 0.005 0.02
    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->symmetry;
    cdl->rightBCValue = ey;

    cdl->topBC = cdl->symmetry;
    cdl->topBCValue = ex;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;
  }
  if (test->Nom == test->SodCaseY || test->Nom == test->BiSodCaseY) {
    // maillage 5 200 0.02 0.005
    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->symmetry;
    cdl->rightBCValue = ey;

    cdl->topBC = cdl->symmetry;
    cdl->topBCValue = ex;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;
  }
}

/**
 * Job init called @2.0 in simulate method.
 * In variables: cellPos_n0, gamma
 * Out variables: c_n0, p_n0, rho_n0, u_n0
 */
void VnrRemap::init() noexcept
{
  if (test->Nom == test->SodCaseX || test->Nom == test->SodCaseY) {
    Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  double r(0.);
	  if (test->Nom == test->SodCaseX) r = cellPos_n0(cCells)[0];
	  if (test->Nom == test->SodCaseY) r = cellPos_n0(cCells)[1];
	  if (r < 0.5) 
	    {
	      rho_n0(cCells) = 1.0;
	      p_n0(cCells) = 1.0;
	      c_n0(cCells) = std::sqrt(eos->gamma);
	    }
	  else
	    {
	      rho_n0(cCells) = 0.1;
	      p_n0(cCells) = 0.125;
	      c_n0(cCells) = std::sqrt(eos->gamma * 0.125 / 0.1);
	    }
	  for (size_t pNodes=0; pNodes<nbNodes; pNodes++)
	    {
	      u_n0(pNodes) = {0.0, 0.0};
	    }
	});
  } else {
    std::cout << "Pas d'autres cas test que SODX ou SODY" << std::endl;
    exit(1);
  }							     
}
/**
 * Job initSubVol called @2.0 in simulate method.
 * In variables: X_n0, cellPos_n0
 * Out variables: SubVol_n0
 */
void VnrRemap::initSubVol() noexcept
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
				const RealArray1D<2> x1(cellPos_n0(cCells));
				const RealArray1D<2> x2(0.5 * (X_n0(pMinus1Nodes) + X_n0(pNodes)));
				const RealArray1D<2> x3(X_n0(pNodes));
				const RealArray1D<2> x4(0.5 * (X_n0(pPlus1Nodes) + X_n0(pNodes)));
				SubVol_n0(cCells,pNodesOfCellC) = 0.5 * (crossProduct2d(x1, x2) + crossProduct2d(x2, x3) + crossProduct2d(x3, x4) + crossProduct2d(x4, x1));
			}
		}
	});
}
/**
 * Job initDeltaT called @3.0 in simulate method.
 * In variables: SubVol_n0, c_n0
 * Out variables: deltat_init
 */
void VnrRemap::initDeltaT() noexcept
{
	double reduction0;
	Kokkos::parallel_reduce(nbCells, KOKKOS_LAMBDA(const size_t& cCells, double& accu)
	{
		const Id cId(cCells);
		double reduction1(0.0);
		{
			const auto nodesOfCellC(mesh->getNodesOfCell(cId));
			const size_t nbNodesOfCellC(nodesOfCellC.size());
			for (size_t pNodesOfCellC=0; pNodesOfCellC<nbNodesOfCellC; pNodesOfCellC++)
			{
				reduction1 = sumR0(reduction1, SubVol_n0(cCells,pNodesOfCellC));
			}
		}
		accu = minR0(accu, 0.1 * std::sqrt(reduction1) / c_n0(cCells));
	}, KokkosJoiner<double>(reduction0, numeric_limits<double>::max(), &minR0));
	gt->deltat_init = reduction0 * 1.0E-6;
}
/**
 * Job initInternalEnergy called @3.0 in simulate method.
 * In variables: gamma, p_n0, rho_n0
 * Out variables: e_n0
 */
void VnrRemap::initInternalEnergy() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		e_n0(cCells) = p_n0(cCells) / ((eos->gamma - 1.0) * rho_n0(cCells));
	});
}

/**
 * Job initPseudo called @3.0 in simulate method.
 * In variables: rho_n0
 * Out variables: divU_n0, tau_n0
 */
void VnrRemap::initPseudo() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		tau_n0(cCells) = 1 / rho_n0(cCells);
		divU_n0(cCells) = 0.0;
		Q_n0(cCells) = 0.0;
	});
}
/**
 * Job initCellPos called @1.0 in simulate method.
 * In variables: X_n0
 * Out variables: cellPos_n0
 */
void VnrRemap::initCellPos() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		const Id cId(cCells);
		RealArray1D<2> reduction0({0.0, 0.0});
		{
			const auto nodesOfCellC(mesh->getNodesOfCell(cId));
			const size_t nbNodesOfCellC(nodesOfCellC.size());
			for (size_t pNodesOfCellC=0; pNodesOfCellC<nbNodesOfCellC; pNodesOfCellC++)
			{
				const Id pId(nodesOfCellC[pNodesOfCellC]);
				const size_t pNodes(pId);
				reduction0 = sumR1(reduction0, X_n0(pNodes));
			}
		}
		cellPos_n0(cCells) = 0.25 * reduction0;
	});
}
