#include "Vnr.h"

using namespace nablalib;

#include "../includes/Freefunctions.h"
#include "types/MathFunctions.h"    // for max, norm, dot
#include "utils/Utils.h"            // for indexOf

void Vnr::initBoundaryConditions() noexcept {
  if (test->Nom == test->SodCaseX || test->Nom == test->BiSodCaseX){
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
  if (test->Nom == test->AdvectionX || test->Nom == test->BiAdvectionX
      || test->Nom == test->AdvectionVitX || test->Nom == test->BiAdvectionVitX) {
    cdl->rightBC = cdl->imposedVelocity;
    cdl->rightBCValue = {{1.0, 0.0}};
    cdl->rightCellBC = cdl->periodic;
    cdl->leftBC = cdl->imposedVelocity;
    cdl->leftBCValue = {{1.0, 0.0}};
    //cdl->leftCellBC = cdl->periodic;
    cdl->topBC = cdl->symmetry;
    cdl->topBCValue = ex;
    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;
  }
  if (test->Nom == test->AdvectionY || test->Nom == test->BiAdvectionY) {
    cdl->bottomBC = cdl->imposedVelocity;
    cdl->bottomBCValue = {{0.0, 1.0}};
    cdl->bottomCellBC = cdl->periodic;
    cdl->topBC = cdl->imposedVelocity;
    cdl->topBCValue = {{0.0, 1.0}};
    cdl->topCellBC = cdl->periodic;
    
    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->symmetry;
    cdl->rightBCValue = ey;
  }
  if (test->Nom == test->UnitTestCase  || test->Nom == test->BiUnitTestCase) {
    cdl->rightBC = cdl->imposedVelocity;
    cdl->rightBCValue = {{1.0, 0.0}};
    
    cdl->leftBC = cdl->imposedVelocity;
    cdl->leftBCValue = {{1.0, 0.0}};
    cdl->leftFluxBC = 1;
    cdl->leftFluxBCValue = {
        {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    
    cdl->bottomBC = cdl->imposedVelocity;
    cdl->bottomBCValue = {{1.0, 0.0}};
    
    cdl->topBC = cdl->imposedVelocity;
    cdl->topBCValue = {{1.0, 0.0}};
    
    //cdl->leftBC = cdl->symmetry;
    //cdl->leftBCValue = ey;

    //cdl->rightBC = cdl->symmetry;
    //cdl->rightBCValue = ey;

    //cdl->topBC = cdl->symmetry;
    //cdl->topBCValue = ex;

    //cdl->bottomBC = cdl->symmetry;
    //cdl->bottomBCValue = ex;
  }
   cdl->FluxBC =
      cdl->leftFluxBC + cdl->rightFluxBC + cdl->bottomFluxBC + cdl->topFluxBC;
}

/**
 * Job init called @2.0 in simulate method.
 * In variables: cellPos_n0, gamma
 * Out variables: c_n0, p_n0, rho_n0, u_n0
 */
void Vnr::init() noexcept
{
  Kokkos::parallel_for("initDensity", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    for (int imat = 0; imat < nbmatmax; imat++) {
      fracvol(cCells)[imat] = 0.0;
      fracmass(cCells)[imat] = 0.0;
      rhop_n0(cCells)[imat] = 0.0;
      pp_n0(cCells)[imat] = 0.0;
    }
  });
  if (test->Nom == test->SodCaseX || test->Nom == test->SodCaseY) {
    Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  double r(0.);
	  if (test->Nom == test->SodCaseX) r = cellPos_n0(cCells)[0];
	  if (test->Nom == test->SodCaseY) r = cellPos_n0(cCells)[1];
	  if (r < 0.5) 
	    {
	      fracvol(cCells)[0] = 1.;
	      fracmass(cCells)[0] = 1.;
	      rho_n0(cCells) = 1.0;
	      p_n0(cCells) = 1.0;
	      rhop_n0(cCells)[0] = 1.0;
	      pp_n0(cCells)[0] = 1.0;
	    }
	  else
	    {
	      fracvol(cCells)[0] = 1.;
	      fracmass(cCells)[0] = 1.;
	      rho_n0(cCells) = 0.125;
	      p_n0(cCells) = 0.1;
	      rhop_n0(cCells)[0] = 0.125;
	      pp_n0(cCells)[0] = 0.1;
	    }	  
	  cp_n0(cCells)[0] = std::sqrt(eos->gamma * rhop_n0(cCells)[0] / pp_n0(cCells)[0]);
	  c_n0(cCells) = cp_n0(cCells)[0];
	});
    for (size_t pNodes=0; pNodes<nbNodes; pNodes++)
      {
	u_n0(pNodes) = {0.0, 0.0};
	
	ux(pNodes) = u_n0(pNodes)[0];
	uy(pNodes) = u_n0(pNodes)[1];
      }
  } else if (test->Nom == test->BiSodCaseX || test->Nom == test->BiSodCaseY) {
    Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  double r(0.);
	  if (test->Nom == test->BiSodCaseX) r = cellPos_n0(cCells)[0];
	  if (test->Nom == test->BiSodCaseY) r = cellPos_n0(cCells)[1];
	  if (r < 0.5) 
	    {
	      fracvol(cCells)[0] = 1.;
	      fracvol(cCells)[1] = 0.;
	      fracmass(cCells)[0] = 1.;
	      fracmass(cCells)[1] = 0.;
	      rho_n0(cCells) = 1.0;
	      p_n0(cCells) = 1.0;
	      rhop_n0(cCells)[0] = 1.0;
	      pp_n0(cCells)[0] = 1.0;
	    }
	  else
	    {
	      fracvol(cCells)[0] = 0.;
	      fracvol(cCells)[1] = 1.;
	      fracmass(cCells)[0] = 0.;
	      fracmass(cCells)[1] = 1.;
	      rho_n0(cCells) = 0.125;
	      p_n0(cCells) = 0.1;
	      rhop_n0(cCells)[1] = 0.125;
	      pp_n0(cCells)[1] = 0.1;
	    }
	  cp_n0(cCells)[0] = std::sqrt(eos->gamma * rhop_n0(cCells)[0] / pp_n0(cCells)[0]);
	  cp_n0(cCells)[1] = std::sqrt(eos->gamma * rhop_n0(cCells)[1] / pp_n0(cCells)[1]);
	  c_n0(cCells) = min(cp_n0(cCells)[0], cp_n0(cCells)[1]);
	});
        for (size_t pNodes=0; pNodes<nbNodes; pNodes++)
	    {
	      u_n0(pNodes) = {0.0, 0.0};
	
	      ux(pNodes) = u_n0(pNodes)[0];
	      uy(pNodes) = u_n0(pNodes)[1];
	    }
  } else if (test->Nom == test->AdvectionX || test->Nom == test->AdvectionY) {
    
    Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  double r(0.);
	  if (test->Nom == test->AdvectionX) r = cellPos_n0(cCells)[0];
	  if (test->Nom == test->AdvectionY) r = cellPos_n0(cCells)[1];
	  if (r < 0.3) 
	    {
	      fracvol(cCells)[0] = 1.;
	      fracmass(cCells)[0] = 1.;
	      rho_n0(cCells) = 1.;
	      p_n0(cCells) = 0.0;
	      rhop_n0(cCells)[0] = 1.;
	      pp_n0(cCells)[0] = 0.0;
	    }
	  else if ((r > 0.3) && (r < 0.5))
	    {
	      fracvol(cCells)[0] = 1.;
	      fracmass(cCells)[0] = 1.;
	      rho_n0(cCells) = 10.;
	      p_n0(cCells) = 0.0;
	      rhop_n0(cCells)[0] = 10.0;
	      pp_n0(cCells)[0] = 0.0;
	    }
	  else if (r > 0.5)
	    {
	      fracvol(cCells)[0] = 1.;
	      fracmass(cCells)[0] = 1.;
	      rho_n0(cCells) = 1.;
	      p_n0(cCells) = 0.0;
	      rhop_n0(cCells)[0] = 1.;
	      pp_n0(cCells)[0] = 0.0;
	    }
	  cp_n0(cCells)[0] = 1.;
	  cp_n0(cCells)[1] = 1.;
	  c_n0(cCells) = min(cp_n0(cCells)[0], cp_n0(cCells)[1]);
	  ep_n0(cCells)[0] = 1.;
	  ep_n0(cCells)[1] = 1.;
	  e_n0(cCells) = 1.;
	});
    const RealArray1D<dim> ex = {{1.0, 0.0}};
    const RealArray1D<dim> ey = {{0.0, 1.0}};
    RealArray1D<dim> u;
    if (test->Nom == test->AdvectionX) u = ex;
    if (test->Nom == test->AdvectionY) u = ey;
        for (size_t pNodes=0; pNodes<nbNodes; pNodes++)
	    {
	      u_n0(pNodes) = u;
	
	      ux(pNodes) = u_n0(pNodes)[0];
	      uy(pNodes) = u_n0(pNodes)[1];
	    }
  } else if (test->Nom == test->BiAdvectionX || test->Nom == test->BiAdvectionY) {
    
    Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  double r(0.);
	  if (test->Nom == test->BiAdvectionX) r = cellPos_n0(cCells)[0];
	  if (test->Nom == test->BiAdvectionY) r = cellPos_n0(cCells)[1];
	  if (r < 0.3) 
	    {
	      fracvol(cCells)[0] = 1.;
	      fracmass(cCells)[0] = 1.;
	      rho_n0(cCells) = 1.;
	      p_n0(cCells) = 0.0;
	      rhop_n0(cCells)[0] = 1.0;
	      pp_n0(cCells)[0] = 0.0;
	    }
	  else if ((r > 0.3) && (r < 0.5))
	    {
	      fracvol(cCells)[1] = 1.;
	      fracmass(cCells)[1] = 1.;
	      rho_n0(cCells) = 1.;
	      p_n0(cCells) = 0.0;
	      rhop_n0(cCells)[1] = 1.0;
	      pp_n0(cCells)[1] = 0.0;
	    }
	  else if (r > 0.5)
	    {
	      fracvol(cCells)[0] = 1.;
	      fracmass(cCells)[0] = 1.;
	      rho_n0(cCells) = 1.;
	      p_n0(cCells) = 0.0;
	      rhop_n0(cCells)[0] = 1.0;
	      pp_n0(cCells)[0] = 0.0;
	    }
	  cp_n0(cCells)[0] = 1.;
	  cp_n0(cCells)[1] = 1.;
	  c_n0(cCells) = min(cp_n0(cCells)[0], cp_n0(cCells)[1]);
	  ep_n0(cCells)[0] = 1.;
	  ep_n0(cCells)[1] = 1.;
	  e_n0(cCells) = 1.;
	});
    const RealArray1D<dim> ex = {{1.0, 0.0}};
    const RealArray1D<dim> ey = {{0.0, 1.0}};
    RealArray1D<dim> u;
    if (test->Nom == test->BiAdvectionX) u = ex;
    if (test->Nom == test->BiAdvectionY) u = ey;
        for (size_t pNodes=0; pNodes<nbNodes; pNodes++)
	    {
	      u_n0(pNodes) = u;
	
	      ux(pNodes) = u_n0(pNodes)[0];
	      uy(pNodes) = u_n0(pNodes)[1];
	    }
  }  else if (test->Nom == test->BiAdvectionVitX || test->Nom == test->BiAdvectionVitY) {
    
    Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  double r(0.);
	  if (test->Nom == test->BiAdvectionVitX) r = cellPos_n0(cCells)[0];
	  if (test->Nom == test->BiAdvectionVitY) r = cellPos_n0(cCells)[1];
	  if (r < 0.3) 
	    {
	      fracvol(cCells)[0] = 1.;
	      fracmass(cCells)[0] = 1.;
	      rho_n0(cCells) = 4.;
	      p_n0(cCells) = 0.0;
	      rhop_n0(cCells)[0] = 4.0;
	      pp_n0(cCells)[0] = 0.0;
	    }
	  else if ((r > 0.3) && (r < 0.5))
	    {
	      fracvol(cCells)[1] = 1.;
	      fracmass(cCells)[1] = 1.;
	      rho_n0(cCells) = 4.;
	      p_n0(cCells) = 0.0;
	      rhop_n0(cCells)[1] = 1.0;
	      pp_n0(cCells)[1] = 0.0;
	    }
	  else if (r > 0.5)
	    {
	      fracvol(cCells)[0] = 1.;
	      fracmass(cCells)[0] = 1.;
	      rho_n0(cCells) = 4.;
	      p_n0(cCells) = 0.0;
	      rhop_n0(cCells)[0] = 4.0;
	      pp_n0(cCells)[0] = 0.0;
	    }
	  cp_n0(cCells)[0] = 1.;
	  cp_n0(cCells)[1] = 1.;
	  c_n0(cCells) = min(cp_n0(cCells)[0], cp_n0(cCells)[1]);
	  ep_n0(cCells)[0] = 1.;
	  ep_n0(cCells)[1] = 1.;
	  e_n0(cCells) = 1.;
	});
    const RealArray1D<dim> ex = {{1.0, 0.0}};
    const RealArray1D<dim> ey = {{0.0, 1.0}};
    RealArray1D<dim> u;
    if (test->Nom == test->BiAdvectionVitX) u = ex;
    if (test->Nom == test->BiAdvectionVitY) u = ey;
        for (size_t pNodes=0; pNodes<nbNodes; pNodes++)
	    {
	      double r(0.);
	      if (test->Nom == test->BiAdvectionVitX) r = X_n0(pNodes)[0];
	      if (test->Nom == test->BiAdvectionVitY) r = X_n0(pNodes)[1];
	  if (r <= 0.3) 
	    {
	      u_n0(pNodes) = 0.2 * u;
	    }
	  else if ((r > 0.3) && (r <= 0.5))
	    {
	      u_n0(pNodes) = (4. * r - 1.) * u;
	    }
	  else if (r > 0.5)
	    {
	      u_n0(pNodes) = u;
	    }
	      ux(pNodes) = u_n0(pNodes)[0];
	      uy(pNodes) = u_n0(pNodes)[1];
	    }
  } else if (test->Nom == test->UnitTestCase  || test->Nom == test->BiUnitTestCase) {
    Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{	 
	  fracvol(cCells)[0] = 1.;
	  fracvol(cCells)[1] = 0.;
	  fracmass(cCells)[0] = 1.;
	  fracmass(cCells)[1] = 0.;
	  p_n0(cCells) = 1.0;
	  pp_n0(cCells)[0] = 1.0;
	  pp_n0(cCells)[1] = 0.0;
	  c_n0(cCells) = std::sqrt(eos->gamma);
	  cp_n0(cCells)[0] = std::sqrt(eos->gamma);
	  cp_n0(cCells)[1] = std::sqrt(eos->gamma);
	  rhop_n0(cCells)[1] = 0.;
	  double r(0.);
	  r = cellPos_n0(cCells)[0];
	  if (r < 0.4) 
	    {
	      rho_n0(cCells) = 1.;
	      rhop_n0(cCells)[0] = 1.;
	    } else if (r > 0.4) {
	    rho_n0(cCells) = 0.1;
	    rhop_n0(cCells)[0] = 0.1;
	  }
	  if (r > 0.6) {
	    rho_n0(cCells) = 0.1;
	    rhop_n0(cCells)[0] = 0.1;
	  }
	    
	      
	});
    for (size_t pNodes=0; pNodes<nbNodes; pNodes++)
      {
	if (X_n0(pNodes)[0] < 0.5) 
	  u_n0(pNodes) = {1.0, 0.0};
	else
	  u_n0(pNodes) = {1.0, 0.0};
	
	ux(pNodes) = u_n0(pNodes)[0];
	uy(pNodes) = u_n0(pNodes)[1];
      }
  } else {
    std::cout << "Cas test inconnu " << std::endl;
    exit(1);
  }
  Kokkos::parallel_for("init", nbCells, KOKKOS_LAMBDA(const int& cCells)
       {
         // pour les sorties au temps 0:
	 fracvol1(cCells) = fracvol(cCells)[0];
	 fracvol2(cCells) = fracvol(cCells)[1];
	 fracvol3(cCells) = fracvol(cCells)[2];
	 // indicateur mailles mixtes
	 int matcell(0);
	 int imatpure(-1);
	 for (int imat = 0; imat < nbmatmax; imat++)
	   if (fracvol(cCells)[imat] > options->threshold) {
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
       });
}
/**
 * Job initSubVol called @2.0 in simulate method.
 * In variables: X_n0, cellPos_n0
 * Out variables: SubVol_n0
 */
void Vnr::initSubVol() noexcept
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
		volE(cCells) = cstmesh->X_EDGE_LENGTH * cstmesh->Y_EDGE_LENGTH;
	});
}
/**
 * Job initDeltaT called @3.0 in simulate method.
 * In variables: SubVol_n0, c_n0
 * Out variables: deltat_init
 */
void Vnr::initDeltaT() noexcept
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
;
}
/**
 * Job initInternalEnergy called @3.0 in simulate method.
 * In variables: gamma, p_n0, rho_n0
 * Out variables: e_n0
 */
void Vnr::initInternalEnergy() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  for (int imat = 0; imat < nbmatmax; imat++)
		ep_n0(cCells)[imat] = pp_n0(cCells)[imat] / ((eos->gammap[imat] - 1.0) * rhop_n0(cCells)[imat]);
	  e_n0(cCells) = p_n0(cCells) / ((eos->gamma - 1.0) * rho_n0(cCells));
	});
}

/**
 * Job initPseudo called @3.0 in simulate method.
 * In variables: rho_n0
 * Out variables: divU_n0, tau_n0
 */
void Vnr::initPseudo() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		tau_n0(cCells) = 1 / rho_n0(cCells);
		divU_n0(cCells) = 0.0;
		Q_n0(cCells) = 0.0;
		for (int imat = 0; imat < nbmatmax; imat++) {
		  taup_n0(cCells)[imat] = 1 / rhop_n0(cCells)[imat];
		  Qp_n0(cCells)[imat] = 0.0;
		}
	});
}
/**
 * Job initCellPos called @1.0 in simulate method.
 * In variables: X_n0
 * Out variables: cellPos_n0
 */
void Vnr::initCellPos() noexcept
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
/**
 * Job initMeshGeometryForFaces called @2.0 in simulate method.
 * In variables: X, Xc, ex, ey, threshold
 * Out variables: Xf, faceLength, faceNormal, outerFaceNormal
 */
void Vnr::initMeshGeometryForFaces() noexcept {
  auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "initMeshGeometryForFaces", nbFaces, KOKKOS_LAMBDA(const int& fFaces) {
        size_t fId(faces[fFaces]);
        int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
        int n1Id(n1FirstNodeOfFaceF);
        int n1Nodes(n1Id);
        int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
        int n2Id(n2SecondNodeOfFaceF);
        int n2Nodes(n2Id);
        RealArray1D<dim> X_face = (0.5 * ((X_n0(n1Nodes) + X_n0(n2Nodes))));
        RealArray1D<dim> face_vec = (X_n0(n2Nodes) - X_n0(n1Nodes));
        varlp->Xf(fFaces) = X_face;
        varlp->faceLength(fFaces) = MathFunctions::norm(face_vec);
        {
          auto cellsOfFaceF(mesh->getCellsOfFace(fId));
          for (int cCellsOfFaceF = 0; cCellsOfFaceF < cellsOfFaceF.size();
               cCellsOfFaceF++) {
            int cId(cellsOfFaceF[cCellsOfFaceF]);
            int cCells(cId);
            int fFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), fId));
            varlp->outerFaceNormal(cCells, fFacesOfCellC) = (
                ((X_face - cellPos_n0(cCells))) / MathFunctions::norm((X_face - cellPos_n0(cCells))));
          }
        }
        RealArray1D<dim> face_normal;
        if (MathFunctions::fabs(dot(face_vec, ex)) <
            options->threshold)
          face_normal = ex;
        else
          face_normal = ey;
	
        varlp->faceNormal(fFaces) = face_normal;
      });
}
