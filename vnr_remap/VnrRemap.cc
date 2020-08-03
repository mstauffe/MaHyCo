#include "VnrRemap.h"

using namespace nablalib;

#include "../includes/Freefunctions.h"
#include <iomanip>          // for operator<<, setw, setiosflags
#include "utils/Utils.h"          // for __RESET__, __BOLD__, __GREEN__

/**
 * Job computeArtificialViscosity called @1.0 in executeTimeLoopN method.
 * In variables: SubVol_n, c_n, divU_n, gamma, tau_n
 * Out variables: Q_nplus1
 */
void VnrRemap::computeArtificialViscosity() noexcept
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
			Q_nplus1(cCells) = 1.0 / tau_n(cCells) * (-0.5 * std::sqrt(reduction0) * c_n(cCells) * divU_n(cCells) + (eos->gamma + 1) / 2.0 * reduction1 * divU_n(cCells) * divU_n(cCells));
		}
		else
			Q_nplus1(cCells) = 0.0;
	});
}

/**
 * Job computeCornerNormal called @1.0 in executeTimeLoopN method.
 * In variables: X_n
 * Out variables: C
 */
void VnrRemap::computeCornerNormal() noexcept
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
 * Job computeDeltaT called @1.0 in executeTimeLoopN method.
 * In variables: SubVol_n, c_n, deltat_n
 * Out variables: deltat_nplus1
 */
void VnrRemap::computeDeltaT() noexcept
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
				reduction1 = sumR0(reduction1, SubVol_n(cCells,pNodesOfCellC));
			}
		}
		accu = minR0(accu, 0.1 * std::sqrt(reduction1) / c_n(cCells));
	}, KokkosJoiner<double>(reduction0, numeric_limits<double>::max(), &minR0));
	gt->deltat_nplus1 = std::min(reduction0, 1.05 * gt->deltat_n);
}

/**
 * Job computeNodeVolume called @1.0 in executeTimeLoopN method.
 * In variables: SubVol_n
 * Out variables: V
 */
void VnrRemap::computeNodeVolume() noexcept
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

/**
 * Job computeTime called @2.0 in executeTimeLoopN method.
 * In variables: deltat_nplus1, t_n
 * Out variables: t_nplus1
 */
void VnrRemap::computeTime() noexcept
{
	gt->t_nplus1 = gt->t_n + gt->deltat_nplus1;
}

/**
 * Job init called @2.0 in simulate method.
 * In variables: cellPos_n0, gamma
 * Out variables: c_n0, p_n0, rho_n0, u_n0
 */
void VnrRemap::init() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		if (cellPos_n0(cCells)[0] < 0.5) 
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
 * Job updateVelocity called @2.0 in executeTimeLoopN method.
 * In variables: C, Q_nplus1, deltat_n, deltat_nplus1, m, p_n, u_n
 * Out variables: u_nplus1
 */
void VnrRemap::updateVelocity() noexcept
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
					reduction0 = sumR1(reduction0, (p_n(cCells) + Q_nplus1(cCells)) * C(cCells,pNodesOfCellC));
				}
			}
			u_nplus1(pNodes) = u_n(pNodes) + dt / m(pNodes) * reduction0;
		});
	}
	{
		const auto bottomNodes(mesh->getBottomNodes());
		const size_t nbBottomNodes(bottomNodes.size());
		Kokkos::parallel_for(nbBottomNodes, KOKKOS_LAMBDA(const size_t& pBottomNodes)
		{
			const Id pId(bottomNodes[pBottomNodes]);
			const size_t pNodes(pId);
			RealArray1D<2> reduction1({0.0, 0.0});
			{
				const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
				const size_t nbCellsOfNodeP(cellsOfNodeP.size());
				for (size_t cCellsOfNodeP=0; cCellsOfNodeP<nbCellsOfNodeP; cCellsOfNodeP++)
				{
					const Id cId(cellsOfNodeP[cCellsOfNodeP]);
					const size_t cCells(cId);
					const size_t pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
					reduction1 = sumR1(reduction1, (p_n(cCells) + Q_nplus1(cCells)) * (C(cCells,pNodesOfCellC) + symmetricVector(C(cCells,pNodesOfCellC), {1.0, 0.0})));
				}
			}
			u_nplus1(pNodes) = u_n(pNodes) + dt / (m(pNodes)) * reduction1;
		});
	}
	{
		const auto topNodes(mesh->getTopNodes());
		const size_t nbTopNodes(topNodes.size());
		Kokkos::parallel_for(nbTopNodes, KOKKOS_LAMBDA(const size_t& pTopNodes)
		{
			const Id pId(topNodes[pTopNodes]);
			const size_t pNodes(pId);
			RealArray1D<2> reduction2({0.0, 0.0});
			{
				const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
				const size_t nbCellsOfNodeP(cellsOfNodeP.size());
				for (size_t cCellsOfNodeP=0; cCellsOfNodeP<nbCellsOfNodeP; cCellsOfNodeP++)
				{
					const Id cId(cellsOfNodeP[cCellsOfNodeP]);
					const size_t cCells(cId);
					const size_t pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
					reduction2 = sumR1(reduction2, (p_n(cCells) + Q_nplus1(cCells)) * (C(cCells,pNodesOfCellC) + symmetricVector(C(cCells,pNodesOfCellC), {1.0, 0.0})));
				}
			}
			u_nplus1(pNodes) = u_n(pNodes) + dt / (m(pNodes)) * reduction2;
		});
	}
	{
		const auto leftNodes(mesh->getLeftNodes());
		const size_t nbLeftNodes(leftNodes.size());
		Kokkos::parallel_for(nbLeftNodes, KOKKOS_LAMBDA(const size_t& pLeftNodes)
		{
			const Id pId(leftNodes[pLeftNodes]);
			const size_t pNodes(pId);
			u_nplus1(pNodes) = {0.0, 0.0};
		});
	}
	{
		const auto rightNodes(mesh->getRightNodes());
		const size_t nbRightNodes(rightNodes.size());
		Kokkos::parallel_for(nbRightNodes, KOKKOS_LAMBDA(const size_t& pRightNodes)
		{
			const Id pId(rightNodes[pRightNodes]);
			const size_t pNodes(pId);
			u_nplus1(pNodes) = {0.0, 0.0};
		});
	}
}

/**
 * Job computeCellMass called @3.0 in simulate method.
 * In variables: X_EDGE_LENGTH, Y_EDGE_LENGTH, rho_n0
 * Out variables: cellMass
 */
void VnrRemap::computeCellMass() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		cellMass(cCells) = cstmesh->X_EDGE_LENGTH * cstmesh->Y_EDGE_LENGTH * rho_n0(cCells);
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
	});
}

/**
 * Job updatePosition called @3.0 in executeTimeLoopN method.
 * In variables: X_n, deltat_nplus1, u_nplus1
 * Out variables: X_nplus1
 */
void VnrRemap::updatePosition() noexcept
{
	Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes)
	{
		X_nplus1(pNodes) = X_n(pNodes) + gt->deltat_nplus1 * u_nplus1(pNodes);
	});
}

/**
 * Job SetUpTimeLoopN called @4.0 in simulate method.
 * In variables: SubVol_n0, X_n0, c_n0, cellPos_n0, deltat_init, divU_n0, e_n0, p_n0, rho_n0, tau_n0, u_n0
 * Out variables: SubVol_n, X_n, c_n, cellPos_n, deltat_n, divU_n, e_n, p_n, rho_n, tau_n, u_n
 */
void VnrRemap::setUpTimeLoopN() noexcept
{
	gt->deltat_n = gt->deltat_init;
	deep_copy(X_n, X_n0);
	deep_copy(SubVol_n, SubVol_n0);
	deep_copy(rho_n, rho_n0);
	deep_copy(p_n, p_n0);
	deep_copy(tau_n, tau_n0);
	deep_copy(divU_n, divU_n0);
	deep_copy(c_n, c_n0);
	deep_copy(e_n, e_n0);
	deep_copy(u_n, u_n0);
	deep_copy(cellPos_n, cellPos_n0);
}

/**
 * Job computeNodeMass called @4.0 in simulate method.
 * In variables: cellMass
 * Out variables: m
 */
void VnrRemap::computeNodeMass() noexcept
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
 * Job computeSubVol called @4.0 in executeTimeLoopN method.
 * In variables: X_nplus1, cellPos_nplus1
 * Out variables: SubVol_nplus1
 */
void VnrRemap::computeSubVol() noexcept
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
 * Job ExecuteTimeLoopN called @5.0 in simulate method.
 * In variables: C, Q_nplus1, SubVol_n, SubVol_nplus1, X_n, X_nplus1, c_n, cellMass, cellPos_nplus1, deltat_n, deltat_nplus1, divU_n, e_n, e_nplus1, gamma, m, p_n, rho_n, rho_nplus1, t_n, tau_n, tau_nplus1, u_n, u_nplus1
 * Out variables: C, Q_nplus1, SubVol_nplus1, V, X_nplus1, c_nplus1, deltat_nplus1, divU_nplus1, e_nplus1, p_nplus1, rho_nplus1, t_nplus1, tau_nplus1, u_nplus1
 */
void VnrRemap::executeTimeLoopN() noexcept
{
	n = 0;
	bool continueLoop = true;
	do
	{
		global_timer.start();
		cpu_timer.start();
		n++;
		if (n!=1)
			std::cout << "[" << __CYAN__ << __BOLD__ << setw(3) << n << __RESET__ "] t = " << __BOLD__
				<< setiosflags(std::ios::scientific) << setprecision(8) << setw(16) << gt->t_n << __RESET__;
	
		computeArtificialViscosity(); // @1.0
		computeCornerNormal(); // @1.0
		computeDeltaT(); // @1.0
		computeNodeVolume(); // @1.0
		computeTime(); // @2.0
		updateVelocity(); // @2.0
		updatePosition(); // @3.0
		computeSubVol(); // @4.0
		updateRho(); // @5.0
		computeTau(); // @6.0
		updateEnergy(); // @6.0
		computeDivU(); // @7.0
		computeEos(); // @7.0
		
	
		// Evaluate loop condition with variables at time n
		continueLoop = (n + 1 < gt->max_time_iterations && gt->t_nplus1 < gt->final_time);
	
		if (continueLoop)
		{
			// Switch variables to prepare next iteration
			std::swap(gt->t_nplus1, gt->t_n);
			std::swap(gt->deltat_nplus1, gt->deltat_n);
			std::swap(X_nplus1, X_n);
			std::swap(SubVol_nplus1, SubVol_n);
			std::swap(rho_nplus1, rho_n);
			std::swap(p_nplus1, p_n);
			std::swap(Q_nplus1, Q_n);
			std::swap(tau_nplus1, tau_n);
			std::swap(divU_nplus1, divU_n);
			std::swap(c_nplus1, c_n);
			std::swap(e_nplus1, e_n);
			std::swap(u_nplus1, u_n);
			std::swap(cellPos_nplus1, cellPos_n);
		}
	
		cpu_timer.stop();
		global_timer.stop();
	
		// Timers display
		if (!writer.isDisabled())
			std::cout << " {CPU: " << __BLUE__ << cpu_timer.print(true) << __RESET__ ", IO: " << __BLUE__ << io_timer.print(true) << __RESET__ "} ";
		else
			std::cout << " {CPU: " << __BLUE__ << cpu_timer.print(true) << __RESET__ ", IO: " << __RED__ << "none" << __RESET__ << "} ";
		
		// Progress
		std::cout << utils::progress_bar(n, gt->max_time_iterations, gt->t_n, gt->final_time , 25);
		std::cout << __BOLD__ << __CYAN__ << utils::Timer::print(
			utils::eta(n, gt->max_time_iterations, gt->t_n, gt->final_time, gt->deltat_n, global_timer), true)
			<< __RESET__ << "\r";
		std::cout.flush();
	
		cpu_timer.reset();
		io_timer.reset();
	} while (continueLoop);
	// force a last output at the end
	dumpVariables();
}

/**
 * Job updateRho called @5.0 in executeTimeLoopN method.
 * In variables: SubVol_nplus1, cellMass
 * Out variables: rho_nplus1
 */
void VnrRemap::updateRho() noexcept
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
		rho_nplus1(cCells) = cellMass(cCells) / reduction0;
	});
}

/**
 * Job computeTau called @6.0 in executeTimeLoopN method.
 * In variables: rho_n, rho_nplus1
 * Out variables: tau_nplus1
 */
void VnrRemap::computeTau() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		tau_nplus1(cCells) = 0.5 * (1.0 / rho_nplus1(cCells) + 1.0 / rho_n(cCells));
	});
}

/**
 * Job updateEnergy called @6.0 in executeTimeLoopN method.
 * In variables: Q_nplus1, e_n, gamma, p_n, rho_n, rho_nplus1
 * Out variables: e_nplus1
 */
void VnrRemap::updateEnergy() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
	  const double den(1 + 0.5 * (eos->gamma - 1.0) * rho_nplus1(cCells) * (1.0 / rho_nplus1(cCells) - 1.0 / rho_n(cCells)));
		const double num(e_n(cCells) - (0.5 * p_n(cCells) + Q_nplus1(cCells)) * (1.0 / rho_nplus1(cCells) - 1.0 / rho_n(cCells)));
		e_nplus1(cCells) = num / den;
	});
}

/**
 * Job computeDivU called @7.0 in executeTimeLoopN method.
 * In variables: deltat_nplus1, rho_n, rho_nplus1, tau_nplus1
 * Out variables: divU_nplus1
 */
void VnrRemap::computeDivU() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		divU_nplus1(cCells) = 1.0 / gt->deltat_nplus1 * (1.0 / rho_nplus1(cCells) - 1.0 / rho_n(cCells)) / tau_nplus1(cCells);
	});
}

/**
 * Job computeEos called @7.0 in executeTimeLoopN method.
 * In variables: e_nplus1, gamma, rho_nplus1
 * Out variables: c_nplus1, p_nplus1
 */
void VnrRemap::computeEos() noexcept
{
	Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells)
	{
		p_nplus1(cCells) = (eos->gamma - 1.0) * rho_nplus1(cCells) * e_nplus1(cCells);
		c_nplus1(cCells) = std::sqrt(eos->gamma * (eos->gamma - 1.0) * e_nplus1(cCells));
	});
}

void VnrRemap::dumpVariables() noexcept {
  nbCalls++;
  if (!writer.isDisabled() &&
      (gt->t_n >= lastDump + gt->output_time || gt->t_n == 0.)) {
    cpu_timer.stop();
    io_timer.start();
    std::map<string, double*> cellVariables;
    std::map<string, double*> nodeVariables;
    std::map<string, double*> partVariables;
    cellVariables.insert(pair<string, double*>("Pressure", p_n.data()));
    cellVariables.insert(pair<string, double*>("Density", rho_n.data()));
    auto quads = mesh->getGeometry()->getQuads();
    writer.writeFile(nbCalls, gt->t_n, nbNodes, X_n.data(), nbCells, quads.data(),
                     cellVariables, nodeVariables);
    lastDump = gt->t_n;
    std::cout << " time = " << gt->t_n << " sortie demandée " << std::endl;
    io_timer.stop();
    cpu_timer.start();
  }
}

void VnrRemap::simulate()
{
	std::cout << "\n" << __BLUE_BKG__ << __YELLOW__ << __BOLD__ <<"\tStarting VnrRemap ..." << __RESET__ << "\n\n";
	
	std::cout << "[" << __GREEN__ << "MESH" << __RESET__ << "]      X=" << __BOLD__ << cstmesh->X_EDGE_ELEMS << __RESET__ << ", Y=" << __BOLD__ << cstmesh->Y_EDGE_ELEMS
		<< __RESET__ << ", X length=" << __BOLD__ << cstmesh->X_EDGE_LENGTH << __RESET__ << ", Y length=" << __BOLD__ << cstmesh->Y_EDGE_LENGTH << __RESET__ << std::endl;
	
	if (Kokkos::hwloc::available())
	{
		std::cout << "[" << __GREEN__ << "TOPOLOGY" << __RESET__ << "]  NUMA=" << __BOLD__ << Kokkos::hwloc::get_available_numa_count()
			<< __RESET__ << ", Cores/NUMA=" << __BOLD__ << Kokkos::hwloc::get_available_cores_per_numa()
			<< __RESET__ << ", Threads/Core=" << __BOLD__ << Kokkos::hwloc::get_available_threads_per_core() << __RESET__ << std::endl;
	}
	else
	{
		std::cout << "[" << __GREEN__ << "TOPOLOGY" << __RESET__ << "]  HWLOC unavailable cannot get topological informations" << std::endl;
	}
	
	// std::cout << "[" << __GREEN__ << "KOKKOS" << __RESET__ << "]    " << __BOLD__ << (is_same<MyLayout,Kokkos::LayoutLeft>::value?"Left":"Right")" << __RESET__ << " layout" << std::endl;
	
	if (!writer.isDisabled())
		std::cout << "[" << __GREEN__ << "OUTPUT" << __RESET__ << "]    VTK files stored in " << __BOLD__ << writer.outputDirectory() << __RESET__ << " directory" << std::endl;
	else
		std::cout << "[" << __GREEN__ << "OUTPUT" << __RESET__ << "]    " << __BOLD__ << "Disabled" << __RESET__ << std::endl;

	initCellPos(); // @1.0
	init(); // @2.0
	initSubVol(); // @2.0
	computeCellMass(); // @3.0
	initDeltaT(); // @3.0
	initInternalEnergy(); // @3.0
	initPseudo(); // @3.0
	setUpTimeLoopN(); // @4.0
	computeNodeMass(); // @4.0
	executeTimeLoopN(); // @5.0
	
	std::cout << __YELLOW__ << "\n\tDone ! Took " << __MAGENTA__ << __BOLD__ << global_timer.print() << __RESET__ << std::endl;
}
