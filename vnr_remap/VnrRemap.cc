#include "Vnr.h"

using namespace nablalib;

#include "../includes/Freefunctions.h"
#include <iomanip>          // for operator<<, setw, setiosflags
#include "utils/Utils.h"          // for __RESET__, __BOLD__, __GREEN__


/**
 * Job computeDeltaT called @1.0 in executeTimeLoopN method.
 * In variables: SubVol_n, c_n, deltat_n
 * Out variables: deltat_nplus1
 */
void Vnr::computeDeltaT() noexcept
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
 * Job computeTime called @2.0 in executeTimeLoopN method.
 * In variables: deltat_nplus1, t_n
 * Out variables: t_nplus1
 */
void Vnr::computeTime() noexcept
{
	gt->t_nplus1 = gt->t_n + gt->deltat_nplus1;
}




/**
 * Job SetUpTimeLoopN called @4.0 in simulate method.
 * In variables: SubVol_n0, X_n0, c_n0, cellPos_n0, deltat_init, divU_n0, e_n0, p_n0, rho_n0, tau_n0, u_n0
 * Out variables: SubVol_n, X_n, c_n, cellPos_n, deltat_n, divU_n, e_n, p_n, rho_n, tau_n, u_n
 */
void Vnr::setUpTimeLoopN() noexcept
{
	gt->deltat_n = gt->deltat_init;
	deep_copy(X_n, X_n0);
	deep_copy(SubVol_n, SubVol_n0);
	deep_copy(rho_n, rho_n0);
	deep_copy(p_n, p_n0);
	deep_copy(Q_n, Q_n0);
	deep_copy(tau_n, tau_n0);
	deep_copy(divU_n, divU_n0);
	deep_copy(c_n, c_n0);
	deep_copy(e_n, e_n0);
	deep_copy(u_n, u_n0);
	deep_copy(cellPos_n, cellPos_n0);
}

/**
 * Job ExecuteTimeLoopN called @5.0 in simulate method.
 * In variables: C, Q_nplus1, SubVol_n, SubVol_nplus1, X_n, X_nplus1, c_n, cellMass, cellPos_nplus1, deltat_n, deltat_nplus1, divU_n, e_n, e_nplus1, gamma, m, p_n, rho_n, rho_nplus1, t_n, tau_n, tau_nplus1, u_n, u_nplus1
 * Out variables: C, Q_nplus1, SubVol_nplus1, V, X_nplus1, c_nplus1, deltat_nplus1, divU_nplus1, e_nplus1, p_nplus1, rho_nplus1, t_nplus1, tau_nplus1, u_nplus1
 */
void Vnr::executeTimeLoopN() noexcept
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
	
		computeCornerNormal(); // @1.0
		computeDeltaT(); // @1.0
		computeNodeVolume(); // @1.0
		computeTime(); // @2.0
		dumpVariables();  // @2.0
		updateVelocity(); // @2.0
		updateVelocityBoundaryConditions(); // @2.0
		updatePosition(); // @3.0
		computeSubVol(); // @4.0
		updateRho(); // @5.0
		computeTau(); // @6.0
		computeDivU(); // @7.0
		computeArtificialViscosity(); // @1.0
		updateEnergy(); // @6.0
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

void Vnr::dumpVariables() noexcept {
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

void Vnr::simulate()
{
	std::cout << "\n" << __BLUE_BKG__ << __YELLOW__ << __BOLD__ <<"\tStarting Vnr ..." << __RESET__ << "\n\n";
	
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
	initBoundaryConditions();
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
