#include <math.h>  // for sqrt

#include <Kokkos_Core.hpp>
#include <algorithm>  // for copy
#include <array>      // for array
#include <iostream>   // for operator<<, basic_ostream::operat...
#include <vector>     // for allocator, vector

#include "Remap.h"          // for Remap, Remap::Opt...
#include "UtilesRemap-Impl.h"       // for Remap::computeFluxPP
#include "mesh/CartesianMesh2D.h"   // for CartesianMesh2D
#include "types/MathFunctions.h"    // for dot
#include "types/MultiArray.h"       // for operator<<
#include "utils/Utils.h"            // for indexOf

#include "../includes/VariablesLagRemap.h"

void Remap::computeDualUremap1() noexcept {
  int nbmat = options->nbmat;
  RealArray1D<dim> exy = xThenYToDirection(varlp->x_then_y_n);
  if (varlp->x_then_y_n) {
    // projection horizontale
    Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes)
      {
	getRightAndLeftFluxMasse1(nbmat, pNodes);
	UDualremap1(pNodes)[2] = varlp->UDualLagrange(pNodes)[2] + LeftFluxMasse(pNodes) - RightFluxMasse(pNodes);

	if (options->projectionOrder == 1) {
	  // recherche de la vitesse du decentrement upwind
	  // Rightvitesse = vitesse(pNodes) si RightFluxMasse(pNodes) > 0 et vitesse(voisin de droite) sinon
	  // Leftvitesse = vitesse(voisin de gauche) si LeftFluxMasse(pNodes) > 0 et vitesse(pNodes) sinon
	  int LeftNode = mesh->getLeftNode(pNodes);
	  int RightNode = mesh->getRightNode(pNodes);
	  
	  if (RightFluxMasse(pNodes) < 0) {
	    RightupwindVelocity(pNodes)[0] = varlp->DualPhi(RightNode)[0];
	    RightupwindVelocity(pNodes)[1] = varlp->DualPhi(RightNode)[1];
	  } else {
	    RightupwindVelocity(pNodes)[0] = varlp->DualPhi(pNodes)[0];
	    RightupwindVelocity(pNodes)[1] = varlp->DualPhi(pNodes)[1];
	  }	    
	  if (LeftFluxMasse(pNodes) > 0) {
	    LeftupwindVelocity(pNodes)[0] = varlp->DualPhi(LeftNode)[0];
	    LeftupwindVelocity(pNodes)[1] = varlp->DualPhi(LeftNode)[1];
	  } else {
	    LeftupwindVelocity(pNodes)[0] = varlp->DualPhi(pNodes)[0];
	    LeftupwindVelocity(pNodes)[1] = varlp->DualPhi(pNodes)[1];
	  }	    
	}
   
	UDualremap1(pNodes)[0] = varlp->UDualLagrange(pNodes)[0]
	  + LeftFluxMasse(pNodes) * LeftupwindVelocity(pNodes)[0]
	  - RightFluxMasse(pNodes) * RightupwindVelocity(pNodes)[0];
	
	UDualremap1(pNodes)[1] = varlp->UDualLagrange(pNodes)[1]
	   + LeftFluxMasse(pNodes) * LeftupwindVelocity(pNodes)[1]
	   - RightFluxMasse(pNodes) * RightupwindVelocity(pNodes)[1];

      });
  } else {
    // projection verticale
    Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes)
      {
	getTopAndBottomFluxMasse1(nbmat, pNodes);
	UDualremap1(pNodes)[2] = varlp->UDualLagrange(pNodes)[2] + BottomFluxMasse(pNodes) - TopFluxMasse(pNodes);
	if (options->projectionOrder == 1) {
	  // recherche de la vitesse du decentrement upwind
	  // Topvitesse = vitesse(pNodes) si TopFluxMasse(pNodes) > 0 et vitesse(voisin du haut) sinon
	  // Bottomvitesse = vitesse(voisin du bas) si BottomFluxMasse(pNodes) > 0 et vitesse(pNodes) sinon
	  int TopNode = mesh->getTopNode(pNodes);
	  int BottomNode = mesh->getBottomNode(pNodes);
	  if (TopFluxMasse(pNodes) < 0) {
	    TopupwindVelocity(pNodes)[0] = varlp->DualPhi(TopNode)[0];
	    TopupwindVelocity(pNodes)[1] = varlp->DualPhi(TopNode)[1];
	  } else {
	    TopupwindVelocity(pNodes)[0] = varlp->DualPhi(pNodes)[0];
	    TopupwindVelocity(pNodes)[1] = varlp->DualPhi(pNodes)[1];
	  }	    
	  if (BottomFluxMasse(pNodes) > 0) {
	    BottomupwindVelocity(pNodes)[0] = varlp->DualPhi(BottomNode)[0];
	    BottomupwindVelocity(pNodes)[1] = varlp->DualPhi(BottomNode)[1];
	  } else {
	    BottomupwindVelocity(pNodes)[0] = varlp->DualPhi(pNodes)[0];
	    BottomupwindVelocity(pNodes)[1] = varlp->DualPhi(pNodes)[1];
	  }
	}
	UDualremap1(pNodes)[0] = varlp->UDualLagrange(pNodes)[0]
	  + BottomFluxMasse(pNodes) * BottomupwindVelocity(pNodes)[0]
	  - TopFluxMasse(pNodes) * TopupwindVelocity(pNodes)[0];
	
	UDualremap1(pNodes)[1] = varlp->UDualLagrange(pNodes)[1]
	  + BottomFluxMasse(pNodes) * BottomupwindVelocity(pNodes)[1]
	  - TopFluxMasse(pNodes) * TopupwindVelocity(pNodes)[1];

    });
  }
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes)
    {
      // vitesse x Y
      varlp->DualPhi(pNodes)[0] = UDualremap1(pNodes)[0] / UDualremap1(pNodes)[2];
      varlp->DualPhi(pNodes)[1] = UDualremap1(pNodes)[1] / UDualremap1(pNodes)[2];
      // energie cinétique
      varlp->DualPhi(pNodes)[3] = UDualremap1(pNodes)[3] / UDualremap1(pNodes)[2];
      
      // if ((pNodes == 30) || (pNodes == 31) || (pNodes == 32)) 
      // std::cout << " pNodes " <<  pNodes << " sortie 1 remaillage "
      // 		<<   UDualremap1(pNodes)[0] / UDualremap1(pNodes)[2]
      // 		<< "   " << UDualremap1(pNodes)[0] / UDualremap1(pNodes)[2]
      // 		<< "   " << UDualremap1(pNodes)[2] << std::endl;
		  
    });
}
	    
			     
