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

void Remap::computeDualUremap2() noexcept {
  int nbmat = options->nbmat;
  RealArray1D<dim> exy = xThenYToDirection(varlp->x_then_y_n);
  // calcul des flux de masses partielles
  
  if (varlp->x_then_y_n) {
    // projection verticale
    Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes)
      {
	int cfCell1(-1), cbCell1(-1), cfCell2(-1), cbCell2(-1);
	int fOfcfCell1(-1), ftOfcbCell1(-1), fOfcbCell1(-1), fbOfcfCell1(-1);
	int fOfcfCell2(-1), ftOfcbCell2(-1), fOfcbCell2(-1), fbOfcfCell2(-1);
	int fFace1(-1),fbFace1(-1),ftFace1(-1),fFace2(-1),fbFace2(-1),ftFace2(-1);
	int nbfaces(0);
	// projection verticale
	if (HorizontalFaceOfNode(pNodes)[0] !=-1) {
	  // attention backCell est au dessus
	  // attention FrontCell est en dessus
	  // pour la premiere face horizontale
	  size_t fId1(HorizontalFaceOfNode(pNodes)[0]);
	  fFace1 = utils::indexOf(mesh->getFaces(), fId1);  
     
	  // on recupere la cellule au dessus
	  int cbBackCellF1(mesh->getBackCell(fId1));
	  int cbId1(cbBackCellF1);
	  cbCell1 = cbId1;
	  // on recupere la face au dessus de la cellule au dessus
	  int ftTopFaceOfCbCell1(mesh->getTopFaceOfCell(cbId1));
	  size_t ftId1(ftTopFaceOfCbCell1);
	  ftFace1 = utils::indexOf(mesh->getFaces(), ftId1);

	  fOfcbCell1 = utils::indexOf(mesh->getFacesOfCell(cbId1), fId1); 
          ftOfcbCell1 = utils::indexOf(mesh->getFacesOfCell(cbId1), ftId1);
	  
	  // on recupere la cellule en dessous 
	  int cfFrontCellF1(mesh->getFrontCell(fId1));
	  int cfId1(cfFrontCellF1);
	  cfCell1 = cfId1;
	  // on recupere la face en dessous de la cellule en dessous
	  int fbBottomFaceOfCell1(mesh->getBottomFaceOfCell(cfId1));
	  size_t fbId1(fbBottomFaceOfCell1);
	  fbFace1 = utils::indexOf(mesh->getFaces(), fbId1);
	  
          fOfcfCell1 = utils::indexOf(mesh->getFacesOfCell(cfId1), fId1); 
          fbOfcfCell1 = utils::indexOf(mesh->getFacesOfCell(cfId1), fbId1); 

	  nbfaces = nbfaces + 2;
	}
      
	if (HorizontalFaceOfNode(pNodes)[1] !=-1) {
	  // attention backCell est au dessus
	  // attention FrontCell est en dessus
	// pour la seconde face horizontale
	size_t fId2(HorizontalFaceOfNode(pNodes)[1]);
	fFace2 = utils::indexOf(mesh->getFaces(), fId2);
      
	// on recupere la cellule au dessus
	int cbBackCellF2(mesh->getBackCell(fId2));
	int cbId2(cbBackCellF2);
	cbCell2 = cbId2;
	// on recupere la face au dessus de la cellule au dessus
	int ftTopFaceOfCfCells2(mesh->getTopFaceOfCell(cbId2));
	size_t ftId2(ftTopFaceOfCfCells2);
	ftFace2 = utils::indexOf(mesh->getFaces(), ftId2);
	  
	fOfcbCell2 = utils::indexOf(mesh->getFacesOfCell(cbId2), fId2); 
	ftOfcbCell2 = utils::indexOf(mesh->getFacesOfCell(cbId2), ftId2); 
      
	// on recupere la cellule en dessous "cf"
	int cfFrontCellF2(mesh->getFrontCell(fId2));
	int cfId2(cfFrontCellF2);
	cfCell2 = cfId2;
	// on recupere la face en dessous de la cellule en dessous
	int fbBottomFaceOfCell2(mesh->getBottomFaceOfCell(cfId2));
	size_t fbId2(fbBottomFaceOfCell2);
	fbFace2 = utils::indexOf(mesh->getFaces(), fbId2);

	fOfcfCell2 = utils::indexOf(mesh->getFacesOfCell(cfId2), fId2); 
	fbOfcfCell2 = utils::indexOf(mesh->getFacesOfCell(cfId2), fbId2); 

	nbfaces = nbfaces + 2;
	}
      
	// on prend moyenne les flux de masses (nbmat + imat)
	// des 4 faces verticales des 2 mailles à droite du noeud
	// ou
	// des 4 faces verticales des 2 mailles à gauche du noeud
	TopFluxMasse(pNodes) = 0.;
	BottomFluxMasse(pNodes) = 0;
	for (int imat = 0; imat < nbmat; imat++) {
	  TopFluxMassePartielle(pNodes)[imat] = 0.;
	  BottomFluxMassePartielle(pNodes)[imat] = 0;

	  if (HorizontalFaceOfNode(pNodes)[0] !=-1) {
	     TopFluxMassePartielle(pNodes)[imat] += 
	      (FluxFace2(cbCell1, fOfcbCell1)[nbmat+imat] * varlp->outerFaceNormal(cbCell1, fOfcbCell1)[1]
	       + FluxFace2(cbCell1, ftOfcbCell1)[nbmat+imat] * varlp->outerFaceNormal(cbCell1, ftOfcbCell1)[1]);
	    BottomFluxMassePartielle(pNodes)[imat] +=
	      (FluxFace2(cfCell1, fOfcfCell1)[nbmat+imat] * varlp->outerFaceNormal(cfCell1, fOfcfCell1)[1]
	       + FluxFace2(cfCell1, fbOfcfCell1)[nbmat+imat] * varlp->outerFaceNormal(cfCell1, fOfcfCell1)[1]);
	  }
	  if (HorizontalFaceOfNode(pNodes)[1] !=-1) {
	    TopFluxMassePartielle(pNodes)[imat] += 
	      (FluxFace2(cbCell2, fOfcbCell2)[nbmat+imat] * varlp->outerFaceNormal(cbCell2, fOfcbCell2)[1]
	       + FluxFace2(cbCell2, ftOfcbCell2)[nbmat+imat]* varlp->outerFaceNormal(cbCell2, ftOfcbCell2)[1]);
	    BottomFluxMassePartielle(pNodes)[imat] +=
	      (FluxFace2(cfCell2, fOfcfCell2)[nbmat+imat] * varlp->outerFaceNormal(cfCell2, fOfcfCell2)[1]
	       + FluxFace2(cfCell2, fbOfcfCell2)[nbmat+imat]* varlp->outerFaceNormal(cfCell2, fOfcfCell2)[1]);
	  }
	  if (nbfaces !=0) {
	    TopFluxMassePartielle(pNodes)[imat] /= nbfaces;
	    BottomFluxMassePartielle(pNodes)[imat] /= nbfaces;
	  }
	  
	  TopFluxMasse(pNodes) +=  TopFluxMassePartielle(pNodes)[imat];	     
	  BottomFluxMasse(pNodes) +=  BottomFluxMassePartielle(pNodes)[imat];
	}
	varlp->UDualremap2(pNodes)[2] = UDualremap1(pNodes)[2] + BottomFluxMasse(pNodes) - TopFluxMasse(pNodes);
	// std::cout << " pNodes " <<  pNodes << " avant proj2V " << UDualremap1(pNodes)[2]
	// 	  << " apres proj2V " << varlp->UDualremap2(pNodes)[2] << std::endl;
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
        varlp->UDualremap2(pNodes)[0] = UDualremap1(pNodes)[0]
	  + BottomFluxMasse(pNodes) * BottomupwindVelocity(pNodes)[0]
	  - TopFluxMasse(pNodes) * TopupwindVelocity(pNodes)[0];
	
	varlp->UDualremap2(pNodes)[1] = UDualremap1(pNodes)[1]
	  + BottomFluxMasse(pNodes) * BottomupwindVelocity(pNodes)[1]
	  - TopFluxMasse(pNodes) * TopupwindVelocity(pNodes)[1];
	
	
    });
  } else {
    // projection horizontale
    Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes)
      {
	int cfCell1(-1), cbCell1(-1), cfCell2(-1), cbCell2(-1);
	int fOfcfCell1(-1), frOfcfCell1(-1), fOfcbCell1(-1), flOfcbCell1(-1);
	int fOfcfCell2(-1), frOfcfCell2(-1), fOfcbCell2(-1), flOfcbCell2(-1);
	int fFace1(-1),frFace1(-1),flFace1(-1),fFace2(-1),frFace2(-1),flFace2(-1);
	int nbfaces(0);
	if (VerticalFaceOfNode(pNodes)[0] != -1) {
	  // pour la premiere face verticale 
	  size_t fId1(VerticalFaceOfNode(pNodes)[0]);
	  fFace1 = utils::indexOf(mesh->getFaces(), fId1);      
	  // on recupere la cellule devant 
	  int cfFrontCellF1(mesh->getFrontCell(fId1));
	  int cfId1(cfFrontCellF1);
	  cfCell1 = cfId1;
	  // on recupere la face à droite de la cellule devant
	  int frRightFaceOfCfCell1(mesh->getRightFaceOfCell(cfId1));
	  size_t frId1(frRightFaceOfCfCell1);
	  frFace1 = utils::indexOf(mesh->getFaces(), frId1);
	  
          fOfcfCell1 = utils::indexOf(mesh->getFacesOfCell(cfId1), fId1); 
          frOfcfCell1 = utils::indexOf(mesh->getFacesOfCell(cfId1), frId1); 
     
	  // on recupere la cellule derriere
	  int cbBackCellF1(mesh->getBackCell(fId1));
	  int cbId1(cbBackCellF1);
	  cbCell1 = cbId1;
	  // on recupere la face à gauche de la cellule derriere
	  int flLeftFaceOfcbCell1(mesh->getLeftFaceOfCell(cbId1));
	  size_t flId1(flLeftFaceOfcbCell1);
	  flFace1 = utils::indexOf(mesh->getFaces(), flId1);
	  
	  fOfcbCell1 = utils::indexOf(mesh->getFacesOfCell(cbId1), fId1); 
	  flOfcbCell1 = utils::indexOf(mesh->getFacesOfCell(cbId1), flId1); 

	  nbfaces = nbfaces + 2;
	}
	if (VerticalFaceOfNode(pNodes)[1] != -1) {
	  // pour la premiere face verticale
	  size_t fId2(VerticalFaceOfNode(pNodes)[1]);
	  fFace2 = utils::indexOf(mesh->getFaces(), fId2);
	  // on recupere la cellule devant 
	  int cfFrontCellF2(mesh->getFrontCell(fId2));
	  int cfId2(cfFrontCellF2);
	  cfCell2 = cfId2;
	  // on recupere la face à droite de la cellule devant
	  int frRightFaceOfCfCell2(mesh->getRightFaceOfCell(cfId2));
	  size_t frId2(frRightFaceOfCfCell2);
	  frFace2 = utils::indexOf(mesh->getFaces(), frId2);
          fOfcfCell2 = utils::indexOf(mesh->getFacesOfCell(cfId2), fId2); 
          frOfcfCell2 = utils::indexOf(mesh->getFacesOfCell(cfId2), frId2);
	  
	  // on recupere la cellule derriere
	  int cbBackCellF2(mesh->getBackCell(fId2));
	  int cbId2(cbBackCellF2);
	  cbCell2 = cbId2;
	  // on recupere la face à gauche de la cellule derriere
	  int flLeftFaceOfcbCell2(mesh->getLeftFaceOfCell(cbId2));
	  size_t flId2(flLeftFaceOfcbCell2);
	  flFace2 = utils::indexOf(mesh->getFaces(), flId2);
	  
	  fOfcbCell2 = utils::indexOf(mesh->getFacesOfCell(cbId2), fId2); 
	  flOfcbCell2 = utils::indexOf(mesh->getFacesOfCell(cbId2), flId2); 

	  nbfaces = nbfaces + 2;
	}
      
	// on prend moyenne les flux de masses (nbmat + imat)
	// des 4 faces verticales des 2 mailles à droite du noeud
	// ou
	// des 4 faces verticales des 2 mailles à gauche du noeud
	RightFluxMasse(pNodes) = 0.;
	LeftFluxMasse(pNodes) = 0.;
	for (int imat = 0; imat < nbmat; imat++) {
	  RightFluxMassePartielle(pNodes)[imat] = 0;
	  LeftFluxMassePartielle(pNodes)[imat] = 0;
	  
	  if (VerticalFaceOfNode(pNodes)[0] != -1) {
	    RightFluxMassePartielle(pNodes)[imat] += 
	      (FluxFace2(cfCell1, fOfcfCell1)[nbmat+imat] * varlp->outerFaceNormal(cfCell1, fOfcfCell1)[0]
	       + FluxFace2(cfCell1, frOfcfCell1)[nbmat+imat] * varlp->outerFaceNormal(cfCell1, frOfcfCell1)[0]);
	    LeftFluxMassePartielle(pNodes)[imat] +=
	      (FluxFace2(cbCell1, fOfcbCell1)[nbmat+imat] * varlp->outerFaceNormal(cbCell1, fOfcbCell1)[0]
	       + FluxFace2(cbCell1, flOfcbCell1)[nbmat+imat] * varlp->outerFaceNormal(cbCell1, flOfcbCell1)[0]);
	  }
	  
	  if (VerticalFaceOfNode(pNodes)[1] != -1) {
	   RightFluxMassePartielle(pNodes)[imat] +=
	     (FluxFace2(cfCell2, fOfcfCell2)[nbmat+imat] * varlp->outerFaceNormal(cfCell2, fOfcfCell2)[0]
	      + FluxFace2(cfCell2, frOfcfCell2)[nbmat+imat] * varlp->outerFaceNormal(cfCell2, frOfcfCell2)[0]);
	    LeftFluxMassePartielle(pNodes)[imat] +=
	      (FluxFace2(cbCell2, fOfcbCell2)[nbmat+imat] * varlp->outerFaceNormal(cbCell2, fOfcbCell2)[0]
	       + FluxFace2(cbCell2, flOfcbCell2)[nbmat+imat] * varlp->outerFaceNormal(cbCell2, flOfcbCell2)[0]);
	  }

	  if (nbfaces !=0) {
	    RightFluxMassePartielle(pNodes)[imat] /= nbfaces;
	    LeftFluxMassePartielle(pNodes)[imat] /= nbfaces;
	  }
	  
	  RightFluxMasse(pNodes) +=  RightFluxMassePartielle(pNodes)[imat];  
	  LeftFluxMasse(pNodes) +=  LeftFluxMassePartielle(pNodes)[imat];
	}
	varlp->UDualremap2(pNodes)[2] = UDualremap1(pNodes)[2] + LeftFluxMasse(pNodes) - RightFluxMasse(pNodes);
	// std::cout << " pNodes " <<  pNodes << " avant proj2H " << UDualremap1(pNodes)[2]
	//	  << " apres proj2H " << varlp->UDualremap2(pNodes)[2] << std::endl;
	if (options->projectionOrder == 1) {
	  // recherche de la vitesse du decentrement upwind
	  // Rightvitesse = vitesse(pNodes) si RightFluxMasse(pNodes) > 0 et vitesse(voisin de droite) sinon
	  // Leftvitesse = vitesse(voisin de gauche) si LeftFluxMasse(pNodes) > 0 et vitesse(pNodes) sinon
	  int LeftNode = mesh->getLeftNode(pNodes);
	  int RightNode = mesh->getRightNode(pNodes);
	  if (RightFluxMasse(pNodes) < 0) {
	    RightupwindVelocity(pNodes)[0] = varlp->DualPhi(RightNode)[0];
	    RightupwindVelocity(pNodes)[1] = varlp->DualPhi(RightNode)[1];
	  }  else {
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
	varlp->UDualremap2(pNodes)[0] = UDualremap1(pNodes)[0]
	  + LeftFluxMasse(pNodes) * LeftupwindVelocity(pNodes)[0]
	  - RightFluxMasse(pNodes) * RightupwindVelocity(pNodes)[0];
	  
	varlp->UDualremap2(pNodes)[1] = UDualremap1(pNodes)[1]
	  + LeftFluxMasse(pNodes) * LeftupwindVelocity(pNodes)[1]
	  - RightFluxMasse(pNodes) * RightupwindVelocity(pNodes)[1];

	  //if ((pNodes == 30) || (pNodes == 31) || (pNodes == 32) )
	  // std::cout << gt->deltat_n << " pNodes " <<  pNodes << " sortie 2 remaillage vitesse "
	  // 	    << UDualremap1(pNodes)[0] / UDualremap1(pNodes)[2]
	  // 	    << " quant "
	  // 	    << UDualremap1(pNodes)[0] << "  "
	  // 	    << varlp->UDualremap2(pNodes)[0] << " lfm " << LeftFluxMasse(pNodes)
	  // 	    << " lv " <<  LeftupwindVelocity(pNodes)[0]
	  // 	    << " rfm " << RightFluxMasse(pNodes) << " rv " << RightupwindVelocity(pNodes)[0]
	  // 	    << std::endl;
    });
  }
}
	    
			     
