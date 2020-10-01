#include <math.h>    // for fabs
#include <stdlib.h>  // for abs

#include <array>  // for array

#include "Remap.h"        // for Remap, Remap::Options
#include "types/MathFunctions.h"  // for min, max
#include "utils/Utils.h"            // for indexOf

void Remap::getRightAndLeftFluxMasse1(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces à gauche et à droite
#include "FacesLeftAndRight.h"
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
	(FluxFace1(cfCell1, fOfcfCell1)[nbmat+imat] * varlp->outerFaceNormal(cfCell1, fOfcfCell1)[0]
	 + FluxFace1(cfCell1, frOfcfCell1)[nbmat+imat] * varlp->outerFaceNormal(cfCell1, frOfcfCell1)[0]);
      LeftFluxMassePartielle(pNodes)[imat] +=
	(FluxFace1(cbCell1, fOfcbCell1)[nbmat+imat] * varlp->outerFaceNormal(cbCell1, fOfcbCell1)[0]
	 + FluxFace1(cbCell1, flOfcbCell1)[nbmat+imat] * varlp->outerFaceNormal(cbCell1, flOfcbCell1)[0]);	      
    }
	  
    if (VerticalFaceOfNode(pNodes)[1] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
	(FluxFace1(cfCell2, fOfcfCell2)[nbmat+imat] * varlp->outerFaceNormal(cfCell2, fOfcfCell2)[0]
	 + FluxFace1(cfCell2, frOfcfCell2)[nbmat+imat] * varlp->outerFaceNormal(cfCell2, frOfcfCell2)[0]);
      LeftFluxMassePartielle(pNodes)[imat] +=
	(FluxFace1(cbCell2, fOfcbCell2)[nbmat+imat] * varlp->outerFaceNormal(cbCell2, fOfcbCell2)[0]
	 + FluxFace1(cbCell2, flOfcbCell2)[nbmat+imat] * varlp->outerFaceNormal(cbCell2, flOfcbCell2)[0]);
    }
	    
    if (nbfaces !=0) {
      RightFluxMassePartielle(pNodes)[imat] /= nbfaces;
      LeftFluxMassePartielle(pNodes)[imat] /= nbfaces;
    }
	  
    RightFluxMasse(pNodes) +=  RightFluxMassePartielle(pNodes)[imat];  
    LeftFluxMasse(pNodes) +=  LeftFluxMassePartielle(pNodes)[imat];
  }
}
void Remap::getRightAndLeftFluxMasse2(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces à gauche et à droite
#include "FacesLeftAndRight.h"
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
}

void Remap::getTopAndBottomFluxMasse1(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces dessus et dessous
#include "FacesTopAndBottom.h"
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
	(FluxFace1(cbCell1, fOfcbCell1)[nbmat+imat] * varlp->outerFaceNormal(cbCell1, fOfcbCell1)[1]
	 + FluxFace1(cbCell1, ftOfcbCell1)[nbmat+imat] * varlp->outerFaceNormal(cbCell1, ftOfcbCell1)[1]);
      BottomFluxMassePartielle(pNodes)[imat] +=
	(FluxFace1(cfCell1, fOfcfCell1)[nbmat+imat] * varlp->outerFaceNormal(cfCell1, fOfcfCell1)[1]
	 + FluxFace1(cfCell1, fbOfcfCell1)[nbmat+imat] * varlp->outerFaceNormal(cfCell1, fbOfcfCell1)[1]);
    }
    if (HorizontalFaceOfNode(pNodes)[1] !=-1) {
      TopFluxMassePartielle(pNodes)[imat] += 
	(FluxFace1(cbCell2, fOfcbCell2)[nbmat+imat]  * varlp->outerFaceNormal(cbCell2, fOfcbCell2)[1]
	 + FluxFace1(cbCell2, ftOfcbCell2)[nbmat+imat] * varlp->outerFaceNormal(cbCell2, ftOfcbCell2)[1]);
      BottomFluxMassePartielle(pNodes)[imat] +=
	(FluxFace1(cfCell2, fOfcfCell2)[nbmat+imat] * varlp->outerFaceNormal(cfCell2, fOfcfCell2)[1]
	 + FluxFace1(cfCell2, fbOfcfCell2)[nbmat+imat] * varlp->outerFaceNormal(cfCell2, fbOfcfCell2)[1]);
    }
    if (nbfaces !=0) {
      TopFluxMassePartielle(pNodes)[imat] /= nbfaces;
      BottomFluxMassePartielle(pNodes)[imat] /= nbfaces;
    }
	  
    TopFluxMasse(pNodes) +=  TopFluxMassePartielle(pNodes)[imat];	     
    BottomFluxMasse(pNodes) +=  BottomFluxMassePartielle(pNodes)[imat];
  }
}
void Remap::getTopAndBottomFluxMasse2(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces dessus et dessous
#include "FacesTopAndBottom.h"
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
	 + FluxFace2(cfCell1, fbOfcfCell1)[nbmat+imat] * varlp->outerFaceNormal(cfCell1, fbOfcfCell1)[1]);
    }
    if (HorizontalFaceOfNode(pNodes)[1] !=-1) {
      TopFluxMassePartielle(pNodes)[imat] += 
	(FluxFace2(cbCell2, fOfcbCell2)[nbmat+imat] * varlp->outerFaceNormal(cbCell2, fOfcbCell2)[1]
	 + FluxFace2(cbCell2, ftOfcbCell2)[nbmat+imat]* varlp->outerFaceNormal(cbCell2, ftOfcbCell2)[1]);
      BottomFluxMassePartielle(pNodes)[imat] +=
	(FluxFace2(cfCell2, fOfcfCell2)[nbmat+imat] * varlp->outerFaceNormal(cfCell2, fOfcfCell2)[1]
	 + FluxFace2(cfCell2, fbOfcfCell2)[nbmat+imat] * varlp->outerFaceNormal(cfCell2, fbOfcfCell2)[1]);
    }
    if (nbfaces !=0) {
      TopFluxMassePartielle(pNodes)[imat] /= nbfaces;
      BottomFluxMassePartielle(pNodes)[imat] /= nbfaces;
    }
	  
    TopFluxMasse(pNodes) +=  TopFluxMassePartielle(pNodes)[imat];	     
    BottomFluxMasse(pNodes) +=  BottomFluxMassePartielle(pNodes)[imat];
  }
}
