#ifndef UTILESREMAP_IMPL_H
#define UTILESREMAP_IMPL_H

#include "Remap.h"          // for Remap, Remap::Opt...
//#include "types/ArrayOperations.h"
#include "types/MathFunctions.h"

template <size_t d>
RealArray1D<d> Remap::computeDualVerticalGradPhi(RealArray1D<d> grad_top, RealArray1D<d> grad_bottom, const size_t pNode) {
  int limiter = limiteurs->projectionLimiterId;
  int TopNode = mesh->getTopNode(pNode);
  int BottomNode = mesh->getBottomNode(pNode);

  if (TopNode == -1 || BottomNode == -1) return Uzero;
  
  grad_top[0] = (varlp->DualPhi(TopNode)[0] - varlp->DualPhi(pNode)[0]) /
    (varlp->XLagrange(TopNode)[1] - varlp->XLagrange(pNode)[1]);
  
  grad_bottom[0] = (varlp->DualPhi(pNode)[0] - varlp->DualPhi(BottomNode)[0]) /
    (varlp->XLagrange(pNode)[1] - varlp->XLagrange(BottomNode)[1]);
  
  grad_top[1] = (varlp->DualPhi(TopNode)[1] - varlp->DualPhi(pNode)[1]) /
    (varlp->XLagrange(TopNode)[1] - varlp->XLagrange(pNode)[1]);
  
  grad_bottom[1] = (varlp->DualPhi(pNode)[1] - varlp->DualPhi(BottomNode)[1]) /
    (varlp->XLagrange(pNode)[1] - varlp->XLagrange(BottomNode)[1]);
  
  int BottomBottomNode = mesh->getBottomNode(BottomNode);
  int TopTopNode = mesh->getTopNode(TopNode);
  // largeurs des mailles duales
  double hmoins, h0, hplus;
  h0 = 0.5 * (varlp->XLagrange(TopNode)[1] - varlp->XLagrange(BottomNode)[1]);
  if (BottomBottomNode == -1) {
    hmoins = 0.;
    hplus = 0.5 * (varlp->XLagrange(TopTopNode)[1] - varlp->XLagrange(pNode)[1]);
  } else if (TopTopNode == -1) {
    hplus = 0.;
    hmoins = 0.5 * (varlp->XLagrange(pNode)[1] - varlp->XLagrange(BottomBottomNode)[1]);
  } else {	
    hmoins = 0.5 * (varlp->XLagrange(pNode)[1] - varlp->XLagrange(BottomBottomNode)[1]);
    hplus = 0.5 * (varlp->XLagrange(TopTopNode)[1] - varlp->XLagrange(pNode)[1]);
  }
  
  RealArray1D<d> res;
  res = computeAndLimitGradPhi(
	  limiter, grad_top, grad_bottom,
	  varlp->DualPhi(pNode), varlp->DualPhi(TopNode),  varlp->DualPhi(BottomNode),
	  h0, hplus, hmoins);
  // std::cout << " Nodes " << pNode << " gradV " << res << std::endl;
  //std::cout << " Nodes " << BottomBottomNode << " " << BottomNode << " " <<
  //   pNode << " " <<  TopNode << " " << TopTopNode << std::endl;
  return res;
}
template <size_t d>
RealArray1D<d> Remap::computeDualHorizontalGradPhi(RealArray1D<d> grad_right, RealArray1D<d> grad_left, const size_t pNode) {
  int limiter = limiteurs->projectionLimiterId;
  int LeftNode = mesh->getLeftNode(pNode);
  int RightNode = mesh->getRightNode(pNode);
  
  if (LeftNode == -1 || RightNode == -1) return Uzero;
  
  grad_right[0] = (varlp->DualPhi(RightNode)[0] - varlp->DualPhi(pNode)[0]) /
    (varlp->XLagrange(RightNode)[0] - varlp->XLagrange(pNode)[0]);
  
  grad_left[0] = (varlp->DualPhi(pNode)[0] - varlp->DualPhi(LeftNode)[0]) /
    (varlp->XLagrange(pNode)[0] - varlp->XLagrange(LeftNode)[0]);
  
  grad_right[1] = (varlp->DualPhi(RightNode)[1] - varlp->DualPhi(pNode)[1]) /
    (varlp->XLagrange(RightNode)[0] - varlp->XLagrange(pNode)[0]);
  
  grad_left[1] = (varlp->DualPhi(pNode)[1] - varlp->DualPhi(LeftNode)[1]) /
    (varlp->XLagrange(pNode)[0] - varlp->XLagrange(LeftNode)[0]);
  
  int LeftLeftNode = mesh->getLeftNode(LeftNode);
  int RightRightNode = mesh->getRightNode(RightNode);
  // largeurs des mailles duales
  double hmoins, h0, hplus;
  h0 = 0.5 * (varlp->XLagrange(RightNode)[0] - varlp->XLagrange(LeftNode)[0]);
  if (LeftLeftNode == -1) {
    hmoins = 0.;
    hplus = 0.5 * (varlp->XLagrange(RightRightNode)[0] - varlp->XLagrange(pNode)[0]);
  } else if (RightRightNode == -1) {
    hplus = 0.;
    hmoins = 0.5 * (varlp->XLagrange(pNode)[0] - varlp->XLagrange(LeftLeftNode)[0]);
  } else {	
    hmoins = 0.5 * (varlp->XLagrange(pNode)[0] - varlp->XLagrange(LeftLeftNode)[0]);
    hplus = 0.5 * (varlp->XLagrange(RightRightNode)[0] - varlp->XLagrange(pNode)[0]);
  }
  
  RealArray1D<d> res;
  res = computeAndLimitGradPhi(
	  limiter, grad_right, grad_left,
	  varlp->DualPhi(pNode), varlp->DualPhi(RightNode),  varlp->DualPhi(LeftNode),
	  h0, hplus, hmoins);
  // std::cout << " Nodes " << pNode << " gradH " << res << std::endl;
  return res;
}

template <size_t d>
RealArray1D<d> Remap::computeAndLimitGradPhi(
    int projectionLimiterId, RealArray1D<d> gradphiplus,
    RealArray1D<d> gradphimoins, RealArray1D<d> phi, RealArray1D<d> phiplus,
    RealArray1D<d> phimoins, double h0, double hplus, double hmoins) {
  RealArray1D<d> res;
  if (projectionLimiterId < limiteurs->minmodG) {
    // std::cout << " Passage gradient limite Classique " << std::endl;
    for (size_t i = 0; i < d; i++) {
      res[i] = (fluxLimiter(projectionLimiterId,
                            divideNoExcept(gradphiplus[i], gradphimoins[i])) *
                    gradphimoins[i] +
                fluxLimiter(projectionLimiterId,
                            divideNoExcept(gradphimoins[i], gradphiplus[i])) *
                    gradphiplus[i]) /
               2.0;
    }
    return res;
  } else {
    // std::cout << " Passage gradient limite Genéralisé " << std::endl;
    for (size_t i = 0; i < d; i++) {
      res[i] =
          fluxLimiterPP(projectionLimiterId, gradphiplus[i], gradphimoins[i],
                        phi[i], phiplus[i], phimoins[i], h0, hplus, hmoins);
    }
    return res;
  }
}

template <size_t d>
RealArray1D<d> Remap::computeVecFluxOrdre3(
    RealArray1D<d> phimmm, RealArray1D<d> phimm, RealArray1D<d> phim,
    RealArray1D<d> phip, RealArray1D<d> phipp, RealArray1D<d> phippp,
    double hmmm, double hmm, double hm, double hp, double hpp, double hppp,
    double face_normal_velocity, double deltat_n) {
  RealArray1D<d> res;
  double v_dt = face_normal_velocity * deltat_n;
  for (size_t i = 0; i < d; i++) {
    res[i] = ComputeFluxOrdre3(phimmm[i], phimm[i], phim[i], phip[i], phipp[i],
                               phippp[i], hmmm, hmm, hm, hp, hpp, hppp, v_dt);
  }
  return res;
}
template <size_t d>
void Remap::computeFluxPP(
    RealArray1D<d> gradphi, RealArray1D<d> phi, RealArray1D<d> phiplus,
    RealArray1D<d> phimoins, double h0, double hplus, double hmoins,
    double face_normal_velocity, double deltat_n, int type, int cell,
    double flux_threhold, int projectionPlateauPenteComplet,
    double dual_normal_velocity, int calcul_flux_dual,
    RealArray1D<d> *pFlux,  RealArray1D<d> *pFlux_dual) {

  RealArray1D<d> Flux = *pFlux;
  RealArray1D<d> Flux_dual = *pFlux_dual;
  
  Flux = Uzero;
  Flux_dual = Uzero;
  int nbmat = options->nbmat;
  double y0plus, y0moins, xd, xg, yd, yg;
  double flux1, flux2, flux3, flux1m, flux2m, flux3m;
  double partie_positive_v =
      0.5 * (face_normal_velocity + abs(face_normal_velocity)) * deltat_n;
  double partie_positive_dual_v =
      0.5 * (dual_normal_velocity + abs(dual_normal_velocity)) * deltat_n;
  int cas_PP = 0;
  for (size_t i = 0; i < nbequamax; i++) {
    // calcul des seuils y0plus, y0moins pour cCells
    y0plus = computeY0(limiteurs->projectionLimiterId, phi[i], phiplus[i],
                       phimoins[i], h0, hplus, hmoins, 0);
    y0moins = computeY0(limiteurs->projectionLimiterId, phi[i], phiplus[i],
                        phimoins[i], h0, hplus, hmoins, 1);

    // calcul des points d'intersections xd,xg
    xg = computexgxd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins, 0);
    xd = computexgxd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins, 1);

    // calcul des valeurs sur ces points d'intersections
    yg = computeygyd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins,
                     gradphi[i], 0);
    yd = computeygyd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins,
                     gradphi[i], 1);


    if (type == 0)  // flux arriere ou en dessous de cCells, integration entre
                    // -h0/2. et -h0/2.+abs(face_normal_velocity)*deltat_n
    {
      // Flux1m : integrale -inf,  -h0/2.+partie_positive_v
      flux1m =
          INT2Y(-h0 / 2. + partie_positive_v, -h0 / 2., phimoins[i], xg, yg);
      // Flux1m : integrale -inf,  -h0/2.
      flux1 = INT2Y(-h0 / 2., -h0 / 2., phimoins[i], xg, yg);
      // Flux2m : integrale -inf,  -h0/2.+partie_positive_v
      flux2m = INT2Y(-h0 / 2. + partie_positive_v, xg, yg, xd, yd);
      // Flux2 : integrale -inf,  -h0/2.
      flux2 = INT2Y(-h0 / 2., xg, yg, xd, yd);
      // Flux3m : integrale -inf,  -h0/2.+partie_positive_v
      flux3m = INT2Y(-h0 / 2. + partie_positive_v, xd, yd, h0 / 2., phiplus[i]);
      // Flux3 : integrale -inf,  -h0/2.
      flux3 = INT2Y(-h0 / 2., xd, yd, h0 / 2., phiplus[i]);
      // integrale positive
      Flux[i] = MathFunctions::max(
          ((flux1m - flux1) + (flux2m - flux2) + (flux3m - flux3)), 0.);
      // formule 16
      if (((phiplus[i] - phi[i]) * (phimoins[i] - phi[i])) >= 0.)
        Flux[i] = phi[i] * partie_positive_v;
      //
      // et calcul du flux dual si calcul_flux_dual=1
      if (calcul_flux_dual == 1) {
	// Flux1m : integrale -inf, partie_positive_dual_v
	flux1m = INT2Y(partie_positive_dual_v, -h0 / 2., phimoins[i], xg, yg);
	// Flux1m : integrale -inf,  0..
	flux1 = INT2Y(0., -h0 / 2., phimoins[i], xg, yg);
	// Flux2m : integrale -inf, partie_positive_dual_v  
	flux2m = INT2Y(partie_positive_dual_v, xg, yg, xd, yd);
	// Flux2 : integrale -inf, 0.
	flux2 = INT2Y(0., xg, yg, xd, yd);
	// Flux3m : integrale -inf, partie_positive_dual_v  
	flux3m = INT2Y(partie_positive_dual_v, xd, yd, h0 / 2., phiplus[i]);
	// Flux3 : integrale -inf, 0.
	flux3 = INT2Y(0., xd, yd, h0 / 2., phiplus[i]);
	// integrale positive
	Flux_dual[i] = MathFunctions::max(
				     ((flux1m - flux1) + (flux2m - flux2) + (flux3m - flux3)), 0.);
	// formule 16
	if (((phiplus[i] - phi[i]) * (phimoins[i] - phi[i])) >= 0.)
	  Flux_dual[i] = phi[i] * partie_positive_dual_v;
	//
      }
    } else if (type == 1) {
      // flux devant ou au dessus de cCells, integration entre
      // h0/2.-abs(face_normal_velocity)*deltat_n et h0/2.
      // Flux1 : integrale -inf,  h0/2.-partie_positive_v
      flux1 = INT2Y(h0 / 2. - partie_positive_v, -h0 / 2., phimoins[i], xg, yg);
      // Flux1m : integrale -inf,  -h0/2.
      flux1m = INT2Y(h0 / 2., -h0 / 2., phimoins[i], xg, yg);
      //
      // Flux2 : integrale -inf,  h0/2.-partie_positive_v
      flux2 = INT2Y(h0 / 2. - partie_positive_v, xg, yg, xd, yd);
      // Flux2m : integrale -inf,  -h0/2.
      flux2m = INT2Y(h0 / 2., xg, yg, xd, yd);
      //
      // Flux3 : integrale -inf,  h0/2.-partie_positive_v
      flux3 = INT2Y(h0 / 2. - partie_positive_v, xd, yd, h0 / 2., phiplus[i]);
      // Flux3m : integrale -inf,  -h0/2.
      flux3m = INT2Y(h0 / 2., xd, yd, h0 / 2., phiplus[i]);
      //
      // integrale positive
      Flux[i] = MathFunctions::max(
          ((flux1m - flux1) + (flux2m - flux2) + (flux3m - flux3)), 0.);
      // formule 16
      if (((phiplus[i] - phi[i]) * (phimoins[i] - phi[i])) >= 0.)
        Flux[i] = phi[i] * partie_positive_v;
      //
      // et calcul du flux dual si calcul_flux_dual=1
      if (calcul_flux_dual == 1) {
	// flux dual deja calculé lors du premier appel à la fonction
	  Flux_dual[i] = 0.;
      }
    }
  }
  if (projectionPlateauPenteComplet == 1) {
   // les flux de masse se déduisent des flux de volume en utilisant une valeur moyenne de Rho calculée par le flux de masse / flux de volume
   // Celles des energies massiques avec une valeur moyenne de e calculée par le flux d'energie / flux de volume
   // Celles de quantité de mouvement avec une valeur moyenne de u calculée par le flux de vitesse / flux de volume
    double somme_flux_masse = 0.;
    double somme_flux_volume = 0.;
    for (size_t imat = 0; imat < nbmat; imat++)
      somme_flux_volume += Flux[imat];
    
    if (MathFunctions::fabs(somme_flux_volume) > flux_threhold ) {
      for (size_t imat = 0; imat < nbmat; imat++) {
	Flux[nbmat + imat] = (Flux[nbmat + imat] / somme_flux_volume) * Flux[imat];
	Flux[2.*nbmat + imat] = (Flux[2.*nbmat + imat] / somme_flux_volume) * Flux[nbmat + imat];
	somme_flux_masse += Flux[nbmat + imat];
      }
    
      Flux[3 * nbmat] =
	(Flux[3 * nbmat]/somme_flux_volume) * somme_flux_masse;  // flux de quantité de mouvement x
      Flux[3 * nbmat +1] =
	(Flux[3 * nbmat +1 ]/somme_flux_volume) * somme_flux_masse;  // flux de quantité de mouvement y
      Flux[3 * nbmat +2] =
	(Flux[3 * nbmat +2 ]/somme_flux_volume) * somme_flux_masse;   
      Flux[3 * nbmat + 3] =
	phi[3 * nbmat + 3] * somme_flux_volume; // flux pour la pseudo VNR
    } else {
      Flux = Uzero;
    }
  } else {
    // les flux de masse, de quantité de mouvement et d'energie massique se
    // deduisent des flux de volumes avec la valeur de rho, e et u à la maille
    double somme_flux_masse = 0.;
    double somme_flux_volume = 0.;
    for (size_t imat = 0; imat < nbmat; imat++) {
      Flux[nbmat + imat] =
        phi[nbmat + imat] * Flux[imat];  // flux de masse de imat
      Flux[2 * nbmat + imat] =
        phi[2 * nbmat + imat] *
        Flux[nbmat + imat];  // flux de masse energy de imat
      somme_flux_masse += Flux[nbmat + imat];
      somme_flux_volume += Flux[imat];
    }
    Flux[3 * nbmat] =
      phi[3 * nbmat] * somme_flux_masse;  // flux de quantité de mouvement x
    Flux[3 * nbmat + 1] =
      phi[3 * nbmat + 1] * somme_flux_masse;  // flux de quantité de mouvement y
    Flux[3 * nbmat + 2] =
      phi[3 * nbmat + 2] * somme_flux_masse;  // flux d'energie cinetique
    Flux[3 * nbmat + 3] =
      phi[3 * nbmat + 3] * somme_flux_volume; // flux pour la pseudo VNR
  }

  *pFlux = Flux;
  *pFlux_dual = Flux_dual;
  if (partie_positive_v == 0.) *pFlux = Uzero;
  if (partie_positive_dual_v == 0.) *pFlux_dual = Uzero;
  
  return;
}
template <size_t d>
void Remap::computeFluxPPPure(
    RealArray1D<d> gradphi, RealArray1D<d> phi, RealArray1D<d> phiplus,
    RealArray1D<d> phimoins, double h0, double hplus, double hmoins,
    double face_normal_velocity, double deltat_n, int type, int cell,
    double flux_threhold, int projectionPlateauPenteComplet,
    double dual_normal_velocity, int calcul_flux_dual,
    RealArray1D<d> *pFlux,  RealArray1D<d> *pFlux_dual) {

  
  
  RealArray1D<d> Flux = *pFlux;
  RealArray1D<d> Flux_dual = *pFlux_dual;
  
  Flux = Uzero;
  Flux_dual  = Uzero;
  int nbmat = options->nbmat;
  double y0plus, y0moins, xd, xg, yd, yg;
  double flux1, flux2, flux3, flux1m, flux2m, flux3m;
  double partie_positive_v =
      0.5 * (face_normal_velocity + abs(face_normal_velocity)) * deltat_n;
  double partie_positive_dual_v =
      0.5 * (dual_normal_velocity + abs(dual_normal_velocity)) * deltat_n;
  int cas_PP = 0;
  // on ne fait que la projection des volumes et masses
  for (size_t i = 0; i < nbequamax; i++) {
    // calcul des seuils y0plus, y0moins pour cCells
    y0plus = computeY0(limiteurs->projectionLimiterIdPure, phi[i], phiplus[i],
                       phimoins[i], h0, hplus, hmoins, 0);
    y0moins = computeY0(limiteurs->projectionLimiterIdPure, phi[i], phiplus[i],
                        phimoins[i], h0, hplus, hmoins, 1);

    // calcul des points d'intersections xd,xg
    xg = computexgxd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins, 0);
    xd = computexgxd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins, 1);

    // calcul des valeurs sur ces points d'intersections
    yg = computeygyd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins,
                     gradphi[i], 0);
    yd = computeygyd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins,
                     gradphi[i], 1);

    if (type == 0)  // flux arriere ou en dessous de cCells, integration entre
                    // -h0/2. et -h0/2.+abs(face_normal_velocity)*deltat_n
    {
      // Flux1m : integrale -inf,  -h0/2.+partie_positive_v
      flux1m =
          INT2Y(-h0 / 2. + partie_positive_v, -h0 / 2., phimoins[i], xg, yg);
      // Flux1m : integrale -inf,  -h0/2.
      flux1 = INT2Y(-h0 / 2., -h0 / 2., phimoins[i], xg, yg);
      //
      // Flux2m : integrale -inf,  -h0/2.+partie_positive_v
      flux2m = INT2Y(-h0 / 2. + partie_positive_v, xg, yg, xd, yd);
      // Flux2 : integrale -inf,  -h0/2.
      flux2 = INT2Y(-h0 / 2., xg, yg, xd, yd);
      //
      // Flux3m : integrale -inf,  -h0/2.+partie_positive_v
      flux3m = INT2Y(-h0 / 2. + partie_positive_v, xd, yd, h0 / 2., phiplus[i]);
      // Flux3 : integrale -inf,  -h0/2.
      flux3 = INT2Y(-h0 / 2., xd, yd, h0 / 2., phiplus[i]);
      //
      // integrale positive
      Flux[i] = MathFunctions::max(
          ((flux1m - flux1) + (flux2m - flux2) + (flux3m - flux3)), 0.);
      // formule 16
      if (((phiplus[i] - phi[i]) * (phimoins[i] - phi[i])) >= 0.)
        Flux[i] = phi[i] * partie_positive_v;
      //
       // et calcul du flux dual si calcul_flux_dual=1
      if (calcul_flux_dual == 1) {
	// Flux1m : integrale -inf, partie_positive_dual_v
	flux1m = INT2Y(partie_positive_dual_v, -h0 / 2., phimoins[i], xg, yg);
	// Flux1m : integrale -inf,  0..
	flux1 = INT2Y(0., -h0 / 2., phimoins[i], xg, yg);
	// Flux2m : integrale -inf, partie_positive_dual_v  
	flux2m = INT2Y(partie_positive_dual_v, xg, yg, xd, yd);
	// Flux2 : integrale -inf, 0.
	flux2 = INT2Y(0., xg, yg, xd, yd);
	// Flux3m : integrale -inf, partie_positive_dual_v  
	flux3m = INT2Y(partie_positive_dual_v, xd, yd, h0 / 2., phiplus[i]);
	// Flux3 : integrale -inf, 0.
	flux3 = INT2Y(0., xd, yd, h0 / 2., phiplus[i]);
	// integrale positive
	Flux_dual[i] = MathFunctions::max(
				     ((flux1m - flux1) + (flux2m - flux2) + (flux3m - flux3)), 0.);
	// formule 16
	if (((phiplus[i] - phi[i]) * (phimoins[i] - phi[i])) >= 0.)
	  Flux_dual[i] = phi[i] * partie_positive_dual_v;
	//
      }
    } else if (type ==1) {
      // flux devant ou au dessus de cCells, integration entre
      // h0/2.-abs(face_normal_velocity)*deltat_n et h0/2.
      // Flux1 : integrale -inf,  h0/2.-partie_positive_v
      flux1 = INT2Y(h0 / 2. - partie_positive_v, -h0 / 2., phimoins[i], xg, yg);
      // Flux1m : integrale -inf,  -h0/2.
      flux1m = INT2Y(h0 / 2., -h0 / 2., phimoins[i], xg, yg);
      // Flux2 : integrale -inf,  h0/2.-partie_positive_v
      flux2 = INT2Y(h0 / 2. - partie_positive_v, xg, yg, xd, yd);
      // Flux2m : integrale -inf,  -h0/2.
      flux2m = INT2Y(h0 / 2., xg, yg, xd, yd);
      //
      // Flux3 : integrale -inf,  h0/2.-partie_positive_v
      flux3 = INT2Y(h0 / 2. - partie_positive_v, xd, yd, h0 / 2., phiplus[i]);
      // Flux3m : integrale -inf,  -h0/2.
      flux3m = INT2Y(h0 / 2., xd, yd, h0 / 2., phiplus[i]);
      // integrale positive
      Flux[i] = MathFunctions::max(
          ((flux1m - flux1) + (flux2m - flux2) + (flux3m - flux3)), 0.);
      // formule 16
      if (((phiplus[i] - phi[i]) * (phimoins[i] - phi[i])) >= 0.)
        Flux[i] = phi[i] * partie_positive_v;
      // et calcul du flux dual si calcul_flux_dual=1
      if (calcul_flux_dual == 1) {
	// flux dual deja calculé lors du premier appel à la fonction
	  Flux_dual[i] = 0.;
      }
    }
  }
  if (projectionPlateauPenteComplet == 1) {
   // les flux de masse se déduisent des flux de volume en utilisant une valeur moyenne de Rho calculée par le flux de masse / flux de volume
   // Celles des energies massiques avec une valeur moyenne de e calculée par le flux d'energie / flux de volume
   // Celles de quantité de mouvement avec une valeur moyenne de u calculée par le flux de vitesse / flux de volume
    double somme_flux_masse = 0.;
    double somme_flux_volume = 0.;
    for (size_t imat = 0; imat < nbmat; imat++)
      somme_flux_volume += Flux[imat];
    
    if (MathFunctions::fabs(somme_flux_volume) > flux_threhold) {
      for (size_t imat = 0; imat < nbmat; imat++) {
	Flux[2.*nbmat + imat] = (Flux[2.*nbmat + imat] / somme_flux_volume) * Flux[nbmat + imat];
	somme_flux_masse += Flux[nbmat + imat];
      }
    
      Flux[3 * nbmat] =
	(Flux[3 * nbmat]/somme_flux_volume) * somme_flux_masse;  // flux de quantité de mouvement x
      Flux[3 * nbmat +1] =
	(Flux[3 * nbmat +1 ]/somme_flux_volume) * somme_flux_masse;  // flux de quantité de mouvement y
      Flux[3 * nbmat +2] =
	(Flux[3 * nbmat +2 ]/somme_flux_volume) * somme_flux_masse;
      Flux[3 * nbmat + 3] =
	phi[3 * nbmat + 3] * somme_flux_volume; // flux pour la pseudo VNR
    } else {
      Flux = Uzero;
    }
  } else {
    // les flux de masse, de quantité de mouvement et d'energie massique se
    // deduisent des flux de volumes
    double somme_flux_masse = 0.;
    double somme_flux_volume = 0.;
    for (size_t imat = 0; imat < nbmat; imat++) {
      Flux[2 * nbmat + imat] =
        phi[2 * nbmat + imat] *
        Flux[nbmat + imat];  // flux de masse energy de imat
      somme_flux_masse += Flux[nbmat + imat];
      somme_flux_volume += Flux[imat];
    }
    Flux[3 * nbmat] =
      phi[3 * nbmat] * somme_flux_masse;  // flux de quantité de mouvement x
    Flux[3 * nbmat + 1] =
      phi[3 * nbmat + 1] * somme_flux_masse;  // flux de quantité de mouvement y
    Flux[3 * nbmat + 2] =
      phi[3 * nbmat + 2] * somme_flux_masse;  // flux d'energie cinetique
    Flux[3 * nbmat + 3] =
      phi[3 * nbmat + 3] * somme_flux_volume; // flux pour la pseudo VNR
  }
  
  *pFlux = Flux;
  *pFlux_dual = Flux_dual;
  if (partie_positive_v == 0.) *pFlux = Uzero;
  if (partie_positive_dual_v == 0.) *pFlux_dual = Uzero;
  return;
}

template <size_t d>
RealArray1D<d> Remap::computeUpwindFaceQuantities(
    RealArray1D<dim> face_normal, double face_normal_velocity, double delta_x,
    RealArray1D<dim> x_f, RealArray1D<d> phi_cb, RealArray1D<d> grad_phi_cb,
    RealArray1D<dim> x_cb, RealArray1D<d> phi_cf, RealArray1D<d> grad_phi_cf,
    RealArray1D<dim> x_cf) {
  if (face_normal_velocity * delta_x > 0.0)
    return phi_cb + (dot((x_f - x_cb), face_normal) * grad_phi_cb);
  else
    return phi_cf + (dot((x_f - x_cf), face_normal) * grad_phi_cf);
}

template <size_t d>
RealArray1D<d> Remap::computeRemapFlux(
    int projectionOrder, int projectionAvecPlateauPente,
    double face_normal_velocity, RealArray1D<dim> face_normal,
    double face_length, RealArray1D<d> phi_face,
    RealArray1D<dim> outer_face_normal, RealArray1D<dim> exy, double deltat_n) {
  RealArray1D<d> Flux;
  if (projectionAvecPlateauPente == 0) {
    // cas projection ordre 3 ou 1 ou 2 sans plateau pente (flux calculé ici
    // avec phi_face)
    if (MathFunctions::fabs(dot(face_normal, exy)) < 1.0E-10)
      return (0.0 * phi_face);
    return (dot(outer_face_normal, exy) * face_normal_velocity *
            face_length * deltat_n * phi_face);
  } else {
    // cas projection ordre 2 avec plateau pente (flux dans la variable
    // phi_face)
    if (MathFunctions::fabs(dot(face_normal, exy)) < 1.0E-10)
      return (0.0 * phi_face);
    return (dot(outer_face_normal, exy) * face_length * phi_face);
  }
}

#endif  // UTILESREMAP_IMPL_H
