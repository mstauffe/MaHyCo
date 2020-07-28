#include <math.h>    // for fabs
#include <stdlib.h>  // for abs

#include <array>  // for array

#include "EucclhydRemap.h"        // for EucclhydRemap, EucclhydRemap::Options
#include "types/MathFunctions.h"  // for min, max

double EucclhydRemap::fluxLimiter(int projectionLimiterId, double r) {
  if (projectionLimiterId == limiteurs->minmod) {
    return MathFunctions::max(0.0, MathFunctions::min(1.0, r));
  } else if (projectionLimiterId == limiteurs->superBee) {
    return MathFunctions::max(
        0.0, MathFunctions::max(MathFunctions::min(2.0 * r, 1.0),
                                MathFunctions::min(r, 2.0)));
  } else if (projectionLimiterId == limiteurs->vanLeer) {
    if (r <= 0.0)
      return 0.0;
    else
      return 2.0 * r / (1.0 + r);
  } else
    return 0.0;  // ordre 1
}

double EucclhydRemap::fluxLimiterPP(int projectionLimiterId, double gradplus,
                                    double gradmoins, double y0, double yplus,
                                    double ymoins, double h0, double hplus,
                                    double hmoins) {
  double grady, gradM, gradMplus, gradMmoins;
  // limitation rupture de pente (formule 16 si on utilise pas le plateau pente)
  if (gradplus * gradmoins < 0.0) return 0.;

  if (projectionLimiterId == limiteurs->minmodG)  // formule 9c
  {
    if ((yplus - ymoins) > 0.)
      grady = MathFunctions::min(fabs(gradplus), fabs(gradmoins));
    else
      grady = -MathFunctions::min(fabs(gradplus), fabs(gradmoins));
  } else if (projectionLimiterId == limiteurs->superBeeG)  // formule 9g
  {
    if ((yplus - ymoins) > 0.)
      grady = MathFunctions::max(fabs(gradplus), fabs(gradmoins));
    else
      grady = -MathFunctions::max(fabs(gradplus), fabs(gradmoins));
  } else if (projectionLimiterId == limiteurs->vanLeerG)  // formule 9e
  {
    double lambdaplus = (h0 / 2 + hplus) / (h0 + hplus + hmoins);
    double lambdamoins = (h0 / 2 + hmoins) / (h0 + hplus + hmoins);
    if ((lambdaplus * gradplus + lambdamoins * gradmoins) != 0.) {
      grady = gradplus * gradmoins /
              (lambdaplus * gradplus + lambdamoins * gradmoins);
    } else
      grady = 0.;
  } else if (projectionLimiterId == limiteurs->arithmeticG) {
    double lambdaplus = (h0 / 2 + hplus) / (h0 + hplus + hmoins);
    double lambdamoins = (h0 / 2 + hmoins) / (h0 + hplus + hmoins);
    grady = lambdamoins * gradplus + lambdaplus * gradmoins;
  }

  // limitation simple-pente (formule 10)
  gradMplus = gradplus * (h0 + hplus) / h0;
  gradMmoins = gradmoins * (h0 + hmoins) / h0;
  gradM = MathFunctions::min(fabs(gradMplus), fabs(gradMmoins));
  if ((yplus - ymoins) > 0.)
    grady = MathFunctions::min(fabs(gradM), fabs(grady));
  else
    grady = -MathFunctions::min(fabs(gradM), fabs(grady));

  return grady;
}

double EucclhydRemap::computeY0(int projectionLimiterId, double y0,
                                double yplus, double ymoins, double h0,
                                double hplus, double hmoins, int type) {
  // retourne {{y0plus, y0moins}}
  double y0plus = 0., y0moins = 0.;
  if (projectionLimiterId == limiteurs->minmodG ||
      projectionLimiterId == limiteurs->minmod)  // minmod
  {
    y0plus = yplus;
    y0moins = ymoins;
  } else if (projectionLimiterId == limiteurs->superBeeG ||
             projectionLimiterId == limiteurs->superBee)  // superbee
  {
    y0plus = ((h0 + hmoins) * yplus + h0 * ymoins) / (2 * h0 + hmoins);
    y0moins = ((h0 + hplus) * ymoins + h0 * yplus) / (2 * h0 + hplus);
  } else if (projectionLimiterId == limiteurs->vanLeerG ||
             projectionLimiterId == limiteurs->vanLeer)  // vanleer
  {
    double a = MathFunctions::min(yplus, ymoins);
    double b = MathFunctions::max(yplus, ymoins);
    double xplus = (h0 * h0 + 3 * h0 * hmoins + 2 * hmoins * hmoins) * yplus;
    double xmoins = (h0 * h0 + 3 * h0 * hplus + 2 * hplus * hplus) * ymoins;
    xplus +=
        (h0 * h0 - h0 * hplus - 2 * hplus * hplus + 2 * h0 * hmoins) * ymoins;
    xmoins +=
        (h0 * h0 - h0 * hmoins - 2 * hmoins * hmoins + 2 * h0 * hplus) * yplus;
    xplus /= (2 * h0 * h0 + 5 * h0 * hmoins + 2 * hmoins * hmoins - h0 * hplus -
              2 * hplus * hplus);
    xmoins /= (2 * h0 * h0 + 5 * h0 * hplus + 2 * hplus * hplus - h0 * hmoins -
               2 * hmoins * hmoins);

    y0plus = MathFunctions::min(MathFunctions::max(xplus, a), b);
    y0moins = MathFunctions::min(MathFunctions::max(xmoins, a), b);
  } else if (projectionLimiterId == limiteurs->arithmeticG) {
    y0plus = ((h0 + hmoins + hplus) * yplus + h0 * ymoins) /
             (2 * h0 + hmoins + hplus);
    y0moins = ((h0 + hmoins + hplus) * ymoins + h0 * yplus) /
              (2 * h0 + hmoins + hplus);
  } else if (projectionLimiterId == 3000) {
    y0plus = yplus;
    y0moins = ymoins;
  }
  if (type == 0)
    return y0plus;
  else if (type == 1)
    return y0moins;
  else
    return 0.0;  // lancer forcement avec type 0 ou 1 mais warning compile
}

double EucclhydRemap::computexgxd(double y0, double yplus, double ymoins,
                                  double h0, double y0plus, double y0moins,
                                  int type) {
  // retourne {{xg, xd}}
  double xd = 0., xg = 0.;
  double xplus = 1.;
  if (abs(y0plus - yplus) > options->threshold)
    xplus = (y0 - yplus) / (y0plus - yplus) - 1. / 2.;
  double xmoins = 1.;
  if (abs(y0moins - ymoins) > options->threshold)
    xmoins = (y0 - ymoins) / (y0moins - ymoins) - 1. / 2.;
  xd = +h0 * MathFunctions::min(MathFunctions::max(xplus, -1. / 2.), 1. / 2.);
  xg = -h0 * MathFunctions::min(MathFunctions::max(xmoins, -1. / 2.), 1. / 2.);
  if (type == 0)
    return xg;
  else if (type == 1)
    return xd;
  else
    return 0.0;  // lancer forcement avec type 0 ou 1 mais warning compile
}

double EucclhydRemap::computeygyd(double y0, double yplus, double ymoins,
                                  double h0, double y0plus, double y0moins,
                                  double grady, int type) {
  // retourne {{yg, yd}}
  double yd, yg;
  double xtd = y0 + h0 / 2 * grady;
  double xtg = y0 - h0 / 2 * grady;
  double ad = MathFunctions::min(yplus, 2. * y0moins - ymoins);
  double bd = MathFunctions::max(yplus, 2. * y0moins - ymoins);
  double ag = MathFunctions::min(ymoins, 2. * y0plus - yplus);
  double bg = MathFunctions::max(ymoins, 2. * y0plus - yplus);
  yd = MathFunctions::min(MathFunctions::max(xtd, ad), bd);
  yg = MathFunctions::min(MathFunctions::max(xtg, ag), bg);
  if (type == 0)
    return yg;
  else if (type == 1)
    return yd;
  else
    return 0.0;  // lancer forcement avec type 0 ou 1 mais warning compile
}

double EucclhydRemap::INTY(double X, double x0, double y0, double x1,
                           double y1) {
  double flux = 0.;
  double Xbar = MathFunctions::min(MathFunctions::max(x0, X), x1);
  // std::cout << " Xbar  " << Xbar << std::endl;
  if (abs(x1 - x0) > 1.e-14)
    flux = (y0 + 0.5 * ((Xbar - x0) / (x1 - x0)) * (y1 - y0)) * (Xbar - x0);
  return flux;
}

double EucclhydRemap::INT2Y(double X, double x0, double y0, double x1,
                            double y1) {
  double flux = 0.;
  // std::cout << " x0 " << x0 << std::endl;
  // std::cout << " x1 " << x1 << std::endl;
  if (abs(x1 - x0) > 1.e-14) {
    double eta =
        MathFunctions::min(MathFunctions::max(0., (X - x0) / (x1 - x0)), 1.);
    // std::cout << " eta " << eta << std::endl;
    flux = (y0 + 0.5 * eta * (y1 - y0)) * (x1 - x0) * eta;
  }
  return flux;
}

RealArray1D<dim> EucclhydRemap::xThenYToDirection(bool x_then_y_) {
  if (x_then_y_)
    return {{1.0, 0.0}};

  else
    return {{0.0, 1.0}};
}
// fonctions pour l'ordre 3
// ----------------------------------
// fonction pour evaluer le gradient
double EucclhydRemap::evaluate_grad(double hm, double h0, double hp, double ym,
                                    double y0, double yp) {
  double grad;
  grad = h0 / (hm + h0 + hp) *
         ((2. * hm + h0) / (h0 + hp) * (yp - y0) +
          (h0 + 2. * hp) / (hm + h0) * (y0 - ym));
  return grad;
}
// ----------------------------------
// fonction pour évaluer ystar
double EucclhydRemap::evaluate_ystar(double hmm, double hm, double hp,
                                     double hpp, double ymm, double ym,
                                     double yp, double ypp, double gradm,
                                     double gradp) {
  double ystar, tmp1, tmp2;
  tmp1 = (2. * hp * hm) / (hm + hp) *
         ((hmm + hm) / (2. * hm + hp) - (hpp + hp) / (2. * hp + hm)) *
         (yp - ym);
  tmp2 = -hm * (hmm + hm) / (2. * hm + hp) * gradp +
         hp * (hp + hpp) / (hm + 2. * hp) * gradm;
  ystar = ym + hm / (hm + hp) * (yp - ym) +
          1. / (hmm + hm + hp + hpp) * (tmp1 + tmp2);
  return ystar;
}
// ----------------------------------
// fonction pour évaluer fm
double EucclhydRemap::evaluate_fm(double x, double dx, double up, double du,
                                  double u6) {
  double fm;
  fm = up - 0.5 * x / dx * (du - (1. - 2. / 3. * x / dx) * u6);
  return fm;
}
// ----------------------------------
// fonction pour évaluer fr
double EucclhydRemap::evaluate_fp(double x, double dx, double um, double du,
                                  double u6) {
  double fp;
  fp = um + 0.5 * x / dx * (du - (1. - 2. / 3. * x / dx) * u6);
  return fp;
}
// ----------------------------------
// fonction pour initialiser la structure interval
EucclhydRemap::interval EucclhydRemap::define_interval(double a, double b) {
  interval I;
  I.inf = MathFunctions::min(a, b);
  I.sup = MathFunctions::max(a, b);
  return I;
}
// ----------------------------------
// fonction pour calculer l'intersection entre deux intervals
EucclhydRemap::interval EucclhydRemap::intersection(interval I1, interval I2) {
  interval I;
  if ((I1.sup < I2.inf) || (I2.sup < I1.inf)) {
    I.inf = 0.;
    I.sup = 0.;
  } else {
    I.inf = MathFunctions::max(I1.inf, I2.inf);
    I.sup = MathFunctions::min(I1.sup, I2.sup);
  }
  return I;
}
// ----------------------------------
// fonction pour calculer le flux
double EucclhydRemap::ComputeFluxOrdre3(double ymmm, double ymm, double ym,
                                        double yp, double ypp, double yppp,
                                        double hmmm, double hmm, double hm,
                                        double hp, double hpp, double hppp,
                                        double vdt) {
  double flux;
  double gradmm, gradm, gradp, gradpp;
  double ystarm, ystar, ystarp;
  double ym_m, ym_p, yp_m, yp_p;
  double grad_m, grad_p, ym6, yp6;
  //
  gradmm = evaluate_grad(hmmm, hmm, hm, ymmm, ymm, ym);
  gradm = evaluate_grad(hmm, hm, hp, ymm, ym, yp);
  gradp = evaluate_grad(hm, hp, hpp, ym, yp, ypp);
  gradpp = evaluate_grad(hp, hpp, hppp, yp, ypp, yppp);
  //
  ystarm = evaluate_ystar(hmmm, hmm, hm, hp, ymmm, ymm, ym, yp, gradmm, gradm);
  ystar = evaluate_ystar(hmm, hm, hp, hpp, ymm, ym, yp, ypp, gradm, gradp);
  ystarp = evaluate_ystar(hm, hp, hpp, hppp, ym, yp, ypp, yppp, gradp, gradpp);
  //
  ym_m = ystarm;
  ym_p = ystar;
  yp_m = ystar;
  yp_p = ystarp;
  //
  grad_m = ym_p - ym_m;
  grad_p = yp_p - yp_m;
  //
  ym6 = 6. * (ym - 0.5 * (ym_m + ym_p));
  yp6 = 6. * (yp - 0.5 * (yp_m + yp_p));
  //
  if (vdt >= 0.) {
    flux = evaluate_fm(vdt, hm, ym_p, grad_m, ym6);
  } else {
    flux = evaluate_fp(-vdt, hp, yp_m, grad_p, yp6);
  }
  // Limitation TVD
  double num, nup, ym_ym, yp_ym;
  interval I1, I2, limiteur;
  num = vdt / hm;
  nup = vdt / hp;
  ym_ym = ym + (1. - num) / num * (ym - ymm);
  yp_ym = yp - (1. + nup) / nup * (yp - ypp);
  if (vdt >= 0.) {
    I1 = define_interval(ym, yp);
    I2 = define_interval(ym, ym_ym);
  } else {
    I1 = define_interval(ym, yp);
    I2 = define_interval(yp, yp_ym);
  }
  limiteur = intersection(I1, I2);
  if (flux < limiteur.inf) {
    flux = limiteur.inf;
  }
  if (flux > limiteur.sup) {
    flux = limiteur.sup;
  }
  //
  return flux;
}
