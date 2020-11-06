#ifndef EOS_H
#define EOS_H

namespace eoslib {

class EquationDetat {
 public:
  // EOS
  int Void = 100;
  int PerfectGas = 101;
  int StiffenedGas = 102;
  int Murnhagan = 103;
  int SolidLinear = 104;
  IntArray1D<nbmatmax> Nom = {{PerfectGas, PerfectGas, PerfectGas}};
  RealArray1D<nbmatmax> gamma = {{1.4, 1.4, 1.4}};
  RealArray1D<nbmatmax> tension_limit = {{0.01, 0.01, 0.01}};

  /**
   * 
   */
  RealArray1D<2> computeEOSGP(double gamma, double rho, double energy) {
    double pression = (gamma -1.) * rho * energy;
    double sound_speed = std::sqrt(gamma * (gamma -1.) * energy);
    return {pression, sound_speed};
  }
  
  RealArray1D<2> computeEOSVoid(double rho, double energy) {
    double pression = 0.;
    double sound_speed = 0.;
    return {pression, sound_speed};
  }
  
  RealArray1D<2> computeEOSSTIFG(double gamma, double limit_tension, double rho, double energy) {
    double pression;
    double sound_speed ;
    if (rho !=0.) {
      pression = ((gamma- 1.) * rho * energy) - (gamma * limit_tension);
      sound_speed = sqrt((gamma/rho)*(pression+limit_tension));
    } else {
      pression = 0.;
      sound_speed = 0.;
    }
    //std::cout << " rho " << rho << "  pression " << pression << std::endl;
    return {pression, sound_speed};   
  }
  
  RealArray1D<2> computeEOSMur(double rho, double energy) {
    std::cout << " Pas encore programmée" << std::endl;
  }
  
  RealArray1D<2> computeEOSSL(double rho, double energy) {
    std::cout << " Pas encore programmée" << std::endl;
  } 
 private:
};
}  // namespace eoslib
#endif  // EOS_H
