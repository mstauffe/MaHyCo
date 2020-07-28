#ifndef GESTIONTEMPS_H
#define GESTIONTEMPS_H

namespace gesttempslib {

class GestionTempsClass {
 public:
  struct GestTemps {
    double final_time = 1.0;
    double output_time = final_time;
    double cfl = 0.45;
    int max_time_iterations = 500000000;
    double deltat_init = 0.;
    double deltat_min = 1.0E-10;
    double deltat_n;
    double deltat_nplus1;
    double t_n;
    double t_nplus1;
  };
  GestTemps* gt;

 private:
};
}  // namespace gesttempslib
#endif  // LIMITEURS_H
