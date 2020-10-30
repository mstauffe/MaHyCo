#ifndef EOS_H
#define EOS_H

namespace eoslib {

class EquationDetat {
 public:
  struct Eos {
    // EOS
    int Void = 100;
    int PerfectGas = 101;
    int StiffenedGas = 102;
    int Murnhagan = 103;
    int SolidLinear = 104;
    IntArray1D<nbmatmax> Nom = {{PerfectGas, PerfectGas, PerfectGas}};
    RealArray1D<nbmatmax> gammap = {{1.4, 1.4, 1.4}};
    double gamma = 1.4;
  };
  Eos* eos;

 private:
};
}  // namespace eoslib
#endif  // EOS_H
