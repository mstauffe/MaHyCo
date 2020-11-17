#ifndef SORTIE_H
#define SORTIE_H

namespace sortielib {

class Sortie {
 public:
  struct SortieVariables {
    bool pression = false;
    bool energie_interne = false;
    bool densite = false;
    bool vitesse  = false;
    bool pseudo = false;
    bool fraction_volumique = false;
    bool interface = false;
  };
  SortieVariables* sortievariable;

 private:
};
}  // namespace sortielib
#endif  // SORTIE_H
