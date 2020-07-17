#ifndef LIMITEURS_H
#define LIMITEURS_H


namespace limiteurslib
{

class LimiteursClass {
 public:
  struct Limiteurs {   
    // limiteur
    int minmod = 300;
    int superBee = 301;
    int vanLeer = 302;
    int minmodG = 1300;
    int superBeeG = 1301;
    int vanLeerG = 1302;
    int arithmeticG = 1303;
    
    int projectionAvecPlateauPente = 0;
    
    int projectionLimiterId = superBee;
    int projectionLimiterIdPure = arithmeticG;
    
    int projectionLimiteurMixte = 0;
  };
  Limiteurs* limiteurs;

 private:

};
} // namespace
#endif  // LIMITEURS_H
