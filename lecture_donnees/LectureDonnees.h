#ifndef LECTURE_DONNEES_H
#define LECTURE_DONNEES_H

#include <string>         // for string
#include <unordered_map>  // for unordered_map

#include "../eucclhyd_remap/Eucclhyd.h"  // for Eucclhyd and all
#include "../includes/SchemaLagrange.h"

class LectureDonneesClass {
 public:
  void LectureDonnees(
      string Fichier, schemalagrangelib::SchemaLagrangeClass::SchemaLagrange* s,
      optionschemalib::OptionsSchema::Options* o,
      sortielib::Sortie::SortieVariables* so,
      cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh,
      gesttempslib::GestionTempsClass::GestTemps* gt,
      limiteurslib::LimiteursClass::Limiteurs* l, eoslib::EquationDetat* eos,
      castestlib::CasTest::Test* test);

 private:
  std::unordered_map<string, int> castestToOptions{
      {"UnitTestCase", 0},     {"SedovTestCase", 1},  {"TriplePoint", 2},
      {"Implosion", 3},        {"SodCaseX", 4},       {"SodCaseY", 5},
      {"NohTestCase", 6},      {"AdvectionX", 7},     {"AdvectionY", 8},
      {"AdvectionVitX", 9},    {"AdvectionVitY", 10}, {"BiUnitTestCase", 11},
      {"BiSedovTestCase", 12}, {"BiTriplePoint", 13}, {"BiShockBubble", 14},
      {"BiSodCaseX", 15},      {"BiSodCaseY", 16},    {"BiNohTestCase", 17},
      {"BiAdvectionX", 18},    {"BiAdvectionY", 19},  {"BiAdvectionVitX", 20},
      {"BiAdvectionVitY", 21}, {"BiImplosion", 22},   {"MonoRiderTx", 23},
      {"MonoRiderTy", 24},     {"MonoRiderT45", 25},      {"MonoRiderRotation", 26},
      {"MonoRiderVortex", 27}, {"MonoRiderDeformation", 28}, {"MonoRiderVortexTimeReverse", 29},
      {"MonoRiderDeformationTimeReverse", 30}, {"RiderTx", 31},
      {"RiderTy", 32},         {"RiderT45", 33},      {"RiderRotation", 34},
      {"RiderVortex", 35},     {"RiderDeformation", 36}, {"RiderVortexTimeReverse", 37},
      {"RiderDeformationTimeReverse", 38}};

  std::unordered_map<string, int> schema_lagrange{
    {"Eucclhyd", 2000}, {"VNR", 2001}, {"CSTS", 2002}, {"MYR", 2003}, {"AUCUN", 2004}};

  std::unordered_map<string, int> limiteur{
      {"minmod", 300},       {"superBee", 301},   {"vanLeer", 302},
      {"minmodG", 1300},     {"superBeeG", 1301}, {"vanLeerG", 1302},
      {"arithmeticG", 1303}, {"ultrabeeG", 1304}};

  std::unordered_map<string, int> ouiOUnon{{"non", 0}, {"oui", 1}};

  std::unordered_map<string, int> A1OUA2OUPB{{"A1", 0}, {"A2", 1}, {"PB", 2}};

  std::unordered_map<string, int> liste_eos{{"Void", 100},
                                            {"PerfectGas", 101},
                                            {"StiffenedGas", 102},
                                            {"Fictif", 103},
                                            {"SolidLinear", 104}};

  std::unordered_map<string, int> equilibrage{
      {"sans", 0}, {"Isotherme", 1}, {"Adiabatique", 2}};
};
#endif  // LECTURE_DONNEES_H
