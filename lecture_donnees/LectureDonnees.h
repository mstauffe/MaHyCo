#ifndef LECTURE_DONNEES_H
#define LECTURE_DONNEES_H

#include <string>  // for string
#include <unordered_map>  // for unordered_map

#include "../eucclhyd_remap/EucclhydRemap.h"  // for EucclhydRemap::Options
#include "../includes/SchemaLagrange.h"

class LectureDonneesClass {
 public:
  void LectureDonnees(string Fichier,
		    schemalagrangelib::SchemaLagrangeClass::SchemaLagrange* s,
		    optionschemalib::OptionsSchema::Options* o,
		    cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh,
		    gesttempslib::GestionTempsClass::GestTemps* gt,
		    limiteurslib::LimiteursClass::Limiteurs* l,
		    eoslib::EquationDetat::Eos* eos,
		    castestlib::CasTest::Test* test);
  
 private:
  std::unordered_map<string, int> castestToOptions
  {
    {"UnitTestCase", 0},
      {"SedovTestCase", 1},
	{"TriplePoint", 2},
	  {"SodCaseX", 4},
	    {"SodCaseY", 5},
	      {"NohTestCase", 6},
		{"BiUnitTestCase", 10},
		  {"BiSedovTestCase", 11},
		    {"BiTriplePoint", 12},
		      {"BiShockBubble", 13},
			{"BiSodCaseX", 14},
			  {"BiSodCaseY", 15},
			    {"BiNohTestCase", 16}
  };

  std::unordered_map<string, int> schema_lagrange
  {
    {"Eucclhyd", 2000},
      {"VNR", 2001},
       {"CSTS", 2002},
	 {"MYR", 2003}
  };

  std::unordered_map<string, int> limiteur{
    {"minmod", 300},
      {"superBee", 301},
	{"vanLeer", 302},
	  {"minmodG", 1300},
	    {"superBeeG", 1301},
	      {"vanLeerG", 1302},
		{"arithmeticG", 1303},
		  {"ultrabeeG", 1304}
  };

  std::unordered_map<string, int> ouiOUnon
  {
    {"non", 0},
      {"oui", 1}
  };

  std::unordered_map<string, int> liste_eos
  {
    {"Void", 100},
      {"PerfectGas", 101},
	{"StiffenedGas", 102},
	  {"Murnhagan", 103},
	    {"SolidLinear", 104}
  };

  std::unordered_map<string, int> equilibrage
  {
    {"sans", 0},
      {"Isotherme", 1},
	{"Adiabatique", 2}
  };
};
#endif  // LECTURE_DONNEES_H
