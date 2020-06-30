#include <stdlib.h>  // for std

#include <fstream>        // for ifstream
#include <iostream>       // for operator<<, endl, basic_o...
#include <string>         // for string
#include <unordered_map>  // for unordered_map

#include "../eucclhyd_remap/EucclhydRemap.h"  // for EucclhydRemap::Options
/**
 * Job LectureDonnees called by main.
 * In variables: fichier
 * Out variables: Options
 */
void LectureDonnees(string Fichier, EucclhydRemap::Options* o) {
  std::unordered_map<string, int> castestToOptions;
  castestToOptions["UnitTestCase"] = 0;
  castestToOptions["SedovTestCase"] = 1;
  castestToOptions["TriplePoint"] = 2;
  castestToOptions["SodCaseX"] = 4;
  castestToOptions["SodCaseY"] = 5;
  castestToOptions["NohTestCase"] = 6;
  castestToOptions["BiUnitTestCase"] = 10;
  castestToOptions["BiSedovTestCase"] = 11;
  castestToOptions["BiTriplePoint"] = 12;
  castestToOptions["BiShockBubble"] = 13;
  castestToOptions["BiSodCaseX"] = 14;
  castestToOptions["BiSodCaseY"] = 15;
  castestToOptions["BiNohTestCase"] = 16;

  std::unordered_map<string, int> schema_lagrange;
  schema_lagrange["Eucclhyd"] = 2000;
  schema_lagrange["CSTS"] = 2001;
  schema_lagrange["MYR"] = 2002;

  std::unordered_map<string, int> limiteur;
  limiteur["minmod"] = 300;
  limiteur["superBee"] = 301;
  limiteur["vanLeer"] = 302;
  limiteur["minmodG"] = 1300;
  limiteur["superBeeG"] = 1301;
  limiteur["vanLeerG"] = 1302;
  limiteur["arithmeticG"] = 1303;

  std::unordered_map<string, int> ouiOUnon;
  ouiOUnon["non"] = 0;
  ouiOUnon["oui"] = 1;

  std::unordered_map<string, int> eosToOptions;
  eosToOptions["Void"] = 100;
  eosToOptions["PerfectGas"] = 101;
  eosToOptions["StiffenedGas"] = 102;
  eosToOptions["Murnhagan"] = 103;
  eosToOptions["SolidLinear"] = 104;

  std::unordered_map<string, int> equilibrage;
  equilibrage["sans"] = 0;
  equilibrage["Isotherme"] = 1;
  equilibrage["Adiabatique"] = 2;
  // string Fichier=argv[1];
  ifstream mesdonnees(Fichier);  // Ouverture d'un fichier en lecture
  if (mesdonnees) {
    // Tout est prêt pour la lecture.
    string ligne;
    string mot;
    int entier;
    getline(mesdonnees, ligne);  // ligne de commentaire numero du cas test
    mesdonnees >> mot;
    o->testCase = castestToOptions[mot];
    std::cout << " Cas test " << mot << " ( " << o->testCase << " ) "
              << std::endl;
    mesdonnees.ignore();

    // on en deduit le nombre de materiaux du calcul
    if (o->testCase < o->BiUnitTestCase)
      o->nbmat = 1;
    else o->nbmat = 2;
    if (o->testCase == o->BiTriplePoint) o->nbmat = 3;
      

    getline(mesdonnees, ligne);  // ligne de commentaire Nombre de Mailles en X
    mesdonnees >> entier;
    o->X_EDGE_ELEMS = entier;
    std::cout << " Nombre de Mailles en X " << o->X_EDGE_ELEMS << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // ligne de commentaire Nombre de Mailles en Y
    mesdonnees >> o->Y_EDGE_ELEMS;
    std::cout << " Nombre de Mailles en Y " << o->Y_EDGE_ELEMS << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // ligne de commentaire DELTA_X
    mesdonnees >> o->X_EDGE_LENGTH;
    std::cout << " DELTA X " << o->X_EDGE_LENGTH << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // ligne de commentaire DELTA_Y
    mesdonnees >> o->Y_EDGE_LENGTH;
    std::cout << " DELTA Y " << o->Y_EDGE_LENGTH << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // ligne de commentaire Temps-final
    mesdonnees >> o->final_time;
    std::cout << " Temps-final " << o->final_time << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees,
            ligne);  // ligne de commentaire Temps entre deux sorties
    mesdonnees >> o->output_time;
    std::cout << " Temps entre deux sorties " << o->output_time << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Pas de temps initial
    mesdonnees >> o->deltat_init;
    std::cout << " Pas de temps initial " << o->deltat_init << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Pas de temps minimal
    mesdonnees >> o->deltat_min;
    std::cout << " Pas de temps minimal " << o->deltat_min << std::endl;
    mesdonnees.ignore();

    for (int imat = 0; imat < o->nbmat; ++imat) {
      getline(mesdonnees, ligne); // Equation d'etat
      mesdonnees >> mot;
      o->eos[imat] = eosToOptions[mot];
      std::cout << " Equation d'etat " << mot << " ( " << o->eos[imat] << " ) "
                << std::endl; mesdonnees.ignore();
      if (o->eos[imat] == o->StiffenedGas) {
        o->gammap[imat] = 6.1;
        o->pip[imat] = 2.e4;
      }
    }
    
    // getline(mesdonnees, ligne); // Equilibrage des pressions
    // mesdonnees >> mot;
    // o->AvecEquilibrage = equilibrage[mot];
    // std::cout << " Equilibrage des pressions " << mot << " ( " <<
    // o->AvecEquilibrage << " ) " << std::endl; mesdonnees.ignore();

    // getline(mesdonnees, ligne); // Schéma Lagrange
    // mesdonnees >> mot;
    // o->schemaLagrange = schema_lagrange[mot];
    // std::cout << " schema Lagrange " << mot << " ( " << o->schemaLagrange <<
    // " ) " << std::endl; mesdonnees.ignore();

    getline(mesdonnees, ligne);  // ordre en espace du schema Lagrange
    mesdonnees >> o->spaceOrder;
    std::cout << " Ordre en espace du schéma Lagrange  " << o->spaceOrder
              << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne); // Avec projection
    mesdonnees >> mot;
    o->AvecProjection = ouiOUnon[mot];
    std::cout << " Avec Projection " << mot << " ( " << o->AvecProjection
              << " ) " << std::endl; mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Projection conservative
    mesdonnees >> mot;
    o->projectionConservative = ouiOUnon[mot];
    std::cout << " Projection conservative " << mot << " ( "
              << o->projectionConservative << " ) " << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Limite de fraction vol des mailles mixtes
    mesdonnees >> o->threshold;
    std::cout << " Limite de fraction volumique des mailles mixtes  "
              << o->threshold << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // ordre de la projection
    mesdonnees >> o->projectionOrder;
    std::cout << " Ordre en de la phase de projection  " << o->projectionOrder
              << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Limiteur
    mesdonnees >> mot;
    o->projectionLimiterId = limiteur[mot];
    std::cout << " Limiteur " << mot << " ( " << o->projectionLimiterId << " ) "
              << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Projection Avec Plateau Pente
    mesdonnees >> mot;
    o->projectionAvecPlateauPente = ouiOUnon[mot];
    std::cout << " Projection Avec Plateau Pente " << mot << " ( "
              << o->projectionAvecPlateauPente << " ) " << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Projection Avec Plateau Pente Mixte
    mesdonnees >> mot;
    o->projectionLimiteurMixte = ouiOUnon[mot];
    std::cout << " Projection Avec Plateau Pente Mixte " << mot << " ( "
              << o->projectionLimiteurMixte << " ) " << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Projection Limiteur pour Mailles Pures
    mesdonnees >> mot;
    o->projectionLimiterIdPure = limiteur[mot];
    std::cout << " Limiteur pour Mailles Pures " << mot << " ( "
              << o->projectionLimiterIdPure << " ) " << std::endl;
    mesdonnees.ignore();

    // getline(mesdonnees, ligne); // Presence de Particules
    // mesdonnees >> mot;
    // o->AvecParticules = ouiOUnon[mot];
    // std::cout << " Presence de Particules " << mot << " ( " <<
    // o->AvecParticules << " ) " << std::endl; mesdonnees.ignore();

    // getline(mesdonnees, ligne); // Nombre de Particules
    // mesdonnees >> nbPart;
    // std::cout << " Nombre de Particules " << nbPart << std::endl;
    // mesdonnees.ignore();

  } else {
    cout << "ERREUR: Impossible d'ouvrir le fichier en lecture." << endl;
    exit(1);
  }
}
