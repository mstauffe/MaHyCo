#include <stdlib.h>  // for std

#include <fstream>        // for ifstream
#include <iostream>       // for operator<<, endl, basic_o...
#include "LectureDonnees.h"

/**
 * Job LectureDonnees called by main.
 * In variables: fichier
 * Out variables: Options
 */
void LectureDonneesClass::LectureDonnees(string Fichier,
		    EucclhydRemap::Options* o,
		    cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh,
		    gesttempslib::GestionTempsClass::GestTemps* gt,		 
		    limiteurslib::LimiteursClass::Limiteurs* l,
		    eoslib::EquationDetat::Eos* eos,
		    castestlib::CasTest::Test* test) {
 
  // string Fichier=argv[1];
  ifstream mesdonnees(Fichier);  // Ouverture d'un fichier en lecture
  if (mesdonnees) {
    // Tout est prêt pour la lecture.
    string ligne;
    string mot;
    int entier;
    getline(mesdonnees, ligne);  // ligne de commentaire numero du cas test
    mesdonnees >> mot;
    test->Nom = castestToOptions[mot];
    std::cout << " Cas test " << mot << " ( " << test->Nom << " ) "
              << std::endl;
    mesdonnees.ignore();

    // on en deduit le nombre de materiaux du calcul
    if (test->Nom < test->BiUnitTestCase)
      o->nbmat = 1;
    else o->nbmat = 2;
    if (test->Nom == test->BiTriplePoint) o->nbmat = 3;
      

    getline(mesdonnees, ligne);  // ligne de commentaire Nombre de Mailles en X
    mesdonnees >> entier;
    cstmesh->X_EDGE_ELEMS = entier;
    std::cout << " Nombre de Mailles en X " << cstmesh->X_EDGE_ELEMS << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // ligne de commentaire Nombre de Mailles en Y
    mesdonnees >> cstmesh->Y_EDGE_ELEMS;
    std::cout << " Nombre de Mailles en Y " << cstmesh->Y_EDGE_ELEMS << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // ligne de commentaire DELTA_X
    mesdonnees >> cstmesh->X_EDGE_LENGTH;
    std::cout << " DELTA X " << cstmesh->X_EDGE_LENGTH << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // ligne de commentaire DELTA_Y
    mesdonnees >> cstmesh->Y_EDGE_LENGTH;
    std::cout << " DELTA Y " << cstmesh->Y_EDGE_LENGTH << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // ligne de commentaire Temps-final
    mesdonnees >> gt->final_time;
    std::cout << " Temps-final " << gt->final_time << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees,
            ligne);  // ligne de commentaire Temps entre deux sorties
    mesdonnees >> gt->output_time;
    std::cout << " Temps entre deux sorties " << gt->output_time << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Pas de temps initial
    mesdonnees >> gt->deltat_init;
    std::cout << " Pas de temps initial " << gt->deltat_init << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Pas de temps minimal
    mesdonnees >> gt->deltat_min;
    std::cout << " Pas de temps minimal " << gt->deltat_min << std::endl;
    mesdonnees.ignore();

    for (int imat = 0; imat < o->nbmat; ++imat) {
      getline(mesdonnees, ligne); // Equation d'etat
      mesdonnees >> mot;
      eos->Nom[imat] = liste_eos[mot];
      std::cout << " Equation d'etat " << mot << " ( " << eos->Nom[imat] << " ) "
		<< std::endl; mesdonnees.ignore();
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
    std::cout << " Ordre de la phase de projection  " << o->projectionOrder
              << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Limiteur
    mesdonnees >> mot;
    l->projectionLimiterId = limiteur[mot];
    if ((l->projectionLimiterId != -1) && (l->projectionLimiterId != 0)) {
      std::cout << " Limiteur  " << mot << " ( " << l->projectionLimiterId << " ) "
              << std::endl;
    } else {
       cout << "ERREUR: Limiteur " << mot << " non prévu " << endl;
       exit(1);
    }
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Projection Avec Plateau Pente
    mesdonnees >> mot;
    l->projectionAvecPlateauPente = ouiOUnon[mot];
    std::cout << " Projection Avec Plateau Pente " << mot << " ( "
              << l->projectionAvecPlateauPente << " ) " << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Projection Avec Plateau Pente Mixte
    mesdonnees >> mot;
    l->projectionLimiteurMixte = ouiOUnon[mot];
    std::cout << " Projection Avec Plateau Pente Mixte " << mot << " ( "
              << l->projectionLimiteurMixte << " ) " << std::endl;
    mesdonnees.ignore();

    getline(mesdonnees, ligne);  // Projection Limiteur pour Mailles Pures
    mesdonnees >> mot;
    l->projectionLimiterIdPure = limiteur[mot];
    std::cout << " Limiteur pour Mailles Pures " << mot << " ( "
              << l->projectionLimiterIdPure << " ) " << std::endl;
    if ((l->projectionLimiterId == -1) && (l->projectionLimiteurMixte == 1)) {
      cout << "ERREUR: Limiteur pour mailles pures non défini " <<
	" alors que l option projectionLimiteurMixte est demandée" << endl;
       exit(1);
    }
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
