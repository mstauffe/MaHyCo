#include <Kokkos_Core.hpp>  // for finalize, initialize
#include <cstdlib>          // for atof, atoi, exit
#include <iostream>         // for operator<<, endl, basic_o...
#include <string>           // for string
#include "EucclhydRemap.h"  // for EucclhydRemap::Options et Cdl
#include "lecture_donnees/LectureDonnees.h"  // for LectureDonnees
#include "mesh/CartesianMesh2D.h"            // for CartesianMesh2D
#include "mesh/CartesianMesh2DGenerator.h"   // for CartesianMesh2DGenerator

using namespace nablalib;

int main(int argc, char* argv[]) {
  // initialisation de Kokkos
  Kokkos::initialize(argc, argv);
  // class utilisable
  auto o = new EucclhydRemap::Options();
  auto cl = new conditionslimiteslib::ConditionsLimites::Cdl();
  auto lim = new limiteurslib::LimiteursClass::Limiteurs();
  auto part = new particulelib::SchemaParticules::Particules();
  auto eos = new eoslib::EquationDetat::Eos();
  auto test = new castestlib::CasTest::Test();
  auto cstmesh =
      new cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages();
  auto gt = new gesttempslib::GestionTempsClass::GestTemps();
  string output;
  // Lecture des donnees
  if (argc == 2) {
    LectureDonneesClass lecture;
    lecture.LectureDonnees(argv[1], o, cstmesh, gt, lim, eos, test);
  } else if (argc != 1) {
    std::cerr << "[ERREUR] Fichier de donnees non passé en argument "
              << std::endl;
    exit(1);
  }
  // chargement du maillage
  auto nm = CartesianMesh2DGenerator::generate(
      cstmesh->X_EDGE_ELEMS, cstmesh->Y_EDGE_ELEMS, cstmesh->X_EDGE_LENGTH,
      cstmesh->Y_EDGE_LENGTH);

  // appel au schéma Lagrange Eucclhyd + schéma de projection ADI (en option)
  auto c =
      new EucclhydRemap(o, cstmesh, gt, test, cl, lim, part, eos, nm, output);

  c->simulate();

  delete c;
  delete nm;
  delete o;
  delete cl;
  delete lim;
  delete part;
  delete eos;
  delete test;
  delete cstmesh;
  delete gt;

  Kokkos::finalize();

  return 0;
}
