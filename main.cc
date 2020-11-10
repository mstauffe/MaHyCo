#include <Kokkos_Core.hpp>  // for finalize, initialize
#include <cstdlib>          // for atof, atoi, exit
#include <iostream>         // for operator<<, endl, basic_o...
#include <string>           // for string
#include "Eucclhyd.h"  // for EucclhydRemap
#include "Vnr.h"
#include "lecture_donnees/LectureDonnees.h"  // for LectureDonnees
#include "mesh/CartesianMesh2D.h"            // for CartesianMesh2D
#include "mesh/CartesianMesh2DGenerator.h"   // for CartesianMesh2DGenerator

using namespace nablalib;

int main(int argc, char* argv[]) {
  // initialisation de Kokkos
  Kokkos::initialize(argc, argv);
  // class utilisable
  auto scheme = new schemalagrangelib::SchemaLagrangeClass::SchemaLagrange();
  auto o = new optionschemalib::OptionsSchema::Options();
  auto cl = new conditionslimiteslib::ConditionsLimites::Cdl();
  auto lim = new limiteurslib::LimiteursClass::Limiteurs();
  auto eos = new eoslib::EquationDetat();
  auto test = new castestlib::CasTest::Test();
  auto cstmesh =
      new cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages();
  auto gt = new gesttempslib::GestionTempsClass::GestTemps();
  string output;
  // Lecture des donnees
  if (argc == 2) {
    LectureDonneesClass lecture;
    lecture.LectureDonnees(argv[1], scheme, o, cstmesh, gt, lim, eos, test);
  } else if (argc != 1) {
    std::cerr << "[ERREUR] Fichier de donnees non passé en argument "
              << std::endl;
    exit(1);
  }
  // chargement du maillage
  auto nm = CartesianMesh2DGenerator::generate(
      cstmesh->X_EDGE_ELEMS, cstmesh->Y_EDGE_ELEMS, cstmesh->X_EDGE_LENGTH,
      cstmesh->Y_EDGE_LENGTH, cstmesh->cylindrical_mesh);

  // appel au schéma Lagrange Eucclhyd + schéma de projection ADI (en option)
  if (scheme->schema == scheme->Eucclhyd) {
    auto varlp = new variableslagremaplib::VariablesLagRemap(nm);
    auto init = new initlib::Initialisations(o, eos, nm, cstmesh, varlp, cl, test);
    auto part = new particleslib::SchemaParticules(nm, cstmesh, gt, test);
    auto proj = new Remap(o, cstmesh, gt, cl, lim, nm, varlp);
    auto c =
      new Eucclhyd(o, cstmesh, gt, test, cl, lim, part, eos, nm, varlp, proj, init, output);
    c->simulate();
    delete varlp;
    delete init;
    delete part;
    //delete c;
  
  } else if (scheme->schema == scheme->VNR) {
    auto varlp = new variableslagremaplib::VariablesLagRemap(nm);
    auto init = new initlib::Initialisations(o, eos, nm, cstmesh, varlp, cl, test);
    auto part = new particleslib::SchemaParticules(nm, cstmesh, gt, test);
    auto proj = new Remap(o, cstmesh, gt, cl, lim, nm, varlp);
    auto c = new Vnr(o, cstmesh, gt, test, cl, lim, part, eos, nm, varlp, proj, init, output);
    c->simulate();
    delete varlp;
    delete init;
    delete part;
    //delete c;
  }
  delete o;
  // delete nm;
  delete cl;
  delete lim;
  delete eos;
  delete test;
  delete cstmesh;
  delete gt;
  delete scheme;

  Kokkos::finalize();

  return 0;
}
