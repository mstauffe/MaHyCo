#include <Kokkos_Core.hpp>                  // for finalize, initialize
#include <cstdlib>                          // for atof, atoi, exit
#include <iostream>                         // for operator<<, endl, basic_o...
#include <string>                           // for string
#include "lecture_donnees/LectureDonnees.h" // for LectureDonnees
#include "EucclhydRemap.h"                  // for EucclhydRemap::Options
#include "mesh/CartesianMesh2D.h"           // for CartesianMesh2D
#include "mesh/CartesianMesh2DGenerator.h"  // for CartesianMesh2DGenerator

using namespace nablalib;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  auto o = new EucclhydRemap::Options();
  string output;
  if (argc == 2) {
    LectureDonnees(argv[1], o);
  } else if (argc != 1) {
    std::cerr << "[ERREUR] Fichier de donnees non passé en argument " << std::endl;
    exit(1);
  }
  auto nm = CartesianMesh2DGenerator::generate(
      o->X_EDGE_ELEMS, o->Y_EDGE_ELEMS, o->X_EDGE_LENGTH, o->Y_EDGE_LENGTH);
  // appel au schéma Lagrange Eucclhyd + schéma de projection ADI (en option)
  auto c = new EucclhydRemap(o, nm, output);
  c->simulate();
  delete c;
  delete nm;
  delete o;
  Kokkos::finalize();
  return 0;
}
