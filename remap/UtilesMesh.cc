#include "Remap.h"         // for Remap, Remap::Options
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "utils/Utils.h"           // for indexOf

int Remap::getLeftCells(const int cells) {
  int flLeftFaceOfCellC(mesh->getLeftFaceOfCell(cells));
  size_t flId(flLeftFaceOfCellC);
  int flFaces(utils::indexOf(mesh->getFaces(), flId));
  int cbBackCellF(mesh->getBackCell(flId));
  int cbId(cbBackCellF);
  if (cbId == -1)
    return cells;
  else
    return cbId;
}

int Remap::getRightCells(const int cells) {
  int frRightFaceOfCellC(mesh->getRightFaceOfCell(cells));
  size_t frId(frRightFaceOfCellC);
  int frFaces(utils::indexOf(mesh->getFaces(), frId));
  int cfFrontCellF(mesh->getFrontCell(frId));
  int cfId(cfFrontCellF);
  if (cfId == -1)
    return cells;
  else
    return cfId;
}

int Remap::getBottomCells(const int cells) {
  int fbBottomFaceOfCellC(mesh->getBottomFaceOfCell(cells));
  size_t fbId(fbBottomFaceOfCellC);
  int fbFaces(utils::indexOf(mesh->getFaces(), fbId));
  int cfFrontCellF(mesh->getFrontCell(fbId));
  int cfId(cfFrontCellF);
  if (cfId == -1)
    return cells;
  else
    return cfId;
}

int Remap::getTopCells(const int cells) {
  int ftTopFaceOfCellC(mesh->getTopFaceOfCell(cells));
  size_t ftId(ftTopFaceOfCellC);
  int ftFaces(utils::indexOf(mesh->getFaces(), ftId));
  int cbBackCellF(mesh->getBackCell(ftId));
  int cbId(cbBackCellF);
  if (cbId == -1)
    return cells;
  else
    return cbId;
}
