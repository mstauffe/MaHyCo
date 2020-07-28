#include "EucclhydRemap.h"         // for EucclhydRemap, EucclhydRemap::Options
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "utils/Utils.h"           // for indexOf

int EucclhydRemap::getLeftCells(const int cells) {
  int flLeftFaceOfCellC(mesh->getLeftFaceOfCell(cells));
  int flId(flLeftFaceOfCellC);
  int flFaces(utils::indexOf(mesh->getFaces(), flId));
  int cbBackCellF(mesh->getBackCell(flId));
  int cbId(cbBackCellF);
  if (cbId == -1)
    return cells;
  else
    return cbId;
}

int EucclhydRemap::getRightCells(const int cells) {
  int frRightFaceOfCellC(mesh->getRightFaceOfCell(cells));
  int frId(frRightFaceOfCellC);
  int frFaces(utils::indexOf(mesh->getFaces(), frId));
  int cfFrontCellF(mesh->getFrontCell(frId));
  int cfId(cfFrontCellF);
  if (cfId == -1)
    return cells;
  else
    return cfId;
}

int EucclhydRemap::getBottomCells(const int cells) {
  int fbBottomFaceOfCellC(mesh->getBottomFaceOfCell(cells));
  int fbId(fbBottomFaceOfCellC);
  int fbFaces(utils::indexOf(mesh->getFaces(), fbId));
  int cfFrontCellF(mesh->getFrontCell(fbId));
  int cfId(cfFrontCellF);
  if (cfId == -1)
    return cells;
  else
    return cfId;
}

int EucclhydRemap::getTopCells(const int cells) {
  int ftTopFaceOfCellC(mesh->getTopFaceOfCell(cells));
  int ftId(ftTopFaceOfCellC);
  int ftFaces(utils::indexOf(mesh->getFaces(), ftId));
  int cbBackCellF(mesh->getBackCell(ftId));
  int cbId(cbBackCellF);
  if (cbId == -1)
    return cells;
  else
    return cbId;
}
