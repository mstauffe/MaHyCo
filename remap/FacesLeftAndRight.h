int cfCell1(-1), cbCell1(-1), cfCell2(-1), cbCell2(-1);
int fOfcfCell1(-1), frOfcfCell1(-1), fOfcbCell1(-1), flOfcbCell1(-1);
int fOfcfCell2(-1), frOfcfCell2(-1), fOfcbCell2(-1), flOfcbCell2(-1);
int fFace1(-1), frFace1(-1), flFace1(-1), fFace2(-1), frFace2(-1), flFace2(-1);
int nbfaces(0);
if (VerticalFaceOfNode(pNodes)[0] != -1) {
  // pour la premiere face verticale
  size_t fId1(VerticalFaceOfNode(pNodes)[0]);
  fFace1 = utils::indexOf(mesh->getFaces(), fId1);
  // on recupere la cellule devant
  int cfFrontCellF1(mesh->getFrontCell(fId1));
  int cfId1(cfFrontCellF1);
  cfCell1 = cfId1;
  // on recupere la face à droite de la cellule devant
  int frRightFaceOfCfCell1(mesh->getRightFaceOfCell(cfId1));
  size_t frId1(frRightFaceOfCfCell1);
  frFace1 = utils::indexOf(mesh->getFaces(), frId1);

  fOfcfCell1 = utils::indexOf(mesh->getFacesOfCell(cfId1), fId1);
  frOfcfCell1 = utils::indexOf(mesh->getFacesOfCell(cfId1), frId1);

  // on recupere la cellule derriere
  int cbBackCellF1(mesh->getBackCell(fId1));
  int cbId1(cbBackCellF1);
  cbCell1 = cbId1;
  // on recupere la face à gauche de la cellule derriere
  int flLeftFaceOfcbCell1(mesh->getLeftFaceOfCell(cbId1));
  size_t flId1(flLeftFaceOfcbCell1);
  flFace1 = utils::indexOf(mesh->getFaces(), flId1);

  fOfcbCell1 = utils::indexOf(mesh->getFacesOfCell(cbId1), fId1);
  flOfcbCell1 = utils::indexOf(mesh->getFacesOfCell(cbId1), flId1);

  nbfaces = nbfaces + 2;
}
if (VerticalFaceOfNode(pNodes)[1] != -1) {
  // pour la seconde face verticale
  size_t fId2(VerticalFaceOfNode(pNodes)[1]);
  fFace2 = utils::indexOf(mesh->getFaces(), fId2);
  // on recupere la cellule devant
  int cfFrontCellF2(mesh->getFrontCell(fId2));
  int cfId2(cfFrontCellF2);
  cfCell2 = cfId2;
  // on recupere la face à droite de la cellule devant
  int frRightFaceOfCfCell2(mesh->getRightFaceOfCell(cfId2));
  size_t frId2(frRightFaceOfCfCell2);
  frFace2 = utils::indexOf(mesh->getFaces(), frId2);
  fOfcfCell2 = utils::indexOf(mesh->getFacesOfCell(cfId2), fId2);
  frOfcfCell2 = utils::indexOf(mesh->getFacesOfCell(cfId2), frId2);

  // on recupere la cellule derriere
  int cbBackCellF2(mesh->getBackCell(fId2));
  int cbId2(cbBackCellF2);
  cbCell2 = cbId2;
  // on recupere la face à gauche de la cellule derriere
  int flLeftFaceOfcbCell2(mesh->getLeftFaceOfCell(cbId2));
  size_t flId2(flLeftFaceOfcbCell2);
  flFace2 = utils::indexOf(mesh->getFaces(), flId2);
  fOfcbCell2 = utils::indexOf(mesh->getFacesOfCell(cbId2), fId2);
  flOfcbCell2 = utils::indexOf(mesh->getFacesOfCell(cbId2), flId2);
  nbfaces = nbfaces + 2;
}
