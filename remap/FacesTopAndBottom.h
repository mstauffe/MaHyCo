  int cfCell1(-1), cbCell1(-1), cfCell2(-1), cbCell2(-1);
  int fOfcfCell1(-1), ftOfcbCell1(-1), fOfcbCell1(-1), fbOfcfCell1(-1);
  int fOfcfCell2(-1), ftOfcbCell2(-1), fOfcbCell2(-1), fbOfcfCell2(-1);
  int fFace1(-1),fbFace1(-1),ftFace1(-1),fFace2(-1),fbFace2(-1),ftFace2(-1);
  int nbfaces(0);
  // projection verticale
  if (HorizontalFaceOfNode(pNodes)[0] !=-1) {
    // attention backCell est au dessus
    // attention FrontCell est en dessous
    // pour la premiere face horizontale
    size_t fId1(HorizontalFaceOfNode(pNodes)[0]);
    fFace1 = utils::indexOf(mesh->getFaces(), fId1);  
     
    // on recupere la cellule au dessus
    int cbBackCellF1(mesh->getBackCell(fId1));
    int cbId1(cbBackCellF1);
    cbCell1 = cbId1;
    // on recupere la face au dessus de la cellule au dessus
    int ftTopFaceOfCbCell1(mesh->getTopFaceOfCell(cbId1));
    size_t ftId1(ftTopFaceOfCbCell1);
    ftFace1 = utils::indexOf(mesh->getFaces(), ftId1);

    fOfcbCell1 = utils::indexOf(mesh->getFacesOfCell(cbId1), fId1); 
    ftOfcbCell1 = utils::indexOf(mesh->getFacesOfCell(cbId1), ftId1);
	  
    // on recupere la cellule en dessous 
    int cfFrontCellF1(mesh->getFrontCell(fId1));
    int cfId1(cfFrontCellF1);
    cfCell1 = cfId1;
    // on recupere la face en dessous de la cellule en dessous
    int fbBottomFaceOfCell1(mesh->getBottomFaceOfCell(cfId1));
    size_t fbId1(fbBottomFaceOfCell1);
    fbFace1 = utils::indexOf(mesh->getFaces(), fbId1);
	  
    fOfcfCell1 = utils::indexOf(mesh->getFacesOfCell(cfId1), fId1); 
    fbOfcfCell1 = utils::indexOf(mesh->getFacesOfCell(cfId1), fbId1); 

    nbfaces = nbfaces + 2;
  }
      
  if (HorizontalFaceOfNode(pNodes)[1] !=-1) {
    // attention backCell est au dessus
    // attention FrontCell est en dessous
    // pour la seconde face horizontale
    size_t fId2(HorizontalFaceOfNode(pNodes)[1]);
    fFace2 = utils::indexOf(mesh->getFaces(), fId2);
      
    // on recupere la cellule au dessus
    int cbBackCellF2(mesh->getBackCell(fId2));
    int cbId2(cbBackCellF2);
    cbCell2 = cbId2;
    // on recupere la face au dessus de la cellule au dessus
    int ftTopFaceOfCfCells2(mesh->getTopFaceOfCell(cbId2));
    size_t ftId2(ftTopFaceOfCfCells2);
    ftFace2 = utils::indexOf(mesh->getFaces(), ftId2);
	  
    fOfcbCell2 = utils::indexOf(mesh->getFacesOfCell(cbId2), fId2); 
    ftOfcbCell2 = utils::indexOf(mesh->getFacesOfCell(cbId2), ftId2); 
      
    // on recupere la cellule en dessous "cf"
    int cfFrontCellF2(mesh->getFrontCell(fId2));
    int cfId2(cfFrontCellF2);
    cfCell2 = cfId2;
    // on recupere la face en dessous de la cellule en dessous
    int fbBottomFaceOfCell2(mesh->getBottomFaceOfCell(cfId2));
    size_t fbId2(fbBottomFaceOfCell2);
    fbFace2 = utils::indexOf(mesh->getFaces(), fbId2);

    fOfcfCell2 = utils::indexOf(mesh->getFacesOfCell(cfId2), fId2); 
    fbOfcfCell2 = utils::indexOf(mesh->getFacesOfCell(cfId2), fbId2); 

    nbfaces = nbfaces + 2;
  }
