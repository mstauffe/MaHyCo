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

void Remap::FacesOfNode() {
   Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes)
    {
      const Id pId(pNodes);
      VerticalFaceOfNode(pNodes)[0] = -1;
      VerticalFaceOfNode(pNodes)[1] = -1;
      HorizontalFaceOfNode(pNodes)[0] = -1;
      HorizontalFaceOfNode(pNodes)[1] = -1;
    });
  auto innerVerticalFaces(mesh->getInnerVerticalFaces());
  int nbInnerVerticalFaces(mesh->getNbInnerVerticalFaces());
  Kokkos::parallel_for("computenodalfluxes", nbInnerVerticalFaces,
    KOKKOS_LAMBDA(const int& fInnerVerticalFaces) {
    size_t fId(innerVerticalFaces[fInnerVerticalFaces]);
    int fFaces(utils::indexOf(mesh->getFaces(), fId));
    int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
    int n1Id(n1FirstNodeOfFaceF);
    int n1Nodes(n1Id);
    int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
    int n2Id(n2SecondNodeOfFaceF);
    int n2Nodes(n2Id);
    
    if (VerticalFaceOfNode(n1Nodes)[0] == -1) VerticalFaceOfNode(n1Nodes)[0] = fId;
    else VerticalFaceOfNode(n1Nodes)[1] = fId;
    
    if (VerticalFaceOfNode(n2Nodes)[0] == -1) VerticalFaceOfNode(n2Nodes)[0] = fId;
    else VerticalFaceOfNode(n2Nodes)[1] = fId;
  });
  auto innerHorizontalFaces(mesh->getInnerHorizontalFaces());
  int nbInnerHorizontalFaces(mesh->getNbInnerHorizontalFaces());
  Kokkos::parallel_for("computenodalfluxes", nbInnerHorizontalFaces,
    KOKKOS_LAMBDA(const int& fInnerHorizontalFaces) {
    size_t fId(innerHorizontalFaces[fInnerHorizontalFaces]);
    int fFaces(utils::indexOf(mesh->getFaces(), fId));
    int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
    int n1Id(n1FirstNodeOfFaceF);
    int n1Nodes(n1Id);
    int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
    int n2Id(n2SecondNodeOfFaceF);
    int n2Nodes(n2Id);

    if (HorizontalFaceOfNode(n1Nodes)[0] == -1) HorizontalFaceOfNode(n1Nodes)[0] = fId;
    else HorizontalFaceOfNode(n1Nodes)[1] = fId;

    if (HorizontalFaceOfNode(n2Nodes)[0] == -1) HorizontalFaceOfNode(n2Nodes)[0] = fId;
    else HorizontalFaceOfNode(n2Nodes)[1] = fId;
  });
  // Verification 
  // auto faces(mesh->getFaces());
  // Kokkos::parallel_for(
  //     "computeFaceQuantities", nbFaces, KOKKOS_LAMBDA(const int& fFaces) {
  //       int fId(faces[fFaces]);	
  // 	int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
  //       int n1Id(n1FirstNodeOfFaceF);
  //       int n1Nodes(n1Id);
  //       int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
  //       int n2Id(n2SecondNodeOfFaceF);
  //       int n2Nodes(n2Id);
  // 	std::cout << " face " << fFaces << " n1Nodes " <<  n1Nodes << " n2Nodes " << n2Nodes << std::endl;
  //     });
  //   const auto innerNodes(mesh->getInnerNodes());
  //   const size_t nbInnerNodes(mesh->getNbInnerNodes());
  //   Kokkos::parallel_for(nbInnerNodes, KOKKOS_LAMBDA(const size_t& pInnerNodes)
  //     {
  // 	const Id pId(innerNodes[pInnerNodes]);
  // 	const size_t pNodes(pId);
  //     std::cout << " pNodes " <<  pNodes << " 1.H " << HorizontalFaceOfNode(pNodes)[0]
  // 		<< " 2.H " << HorizontalFaceOfNode(pNodes)[1] << std::endl;
  //     std::cout << " pNodes " <<  pNodes << " 1.V " <<  VerticalFaceOfNode(pNodes)[0]
  // 	        << " 2.V " <<  VerticalFaceOfNode(pNodes)[1]   << std::endl;
  //   });	
}
// int Remap::getLeftNode(const int node) {
//    size_t i,j;
//    pair<size_t, size_t>(i,j) = CartesianMesh2D::id2IndexNode(node);
//    return CartesianMesh2D::index2IdNode(i-1,j);
// }
// int Remap::getRightNode(const int node) {
//    size_t i,j;
//    pair<size_t, size_t>(i,j) = CartesianMesh2D::id2IndexNode(node);
//    return CartesianMesh2D::index2IdNode(i+1,j);
// }
// int Remap::getBottomNode(const int node) {
//    size_t i,j;
//    pair<size_t, size_t>(i,j) = CartesianMesh2D::id2IndexNode(node);
//    return index2IdNode(i,j-1);
// }
// int Remap::getTopNode(const int node) {
//    size_t i,j;
//    pair<size_t, size_t>(i,j) = CartesianMesh2D::id2IndexNode(node);
//    return index2IdNode(i,j+1);
// }
