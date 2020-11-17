#include <math.h>  // for floor, sqrt

#include <Kokkos_Core.hpp>  // for deep_copy
#include <algorithm>        // for copy
#include <array>            // for array
#include <iostream>         // for operator<<, basic_ostream::ope...
#include <vector>           // for allocator, vector

#include "Init.h"

#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "types/MathFunctions.h"   // for max, norm, dot
#include "types/MultiArray.h"      // for operator<<
#include "utils/Utils.h"           // for indexOf

namespace initlib {
/**
 * Job initMeshGeometryForCells called @1.0 in simulate method.
 * In variables: X
 * Out variables: m_cell_coord, m_cell_coord_x, m_cell_coord_y,
 * m_cell_perimeter, v
 */
void Initialisations::initMeshGeometryForCells() noexcept {
  Kokkos::parallel_for(
      "initMeshGeometryForCells", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        int cId(cCells);
        double reduction11 = 0.0;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            reduction11 =
                reduction11 + (crossProduct2d(m_node_coord_n0(pNodes),
                                              m_node_coord_n0(pPlus1Nodes)));
          }
        }
        double vol = 0.5 * reduction11;
        RealArray1D<dim> reduction12 = zeroVect;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            reduction12 =
                reduction12 +
                ((crossProduct2d(m_node_coord_n0(pNodes),
                                 m_node_coord_n0(pPlus1Nodes)) *
                  ((m_node_coord_n0(pNodes) + m_node_coord_n0(pPlus1Nodes)))));
          }
        }
        RealArray1D<dim> xc = (1.0 / (6.0 * vol) * reduction12);
        m_cell_coord_n0(cCells) = xc;
        m_euler_volume_n0(cCells) = vol;
        double reduction13 = 0.0;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            reduction13 = reduction13 +
                          (MathFunctions::norm((m_node_coord_n0(pNodes) -
                                                m_node_coord_n0(pPlus1Nodes))));
          }
        }
        m_cell_perimeter_n0(cCells) = reduction13;
      });
}
/**
 * Job initCellPos called @1.0 in simulate method.
 * In variables: m_node_coord_n0
 * Out variables: m_cell_coord_n0
 */
void Initialisations::initCellPos() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    const Id cId(cCells);
    RealArray1D<2> reduction0({0.0, 0.0});
    {
      const auto nodesOfCellC(mesh->getNodesOfCell(cId));
      const size_t nbNodesOfCellC(nodesOfCellC.size());
      for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
           pNodesOfCellC++) {
        const Id pId(nodesOfCellC[pNodesOfCellC]);
        const size_t pNodes(pId);
        reduction0 = sumR1(reduction0, m_node_coord_n0(pNodes));
      }
    }
    m_cell_coord_n0(cCells) = 0.25 * reduction0;
  });
}
/**
 * Job initMeshGeometryForFaces called @2.0 in simulate method.
 * In variables: X, m_cell_coord, ex, ey, threshold
 * Out variables: Xf, faceLength, faceNormal, outerFaceNormal
 */
void Initialisations::initMeshGeometryForFaces() noexcept {
  auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "initMeshGeometryForFaces", nbFaces, KOKKOS_LAMBDA(const int& fFaces) {
        size_t fId(faces[fFaces]);
        int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
        int n1Id(n1FirstNodeOfFaceF);
        int n1Nodes(n1Id);
        int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
        int n2Id(n2SecondNodeOfFaceF);
        int n2Nodes(n2Id);
        RealArray1D<dim> X_face =
            (0.5 * ((m_node_coord_n0(n1Nodes) + m_node_coord_n0(n2Nodes))));
        RealArray1D<dim> face_vec =
            (m_node_coord_n0(n2Nodes) - m_node_coord_n0(n1Nodes));
        varlp->Xf(fFaces) = X_face;
        varlp->faceLength(fFaces) = MathFunctions::norm(face_vec);
        varlp->faceNormal(fFaces) = zeroVect;
        {
          auto cellsOfFaceF(mesh->getCellsOfFace(fId));
          for (int cCellsOfFaceF = 0; cCellsOfFaceF < cellsOfFaceF.size();
               cCellsOfFaceF++) {
            int cId(cellsOfFaceF[cCellsOfFaceF]);
            int cCells(cId);
            int fFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), fId));
            varlp->outerFaceNormal(cCells, fFacesOfCellC) =
                (((X_face - m_cell_coord_n0(cCells))) /
                 MathFunctions::norm((X_face - m_cell_coord_n0(cCells))));
            if (cstmesh->cylindrical_mesh == 1) {
              varlp->faceNormal(fFaces)[0] +=
                  0.5 *
                  std::abs(varlp->outerFaceNormal(cCells, fFacesOfCellC)[0]);
              varlp->faceNormal(fFaces)[1] +=
                  0.5 *
                  std::abs(varlp->outerFaceNormal(cCells, fFacesOfCellC)[1]);
            }
          }
        }
        if (cstmesh->cylindrical_mesh != 1) {
          RealArray1D<dim> face_normal;
          if (MathFunctions::fabs(dot(face_vec, ex)) < options->threshold)
            face_normal = ex;
          else
            face_normal = ey;

          varlp->faceNormal(fFaces) = face_normal;
        }
        // std::cout << nbFaces << " "
        //	  <<  varlp->faceNormal(fFaces) << std::endl;
      });
}

}  // namespace initlib
