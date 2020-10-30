#ifndef CSTMESH_H
#define CSTMESH_H

namespace cstmeshlib {

class ConstantesMaillagesClass {
 public:
  struct ConstantesMaillages {
    double X_LENGTH = 1.2;
    double Y_LENGTH = X_LENGTH;
    int X_EDGE_ELEMS = 30;
    int Y_EDGE_ELEMS = 30;
    double X_EDGE_LENGTH = X_LENGTH / X_EDGE_ELEMS;
    double Y_EDGE_LENGTH = Y_LENGTH / Y_EDGE_ELEMS;
  };
  ConstantesMaillages* cstmesh;

 private:
};
}  // namespace cstmeshlib
#endif  // CSTMESH_H
