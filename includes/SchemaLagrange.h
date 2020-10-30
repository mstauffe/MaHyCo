#ifndef SCHEMALAGRANGE_H
#define SCHEMALAGRANGE_H

namespace schemalagrangelib {

class SchemaLagrangeClass {
 public:
  struct SchemaLagrange {
    int Eucclhyd = 2000;
    int VNR = 2001;
    int CSTS = 2002;
    int MYR = 2003;

    int schema = -1;
  };
  SchemaLagrange* schemas;

 private:
};
}  // namespace schemalagrangelib
#endif  // SCHEMALAGRANGE_H
