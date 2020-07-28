#ifndef CASTEST_H
#define CASTEST_H

namespace castestlib {

class CasTest {
 public:
  struct Test {
    // cas test
    int UnitTestCase = 0;
    int SedovTestCase = 1;
    int TriplePoint = 2;
    int SodCaseX = 4;
    int SodCaseY = 5;
    int NohTestCase = 6;
    int BiUnitTestCase = 10;
    int BiSedovTestCase = 11;
    int BiTriplePoint = 12;
    int BiShockBubble = 13;
    int BiSodCaseX = 14;
    int BiSodCaseY = 15;
    int BiNohTestCase = 16;

    int Nom = -1;
  };
  Test* test;

 private:
};
}  // namespace castestlib
#endif  // CASTEST_H
