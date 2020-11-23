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
    int Implosion = 3;
    int SodCaseX = 4;
    int SodCaseY = 5;
    int NohTestCase = 6;
    int AdvectionX = 7;
    int AdvectionY = 8;
    int AdvectionVitX = 9;
    int AdvectionVitY = 10;
    int BiUnitTestCase = 11;
    int BiSedovTestCase = 12;
    int BiTriplePoint = 13;
    int BiShockBubble = 14;
    int BiSodCaseX = 15;
    int BiSodCaseY = 16;
    int BiNohTestCase = 17;
    int BiAdvectionX = 18;
    int BiAdvectionY = 19;
    int BiAdvectionVitX = 20;
    int BiAdvectionVitY = 21;
    int BiImplosion = 22;
    int MonoRiderTx = 23;
    int MonoRiderTy = 24;
    int MonoRiderT45 = 25;
    int MonoRiderRotation = 26;
    int MonoRiderVortex = 27;
    int MonoRiderDeformation = 28;
    int MonoRiderVortexTimeReverse = 29;
    int MonoRiderDeformationTimeReverse = 30;
    int RiderTx = 31;
    int RiderTy = 32;
    int RiderT45 = 33;
    int RiderRotation = 34;
    int RiderVortex = 35;
    int RiderDeformation = 36;
    int RiderVortexTimeReverse = 37;
    int RiderDeformationTimeReverse = 38;


    int Nom = -1;
  };
  Test* test;

 private:
};
}  // namespace castestlib
#endif  // CASTEST_H
