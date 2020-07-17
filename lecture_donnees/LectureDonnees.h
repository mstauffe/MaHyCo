#ifndef LECTURE_DONNEES_H
#define LECTURE_DONNEES_H

#include <string>  // for string

#include "EucclhydRemap.h"  // for EucclhydRemap::Options

void LectureDonnees(string Fichier, EucclhydRemap::Options* o,
		    limiteurslib::LimiteursClass::Limiteurs* l,
		    eoslib::EquationDetat::Eos* eos,
		    castestlib::CasTest::Test* test);

#endif  // LECTURE_DONNEES_H
