#ifndef __FEM_H__
#define __FEM_H__

#include <Eigen/Dense>
#include "Square.h"

const int dimensions = 3;

const int NumberOfOneDemensionParticles = 3; // 3~5
const int NumberOfParticles = int(pow(NumberOfOneDemensionParticles, 3)); //125

Eigen::Vector3d calConflict(Eigen::Vector3d vel, Eigen::Vector3d pos);
Square fem(Square square, int SimulationTime);
void fem_vector(Square square, int SimulationTime);

#endif