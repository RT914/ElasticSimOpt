#ifndef __FEM_H__
#define __FEM_H__

#include <Eigen/Dense>

const int dimensions = 3;

const int NumberOfOneDemensionParticles = 3; // 3~5
const int NumberOfParticles = int(pow(NumberOfOneDemensionParticles, 3)); //125

void calVelocity();
Eigen::Vector3d calConflict(Eigen::Vector3d vel, Eigen::Vector3d pos);
void calPosition();
void fem(int SimulationTime);

#endif