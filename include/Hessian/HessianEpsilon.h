#ifndef __HESSIANEPSILON_H__
#define __HESSIANEPSILON_H__

#include <Eigen/Dense>
#include "../Square.h"

void recursiveLoopForHesEpsilon(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices);
Eigen::MatrixXd calHessianEpsilon(const Square& square, const Eigen::VectorXd& phi);

#endif