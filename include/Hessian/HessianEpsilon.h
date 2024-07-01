#ifndef __HESSIANEPSILON_H__
#define __HESSIANEPSILON_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::MatrixXd calHessianEpsilon(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
Eigen::MatrixXd calHessianEpsilon1(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
Eigen::MatrixXd calHessianEpsilon2(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
Eigen::MatrixXd calHessianEpsilon3(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
void recursiveLoopForHesEpsilon(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices);

#endif