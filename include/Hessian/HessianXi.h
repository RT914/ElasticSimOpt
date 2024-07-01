#ifndef __HESSIANXI_H__
#define __HESSIANXI_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::MatrixXd calHessianXi(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
Eigen::MatrixXd calHessianXi1(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
Eigen::MatrixXd calHessianXi2(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
Eigen::MatrixXd calHessianXi3(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
void recursiveLoopForHesXi(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices);

#endif
