#ifndef __HESSIANCHI_H__
#define __HESSIANCHI_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::MatrixXd calHessianChi(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
Eigen::MatrixXd calHessianChi1(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
Eigen::MatrixXd calHessianChi2(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
Eigen::MatrixXd calHessianChi3(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
void recursiveLoopForHesChi(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices);

#endif