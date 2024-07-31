#ifndef __HESSIANCHI_H__
#define __HESSIANCHI_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::MatrixXd calHessianChi(Square square, Eigen::VectorXd phi);
Eigen::MatrixXd calHessianChi1(Square square, Eigen::VectorXd phi);
Eigen::MatrixXd calHessianChi2(const Square& square, const Eigen::VectorXd& phi);
Eigen::MatrixXd calHessianChi3(const Square& square, const Eigen::VectorXd& phi);
void recursiveLoopForHesChi(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices);

#endif