#ifndef __HESSIANR_H__
#define __HESSIANR_H__

#include <Eigen/Dense>
#include "../Square.h"

void recursiveLoopForHesR(int depth, int maxDepth, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices);
Eigen::MatrixXd calHessianR(const Square& square, const Eigen::VectorXd& theta);

#endif
