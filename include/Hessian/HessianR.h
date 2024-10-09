#ifndef __HESSIANR_H__
#define __HESSIANR_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::MatrixXd calHessianR(const Square& square, const Eigen::VectorXd& theta);

#endif
