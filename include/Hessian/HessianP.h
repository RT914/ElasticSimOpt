#ifndef __HESSIANP_H__
#define __HESSIANP_H__

#include <Eigen/Dense>
#include "..//Square.h"

Eigen::MatrixXd calHessianP(Square square, Eigen::VectorXd phi, Eigen::VectorXd theta);

#endif
