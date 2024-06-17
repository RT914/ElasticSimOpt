#ifndef __HESSIANQ_H__
#define __HESSIANQ_H__

#include <Eigen/Dense>
#include "..//Square.h"

Eigen::MatrixXd calHessianQ(Square square, Eigen::VectorXd phi, Eigen::VectorXd theta);

#endif
