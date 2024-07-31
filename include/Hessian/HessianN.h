#ifndef __HESSIANN_H__
#define __HESSIANN_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::MatrixXd calHessianN(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);

#endif
