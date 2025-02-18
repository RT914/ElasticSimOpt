#ifndef __HESSIANN_H__
#define __HESSIANN_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::MatrixXd calHessianN(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power);
Eigen::MatrixXd calHessianEpsilon(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);

#endif
