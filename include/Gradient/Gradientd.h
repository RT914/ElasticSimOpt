#ifndef __GRADIENTD_H__
#define __GRADIENTD_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::VectorXd calGradientd(Square square, Eigen::VectorXd phi, Eigen::VectorXd phi_previous, Eigen::VectorXd power);
Eigen::VectorXd calGradientd1(Square square, Eigen::VectorXd phi, Eigen::VectorXd phi_previous);
Eigen::VectorXd calGradientd2(Square square, Eigen::VectorXd phi);
Eigen::VectorXd calGradientd3(Square square, Eigen::VectorXd phi);
Eigen::VectorXd calGradientd4(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);

#endif