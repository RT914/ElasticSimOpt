#ifndef __GRADIENTB_H__
#define __GRADIENTB_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::VectorXd calGradientb(Square square, Eigen::VectorXd phi, Eigen::VectorXd theta);
Eigen::VectorXd calGradientb1(Square square, Eigen::VectorXd theta);
Eigen::VectorXd calGradientb2(Square square, Eigen::VectorXd phi);

#endif