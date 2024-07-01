#ifndef __GRADIENTB_H__
#define __GRADIENTB_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::VectorXd calGradientb(Square square, Eigen::VectorXd phi, Eigen::VectorXd theta);

#endif