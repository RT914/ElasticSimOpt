#ifndef __GRADIENTC_H__
#define __GRADIENTC_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::VectorXd calGradientc(Square square, Eigen::VectorXd power, Eigen::VectorXd theta);

#endif