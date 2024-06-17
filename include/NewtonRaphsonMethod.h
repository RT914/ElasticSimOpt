#ifndef __NEWTONRAPHSONMETHOD_H__
#define __NEWTONRAPHSONMETHOD_H__

#include "Square.h"
#include <Eigen/Dense>

double kappa = 1.0;

Eigen::VectorXd Newton(Square square);

#endif
