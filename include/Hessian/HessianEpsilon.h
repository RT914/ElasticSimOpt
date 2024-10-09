#ifndef __HESSIANEPSILON_H__
#define __HESSIANEPSILON_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::MatrixXd calHessianEpsilon(const Square& square);

#endif