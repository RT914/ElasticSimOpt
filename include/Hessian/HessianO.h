#ifndef __HESSIANO_H__
#define __HESSIANO_H__

#include <Eigen/Dense>
#include "../Square.h"


Eigen::MatrixXd calHessianO(Square square, Eigen::VectorXd phi);

#endif
