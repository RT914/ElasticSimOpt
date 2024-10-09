#ifndef __HESSIANP_H__
#define __HESSIANP_H__

#include <Eigen/Dense>
#include "../Square.h"


Eigen::MatrixXd calHessianP(const Square& square, const Eigen::VectorXd& phi);

#endif
