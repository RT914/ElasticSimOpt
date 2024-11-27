#ifndef __HESSIANO_H__
#define __HESSIANO_H__

#include <Eigen/Dense>
#include "../Square.h"


Eigen::MatrixXd calHessianO(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);

#endif
