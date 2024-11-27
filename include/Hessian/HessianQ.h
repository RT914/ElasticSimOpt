#ifndef __HESSIANQ_H__
#define __HESSIANQ_H__

#include <Eigen/Dense>
#include "../Square.h"


Eigen::MatrixXd calHessianQ(const Square& square, const Eigen::VectorXd& re_phi);

#endif
