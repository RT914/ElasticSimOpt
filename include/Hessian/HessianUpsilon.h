#ifndef __HESSIANUPSILON_H__
#define __HESSIANUPSILON_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::MatrixXd calHessianUpsilon(Square square, Eigen::VectorXd phi, Eigen::VectorXd power);
Eigen::MatrixXd calHessianUpsilon1(const Square& square, const Eigen::VectorXd& phi);
Eigen::MatrixXd calHessianUpsilon2(const Square& square, const Eigen::VectorXd& phi);
Eigen::MatrixXd calHessianUpsilon3(const Square& square, const Eigen::VectorXd& phi, const Eigen::VectorXd& power);

#endif