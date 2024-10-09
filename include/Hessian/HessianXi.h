#ifndef __HESSIANXI_H__
#define __HESSIANXI_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::MatrixXd calHessianXi(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power);
Eigen::MatrixXd calHessianXi1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power);
Eigen::MatrixXd calHessianXi2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);
Eigen::MatrixXd calHessianXi3(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);

#endif
