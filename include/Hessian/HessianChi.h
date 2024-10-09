#ifndef __HESSIANCHI_H__
#define __HESSIANCHI_H__

#include <Eigen/Dense>
#include "../Square.h"

Eigen::MatrixXd calHessianChi(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);
Eigen::MatrixXd calHessianChi1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);
Eigen::MatrixXd calHessianChi2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);
Eigen::MatrixXd calHessianChi3(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);

#endif