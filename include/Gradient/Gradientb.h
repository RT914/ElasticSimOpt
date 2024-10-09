#ifndef __GRADIENTB_H__
#define __GRADIENTB_H__

#include <Eigen/Dense>
#include "../Square.h"


Eigen::VectorXd calGradientb(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& theta);
Eigen::VectorXd calGradientb1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& theta);
Eigen::VectorXd calGradientb2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);

#endif