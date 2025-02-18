#ifndef __GRADIENTD_H__
#define __GRADIENTD_H__

#include <Eigen/Dense>
#include "../Square.h"


Eigen::VectorXd calGradientd(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& phi_current, const Eigen::VectorXd& phi_previous, const Eigen::VectorXd& power);
Eigen::VectorXd calGradientd1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);
Eigen::VectorXd calGradientd2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);
Eigen::VectorXd calGradientd3(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power);
Eigen::VectorXd calGradientd4(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& phi_current, const Eigen::VectorXd& phi_previous);

#endif