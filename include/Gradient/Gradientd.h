#ifndef __GRADIENTD_H__
#define __GRADIENTD_H__

#include <Eigen/Dense>
#include "../Square.h"


Eigen::VectorXd calGradientd(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& phi_previous, const Eigen::VectorXd& power);
Eigen::VectorXd calGradientG1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& phi_previous);
Eigen::VectorXd calGradientG2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power);
Eigen::VectorXd calGradientG2_1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);
Eigen::VectorXd calGradientG2_2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi);
Eigen::VectorXd calGradientG2_3(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power);

#endif