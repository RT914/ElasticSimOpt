#ifndef __GRADIENTC_H__
#define __GRADIENTC_H__

#include <Eigen/Dense>
#include "../Square.h"


Eigen::VectorXd calGradientc(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power, const Eigen::VectorXd& theta);
Eigen::VectorXd calGradientc1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power, const Eigen::VectorXd& theta);
Eigen::VectorXd calGradientc2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& theta);


#endif