#ifndef __NEWTONRAPHSONMETHOD_H__
#define __NEWTONRAPHSONMETHOD_H__

#include "Square.h"
#include <Eigen/Dense>

const double kappa = 1.0;
const double mu = 1.0;
const double rho = 1.0;
const double dt = 1.0;

Eigen::VectorXd Newton(Square square);
void exportMatrix_CSV(Eigen::MatrixXd M, std::string file_name);

#endif
