#ifndef __NEWTONRAPHSONMETHOD_H__
#define __NEWTONRAPHSONMETHOD_H__

#include "Square.h"
#include <Eigen/Dense>

const double kappa = 1.0;
const double mu = 1.0;
const double rho = 1.0;
const double dt = 0.1;

Eigen::MatrixXd calMatrixS(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power, const Eigen::VectorXd& theta);
Eigen::VectorXd calVectore(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& phi_previous, const Eigen::VectorXd& power, const Eigen::VectorXd& theta);
Eigen::VectorXd Newton(Square square);
Eigen::VectorXd NewtonIteration(Square square);
void exportMatrix_CSV(Eigen::MatrixXd M, std::string file_name);
void exportVector_CSV(Eigen::VectorXd V, std::string file_name);
void exportLineSearch_CSV(std::ofstream& file, int line_search_times, double lambda, double f_prime_norm, double armijo_rhs, double nabla_f_prime_p_norm, double curvature_rhs);
void checkHessian(const Eigen::MatrixXd& H);
bool checkArmijo(const Eigen::VectorXd VectorA, const Eigen::VectorXd VectorB);
void createDirectoryIfNeeded(const std::string& dirPath);
void renderAndSave(Square square, int repetitionTime);

#endif
