#include <Eigen/Dense>
#include <iostream>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"

#include "../../include/Hessian/HessianXi.h"
#include "../../include/Hessian/HessianChi.h"
#include "../../include/Hessian/HessianUpsilon.h"
#include "../../include/Hessian/HessianN.h"

// Calculate Hessian N
Eigen::MatrixXd calHessianN(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power)
{
	Eigen::MatrixXd HessianN(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianXi(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianChi(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianUpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);

	HessianXi = calHessianXi(square, re_phi, phi, power);
	exportMatrix_CSV(HessianXi, "csv/HessianXi.csv");
	if (HessianXi.array().isNaN().any()) {
		std::cerr << "NaN detected HessianXi" << std::endl;
	}

	HessianChi = calHessianChi(square, re_phi, phi);
	exportMatrix_CSV(HessianChi, "csv/HessianChi.csv");
	if (HessianChi.array().isNaN().any()) {
		std::cerr << "NaN detected HessianChi" << std::endl;
	}

	HessianUpsilon = calHessianUpsilon(square, re_phi, phi, power);
	exportMatrix_CSV(HessianUpsilon, "csv/HessianUpsilon.csv");
	if (HessianUpsilon.array().isNaN().any()) {
		std::cerr << "NaN detected HessianUpsilon" << std::endl;
	}

	HessianN = HessianXi - HessianChi + HessianUpsilon;

	return HessianN;
}

