#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"

#include "../../include/Hessian/HessianXi.h"
#include "../../include/Hessian/HessianChi.h"
#include "../../include/Hessian/HessianEpsilon.h"
#include "../../include/Hessian/HessianUpsilon.h"
#include "../../include/Hessian/HessianN.h"



// Calculate Hessian N
Eigen::MatrixXd calHessianN(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power)
{
	Eigen::MatrixXd HessianN(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianEpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianXi(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianChi(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianUpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);

	// HessianEpsilon = calHessianEpsilon(square);
	// HessianXi = calHessianXi(square, phi, power);
	HessianChi = calHessianChi(square, re_phi, phi);
	// HessianUpsilon = calHessianUpsilon(square, phi, power);
	
	// HessianN = HessianEpsilon - HessianXi - HessianChi - HessianUpsilon;
	HessianN = HessianChi;

	return HessianN;
}

