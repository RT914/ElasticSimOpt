#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"

// Calculate Hessian N
Eigen::MatrixXd calHessianN(Square square, Eigen::VectorXd phi)
{
	Eigen::MatrixXd HessianN(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianEpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianXi(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianChi(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianUpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);
	
	HessianN = HessianEpsilon + HessianXi + HessianChi + HessianUpsilon;

	return HessianN;
}

