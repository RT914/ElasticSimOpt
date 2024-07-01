#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"

// Calculate Hessian R
Eigen::MatrixXd calHessianR(Square square, Eigen::VectorXd theta)
{
	Eigen::MatrixXd HessianR(NumberOfParticles, NumberOfParticles);

	for (int xi = 0; xi < NumberOfParticles; xi++) {
		Eigen::Vector3i grid_xi = FlatToGrid(xi);

		for (int i = 0; i < NumberOfParticles; i++) {
			HessianR(i, xi) = 0.0;
			Eigen::Vector3i grid_i = FlatToGrid(i);
			Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

			if ((abs(i_minus_xi[0]) <= 1) && (abs(i_minus_xi[1]) <= 1) && (abs(i_minus_xi[2]) <= 1)) {
				HessianR(i, xi) = (kappa/2) * ( RiemannSum1(i_minus_xi, square.dx) + RiemannSum3(i_minus_xi, grid_xi, theta, square.dx) );
			}
		}
	}
	

	return HessianR;
}
