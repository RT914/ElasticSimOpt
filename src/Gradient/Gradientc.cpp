#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"

// Calculate Gradient c
Eigen::VectorXd calGradientc(Square square, Eigen::VectorXd power, Eigen::VectorXd theta)
{
	Eigen::VectorXd Gradientc(NumberOfParticles);

	for (int xi = 0; xi < NumberOfParticles; xi++) {

		Eigen::Vector3i grid_xi = FlatToGrid(xi);
		// ‘æˆê€
		Gradientc(xi) = (kappa / 2) * RiemannSum4(grid_xi, theta, square.dx);

		for (int i = 0; i < NumberOfParticles; i++) {

			Eigen::Vector3i grid_i = FlatToGrid(i);
			Eigen::Vector3i i_minus_xi = grid_i - grid_xi;
			
			// ‘æˆê€
			Gradientc(xi) += (power[i] - (kappa / 2) * theta[i]) * RiemannSum1(i_minus_xi, square.dx);
		}
	}

	return Gradientc;
}
