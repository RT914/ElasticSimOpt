#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/Hessian/HessianQ.h"


// Calculate Hessian Q
Eigen::MatrixXd calHessianQ(Square square)
{
	Eigen::MatrixXd HessianQ(NumberOfParticles, NumberOfParticles);

	for (int xi = 0; xi < NumberOfParticles; xi++) {

		Eigen::Vector3i grid_xi = FlatToGrid(xi);

		for (int i = 0; i < NumberOfParticles; i++) {

			HessianQ(i, xi) = 0.0;

			Eigen::Vector3i grid_i = FlatToGrid(i);
			Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

			if ((abs(i_minus_xi[0]) <= 1) && (abs(i_minus_xi[1]) <= 1) && (abs(i_minus_xi[2]) <= 1)) {
                HessianQ(i, xi) = - RiemannSum1(i_minus_xi, square.dx);
			}
		}
	}

    return HessianQ;
}
