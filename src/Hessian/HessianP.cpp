#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"

// Calculate Hessian P
Eigen::MatrixXd calHessianP(Square square, Eigen::VectorXd phi, Eigen::VectorXd theta)
{
	Eigen::MatrixXd HessianP(3 * NumberOfParticles, NumberOfParticles);

	for (int xi = 0; xi < NumberOfParticles; xi++) {

		Eigen::Vector3i grid_xi = FlatToGrid(xi);

		for (int k = 0; k < NumberOfParticles; k++) {

			// initialize
			for (int d = 0; d < 3; d++) {
				HessianP(3 * k + d, xi) = 0.0;
			}

			Eigen::Vector3i grid_k = FlatToGrid(k);
			Eigen::Vector3i k_minus_xi = grid_k - grid_xi;

			for (int j = 0; j < NumberOfParticles; j++) {

				Eigen::Vector3i grid_j = FlatToGrid(j);
				Eigen::Vector3i j_minus_xi = grid_j - grid_xi;

				for (int i = 0; i < NumberOfParticles; i++) {

					Eigen::Vector3i grid_i = FlatToGrid(i);
					Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

					if ((abs(i_minus_xi[0]) <= 1) && (abs(i_minus_xi[1]) <= 1) && (abs(i_minus_xi[2]) <= 1)) {
						Eigen::Matrix3i matrix;
						matrix << i_minus_xi, j_minus_xi, k_minus_xi;

						Eigen::Vector3i axis(1, 2, 0); // ŠeŽŸŒ³‚Ì”‚©‚ç-1
						double w2w3w1 = RiemannSum2(matrix, axis, square.dx);
						Eigen::Vector3i axis(0, 2, 1); // ŠeŽŸŒ³‚Ì”‚©‚ç-1
						double w1w3w2 = RiemannSum2(matrix, axis, square.dx);
						Eigen::Vector3i axis(0, 1, 2); // ŠeŽŸŒ³‚Ì”‚©‚ç-1
						double w1w2w3 = RiemannSum2(matrix, axis, square.dx);

						Eigen::Vector3d VectorPhi;
						double Lphi1 = phi(3 * i + 1) * phi(3 * j + 2) - phi(3 * i + 2) * phi(3 * j + 1);
						double Lphi2 = phi(3 * i) * phi(3 * j + 2) - phi(3 * i + 2) * phi(3 * j);
						double Lphi3 = phi(3 * i) * phi(3 * j + 1) - phi(3 * i + 1) * phi(3 * j);
						VectorPhi << Lphi1, Lphi2, Lphi3;

						// update
						for (int d = 0; d < 3; d++) {
							HessianP(3 * k + d, xi) += VectorPhi(d) * (w2w3w1 - w1w3w2 + w1w2w3);
						}
					}
				}
			}
		}
	}

	return HessianP;
}