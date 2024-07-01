#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"

// Calculate Hessian R
Eigen::MatrixXd calHessianO(Square square, Eigen::VectorXd phi)
{
	Eigen::MatrixXd HessianO(NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXi grid_matrix;
	Eigen::Vector3d VectorPhi;
	Eigen::Vector3i axis1, axis2, axis3;

	for (int xi = 0; xi < NumberOfParticles; xi++) {
		Eigen::Vector3i grid_xi = FlatToGrid(xi);

		for (int k = 0; k < NumberOfParticles; k++) {
			// initialize
			HessianO(k, xi) = 0.0;

			Eigen::Vector3i grid_k = FlatToGrid(k);
			Eigen::Vector3i k_minus_xi = grid_k - grid_xi;

			for (int j = 0; j < NumberOfParticles; j++) {
				Eigen::Vector3i grid_j = FlatToGrid(j);
				Eigen::Vector3i j_minus_xi = grid_j - grid_xi;

				for (int i = 0; i < NumberOfParticles; i++) {
					Eigen::Vector3i grid_i = FlatToGrid(i);
					Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

					if ((abs(i_minus_xi[0]) <= 1) && (abs(i_minus_xi[1]) <= 1) && (abs(i_minus_xi[2]) <= 1)) {
						grid_matrix << i_minus_xi, j_minus_xi, k_minus_xi;
						double Lphi1 = phi[3 * i + 1] * phi[3 * j + 2] - phi[3 * i + 2] * phi[3 * j + 1];
						double Lphi2 = -(phi[3 * i] * phi[3 * j + 2] - phi[3 * i + 2] * phi[3 * j]);
						double Lphi3 = phi[3 * i] * phi[3 * j + 1] - phi[3 * i + 1] * phi[3 * j];
						VectorPhi << Lphi1, Lphi2, Lphi3;

						axis1 << 1, 2, 0; // ŠeŽŸŒ³‚Ì”‚©‚ç-1
						double w2w3w1 = RiemannSum6(grid_matrix, axis1, square.dx);
						axis2 << 0, 2, 1; // ŠeŽŸŒ³‚Ì”‚©‚ç-1
						double w1w3w2 = RiemannSum6(grid_matrix, axis2, square.dx);
						axis3 << 0, 1, 2; // ŠeŽŸŒ³‚Ì”‚©‚ç-1
						double w1w2w3 = RiemannSum6(grid_matrix, axis3, square.dx);

						// update
						for (int d = 0; d < dimensions; d++) {
							HessianO(k, 3 * xi + d) += VectorPhi(d) * (w2w3w1 - w1w3w2 + w1w2w3);
						}
					}

				}

			}

		}

	}


	return HessianO;
}
