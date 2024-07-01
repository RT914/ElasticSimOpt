#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"


// Calculate Gradient b
Eigen::VectorXd calGradientb(Square square, Eigen::VectorXd phi, Eigen::VectorXd theta)
{
	Eigen::VectorXd Gradientb(NumberOfParticles);
	double Lphi1, Lphi2, Lphi3, Lphi;
	

	for (int xi = 0; xi < NumberOfParticles; xi++) {

		Eigen::Vector3i grid_xi = FlatToGrid(xi);

		// initialize
		for (int d = 0; d < 3; d++) {
			Gradientb(xi) = 0.0;
		}

		for (int k = 0; k < NumberOfParticles; k++) {

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

						Eigen::Vector3i axis(0, 1, 2); // ŠeŽŸŒ³‚Ì”‚©‚ç-1‚µ‚½’l‚ð‘}“ü
						double w1w2w3 = RiemannSum6(matrix, axis, square.dx);

						Lphi1 = phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1);
						Lphi2 = phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k);
						Lphi3 = phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);
						Lphi = phi(3 * i) * Lphi1 - phi(3 * i + 1) * Lphi2 + phi(3 * i + 2) * Lphi3;

						// ‘æ“ñ€‚ÌŒvŽZ
						Gradientb(xi) += Lphi * w1w2w3;
					}
				}
			}

			// ‘æˆê€‚ÌŒvŽZ
			Gradientb(xi) += theta[k] * RiemannSum1(k_minus_xi, square.dx);

		}
	}

	return Gradientb;
}
