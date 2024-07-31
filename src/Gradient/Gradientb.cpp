#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/Gradient/Gradientb.h"

// Calculate Gradient b
Eigen::VectorXd calGradientb(Square square, Eigen::VectorXd phi, Eigen::VectorXd theta) {
	Eigen::VectorXd Gradientb(3 * NumberOfParticles);
	Gradientb.setZero();

	Eigen::VectorXd Gradientb1 = calGradientb1(square, theta);
	Eigen::VectorXd Gradientb2 = calGradientb2(square, phi);

	Gradientb = Gradientb1 + Gradientb2;

	return Gradientb;
}

void recursiveLoopForGradientb(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices) {
	if (level == maxLevel) {
		process(indices);
		return;
	}

	for (int i = 0; i < NumberOfParticles; i++) {
		indices[level] = i;
		recursiveLoopForGradientb(level + 1, maxLevel, process, indices);
	}
}

Eigen::VectorXd calGradientb1(Square square, Eigen::VectorXd theta)
{
	Eigen::VectorXd Gradientb1(NumberOfParticles);
	Gradientb1.setZero();

	auto processIndices = [&](std::vector<int>& indices) {

		int i = indices[1], xi = indices[0];

		Eigen::Vector3i grid_xi = FlatToGrid(xi);

		Eigen::Vector3i grid_i = FlatToGrid(i);
		Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

		// 積分計算
		double W = RiemannSum1(i_minus_xi, square.dx);

		// 更新
		Gradientb1(xi) += theta[i] * W;

		};

	std::vector<int> indices(2); // 2つのインデックス用のベクター
	recursiveLoopForGradientb(0, 2, processIndices, indices); // 2重ループを再帰で実行

	return Gradientb1;
}

Eigen::VectorXd calGradientb2(Square square, Eigen::VectorXd phi)
{
	Eigen::VectorXd Gradientb2(NumberOfParticles);
	Gradientb2.setZero();

	auto processIndices = [&](std::vector<int>& indices) {

		int i = indices[3], j = indices[2], k = indices[1], xi = indices[0];

		Eigen::Vector3i grid_xi = FlatToGrid(xi);

		Eigen::Vector3i grid_k = FlatToGrid(k);
		Eigen::Vector3i k_minus_xi = grid_k - grid_xi;

		Eigen::Vector3i grid_j = FlatToGrid(j);
		Eigen::Vector3i j_minus_xi = grid_j - grid_xi;

		Eigen::Vector3i grid_i = FlatToGrid(i);
		Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

		Eigen::Matrix3i matrix(3,3);
		matrix << i_minus_xi, j_minus_xi, k_minus_xi;

		Eigen::Vector3i axis(0, 1, 2); // 各次元の数から-1した値を挿入
		double W = RiemannSum6(matrix, axis, square.dx);

		// phiの計算
		double Lphi1 = phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1);
		double Lphi2 = phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k);
		double Lphi3 = phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);
		double Lphi = phi(3 * i) * Lphi1 - phi(3 * i + 1) * Lphi2 + phi(3 * i + 2) * Lphi3;

		// 更新
		Gradientb2(xi) += Lphi * W;

		};

	std::vector<int> indices(2); // 2つのインデックス用のベクター
	recursiveLoopForGradientb(0, 4, processIndices, indices); // 2重ループを再帰で実行

	return Gradientb2;
}
