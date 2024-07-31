#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Gradient/Gradientd.h"


Eigen::VectorXd calGradientd(Square square, Eigen::VectorXd phi, Eigen::VectorXd phi_previous, Eigen::VectorXd power) {
    Eigen::VectorXd Gradientd(3 * NumberOfParticles);
    Gradientd.setZero();

    Eigen::VectorXd Gradientd1 = calGradientd1(square, phi, phi_previous);
    Eigen::VectorXd Gradientd2 = calGradientd2(square, phi);
    Eigen::VectorXd Gradientd3 = calGradientd3(square, phi);
    Eigen::VectorXd Gradientd4 = calGradientd4(square, phi, power);

    Gradientd = - Gradientd1 - Gradientd2 - Gradientd3 - Gradientd4;

    return Gradientd;
}

void recursiveLoopForGradientd(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices) {
	if (level == maxLevel) {
		process(indices);
		return;
	}

	for (int i = 0; i < NumberOfParticles; i++) {
		indices[level] = i;
		recursiveLoopForGradientd(level + 1, maxLevel, process, indices);
	}
}

// Calculate Gradient d1
Eigen::VectorXd calGradientd1(Square square, Eigen::VectorXd phi, Eigen::VectorXd phi_previous)
{
    Eigen::VectorXd Gradientd1(3 * NumberOfParticles);
    Gradientd1.setZero();

    auto processIndices = [&](std::vector<int>& indices) {

        int i = indices[1], xi = indices[0];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);

        Eigen::Vector3i grid_i = FlatToGrid(i);
        Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

        // 積分計算
        double W =RiemannSum1(i_minus_xi, square.dx);

        // 更新
        for (int s = 0; s < 3; ++s) {
			Gradientd1(3 * xi + s) += ( - phi[3 * i + s] + phi_previous[i] ) * (rho / dt) * W;
        }

        };

    std::vector<int> indices(2); // 2つのインデックス用のベクター
    recursiveLoopForGradientd(0, 2, processIndices, indices); // 2重ループを再帰で実行

    return Gradientd1;
}

// Calculate Gradient d2
Eigen::VectorXd calGradientd2(Square square, Eigen::VectorXd phi)
{
    Eigen::VectorXd Gradientd2(3 * NumberOfParticles);
    Gradientd2.setZero();

    auto processIndices = [&](std::vector<int>& indices) {

        int i = indices[1], xi = indices[0];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);

        Eigen::Vector3i grid_i = FlatToGrid(i);
        Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

        Eigen::MatrixXi matrix(3,1);
        matrix << i_minus_xi;

        // 積分計算
        Eigen::VectorXi AxisW11(2); // サイズ2のベクトルを作成
        AxisW11 << 0, 0; // 各次元の数から-1した値を挿入
        double W11 = RiemannSum5(matrix, AxisW11, square.dx);

        Eigen::VectorXi AxisW22(2); // サイズ2のベクトルを作成
        AxisW22 << 0, 0; // 各次元の数から-1した値を挿入
        double W22 = RiemannSum5(matrix, AxisW22, square.dx);

        Eigen::VectorXi AxisW33(2); // サイズ2のベクトルを作成
        AxisW33 << 0, 0; // 各次元の数から-1した値を挿入
        double W33 = RiemannSum5(matrix, AxisW33, square.dx);

        double W = W11 + W22 + W33;

        double J = RiemannSumForDetF(phi, grid_xi, square.dx);

        // 更新
        for (int s = 0; s < 3; ++s) {
            Gradientd2(3 * xi + s) += phi[3 * i + s] * (-1) * mu * pow(J, -2/3) * W;
        }

        };

    std::vector<int> indices(2); // 2つのインデックス用のベクター
    recursiveLoopForGradientd(0, 2, processIndices, indices); // 2重ループを再帰で実行

    return Gradientd2;
}

// Calculate Gradient d3
Eigen::VectorXd calGradientd3(Square square, Eigen::VectorXd phi)
{
    Eigen::VectorXd Gradientd3(3 * NumberOfParticles);
    Gradientd3.setZero();

    auto processIndices = [&](std::vector<int>& indices) {

        int i = indices[4], j = indices[3], k = indices[2], l = indices[1], xi = indices[0];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);

        Eigen::Vector3i grid_l = FlatToGrid(l);
        Eigen::Vector3i l_minus_xi = grid_l - grid_xi;

        Eigen::Vector3i grid_k = FlatToGrid(k);
        Eigen::Vector3i k_minus_xi = grid_k - grid_xi;

        Eigen::Vector3i grid_j = FlatToGrid(j);
        Eigen::Vector3i j_minus_xi = grid_j - grid_xi;

        Eigen::Vector3i grid_i = FlatToGrid(i);
        Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

        Eigen::MatrixXi matrix(3,4);
        matrix << i_minus_xi, j_minus_xi, k_minus_xi, l_minus_xi;

        // 積分計算
        Eigen::VectorXi AxisW11231(5); // サイズ5のベクトルを作成
        AxisW11231 << 0, 0, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W11231 = RiemannSum5(matrix, AxisW11231, square.dx);
        Eigen::VectorXi AxisW11132(5); // サイズ5のベクトルを作成
        AxisW11132 << 0, 0, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W11132 = RiemannSum5(matrix, AxisW11132, square.dx);
        Eigen::VectorXi AxisW11123(5); // サイズ5のベクトルを作成
        AxisW11123 << 0, 0, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W11123 = RiemannSum5(matrix, AxisW11123, square.dx);
        Eigen::VectorXi AxisW22231(5); // サイズ5のベクトルを作成
        AxisW22231 << 1, 1, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W22231 = RiemannSum5(matrix, AxisW22231, square.dx);
        Eigen::VectorXi AxisW22132(5); // サイズ5のベクトルを作成
        AxisW22132 << 1, 1, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W22132 = RiemannSum5(matrix, AxisW22132, square.dx);
        Eigen::VectorXi AxisW22123(5); // サイズ5のベクトルを作成
        AxisW22123 << 1, 1, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W22123 = RiemannSum5(matrix, AxisW22123, square.dx);
        Eigen::VectorXi AxisW33231(5); // サイズ5のベクトルを作成
        AxisW33231 << 2, 2, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W33231 = RiemannSum5(matrix, AxisW33231, square.dx);
        Eigen::VectorXi AxisW33132(5); // サイズ5のベクトルを作成
        AxisW33132 << 2, 2, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W33132 = RiemannSum5(matrix, AxisW33132, square.dx);
        Eigen::VectorXi AxisW33123(5); // サイズ5のベクトルを作成
        AxisW33123 << 2, 2, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W33123 = RiemannSum5(matrix, AxisW33123, square.dx);

        double J = RiemannSumForDetF(phi, grid_xi, square.dx);

        double W = W11231 - W11132 + W11123 + W22231 - W22132 + W22123 + W33231 - W33132 + W33123;

        // phiの計算
        Eigen::VectorXd VectorPhi(3);
        VectorPhi <<
            phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1),
            -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l)),
            phi(3 * k)* phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);

        double Phi = phi[3 * i] * phi[3 * j] + phi[3 * i + 1] * phi[3 * j + 1] + phi[3 * i + 2] * phi[3 * j + 2];

        // 更新
        for (int s = 0; s < 3; ++s) {
            Gradientd3(3 * xi + s) += VectorPhi[s] * Phi * (1.0/3.0) * mu * pow(J, -2/3) * W;
        }

        };

    std::vector<int> indices(5); // 5つのインデックス用のベクター
    recursiveLoopForGradientd(0, 5, processIndices, indices); // 5重ループを再帰で実行

    return Gradientd3;
}

// Calculate Gradient d4
Eigen::VectorXd calGradientd4(Square square, Eigen::VectorXd phi, Eigen::VectorXd power)
{
    Eigen::VectorXd Gradientd4(3 * NumberOfParticles);
    Gradientd4.setZero();

    auto processIndices = [&](std::vector<int>& indices) {

        int i = indices[3], j = indices[2], k = indices[1], xi = indices[0];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);

        Eigen::Vector3i grid_k = FlatToGrid(k);
        Eigen::Vector3i k_minus_xi = grid_k - grid_xi;

        Eigen::Vector3i grid_j = FlatToGrid(j);
        Eigen::Vector3i j_minus_xi = grid_j - grid_xi;

        Eigen::Vector3i grid_i = FlatToGrid(i);
        Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

        Eigen::MatrixXi matrix(3,3);
        matrix << i_minus_xi, j_minus_xi, k_minus_xi;

        // 積分計算
        Eigen::VectorXi AxiswW231(3); // サイズ3のベクトルを作成
        AxiswW231 << 1, 2, 0; // 各次元の数から-1した値を挿入
        double wW231 = RiemannSum7(matrix, AxiswW231, square.dx);
        Eigen::VectorXi AxiswW132(3); // サイズ3のベクトルを作成
        AxiswW132 << 0, 2, 1; // 各次元の数から-1した値を挿入
        double wW132 = RiemannSum7(matrix, AxiswW132, square.dx);
        Eigen::VectorXi AxiswW123(3); // サイズ3のベクトルを作成
        AxiswW123 << 0, 1, 2; // 各次元の数から-1した値を挿入
        double wW123 = RiemannSum7(matrix, AxiswW123, square.dx);

        double W = wW231 - wW132 + wW123;

        // phiの計算
        Eigen::VectorXd VectorPhi(3);
        VectorPhi <<
            phi(3 * i + 1) * phi(3 * j + 2) - phi(3 * i + 2) * phi(3 * j + 1),
            -(phi(3 * i) * phi(3 * j + 2) - phi(3 * i + 2) * phi(3 * j)),
            phi(3 * i)* phi(3 * j + 1) - phi(3 * i + 1) * phi(3 * j);

        // 更新
        for (int s = 0; s < 3; ++s) {
            Gradientd4(3 * xi + s) += VectorPhi[s] * power[k] * W;
        }

        };

    std::vector<int> indices(4); // 4つのインデックス用のベクター
    recursiveLoopForGradientd(0, 4, processIndices, indices); // 4重ループを再帰で実行

    return Gradientd4;
}