#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"

Eigen::MatrixXd calHessianChi(Square square, Eigen::VectorXd phi, Eigen::VectorXd power) {
	Eigen::MatrixXd HessianChi(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianChi.setZero();
    Eigen::MatrixXd HessianChi1 = calHessianChi1(square, phi, power);
    Eigen::MatrixXd HessianChi2 = calHessianChi2(square, phi, power);
    Eigen::MatrixXd HessianChi3 = calHessianChi3(square, phi, power);

    HessianChi = HessianChi1 + HessianChi2 + HessianChi3;

    return HessianChi;
}

void recursiveLoopForHesChi(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices) {
    if (level == maxLevel) {
        process(indices);
        return;
    }

    for (int i = 0; i < NumberOfParticles; i++) {
        indices[level] = i;
        recursiveLoopForHesChi(level + 1, maxLevel, process, indices);
    }
};

Eigen::MatrixXd calHessianChi1(Square square, Eigen::VectorXd phi, Eigen::VectorXd power) {
	Eigen::MatrixXd HessianChi1(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianChi1.setZero();

    auto processIndices = [&](std::vector<int>& indices) {

        int i = indices[1], xi = indices[0];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);

        Eigen::Vector3i grid_i = FlatToGrid(i);
        Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

        Eigen::MatrixXi matrix;
        matrix << i_minus_xi;

        // 積分計算
        Eigen::VectorXi AxisW11(2); // サイズ2のベクトルを作成
        AxisW11 << 0, 0; // 各次元の数から-1した値を挿入
        double W11 = RiemannSum5(matrix, AxisW11, square.dx);

        Eigen::VectorXi AxisW22(2); // サイズ2のベクトルを作成
        AxisW22 << 1, 1; // 各次元の数から-1した値を挿入
        double W22 = RiemannSum5(matrix, AxisW22, square.dx);

        Eigen::VectorXi AxisW33(2); // サイズ2のベクトルを作成
        AxisW33 << 2, 2; // 各次元の数から-1した値を挿入
        double W33 = RiemannSum5(matrix, AxisW33, square.dx);

        double J = RiemannSumForDetF(phi, grid_xi, square.dx);

        double W = W11 + W22 + W33;

        // 更新 3 × 3にするために，単位行列をかける
        double ans = mu * pow(J, -2 / 3) * W;
        HessianChi1(3 * i, 3 * xi) += ans;
        HessianChi1(3 * i + 1, 3 * xi + 1) += ans;
        HessianChi1(3 * i + 2, 3 * xi + 2) += ans;

        };

    std::vector<int> indices(2); // 7つのインデックス用のベクター
    recursiveLoopForHesChi(0, 2, processIndices, indices); // 2重ループを再帰で実行

    return HessianChi1;
}

Eigen::MatrixXd calHessianChi2(Square square, Eigen::VectorXd phi, Eigen::VectorXd power) {
    Eigen::MatrixXd HessianChi2(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianChi2.setZero();

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

        Eigen::MatrixXi matrix;
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
        Eigen::VectorXd VectorPhi1(3);
        VectorPhi1 <<
            phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1),
            -(phi(3 * k)* phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l)),
            phi(3 * k)* phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);

        Eigen::VectorXd VectorPhi2(3);
        VectorPhi2 << phi(3 * i), phi(3 * i + 1), phi(3 * i + 2);

        Eigen::MatrixXd PhiMatrix = VectorPhi1 * VectorPhi2;

        // 更新
        for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < 3; ++t) {
                HessianChi2(3 * j + s, 3 * xi + t) += PhiMatrix(s, t) * mu * pow(J, -5/3) * W;
            }
        }

        };

    std::vector<int> indices(5); // 7つのインデックス用のベクター
    recursiveLoopForHesChi(0, 5, processIndices, indices); // 2重ループを再帰で実行

    return HessianChi2;
}

Eigen::MatrixXd calHessianChi3(Square square, Eigen::VectorXd phi, Eigen::VectorXd power) {
    Eigen::MatrixXd HessianChi3(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianChi3.setZero();

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

        Eigen::MatrixXi matrix;
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
        Eigen::VectorXd VectorPhi1(3);
        VectorPhi1 << phi(3 * i), phi(3 * i + 1), phi(3 * i + 2);

        Eigen::VectorXd VectorPhi2(3);
        VectorPhi2 <<
            phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1),
            -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l)),
            phi(3 * k)* phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);

        Eigen::MatrixXd PhiMatrix = VectorPhi1 * VectorPhi2;

        // 更新
        for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < 3; ++t) {
                HessianChi3(3 * j + s, 3 * xi + t) += - PhiMatrix(s, t) * mu * pow(J, -2 / 3) * W;
            }
        }

        };

    std::vector<int> indices(5); // 7つのインデックス用のベクター
    recursiveLoopForHesChi(0, 5, processIndices, indices); // 2重ループを再帰で実行

    return HessianChi3;
}