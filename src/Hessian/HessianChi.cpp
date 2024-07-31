#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Hessian/HessianChi.h"

Eigen::MatrixXd calHessianChi(Square square, Eigen::VectorXd phi) {
	Eigen::MatrixXd HessianChi(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianChi.setZero();
    Eigen::MatrixXd HessianChi1 = calHessianChi1(square, phi);
    Eigen::MatrixXd HessianChi2 = calHessianChi2(square, phi);
    Eigen::MatrixXd HessianChi3 = calHessianChi3(square, phi);

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

Eigen::MatrixXd calHessianChi1(Square square, Eigen::VectorXd phi) {
	Eigen::MatrixXd HessianChi1(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianChi1.setZero();

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
        AxisW22 << 1, 1; // 各次元の数から-1した値を挿入
        double W22 = RiemannSum5(matrix, AxisW22, square.dx);

        Eigen::VectorXi AxisW33(2); // サイズ2のベクトルを作成
        AxisW33 << 2, 2; // 各次元の数から-1した値を挿入
        double W33 = RiemannSum5(matrix, AxisW33, square.dx);

        double J = RiemannSumForDetF(phi, grid_xi, square.dx);

        double W = W11 + W22 + W33;

        // 更新 3 × 3にするために，単位行列をかける
        double ans = mu * pow(J, -2 / 3) * W;
        for (int d = 0; d < 3; ++d) {
            HessianChi1(3 * i + d, 3 * xi + d) += ans;
        }

        };

    std::vector<int> indices(2); // 2つのインデックス用のベクター
    recursiveLoopForHesChi(0, 2, processIndices, indices); // 2重ループを再帰で実行

    return HessianChi1;
}

Eigen::MatrixXd calHessianChi2(const Square& square, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianChi2 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    auto processIndices = [&](const std::vector<int>& indices) {
        int xi = indices[0], l = indices[1], k = indices[2], j = indices[3], i = indices[4];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
        Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;

        if (allElementsWithinOne(l_minus_xi) && allElementsWithinOne(k_minus_xi) &&
            allElementsWithinOne(j_minus_xi) && allElementsWithinOne(i_minus_xi)) {

            Eigen::Matrix<int, 3, 4> matrix;
            matrix << i_minus_xi, j_minus_xi, k_minus_xi, l_minus_xi;

            // 積分計算
            auto calculateW = [&](const Eigen::VectorXi& axis) {
                return RiemannSum5(matrix, axis, square.dx);
                };

            double W = calculateW((Eigen::VectorXi(5) << 0, 0, 1, 2, 0).finished()) -
                calculateW((Eigen::VectorXi(5) << 0, 0, 0, 2, 1).finished()) +
                calculateW((Eigen::VectorXi(5) << 0, 0, 0, 1, 2).finished()) +
                calculateW((Eigen::VectorXi(5) << 1, 1, 1, 2, 0).finished()) -
                calculateW((Eigen::VectorXi(5) << 1, 1, 0, 2, 1).finished()) +
                calculateW((Eigen::VectorXi(5) << 1, 1, 0, 1, 2).finished()) +
                calculateW((Eigen::VectorXi(5) << 2, 2, 1, 2, 0).finished()) -
                calculateW((Eigen::VectorXi(5) << 2, 2, 0, 2, 1).finished()) +
                calculateW((Eigen::VectorXi(5) << 2, 2, 0, 1, 2).finished());

            double J = RiemannSumForDetF(phi, grid_xi, square.dx);

            // phiの計算
            Eigen::Vector3d VectorPhi1;
            VectorPhi1 <<
                phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1),
                -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l)),
                phi(3 * k)* phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);

            Eigen::Vector3d VectorPhi2 = phi.segment<3>(3 * i);

            Eigen::Matrix3d PhiMatrix = VectorPhi1 * VectorPhi2.transpose();

            // 更新
            double factor = mu * std::pow(J, -5.0 / 3.0) * W;
            HessianChi2.block<3, 3>(3 * j, 3 * xi) += PhiMatrix * factor;
        }
        };

    std::vector<int> indices(5);
    recursiveLoopForHesChi(0, 5, processIndices, indices);

    return HessianChi2;
}

Eigen::MatrixXd calHessianChi3(const Square& square, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianChi3 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    auto processIndices = [&](const std::vector<int>& indices) {
        int xi = indices[0], l = indices[1], k = indices[2], j = indices[3], i = indices[4];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
        Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;

        if (allElementsWithinOne(l_minus_xi) && allElementsWithinOne(k_minus_xi) &&
            allElementsWithinOne(j_minus_xi) && allElementsWithinOne(i_minus_xi)) {

            Eigen::Matrix<int, 3, 4> matrix;
            matrix << i_minus_xi, j_minus_xi, k_minus_xi, l_minus_xi;

            // 積分計算
            auto calculateW = [&](const Eigen::VectorXi& axis) {
                return RiemannSum5(matrix, axis, square.dx);
                };

            double W = calculateW((Eigen::VectorXi(5) << 0, 0, 1, 2, 0).finished()) -
                calculateW((Eigen::VectorXi(5) << 0, 0, 0, 2, 1).finished()) +
                calculateW((Eigen::VectorXi(5) << 0, 0, 0, 1, 2).finished()) +
                calculateW((Eigen::VectorXi(5) << 1, 1, 1, 2, 0).finished()) -
                calculateW((Eigen::VectorXi(5) << 1, 1, 0, 2, 1).finished()) +
                calculateW((Eigen::VectorXi(5) << 1, 1, 0, 1, 2).finished()) +
                calculateW((Eigen::VectorXi(5) << 2, 2, 1, 2, 0).finished()) -
                calculateW((Eigen::VectorXi(5) << 2, 2, 0, 2, 1).finished()) +
                calculateW((Eigen::VectorXi(5) << 2, 2, 0, 1, 2).finished());

            double J = RiemannSumForDetF(phi, grid_xi, square.dx);

            // phiの計算
            Eigen::Vector3d VectorPhi1 = phi.segment<3>(3 * i);

            Eigen::Vector3d VectorPhi2;
            VectorPhi2 <<
                phi(3 * k + 1)* phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1),
                -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l)),
                phi(3 * k)* phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);

            Eigen::Matrix3d PhiMatrix = VectorPhi1 * VectorPhi2.transpose();

            // 更新
            double factor = -mu * std::pow(J, -2.0 / 3.0) * W;
            HessianChi3.block<3, 3>(3 * j, 3 * xi) += PhiMatrix * factor;
        }
        };

    std::vector<int> indices(5);
    recursiveLoopForHesChi(0, 5, processIndices, indices);

    return HessianChi3;
}