#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"


// Caluculate HessianXi
Eigen::MatrixXd calHessianEpsilon(Square square, Eigen::VectorXd phi, Eigen::VectorXd power) {
    Eigen::MatrixXd HessianEpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianEpsilon.setZero();
    Eigen::MatrixXd HessianEpsilon1 = calHessianEpsilon1(square, phi, power);
    Eigen::MatrixXd HessianEpsilon2 = calHessianEpsilon2(square, phi, power);
    Eigen::MatrixXd HessianEpsilon3 = calHessianEpsilon3(square, phi, power);

    HessianEpsilon = HessianEpsilon1 + HessianEpsilon2 + HessianEpsilon3;

    return HessianEpsilon;
}

void recursiveLoopForHesEpsilon(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices) {
    if (level == maxLevel) {
        process(indices);
        return;
    }

    for (int i = 0; i < NumberOfParticles; i++) {
        indices[level] = i;
        recursiveLoopForHesEpsilon(level + 1, maxLevel, process, indices);
    }
}

Eigen::MatrixXd calHessianEpsilon1(Square square, Eigen::VectorXd phi, Eigen::VectorXd power)
{
    Eigen::MatrixXd HessianEpsilon1(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianEpsilon1.setZero();

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
            -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l)),
            phi(3 * k)* phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);

        Eigen::VectorXd VectorPhi2(3);
        VectorPhi2 << phi(3 * i), phi(3 * i + 1), phi(3 * i + 2);

        Eigen::MatrixXd PhiMatrix = VectorPhi1 * VectorPhi2;

        // 更新
        for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < 3; ++t) {
                HessianEpsilon1(3 * j + s, 3 * xi + t) += PhiMatrix(s, t) * mu * pow(J, -5 / 3) * W;
            }
        }

        };

    std::vector<int> indices(5); // 7つのインデックス用のベクター
    recursiveLoopForHesEpsilon(0, 5, processIndices, indices); // 7重ループを再帰で実行

    return HessianEpsilon1;
}

Eigen::MatrixXd calHessianEpsilon2(Square square, Eigen::VectorXd phi, Eigen::VectorXd power)
{
    Eigen::MatrixXd HessianEpsilon2(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianEpsilon2.setZero();

    auto processIndices = [&](std::vector<int>& indices) {

        int i = indices[7], j = indices[6], k = indices[5], l = indices[4], m = indices[3], n = indices[2], o = indices[1], xi = indices[0];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);

        Eigen::Vector3i grid_o = FlatToGrid(o);
        Eigen::Vector3i o_minus_xi = grid_o - grid_xi;

        Eigen::Vector3i grid_n = FlatToGrid(n);
        Eigen::Vector3i n_minus_xi = grid_n - grid_xi;

        Eigen::Vector3i grid_m = FlatToGrid(m);
        Eigen::Vector3i m_minus_xi = grid_m - grid_xi;

        Eigen::Vector3i grid_l = FlatToGrid(l);
        Eigen::Vector3i l_minus_xi = grid_l - grid_xi;

        Eigen::Vector3i grid_k = FlatToGrid(k);
        Eigen::Vector3i k_minus_xi = grid_k - grid_xi;

        Eigen::Vector3i grid_j = FlatToGrid(j);
        Eigen::Vector3i j_minus_xi = grid_j - grid_xi;

        Eigen::Vector3i grid_i = FlatToGrid(i);
        Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

        Eigen::MatrixXi matrix;
        matrix << i_minus_xi, j_minus_xi, k_minus_xi, l_minus_xi, m_minus_xi, n_minus_xi, o_minus_xi;

        // 積分計算
        Eigen::VectorXi AxisW11231231(8); // サイズ8のベクトルを作成
        AxisW11231231 << 0, 0, 1, 2, 0, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W11231231 = RiemannSum6(matrix, AxisW11231231, square.dx);

        Eigen::VectorXi AxisW11231132(8); // サイズ8のベクトルを作成
        AxisW11231132 << 0, 0, 1, 2, 0, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W11231132 = RiemannSum6(matrix, AxisW11231132, square.dx);

        Eigen::VectorXi AxisW11231123(8); // サイズ8のベクトルを作成
        AxisW11231123 << 0, 0, 1, 2, 0, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W11231123 = RiemannSum6(matrix, AxisW11231123, square.dx);

        Eigen::VectorXi AxisW11132231(8); // サイズ8のベクトルを作成
        AxisW11132231 << 0, 0, 0, 2, 1, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W11132231 = RiemannSum6(matrix, AxisW11132231, square.dx);

        Eigen::VectorXi AxisW11132132(8); // サイズ8のベクトルを作成
        AxisW11132132 << 0, 0, 0, 2, 1, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W11132132 = RiemannSum6(matrix, AxisW11132132, square.dx);

        Eigen::VectorXi AxisW11132123(8); // サイズ8のベクトルを作成
        AxisW11132123 << 0, 0, 0, 2, 1, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W11132123 = RiemannSum6(matrix, AxisW11132123, square.dx);

        Eigen::VectorXi AxisW11123231(8); // サイズ8のベクトルを作成
        AxisW11123231 << 0, 0, 0, 1, 2, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W11123231 = RiemannSum6(matrix, AxisW11123231, square.dx);

        Eigen::VectorXi AxisW11123132(8); // サイズ8のベクトルを作成
        AxisW11123132 << 0, 0, 0, 1, 2, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W11123132 = RiemannSum6(matrix, AxisW11123132, square.dx);

        Eigen::VectorXi AxisW11123123(8); // サイズ8のベクトルを作成
        AxisW11123123 << 0, 0, 0, 1, 2, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W11123123 = RiemannSum6(matrix, AxisW11123123, square.dx);

        //

        Eigen::VectorXi AxisW22231231(8); // サイズ8のベクトルを作成
        AxisW22231231 << 1, 1, 1, 2, 0, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W22231231 = RiemannSum6(matrix, AxisW22231231, square.dx);

        Eigen::VectorXi AxisW22231132(8); // サイズ8のベクトルを作成
        AxisW22231132 << 1, 1, 1, 2, 0, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W22231132 = RiemannSum6(matrix, AxisW22231132, square.dx);

        Eigen::VectorXi AxisW22231123(8); // サイズ8のベクトルを作成
        AxisW22231123 << 1, 1, 1, 2, 0, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W22231123 = RiemannSum6(matrix, AxisW22231123, square.dx);

        Eigen::VectorXi AxisW22132231(8); // サイズ8のベクトルを作成
        AxisW22132231 << 1, 1, 0, 2, 1, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W22132231 = RiemannSum6(matrix, AxisW22132231, square.dx);

        Eigen::VectorXi AxisW22132132(8); // サイズ8のベクトルを作成
        AxisW22132132 << 1, 1, 0, 2, 1, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W22132132 = RiemannSum6(matrix, AxisW22132132, square.dx);

        Eigen::VectorXi AxisW22132123(8); // サイズ8のベクトルを作成
        AxisW22132123 << 1, 1, 0, 2, 1, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W22132123 = RiemannSum6(matrix, AxisW22132123, square.dx);

        Eigen::VectorXi AxisW22123231(8); // サイズ8のベクトルを作成
        AxisW22123231 << 1, 1, 0, 1, 2, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W22123231 = RiemannSum6(matrix, AxisW22123231, square.dx);

        Eigen::VectorXi AxisW22123132(8); // サイズ8のベクトルを作成
        AxisW22123132 << 1, 1, 0, 1, 2, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W22123132 = RiemannSum6(matrix, AxisW22123132, square.dx);

        Eigen::VectorXi AxisW22123123(8); // サイズ8のベクトルを作成
        AxisW22123123 << 1, 1, 0, 1, 2, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W22123123 = RiemannSum6(matrix, AxisW22123123, square.dx);

        //

        Eigen::VectorXi AxisW33231231(8); // サイズ8のベクトルを作成
        AxisW33231231 << 2, 2, 1, 2, 0, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W33231231 = RiemannSum6(matrix, AxisW33231231, square.dx);

        Eigen::VectorXi AxisW33231132(8); // サイズ8のベクトルを作成
        AxisW33231132 << 2, 2, 1, 2, 0, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W33231132 = RiemannSum6(matrix, AxisW33231132, square.dx);

        Eigen::VectorXi AxisW33231123(8); // サイズ8のベクトルを作成
        AxisW33231123 << 2, 2, 1, 2, 0, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W33231123 = RiemannSum6(matrix, AxisW33231123, square.dx);

        Eigen::VectorXi AxisW33132231(8); // サイズ8のベクトルを作成
        AxisW33132231 << 2, 2, 0, 2, 1, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W33132231 = RiemannSum6(matrix, AxisW33132231, square.dx);

        Eigen::VectorXi AxisW33132132(8); // サイズ8のベクトルを作成
        AxisW33132132 << 2, 2, 0, 2, 1, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W33132132 = RiemannSum6(matrix, AxisW33132132, square.dx);

        Eigen::VectorXi AxisW33132123(8); // サイズ8のベクトルを作成
        AxisW33132123 << 2, 2, 0, 2, 1, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W33132123 = RiemannSum6(matrix, AxisW33132123, square.dx);

        Eigen::VectorXi AxisW33123231(8); // サイズ8のベクトルを作成
        AxisW33123231 << 2, 2, 0, 1, 2, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W33123231 = RiemannSum6(matrix, AxisW33123231, square.dx);

        Eigen::VectorXi AxisW33123132(8); // サイズ8のベクトルを作成
        AxisW33123132 << 2, 2, 0, 1, 2, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W33123132 = RiemannSum6(matrix, AxisW33123132, square.dx);

        Eigen::VectorXi AxisW33123123(8); // サイズ8のベクトルを作成
        AxisW33123123 << 2, 2, 0, 1, 2, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W33123123 = RiemannSum6(matrix, AxisW33123123, square.dx);

        double J = RiemannSumForDetF(phi, grid_xi, square.dx);

        double W1 = W11231231 - W11231132 + W11231123 - W11132231 + W11132132 - W11132123 + W11123231 - W11123132 + W11123123;
        double W2 = W22231231 - W22231132 + W22231123 - W22132231 + W22132132 - W22132123 + W22123231 - W22123132 + W22123123;
        double W3 = W33231231 - W33231132 + W33231123 - W33132231 + W33132132 - W33132123 + W33123231 - W33123132 + W11123123;
        double W = W1 + W2 + W3;

        // phiの計算
        Eigen::VectorXd VectorPhi1(3);
        VectorPhi1 <<
            phi(3 * n + 1) * phi(3 * o + 2) - phi(3 * n + 2) * phi(3 * o + 1),
            -(phi(3 * n) * phi(3 * o + 2) - phi(3 * n + 2) * phi(3 * o)),
            phi(3 * n)* phi(3 * o + 1) - phi(3 * n + 1) * phi(3 * o);

        Eigen::VectorXd VectorPhi2(3);
        VectorPhi2 <<
            phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1),
            -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l)),
            phi(3 * k)* phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);

        Eigen::MatrixXd PhiMatrix = VectorPhi1 * VectorPhi2 * ( phi(3 * i) * phi(3 * J) + phi(3 * i + 1) * phi(3 * J + 1) + phi(3 * i + 2) * phi(3 * J + 2) );

        // 更新
        for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < 3; ++t) {
                HessianEpsilon2(3 * m + s, 3 * xi + t) += (1.0 / 3.0) * mu * PhiMatrix(s, t) * pow(J, -8.0 / 3.0);
            }
        }

        };

    std::vector<int> indices(8); // 7つのインデックス用のベクター
    recursiveLoopForHesEpsilon(0, 8, processIndices, indices); // 7重ループを再帰で実行

    return HessianEpsilon2;
}

Eigen::MatrixXd calHessianEpsilon3(Square square, Eigen::VectorXd phi, Eigen::VectorXd power)
{
    Eigen::MatrixXd HessianEpsilon3(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianEpsilon3.setZero();

    auto processIndices = [&](std::vector<int>& indices) {

        int i = indices[6], j = indices[5], k = indices[4], l = indices[3], m = indices[2], n = indices[1], xi = indices[0];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);

        Eigen::Vector3i grid_n = FlatToGrid(n);
        Eigen::Vector3i n_minus_xi = grid_n - grid_xi;

        Eigen::Vector3i grid_m = FlatToGrid(m);
        Eigen::Vector3i m_minus_xi = grid_m - grid_xi;

        Eigen::Vector3i grid_l = FlatToGrid(l);
        Eigen::Vector3i l_minus_xi = grid_l - grid_xi;

        Eigen::Vector3i grid_k = FlatToGrid(k);
        Eigen::Vector3i k_minus_xi = grid_k - grid_xi;

        Eigen::Vector3i grid_j = FlatToGrid(j);
        Eigen::Vector3i j_minus_xi = grid_j - grid_xi;

        Eigen::Vector3i grid_i = FlatToGrid(i);
        Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

        Eigen::MatrixXi matrix;
        matrix << j_minus_xi, k_minus_xi, l_minus_xi, m_minus_xi, n_minus_xi;

        // 積分計算
        Eigen::VectorXi AxiswW231231(6); // サイズ7のベクトルを作成
        AxiswW231231 << 1, 2, 0, 1, 2, 0; // 各次元の数から-1した値を挿入
        double wW231231 = RiemannSum7(matrix, AxiswW231231, square.dx);

        Eigen::VectorXi AxiswW231132(6); // サイズ7のベクトルを作成
        AxiswW231132 << 1, 2, 0, 0, 2, 1; // 各次元の数から-1した値を挿入
        double wW231132 = RiemannSum7(matrix, AxiswW231132, square.dx);

        Eigen::VectorXi AxiswW231123(6); // サイズ7のベクトルを作成
        AxiswW231123 << 1, 2, 0, 0, 1, 2; // 各次元の数から-1した値を挿入
        double wW231123 = RiemannSum6(matrix, AxiswW231123, square.dx);

        Eigen::VectorXi AxiswW132231(6); // サイズ7のベクトルを作成
        AxiswW132231 << 0, 2, 1, 1, 2, 0; // 各次元の数から-1した値を挿入
        double wW132231 = RiemannSum7(matrix, AxiswW132231, square.dx);

        Eigen::VectorXi AxiswW132132(6); // サイズ7のベクトルを作成
        AxiswW132132 << 0, 2, 1, 0, 2, 1; // 各次元の数から-1した値を挿入
        double wW132132 = RiemannSum7(matrix, AxiswW132132, square.dx);

        Eigen::VectorXi AxiswW132123(6); // サイズ7のベクトルを作成
        AxiswW132123 << 0, 2, 1, 0, 1, 2; // 各次元の数から-1した値を挿入
        double wW132123 = RiemannSum7(matrix, AxiswW132123, square.dx);

        Eigen::VectorXi AxiswW123231(6); // サイズ7のベクトルを作成
        AxiswW123231 << 0, 0, 0, 1, 2, 1, 2, 0; // 各次元の数から-1した値を挿入
        double wW123231 = RiemannSum7(matrix, AxiswW123231, square.dx);

        Eigen::VectorXi AxiswW123132(6); // サイズ7のベクトルを作成
        AxiswW123132 << 0, 1, 2, 0, 2, 1; // 各次元の数から-1した値を挿入
        double wW123132 = RiemannSum7(matrix, AxiswW123132, square.dx);

        Eigen::VectorXi AxiswW123123(6); // サイズ7のベクトルを作成
        AxiswW123123 << 0, 1, 2, 0, 1, 2; // 各次元の数から-1した値を挿入
        double wW123123 = RiemannSum7(matrix, AxiswW123123, square.dx);

        double J = RiemannSumForDetF(phi, grid_xi, square.dx);

        
        double W = wW231231 - wW231132 + wW231123 - wW132231 + wW132132 - wW132123 + wW123231 - wW123132 + wW123123;

        // phiの計算
        Eigen::VectorXd VectorPhi1(3);
        VectorPhi1 <<
            phi(3 * m + 1) * phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n + 1),
            -(phi(3 * m) * phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n)),
            phi(3 * m)* phi(3 * n + 1) - phi(3 * m + 1) * phi(3 * n);

        Eigen::VectorXd VectorPhi2(3);
        VectorPhi2 <<
            phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1),
            -(phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k)),
            phi(3 * j)* phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);

        Eigen::MatrixXd PhiMatrix = VectorPhi1 * VectorPhi2;

        // 更新
        for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < 3; ++t) {
                HessianEpsilon3(3 * l + s, 3 * xi + t) += power[i] * PhiMatrix(s, t) * W * pow(J, -1.0);
            }
        }

        };

    std::vector<int> indices(7); // 7つのインデックス用のベクター
    recursiveLoopForHesEpsilon(0, 7, processIndices, indices); // 7重ループを再帰で実行

    return HessianEpsilon3;
}