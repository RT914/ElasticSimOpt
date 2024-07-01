#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"


// Caluculate HessianXi
Eigen::MatrixXd calHessianXi(Square square, Eigen::VectorXd phi, Eigen::VectorXd power) {
    Eigen::MatrixXd HessianXi(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianXi.setZero();
    Eigen::MatrixXd HessianXi1 = calHessianXi1(square, phi, power);
    Eigen::MatrixXd HessianXi2 = calHessianXi2(square, phi, power);
    Eigen::MatrixXd HessianXi3 = calHessianXi3(square, phi, power);

    HessianXi = HessianXi1 + HessianXi2 + HessianXi3;

    return HessianXi;
}

void recursiveLoopForHesXi(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices) {
    if (level == maxLevel) {
        process(indices);
        return;
    }

    for (int i = 0; i < NumberOfParticles; i++) {
        indices[level] = i;
        recursiveLoopForHesXi(level + 1, maxLevel, process, indices);
    }
}

Eigen::MatrixXd calHessianXi1(Square square, Eigen::VectorXd phi, Eigen::VectorXd power)
{
    Eigen::MatrixXd HessianXi1(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianXi1.setZero();

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
        matrix << i_minus_xi, j_minus_xi, k_minus_xi, l_minus_xi, m_minus_xi, n_minus_xi;

        // 積分計算 RiemannSum6では不適切だから後日修正 wWWWの内挿関数計算を考慮すべき
        Eigen::VectorXi AxisW231231w(6); // サイズ6のベクトルを作成
        AxisW231231w << 1, 2, 0, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W231231w = RiemannSum7(matrix, AxisW231231w, square.dx);

        Eigen::VectorXi AxisW231132w(6); // サイズ6のベクトルを作成
        AxisW231132w << 1, 2, 0, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W231132w = RiemannSum7(matrix, AxisW231132w, square.dx);

        Eigen::VectorXi AxisW231123w(6); // サイズ6のベクトルを作成
        AxisW231123w << 1, 2, 0, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W231123w = RiemannSum7(matrix, AxisW231123w, square.dx);

        Eigen::VectorXi AxisW132231w(6); // サイズ6のベクトルを作成
        AxisW132231w << 0, 2, 1, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W132231w = RiemannSum7(matrix, AxisW132231w, square.dx);

        Eigen::VectorXi AxisW132132w(6); // サイズ6のベクトルを作成
        AxisW132132w << 0, 2, 1, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W132132w = RiemannSum7(matrix, AxisW132132w, square.dx);

        Eigen::VectorXi AxisW132123w(6); // サイズ6のベクトルを作成
        AxisW132123w << 0, 2, 1, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W132123w = RiemannSum7(matrix, AxisW132123w, square.dx);

        Eigen::VectorXi AxisW123231w(6); // サイズ6のベクトルを作成
        AxisW123231w << 0, 1, 2, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W123231w = RiemannSum7(matrix, AxisW123231w, square.dx);

        Eigen::VectorXi AxisW123132w(6); // サイズ6のベクトルを作成
        AxisW123132w << 0, 1, 2, 0, 2, 1; // 各次元の数から-1した値を挿入
        double W123132w = RiemannSum7(matrix, AxisW123132w, square.dx);

        Eigen::VectorXi AxisW123123w(6); // サイズ6のベクトルを作成
        AxisW123123w << 0, 1, 2, 0, 1, 2; // 各次元の数から-1した値を挿入
        double W123123w = RiemannSum7(matrix, AxisW123123w, square.dx);

        double J = RiemannSumForDetF(phi, grid_xi, square.dx);

        double W = W231231w - W231132w + W231123w - W132231w + W132132w - W132123w + W123231w - W123132w + W123123w;

        // phiの計算
        Eigen::VectorXd VectorPhi1(3);
        VectorPhi1 <<
            phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1),
            -(phi(3 * j)* phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k)),
            phi(3 * j)* phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);

        Eigen::VectorXd VectorPhi2(3);
        VectorPhi2 <<
            phi(3 * m + 1) * phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n + 1),
            -(phi(3 * m)* phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n)),
            phi(3 * m)* phi(3 * n + 1) - phi(3 * m + 1) * phi(3 * n);

        Eigen::MatrixXd PhiMatrix = VectorPhi1 * VectorPhi2;

        // 更新
        for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < 3; ++t) {
                HessianXi1(3 * l + s, 3 * xi + t) += PhiMatrix(s, t) * power[i] * (1 / J) * W;
            }
        }

        };

    std::vector<int> indices(7); // 7つのインデックス用のベクター
    recursiveLoopForHesXi(0, 7, processIndices, indices); // 7重ループを再帰で実行

    return HessianXi1;
}

Eigen::MatrixXd calHessianXi2(Square square, Eigen::VectorXd phi, Eigen::VectorXd power)
{
    Eigen::MatrixXd HessianXi2(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianXi2.setZero();

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
        matrix << i_minus_xi, j_minus_xi, k_minus_xi, l_minus_xi, m_minus_xi, n_minus_xi;

        // 積分計算 RiemannSum6では不適切だから後日修正 wWWWの内挿関数計算を考慮すべき
        Eigen::VectorXi AxisW2312311(7); // サイズ6のベクトルを作成
        AxisW2312311 << 1, 2, 0, 1, 2, 0, 0; // 各次元の数から-1した値を挿入
        double W2312311 = RiemannSum6(matrix, AxisW2312311, square.dx);

        Eigen::VectorXi AxisW2311321(7); // サイズ6のベクトルを作成
        AxisW2311321 << 1, 2, 0, 0, 2, 1, 0; // 各次元の数から-1した値を挿入
        double W2311321 = RiemannSum6(matrix, AxisW2311321, square.dx);

        Eigen::VectorXi AxisW2311231(7); // サイズ6のベクトルを作成
        AxisW2311231 << 1, 2, 0, 0, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W2311231 = RiemannSum6(matrix, AxisW2311231, square.dx);

        Eigen::VectorXi AxisW1322311(7); // サイズ6のベクトルを作成
        AxisW1322311 << 0, 2, 1, 1, 2, 0, 0; // 各次元の数から-1した値を挿入
        double W1322311 = RiemannSum6(matrix, AxisW1322311, square.dx);

        Eigen::VectorXi AxisW1321321(7); // サイズ6のベクトルを作成
        AxisW1321321 << 0, 2, 1, 0, 2, 1, 0; // 各次元の数から-1した値を挿入
        double W1321321 = RiemannSum6(matrix, AxisW1321321, square.dx);

        Eigen::VectorXi AxisW1321231(7); // サイズ6のベクトルを作成
        AxisW1321231 << 0, 2, 1, 0, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W1321231 = RiemannSum6(matrix, AxisW1321231, square.dx);

        Eigen::VectorXi AxisW1232311(7); // サイズ6のベクトルを作成
        AxisW1232311 << 0, 1, 2, 1, 2, 0, 0; // 各次元の数から-1した値を挿入
        double W1232311 = RiemannSum6(matrix, AxisW1232311, square.dx);

        Eigen::VectorXi AxisW1231321(7); // サイズ6のベクトルを作成
        AxisW1231321 << 0, 1, 2, 0, 2, 1, 0; // 各次元の数から-1した値を挿入
        double W1231321 = RiemannSum6(matrix, AxisW1231321, square.dx);

        Eigen::VectorXi AxisW1231231(7); // サイズ6のベクトルを作成
        AxisW1231231 << 0, 1, 2, 0, 1, 2, 0; // 各次元の数から-1した値を挿入
        double W1231231 = RiemannSum6(matrix, AxisW1231231, square.dx);

        //

        Eigen::VectorXi AxisW2312312(7); // サイズ7のベクトルを作成
        AxisW2312312 << 1, 2, 0, 1, 2, 0, 1; // 各次元の数から-1した値を挿入
        double W2312312 = RiemannSum6(matrix, AxisW2312312, square.dx);

        Eigen::VectorXi AxisW2311322(7); // サイズ7のベクトルを作成
        AxisW2311322 << 1, 2, 0, 0, 2, 1, 1; // 各次元の数から-1した値を挿入
        double W2311322 = RiemannSum6(matrix, AxisW2311322, square.dx);

        Eigen::VectorXi AxisW2311232(7); // サイズ7のベクトルを作成
        AxisW2311232 << 1, 2, 0, 0, 1, 2, 1; // 各次元の数から-1した値を挿入
        double W2311232 = RiemannSum6(matrix, AxisW2311232, square.dx);

        Eigen::VectorXi AxisW1322312(7); // サイズ7のベクトルを作成
        AxisW1322312 << 0, 2, 1, 1, 2, 0, 1; // 各次元の数から-1した値を挿入
        double W1322312 = RiemannSum6(matrix, AxisW1322312, square.dx);

        Eigen::VectorXi AxisW1321322(7); // サイズ7のベクトルを作成
        AxisW1321322 << 0, 2, 1, 0, 2, 1, 1; // 各次元の数から-1した値を挿入
        double W1321322 = RiemannSum6(matrix, AxisW1321322, square.dx);

        Eigen::VectorXi AxisW1321232(7); // サイズ7のベクトルを作成
        AxisW1321232 << 0, 2, 1, 0, 1, 2, 1; // 各次元の数から-1した値を挿入
        double W1321232 = RiemannSum6(matrix, AxisW1321232, square.dx);

        Eigen::VectorXi AxisW1232312(7); // サイズ7のベクトルを作成
        AxisW1232312 << 0, 1, 2, 1, 2, 0, 1; // 各次元の数から-1した値を挿入
        double W1232312 = RiemannSum6(matrix, AxisW1232312, square.dx);

        Eigen::VectorXi AxisW1231322(7); // サイズ7のベクトルを作成
        AxisW1231322 << 0, 1, 2, 0, 2, 1, 1; // 各次元の数から-1した値を挿入
        double W1231322 = RiemannSum6(matrix, AxisW1231322, square.dx);

        Eigen::VectorXi AxisW1231232(7); // サイズ7のベクトルを作成
        AxisW1231232 << 0, 1, 2, 0, 1, 2, 1; // 各次元の数から-1した値を挿入
        double W1231232 = RiemannSum6(matrix, AxisW1231232, square.dx);

        //

        Eigen::VectorXi AxisW2312313(7); // サイズ7のベクトルを作成
        AxisW2312313 << 1, 2, 0, 1, 2, 0, 2; // 各次元の数から-1した値を挿入
        double W2312313 = RiemannSum6(matrix, AxisW2312313, square.dx);

        Eigen::VectorXi AxisW2311323(7); // サイズ7のベクトルを作成
        AxisW2311323 << 1, 2, 0, 0, 2, 1, 2; // 各次元の数から-1した値を挿入
        double W2311323 = RiemannSum6(matrix, AxisW2311323, square.dx);

        Eigen::VectorXi AxisW2311233(7); // サイズ7のベクトルを作成
        AxisW2311233 << 1, 2, 0, 0, 1, 2, 2; // 各次元の数から-1した値を挿入
        double W2311233 = RiemannSum6(matrix, AxisW2311233, square.dx);

        Eigen::VectorXi AxisW1322313(7); // サイズ7のベクトルを作成
        AxisW1322313 << 0, 2, 1, 1, 2, 0, 2; // 各次元の数から-1した値を挿入
        double W1322313 = RiemannSum6(matrix, AxisW1322313, square.dx);

        Eigen::VectorXi AxisW1321323(7); // サイズ7のベクトルを作成
        AxisW1321323 << 0, 2, 1, 0, 2, 1, 2; // 各次元の数から-1した値を挿入
        double W1321323 = RiemannSum6(matrix, AxisW1321323, square.dx);

        Eigen::VectorXi AxisW1321233(7); // サイズ7のベクトルを作成
        AxisW1321233 << 0, 2, 1, 0, 1, 2, 2; // 各次元の数から-1した値を挿入
        double W1321233 = RiemannSum7(matrix, AxisW1321233, square.dx);

        Eigen::VectorXi AxisW1232313(7); // サイズ7のベクトルを作成
        AxisW1232313 << 0, 1, 2, 1, 2, 0, 2; // 各次元の数から-1した値を挿入
        double W1232313 = RiemannSum7(matrix, AxisW1232313, square.dx);

        Eigen::VectorXi AxisW1231323(7); // サイズ7のベクトルを作成
        AxisW1231323 << 0, 1, 2, 0, 2, 1, 2; // 各次元の数から-1した値を挿入
        double W1231323 = RiemannSum7(matrix, AxisW1231323, square.dx);

        Eigen::VectorXi AxisW1231233(7); // サイズ7のベクトルを作成
        AxisW1231233 << 0, 1, 2, 0, 1, 2, 2; // 各次元の数から-1した値を挿入
        double W1231233 = RiemannSum7(matrix, AxisW1231233, square.dx);

        double J = RiemannSumForDetF(phi, grid_xi, square.dx);

        Eigen::VectorXd VectorW(3);
        VectorW <<
            W2312311 - W2311321 + W2311231 - W1322311 + W1321321 - W1321231 + W1232311 - W1231321 + W1231231,
            W2312312 - W2311322 + W2311232 - W1322312 + W1321322 - W1321232 + W1232312 - W1231322 + W1231232,
            W2312313 - W2311323 + W2311233 - W1322313 + W1321323 - W1321233 + W1232313 - W1231323 + W1231233;

        Eigen::MatrixXd MatrixW(3, 3);
        for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < 3; ++t) {
                MatrixW(s, t) = VectorW(s) * phi(3 * n + t);
            }
        }

        // phiの計算
        Eigen::VectorXd VectorPhi1(3);
        VectorPhi1 <<
            phi(3 * i + 1) * phi(3 * j + 2) - phi(3 * i + 2) * phi(3 * j + 1),
            -(phi(3 * i)* phi(3 * j + 2) - phi(3 * i + 2) * phi(3 * j)),
            phi(3 * i)* phi(3 * j + 1) - phi(3 * i + 1) * phi(3 * j);

        Eigen::VectorXd VectorPhi2(3);
        VectorPhi2 <<
            phi(3 * l + 1) * phi(3 * m + 2) - phi(3 * l + 2) * phi(3 * m + 1),
            -(phi(3 * l)* phi(3 * m + 2) - phi(3 * l + 2) * phi(3 * m)),
            phi(3 * l)* phi(3 * m + 1) - phi(3 * l + 1) * phi(3 * m);

        Eigen::MatrixXd PhiMatrix = VectorPhi1 * VectorPhi2;

        // 更新
        for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < 3; ++t) {
                HessianXi2(3 * k + s, 3 * xi + t) += (2.0 / 3.0) * mu * PhiMatrix(s, t) * MatrixW(s, t) * pow(J, -8.0/3.0);
            }
        }

        };

    std::vector<int> indices(7); // 7つのインデックス用のベクター
    recursiveLoopForHesXi(0, 7, processIndices, indices); // 7重ループを再帰で実行

    return HessianXi2;
}

Eigen::MatrixXd calHessianXi3(Square square, Eigen::VectorXd phi, Eigen::VectorXd power)
{
    Eigen::MatrixXd HessianXi3(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianXi3.setZero();

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
            phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1),
            -(phi(3 * k)* phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l)),
            phi(3 * k)* phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);

        Eigen::VectorXd VectorPhi2(3);
        VectorPhi2 <<
            phi(3 * n + 1) * phi(3 * o + 2) - phi(3 * n + 2) * phi(3 * o + 1),
            -(phi(3 * n) * phi(3 * o + 2) - phi(3 * n + 2) * phi(3 * o)),
            phi(3 * n) * phi(3 * o + 1) - phi(3 * n + 1) * phi(3 * o);

        double P = phi(3 * i) * phi(3 * j) + phi(3 * i + 1) * phi(3 * j + 1) + phi(3 * i + 2) * phi(3 * j + 2);

        Eigen::MatrixXd PhiMatrix = VectorPhi1 * VectorPhi2;

        // 更新
        for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < 3; ++t) {
                HessianXi3(3 * k + s, 3 * xi + t) += (2.0 / 9.0) * mu * PhiMatrix(s, t) * P * W * pow(J, -8.0 / 3.0);
            }
        }

        };

    std::vector<int> indices(7); // 7つのインデックス用のベクター
    recursiveLoopForHesXi(0, 7, processIndices, indices); // 7重ループを再帰で実行

    return HessianXi3;
}