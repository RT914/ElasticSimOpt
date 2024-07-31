#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Hessian/HessianXi.h"

// Caluculate HessianXi
Eigen::MatrixXd calHessianXi(Square square, Eigen::VectorXd phi, Eigen::VectorXd power) {
    Eigen::MatrixXd HessianXi(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianXi.setZero();
    Eigen::MatrixXd HessianXi1 = calHessianXi1(square, phi, power);
    Eigen::MatrixXd HessianXi2 = calHessianXi2(square, phi);
    Eigen::MatrixXd HessianXi3 = calHessianXi3(square, phi);

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

Eigen::MatrixXd calHessianXi1(const Square& square, const Eigen::VectorXd& phi, const Eigen::VectorXd& power)
{
    Eigen::MatrixXd HessianXi1 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    auto processIndices = [&](const std::vector<int>& indices) {
        int xi = indices[0], n = indices[1], m = indices[2], l = indices[3], k = indices[4], j = indices[5], i = indices[6];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        std::vector<Eigen::Vector3i> diff_vectors = {
            FlatToGrid(i) - grid_xi,
            FlatToGrid(j) - grid_xi,
            FlatToGrid(k) - grid_xi,
            FlatToGrid(l) - grid_xi,
            FlatToGrid(m) - grid_xi,
            FlatToGrid(n) - grid_xi
        };

        if (std::all_of(diff_vectors.begin(), diff_vectors.end(), allElementsWithinOne)) {
            Eigen::Matrix<int, 3, 6> matrix;
            for (int col = 0; col < 6; ++col) {
                matrix.col(col) = diff_vectors[col];
            }

            // 積分計算
            auto calculateW = [&](const Eigen::VectorXi& axis) {
                return RiemannSum7(matrix, axis, square.dx);
                };

            double W = calculateW((Eigen::VectorXi(6) << 1, 2, 0, 1, 2, 0).finished()) -
                calculateW((Eigen::VectorXi(6) << 1, 2, 0, 0, 2, 1).finished()) +
                calculateW((Eigen::VectorXi(6) << 1, 2, 0, 0, 1, 2).finished()) -
                calculateW((Eigen::VectorXi(6) << 0, 2, 1, 1, 2, 0).finished()) +
                calculateW((Eigen::VectorXi(6) << 0, 2, 1, 0, 2, 1).finished()) -
                calculateW((Eigen::VectorXi(6) << 0, 2, 1, 0, 1, 2).finished()) +
                calculateW((Eigen::VectorXi(6) << 0, 1, 2, 1, 2, 0).finished()) -
                calculateW((Eigen::VectorXi(6) << 0, 1, 2, 0, 2, 1).finished()) +
                calculateW((Eigen::VectorXi(6) << 0, 1, 2, 0, 1, 2).finished());

            double J = RiemannSumForDetF(phi, grid_xi, square.dx);

            Eigen::Vector3d VectorPhi1;
            VectorPhi1 <<
                phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1),
                -(phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k)),
                phi(3 * j)* phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);

            Eigen::Vector3d VectorPhi2;
            VectorPhi2 <<
                phi(3 * m + 1) * phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n + 1),
                -(phi(3 * m) * phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n)),
                phi(3 * m)* phi(3 * n + 1) - phi(3 * m + 1) * phi(3 * n);

            Eigen::Matrix3d PhiMatrix = VectorPhi1 * VectorPhi2.transpose();

            // 更新
            double factor = power[i] / J * W;
            HessianXi1.block<3, 3>(3 * l, 3 * xi) += PhiMatrix * factor;
        }
        };

    std::vector<int> indices(7);
    recursiveLoopForHesXi(0, 7, processIndices, indices);

    return HessianXi1;
}

Eigen::MatrixXd calHessianXi2(const Square& square, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd HessianXi2 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    auto processIndices = [&](const std::vector<int>& indices) {
        int xi = indices[0], n = indices[1], m = indices[2], l = indices[3], k = indices[4], j = indices[5], i = indices[6];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        std::vector<Eigen::Vector3i> diff_vectors = {
            FlatToGrid(i) - grid_xi,
            FlatToGrid(j) - grid_xi,
            FlatToGrid(k) - grid_xi,
            FlatToGrid(l) - grid_xi,
            FlatToGrid(m) - grid_xi,
            FlatToGrid(n) - grid_xi
        };

        if (std::all_of(diff_vectors.begin(), diff_vectors.end(), allElementsWithinOne)) {
            Eigen::Matrix<int, 3, 6> matrix;
            for (int col = 0; col < 6; ++col) {
                matrix.col(col) = diff_vectors[col];
            }

            auto calculateW = [&](const Eigen::VectorXi& axis) {
                return RiemannSum6(matrix, axis, square.dx);
                };

            std::vector<Eigen::VectorXi> axes = {
                (Eigen::VectorXi(7) << 1,2,0,1,2,0,0).finished(),
                (Eigen::VectorXi(7) << 1,2,0,0,2,1,0).finished(),
                (Eigen::VectorXi(7) << 1,2,0,0,1,2,0).finished(),
                (Eigen::VectorXi(7) << 0,2,1,1,2,0,0).finished(),
                (Eigen::VectorXi(7) << 0,2,1,0,2,1,0).finished(),
                (Eigen::VectorXi(7) << 0,2,1,0,1,2,0).finished(),
                (Eigen::VectorXi(7) << 0,1,2,1,2,0,0).finished(),
                (Eigen::VectorXi(7) << 0,1,2,0,2,1,0).finished(),
                (Eigen::VectorXi(7) << 0,1,2,0,1,2,0).finished(),
                (Eigen::VectorXi(7) << 1,2,0,1,2,0,1).finished(),
                (Eigen::VectorXi(7) << 1,2,0,0,2,1,1).finished(),
                (Eigen::VectorXi(7) << 1,2,0,0,1,2,1).finished(),
                (Eigen::VectorXi(7) << 0,2,1,1,2,0,1).finished(),
                (Eigen::VectorXi(7) << 0,2,1,0,2,1,1).finished(),
                (Eigen::VectorXi(7) << 0,2,1,0,1,2,1).finished(),
                (Eigen::VectorXi(7) << 0,1,2,1,2,0,1).finished(),
                (Eigen::VectorXi(7) << 0,1,2,0,2,1,1).finished(),
                (Eigen::VectorXi(7) << 0,1,2,0,1,2,1).finished(),
                (Eigen::VectorXi(7) << 1,2,0,1,2,0,2).finished(),
                (Eigen::VectorXi(7) << 1,2,0,0,2,1,2).finished(),
                (Eigen::VectorXi(7) << 1,2,0,0,1,2,2).finished(),
                (Eigen::VectorXi(7) << 0,2,1,1,2,0,2).finished(),
                (Eigen::VectorXi(7) << 0,2,1,0,2,1,2).finished(),
                (Eigen::VectorXi(7) << 0,2,1,0,1,2,2).finished(),
                (Eigen::VectorXi(7) << 0,1,2,1,2,0,2).finished(),
                (Eigen::VectorXi(7) << 0,1,2,0,2,1,2).finished(),
                (Eigen::VectorXi(7) << 0,1,2,0,1,2,2).finished()
            };

            std::vector<double> W_values;
            for (const auto& axis : axes) {
                W_values.push_back(calculateW(axis));
            }

            double J = RiemannSumForDetF(phi, grid_xi, square.dx);

            Eigen::Vector3d VectorW;
            for (int t = 0; t < 3; ++t) {
                VectorW(t) = W_values[t * 9] - W_values[t * 9 + 1] + W_values[t * 9 + 2] - W_values[t * 9 + 3] + W_values[t * 9 + 4]
                    - W_values[t * 9 + 5] + W_values[t * 9 + 6] - W_values[t * 9 + 7] + W_values[t * 9 + 8];
            }

            Eigen::Matrix3d MatrixW = VectorW * phi.segment<3>(3 * n).transpose();

            // phiの計算
            Eigen::Vector3d VectorPhi1;
            VectorPhi1 <<
                phi(3 * i + 1) * phi(3 * j + 2) - phi(3 * i + 2) * phi(3 * j + 1),
                -(phi(3 * i) * phi(3 * j + 2) - phi(3 * i + 2) * phi(3 * j)),
                phi(3 * i)* phi(3 * j + 1) - phi(3 * i + 1) * phi(3 * j);

            Eigen::Vector3d VectorPhi2;
            VectorPhi2 <<
                phi(3 * l + 1) * phi(3 * m + 2) - phi(3 * l + 2) * phi(3 * m + 1),
                -(phi(3 * l) * phi(3 * m + 2) - phi(3 * l + 2) * phi(3 * m)),
                phi(3 * l)* phi(3 * m + 1) - phi(3 * l + 1) * phi(3 * m);

            Eigen::Matrix3d PhiMatrix = VectorPhi1 * VectorPhi2.transpose();

            // 更新
            double factor = (2.0 / 3.0) * mu * std::pow(J, -8.0 / 3.0);
            HessianXi2.block<3, 3>(3 * k, 3 * xi) += factor * (PhiMatrix.array() * MatrixW.array()).matrix();
        }

        };

        

    std::vector<int> indices(7);
    recursiveLoopForHesXi(0, 7, processIndices, indices);

    return HessianXi2;
}

Eigen::MatrixXd calHessianXi3(const Square& square, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd HessianXi3 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    auto processIndices = [&](const std::vector<int>& indices) {
        int xi = indices[0], o = indices[1], n = indices[2], m = indices[3], l = indices[4], k = indices[5], j = indices[6], i = indices[7];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        std::vector<Eigen::Vector3i> diff_vectors = {
            FlatToGrid(i) - grid_xi,
            FlatToGrid(j) - grid_xi,
            FlatToGrid(k) - grid_xi,
            FlatToGrid(l) - grid_xi,
            FlatToGrid(m) - grid_xi,
            FlatToGrid(n) - grid_xi,
            FlatToGrid(o) - grid_xi
        };

        if (std::all_of(diff_vectors.begin(), diff_vectors.end(), allElementsWithinOne)) {
            Eigen::Matrix<int, 3, 7> matrix;
            for (int col = 0; col < 7; ++col) {
                matrix.col(col) = diff_vectors[col];
            }

            auto calculateW = [&](const Eigen::VectorXi& axis) {
                return RiemannSum6(matrix, axis, square.dx);
                };

            std::vector<Eigen::VectorXi> axes = {
                (Eigen::VectorXi(8) << 0,0,1,2,0,1,2,0).finished(),
                (Eigen::VectorXi(8) << 0,0,1,2,0,0,2,1).finished(),
                (Eigen::VectorXi(8) << 0,0,1,2,0,0,1,2).finished(),
                (Eigen::VectorXi(8) << 0,0,0,2,1,1,2,0).finished(),
                (Eigen::VectorXi(8) << 0,0,0,2,1,0,2,1).finished(),
                (Eigen::VectorXi(8) << 0,0,0,2,1,0,1,2).finished(),
                (Eigen::VectorXi(8) << 0,0,0,1,2,1,2,0).finished(),
                (Eigen::VectorXi(8) << 0,0,0,1,2,0,2,1).finished(),
                (Eigen::VectorXi(8) << 0,0,0,1,2,0,1,2).finished(),
                (Eigen::VectorXi(8) << 1,1,1,2,0,1,2,0).finished(),
                (Eigen::VectorXi(8) << 1,1,1,2,0,0,2,1).finished(),
                (Eigen::VectorXi(8) << 1,1,1,2,0,0,1,2).finished(),
                (Eigen::VectorXi(8) << 1,1,0,2,1,1,2,0).finished(),
                (Eigen::VectorXi(8) << 1,1,0,2,1,0,2,1).finished(),
                (Eigen::VectorXi(8) << 1,1,0,2,1,0,1,2).finished(),
                (Eigen::VectorXi(8) << 1,1,0,1,2,1,2,0).finished(),
                (Eigen::VectorXi(8) << 1,1,0,1,2,0,2,1).finished(),
                (Eigen::VectorXi(8) << 1,1,0,1,2,0,1,2).finished(),
                (Eigen::VectorXi(8) << 2,2,1,2,0,1,2,0).finished(),
                (Eigen::VectorXi(8) << 2,2,1,2,0,0,2,1).finished(),
                (Eigen::VectorXi(8) << 2,2,1,2,0,0,1,2).finished(),
                (Eigen::VectorXi(8) << 2,2,0,2,1,1,2,0).finished(),
                (Eigen::VectorXi(8) << 2,2,0,2,1,0,2,1).finished(),
                (Eigen::VectorXi(8) << 2,2,0,2,1,0,1,2).finished(),
                (Eigen::VectorXi(8) << 2,2,0,1,2,1,2,0).finished(),
                (Eigen::VectorXi(8) << 2,2,0,1,2,0,2,1).finished(),
                (Eigen::VectorXi(8) << 2,2,0,1,2,0,1,2).finished()
            };

            std::vector<double> W_values;
            for (const auto& axis : axes) {
                W_values.push_back(calculateW(axis));
            }

            double J = RiemannSumForDetF(phi, grid_xi, square.dx);

            double W1 = W_values[0] - W_values[1] + W_values[2] - W_values[3] + W_values[4] - W_values[5] + W_values[6] - W_values[7] + W_values[8];
            double W2 = W_values[9] - W_values[10] + W_values[11] - W_values[12] + W_values[13] - W_values[14] + W_values[15] - W_values[16] + W_values[17];
            double W3 = W_values[18] - W_values[19] + W_values[20] - W_values[21] + W_values[22] - W_values[23] + W_values[24] - W_values[25] + W_values[26];
            double W = W1 + W2 + W3;

            Eigen::Vector3d VectorPhi1;
            VectorPhi1 << phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1),
                -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l)),
                phi(3 * k)* phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);

            Eigen::Vector3d VectorPhi2;
            VectorPhi2 << phi(3 * n + 1) * phi(3 * o + 2) - phi(3 * n + 2) * phi(3 * o + 1),
                -(phi(3 * n) * phi(3 * o + 2) - phi(3 * n + 2) * phi(3 * o)),
                phi(3 * n)* phi(3 * o + 1) - phi(3 * n + 1) * phi(3 * o);

            double Phi = phi.segment<3>(3 * i).dot(phi.segment<3>(3 * j));

            Eigen::Matrix3d PhiMatrix = VectorPhi1 * VectorPhi2.transpose();

            double factor = (2.0 / 9.0) * mu * Phi * W * std::pow(J, -8.0 / 3.0);
            HessianXi3.block<3, 3>(3 * k, 3 * xi) += factor * PhiMatrix;
        }
        };

    std::vector<int> indices(8);
    recursiveLoopForHesXi(0, 8, processIndices, indices);

    return HessianXi3;
}