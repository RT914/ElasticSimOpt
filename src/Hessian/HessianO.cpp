#include <Eigen/Dense>
#include <iostream>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/Hessian/HessianO.h"


// Calculate Hessian O
Eigen::MatrixXd calHessianO(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd HessianO = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);

    const int kNumSection = 3; // 各区間の分割数
    const double kWidth = square.dx / kNumSection; // 分割の正規化
    const int kNum = square.SideNumber * kNumSection; // 全区間の分割数
    const int AllkNum = pow(kNum, 3);// 全次元の全区間分割数
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = 0; offset < square.SideNumber; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = (static_cast<double>(offset) + 1.0 / (2.0 * kNumSection)) * square.dx + divIndex * kWidth - 1.0;
            index++;
        }
    }

    // 係数の初期化
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);

    // 係数の計算
    // 現在座標phiの計算
    for (int j = 0; j < NumberOfParticles; j++) {
        for (int k = 0; k < NumberOfParticles; k++) {
            double Phi1 = phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1);
            double Phi2 = -(phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k));
            double Phi3 = phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_JK(3 * j + a, k) = Phi(a);
            }
        }
    }

    // 内挿関数の計算
    // 区間分割
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;
        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        // Stencil Baseの計算
        Eigen::Vector3d stencil_base = calculateStencilBase(cal_point, square.dx);

        // Stencil行列とstencil_numの生成
        Eigen::MatrixXi stencil;
        std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

        for (int xi = 0; xi < NumberOfParticles; xi++) {
            if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
            Eigen::Vector3i grid_xi = FlatToGrid(xi);
            Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

            // xi関連の内挿関数の計算
            double hat_x_xi = HatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx);
            double diff_hat_x_xi = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx) / square.dx;
            double hat_y_xi = HatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx);
            double diff_hat_y_xi = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx) / square.dx;
            double hat_z_xi = HatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx);
            double diff_hat_z_xi = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx) / square.dx;

            for (int tau = 0; tau < NumberOfParticles; tau++) {
                Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                if (!allElementsWithinOne(tau_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                // tau関連の内挿関数の計算
                double hat_x_tau = HatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx) / square.dx;
                double hat_y_tau = HatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx) / square.dx;
                double hat_z_tau = HatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx) / square.dx;

                double WeightTau = hat_x_tau * hat_y_tau * hat_z_tau;

                Eigen::Vector3d WeightJKTau = Eigen::Vector3d::Zero();

                for (int j = 0; j < NumberOfParticles; j++) {
                    Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                    if (!allElementsWithinOne(j_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_j = { re_phi(3 * j), re_phi(3 * j + 1), re_phi(3 * j + 2) };

                    // j関連の内挿関数の計算
                    double hat_x_j = HatFunction((cal_point(0) - grid_point_coordinates_j(0)) / square.dx);
                    double diff_hat_x_j = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_j(0)) / square.dx) / square.dx;
                    double hat_y_j = HatFunction((cal_point(1) - grid_point_coordinates_j(1)) / square.dx);
                    double diff_hat_y_j = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_j(1)) / square.dx) / square.dx;
                    double hat_z_j = HatFunction((cal_point(2) - grid_point_coordinates_j(2)) / square.dx);

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_k = { re_phi(3 * k), re_phi(3 * k + 1), re_phi(3 * k + 2) };

                        // k関連の内挿関数の計算
                        double hat_x_k = HatFunction((cal_point(0) - grid_point_coordinates_k(0)) / square.dx);
                        double hat_y_k = HatFunction((cal_point(1) - grid_point_coordinates_k(1)) / square.dx);
                        double diff_hat_y_k = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_k(1)) / square.dx) / square.dx;
                        double hat_z_k = HatFunction((cal_point(2) - grid_point_coordinates_k(2)) / square.dx);
                        double diff_hat_z_k = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_k(2)) / square.dx) / square.dx;

                        // 各項の計算
                        double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;
                        double w_k_3 = hat_x_k * hat_y_k * diff_hat_z_k;
                        double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                        double w_j_1 = diff_hat_x_j * hat_y_j * hat_z_j;
                        double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                        double w_k_2 = hat_x_k * diff_hat_y_k * hat_z_k;
                        double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int a = 0; a < dimensions; a++) {
                            WeightJKTau(a) += Phi_JK(3 * j + a, k)
                                * (w_j_2 * w_k_3 * w_xi_1 - w_j_1 * w_k_3 * w_xi_2 + w_j_1 * w_k_2 * w_xi_3);
                        }

                    }
                }

                for (int row = 0; row < dimensions; row++) { // 列数
                    double term = WeightTau * WeightJKTau(row) * volume_element;
                    if (abs(term) < 1e-10) continue;
                    HessianO(3 * xi + row, tau) += term;
                }

            }

        }

    }

    return HessianO;
}

