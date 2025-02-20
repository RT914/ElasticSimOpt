#include <iostream>
#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Hessian/HessianChi.h"

Eigen::MatrixXd calHessianChi(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianChi = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);
    Eigen::MatrixXd HessianChi1 = calHessianChi1(square, re_phi, phi);
    Eigen::MatrixXd HessianChi2 = calHessianChi2(square, re_phi, phi);
    Eigen::MatrixXd HessianChi3 = calHessianChi3(square, re_phi, phi);

    HessianChi = HessianChi1 + HessianChi2 + HessianChi3;

    return HessianChi;
}

Eigen::MatrixXd calHessianChi1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianChi1 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

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

    // 内挿関数の計算
    // 区間分割
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;

        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        /*------  式計算   ------*/
        // Stencil Baseの計算
        Eigen::Vector3d stencil_base = calculateStencilBase(cal_point, square.dx);

        // Stencil行列とstencil_numの生成
        Eigen::MatrixXi stencil;
        std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

        for (int xi = 0; xi < NumberOfParticles; xi++) {
            if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
            Eigen::Vector3i grid_xi = FlatToGrid(xi);
            Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

            double hat_x_xi = HatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx);
            double diff_hat_x_xi = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx) / square.dx;
            double hat_y_xi = HatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx);
            double diff_hat_y_xi = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx) / square.dx;
            double hat_z_xi = HatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx);
            double diff_hat_z_xi = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx) / square.dx;

            double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;
            double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;
            double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

            // 体積変化率の計算
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, square.dx, -2.0/3.0);

            for (int tau = 0; tau < NumberOfParticles; tau++) {
                Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                if (!allElementsWithinOne(tau_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                // 内挿関数の計算
                double hat_x_tau = HatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx);
                double diff_hat_x_tau = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx) / square.dx;
                double hat_y_tau = HatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx);
                double diff_hat_y_tau = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx) / square.dx;
                double hat_z_tau = HatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx);
                double diff_hat_z_tau = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx) / square.dx;

                double w_tau_1 = diff_hat_x_tau * hat_y_tau * hat_z_tau;
                double w_tau_2 = hat_x_tau * diff_hat_y_tau * hat_z_tau;
                double w_tau_3 = hat_x_tau * hat_y_tau * diff_hat_z_tau;

                double WeightTauXi = w_tau_1 * w_xi_1 + w_tau_2 * w_xi_2 + w_tau_3 * w_xi_3;

                // 単位行列
                Eigen::Matrix3d identityMatrix = Eigen::Matrix3d::Identity();

                for (int col = 0; col < dimensions; col++) { // 列数（横の数）
                    for (int row = 0; row < dimensions; row++) { // 行数（縦の数）
                        double term = mu * detF * WeightTauXi * identityMatrix(row, col) * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianChi1(3 * xi + row, 3 * tau + col) += term;
                    }
                }

            }
        }

        /*------  式計算終了   ------*/

    }

    return HessianChi1;
}

Eigen::MatrixXd calHessianChi2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianChi2 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

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
    Eigen::MatrixXd Phi_KL = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // 係数の計算
    // 現在座標phiの計算
    for (int k = 0; k < NumberOfParticles; k++) {
        for (int l = 0; l < NumberOfParticles; l++) {
            double Phi1 = phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1);
            double Phi2 = -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l));
            double Phi3 = phi(3 * k) * phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_KL(k, 3 * l + a) = Phi(a);
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

        /*------  式計算   ------*/
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

            // 体積変化率の計算
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, square.dx, -5.0 / 3.0);

            for (int tau = 0; tau < NumberOfParticles; tau++) {
                Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                if (!allElementsWithinOne(tau_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                // i関連の内挿関数の計算
                double diff_hat_x_tau = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx) / square.dx;
                double diff_hat_y_tau = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx) / square.dx;
                double diff_hat_z_tau = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx) / square.dx;
                double hat_x_tau = HatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx);
                double hat_y_tau = HatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx);
                double hat_z_tau = HatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx);

                double w_tau_1 = diff_hat_x_tau * hat_y_tau * hat_z_tau;
                double w_tau_2 = hat_x_tau * diff_hat_y_tau * hat_z_tau;
                double w_tau_3 = hat_x_tau * hat_y_tau * diff_hat_z_tau;

                Eigen::Vector3d WeightJTau = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightKLXi = Eigen::Vector3d::Zero();

                for (int j = 0; j < NumberOfParticles; j++) {
                    Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                    if (!allElementsWithinOne(j_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_j = { re_phi(3 * j), re_phi(3 * j + 1), re_phi(3 * j + 2) };
                    
                    // j関連の内挿関数の計算
                    double diff_hat_x_j = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_j(0)) / square.dx) / square.dx;
                    double diff_hat_y_j = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_j(1)) / square.dx) / square.dx;
                    double diff_hat_z_j = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_j(2)) / square.dx) / square.dx;
                    double hat_x_j = HatFunction((cal_point(0) - grid_point_coordinates_j(0)) / square.dx);
                    double hat_y_j = HatFunction((cal_point(1) - grid_point_coordinates_j(1)) / square.dx);
                    double hat_z_j = HatFunction((cal_point(2) - grid_point_coordinates_j(2)) / square.dx);

                    double w_j_1 = diff_hat_x_j * hat_y_j * hat_z_j;
                    double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;
                    double w_j_3 = hat_x_j * hat_y_j * diff_hat_z_j;

                    for (int a = 0; a < dimensions; a++) {
                        WeightJTau(a) += phi(3 * j + a) * (w_tau_1 * w_j_1 + w_tau_2 * w_j_2 + w_tau_3 * w_j_3);
                    }

                }

                for (int k = 0; k < NumberOfParticles; k++) {
                    Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                    if (!allElementsWithinOne(k_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_k = { re_phi(3 * k), re_phi(3 * k + 1), re_phi(3 * k + 2) };

                    // k関連の内挿関数の計算
                    double hat_x_k = HatFunction((cal_point(0) - grid_point_coordinates_k(0)) / square.dx);
                    double diff_hat_x_k = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_k(0)) / square.dx) / square.dx;
                    double hat_y_k = HatFunction((cal_point(1) - grid_point_coordinates_k(1)) / square.dx);
                    double diff_hat_y_k = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_k(1)) / square.dx) / square.dx;
                    double hat_z_k = HatFunction((cal_point(2) - grid_point_coordinates_k(2)) / square.dx);

                    for (int l = 0; l < NumberOfParticles; l++) {
                        Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                        if (!allElementsWithinOne(l_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_l = { re_phi(3 * l), re_phi(3 * l + 1), re_phi(3 * l + 2) };

                        // l関連の内挿関数の計算
                        double hat_x_l = HatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx);
                        double hat_y_l = HatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx);
                        double diff_hat_y_l = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx) / square.dx;
                        double hat_z_l = HatFunction((cal_point(2) - grid_point_coordinates_l(2)) / square.dx);
                        double diff_hat_z_l = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_l(2)) / square.dx) / square.dx;

                        // 各項の計算
                        double w_k_2 = hat_x_k * diff_hat_y_k * hat_z_k;
                        double w_l_3 = hat_x_l * hat_y_l * diff_hat_z_l;
                        double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                        double w_k_1 = diff_hat_x_k * hat_y_k * hat_z_k;
                        double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                        double w_l_2 = hat_x_l * diff_hat_y_l * hat_z_l;
                        double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int a = 0; a < dimensions; a++) {
                            WeightKLXi(a) += Phi_KL(k, 3 * l + a)
                                * (w_k_2 * w_l_3 * w_xi_1 - w_k_1 * w_l_3 * w_xi_2 + w_k_1 * w_l_2 * w_xi_3);
                        }

                    }
                }

                for (int col = 0; col < dimensions; col++) { // 列数（横の数）
                    for (int row = 0; row < dimensions; row++) { // 行数（縦の数）
                        double term = mu * detF * WeightJTau(row) * WeightKLXi(col) * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianChi2(3 * xi + row, 3 * tau + col) += term;
                    }
                }

            }
        }

        /*------  式計算終了   ------*/

    }

    return HessianChi2;
}

Eigen::MatrixXd calHessianChi3(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianChi3 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

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

    Eigen::MatrixXd Phi_KL = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);

    // 係数の計算
    // 現在座標phiの計算
    for (int k = 0; k < NumberOfParticles; k++) {
        for (int l = 0; l < NumberOfParticles; l++) {
            double Phi1 = phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1);
            double Phi2 = -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l));
            double Phi3 = phi(3 * k) * phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_KL(3 * k + a, l) = Phi(a);
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

        /*------  式計算   ------*/
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

            // 体積変化率の計算
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, square.dx, -5.0 / 3.0);

            for (int tau = 0; tau < NumberOfParticles; tau++) {
                Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                if (!allElementsWithinOne(tau_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                // tau関連の内挿関数の計算
                double hat_x_tau = HatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx);
                double diff_hat_x_tau = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx) / square.dx;
                double hat_y_tau = HatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx);
                double diff_hat_y_tau = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx) / square.dx;
                double hat_z_tau = HatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx);
                double diff_hat_z_tau = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx) / square.dx;

                double w_tau_1 = diff_hat_x_tau * hat_y_tau * hat_z_tau;
                double w_tau_2 = hat_x_tau * diff_hat_y_tau * hat_z_tau;
                double w_tau_3 = hat_x_tau * hat_y_tau * diff_hat_z_tau;

                Eigen::Vector3d WeightJTau = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightKLXi = Eigen::Vector3d::Zero();

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
                    double diff_hat_z_j = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_j(2)) / square.dx) / square.dx;

                    double w_j_1 = diff_hat_x_j * hat_y_j * hat_z_j;
                    double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;
                    double w_j_3 = hat_x_j * hat_y_j * diff_hat_z_j;

                    for (int a = 0; a < dimensions; a++) {
                        WeightJTau(a) += phi(3 * j + a) * (w_tau_1 * w_j_1 + w_tau_2 * w_j_2 + w_tau_3 * w_j_3);
                    }

                }

                for (int k = 0; k < NumberOfParticles; k++) {
                    Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                    if (!allElementsWithinOne(k_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_k = { re_phi(3 * k), re_phi(3 * k + 1), re_phi(3 * k + 2) };

                    // k関連の内挿関数の計算
                    double hat_x_k = HatFunction((cal_point(0) - grid_point_coordinates_k(0)) / square.dx);
                    double diff_hat_x_k = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_k(0)) / square.dx) / square.dx;
                    double hat_y_k = HatFunction((cal_point(1) - grid_point_coordinates_k(1)) / square.dx);
                    double diff_hat_y_k = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_k(1)) / square.dx) / square.dx;
                    double hat_z_k = HatFunction((cal_point(2) - grid_point_coordinates_k(2)) / square.dx);

                    for (int l = 0; l < NumberOfParticles; l++) {
                        Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                        if (!allElementsWithinOne(l_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_l = { re_phi(3 * l), re_phi(3 * l + 1), re_phi(3 * l + 2) };

                        // l関連の内挿関数の計算
                        double hat_x_l = HatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx);
                        double hat_y_l = HatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx);
                        double diff_hat_y_l = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx) / square.dx;
                        double hat_z_l = HatFunction((cal_point(2) - grid_point_coordinates_l(2)) / square.dx);
                        double diff_hat_z_l = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_l(2)) / square.dx) / square.dx;

                        // 各項の計算
                        double w_k_2 = hat_x_k * diff_hat_y_k * hat_z_k;
                        double w_l_3 = hat_x_l * hat_y_l * diff_hat_z_l;
                        double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                        double w_k_1 = diff_hat_x_k * hat_y_k * hat_z_k;
                        double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                        double w_l_2 = hat_x_l * diff_hat_y_l * hat_z_l;
                        double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int a = 0; a < dimensions; a++) {
                            WeightKLXi(a) += Phi_KL(3 * k + a, l) 
                                * (w_k_2 * w_l_3 * w_xi_1 - w_k_1 * w_l_3 * w_xi_2 + w_k_1 * w_l_2 * w_xi_3);
                        }

                    }
                }

                // 単位行列
                Eigen::Matrix3d identityMatrix = Eigen::Matrix3d::Identity();

                for (int col = 0; col < dimensions; col++) { // 列数（横の数）
                    for (int row = 0; row < dimensions; row++) { // 行数（縦の数）
                        double term = -2.0 / 3.0 * mu * detF * WeightKLXi(row) * WeightJTau(col) * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianChi3(3 * xi + row, 3 * tau + col) += term;
                    }
                }

            }
        }

        /*------  式計算終了   ------*/
    }

    return HessianChi3;
}

