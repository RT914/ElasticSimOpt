#include <iostream>
#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Hessian/HessianXi.h"

// Caluculate HessianXi
Eigen::MatrixXd calHessianXi(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power) {
    Eigen::MatrixXd HessianXi = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    Eigen::MatrixXd HessianXi1 = calHessianXi1(square, re_phi, phi, power);
    if (HessianXi1.array().isNaN().any()) {
        std::cerr << "NaN detected HessianXi1" << std::endl;
    }

    Eigen::MatrixXd HessianXi2 = calHessianXi2(square, re_phi, phi);
    if (HessianXi2.array().isNaN().any()) {
        std::cerr << "NaN detected HessianXi2" << std::endl;
    }

    Eigen::MatrixXd HessianXi3 = calHessianXi3(square, re_phi, phi);
    if (HessianXi3.array().isNaN().any()) {
        std::cerr << "NaN detected HessianXi3" << std::endl;
    }

    HessianXi = HessianXi1 + HessianXi2 + HessianXi3;

    return HessianXi;
}

Eigen::MatrixXd calHessianXi1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power)
{
    Eigen::MatrixXd HessianXi1 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // 各区間の分割数
    const double kWidth = square.dx / kNumSection; // 分割の正規化
    const int kNum = square.SideNumber * kNumSection; // 全区間の分割数
    const int AllkNum = pow(kNum, 3);// 全次元の全区間分割数
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = 0; offset < square.SideNumber; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = ( static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) ) * square.dx + divIndex * kWidth - 1.0;
            index++;
        }
    }

    // std::cout << cal_points << std::endl;

    // 係数の初期化
    Eigen::MatrixXd Phi_NP = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_LO = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // 係数の計算
    // 現在座標phiの計算
    for (int n = 0; n < NumberOfParticles; n++) {
        for (int p = 0; p < NumberOfParticles; p++) {
            double Phi1 = phi(3 * n + 1) * phi(3 * p + 2) - phi(3 * n + 2) * phi(3 * p + 1);
            double Phi2 = -(phi(3 * n) * phi(3 * p + 2) - phi(3 * n + 2) * phi(3 * p));
            double Phi3 = phi(3 * n) * phi(3 * p + 1) - phi(3 * n + 1) * phi(3 * p);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_NP(3 * n + a, p) = Phi(a);
            }
        }
    }

    for (int l = 0; l < NumberOfParticles; l++) {
        for (int o = 0; o < NumberOfParticles; o++) {
            double Phi1 = phi(3 * l + 1) * phi(3 * o + 2) - phi(3 * l + 2) * phi(3 * o + 1);
            double Phi2 = -(phi(3 * l) * phi(3 * o + 2) - phi(3 * l + 2) * phi(3 * o));
            double Phi3 = phi(3 * l) * phi(3 * o + 1) - phi(3 * l + 1) * phi(3 * o);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_LO(l, 3 * o + a) = Phi(a);
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
        /*std::cout << "stencil_base : " << std::endl;
        std::cout << stencil_base << std::endl;
        std::cout << std::endl;*/

        // Stencil行列とstencil_numの生成
        Eigen::MatrixXi stencil;
        std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

        /*for (int s = 0; s < stencil_num.size(); s++) {
            std::cout << stencil_num[s] << std::endl;
        }*/

        // std::cout << "---------------------------" << std::endl;

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

            double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;
            double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;
            double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

            // 体積変化率の計算
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, square.dx, -1.0);
            // std::cout << " xi = " << xi << " : " << detF << std::endl;

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

                Eigen::Vector3d WeightLOTau = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightNPXi = Eigen::Vector3d::Zero();
                double WeightI = 0.0;

                for (int l = 0; l < NumberOfParticles; l++) {
                    Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                    if (!allElementsWithinOne(l_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_l = { re_phi(3 * l), re_phi(3 * l + 1), re_phi(3 * l + 2) };

                    // j関連の内挿関数の計算
                    double hat_x_l = HatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx);
                    double diff_hat_x_l = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx) / square.dx;
                    double hat_y_l = HatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx);
                    double diff_hat_y_l = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx) / square.dx;
                    double hat_z_l = HatFunction((cal_point(2) - grid_point_coordinates_l(2)) / square.dx);

                    double w_l_1 = diff_hat_x_l * hat_y_l * hat_z_l;
                    double w_l_2 = hat_x_l * diff_hat_y_l * hat_z_l;

                    for (int o = 0; o < NumberOfParticles; o++) {
                        Eigen::Vector3i o_minus_xi = FlatToGrid(o) - grid_xi;
                        if (!allElementsWithinOne(o_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_o = { re_phi(3 * o), re_phi(3 * o + 1), re_phi(3 * o + 2) };

                        // o関連の内挿関数の計算
                        double hat_x_o = HatFunction((cal_point(0) - grid_point_coordinates_o(0)) / square.dx);
                        double hat_y_o = HatFunction((cal_point(1) - grid_point_coordinates_o(1)) / square.dx);
                        double diff_hat_y_o = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_o(1)) / square.dx) / square.dx;
                        double hat_z_o = HatFunction((cal_point(2) - grid_point_coordinates_o(2)) / square.dx);
                        double diff_hat_z_o = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_o(2)) / square.dx) / square.dx;

                        double w_o_2 = hat_x_o * diff_hat_y_o * hat_z_o;
                        double w_o_3 = hat_x_o * hat_y_o * diff_hat_z_o;

                        for (int a = 0; a < dimensions; a++) {
                            WeightLOTau(a) += Phi_LO(l, 3 * o + a) *
                                (w_l_2 * w_o_3 * w_tau_1
                                    - w_l_1 * w_o_3 * w_tau_2
                                    + w_l_1 * w_o_2 * w_tau_3);
                        }

                    }
                }

                for (int n = 0; n < NumberOfParticles; n++) {
                    Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                    if (!allElementsWithinOne(n_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_n = { re_phi(3 * n), re_phi(3 * n + 1), re_phi(3 * n + 2) };

                    // n関連の内挿関数の計算
                    double hat_x_n = HatFunction((cal_point(0) - grid_point_coordinates_n(0)) / square.dx);
                    double diff_hat_x_n = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_n(0)) / square.dx) / square.dx;
                    double hat_y_n = HatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx);
                    double diff_hat_y_n = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx) / square.dx;
                    double hat_z_n = HatFunction((cal_point(2) - grid_point_coordinates_n(2)) / square.dx);

                    double w_n_1 = diff_hat_x_n * hat_y_n * hat_z_n;
                    double w_n_2 = hat_x_n * diff_hat_y_n * hat_z_n;

                    for (int p = 0; p < NumberOfParticles; p++) {
                        Eigen::Vector3i p_minus_xi = FlatToGrid(p) - grid_xi;
                        if (!allElementsWithinOne(p_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_p = { re_phi(3 * p), re_phi(3 * p + 1), re_phi(3 * p + 2) };

                        // p関連の内挿関数の計算
                        double hat_x_p = HatFunction((cal_point(0) - grid_point_coordinates_p(0)) / square.dx);
                        double hat_y_p = HatFunction((cal_point(1) - grid_point_coordinates_p(1)) / square.dx);
                        double diff_hat_y_p = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_p(1)) / square.dx) / square.dx;
                        double hat_z_p = HatFunction((cal_point(2) - grid_point_coordinates_p(2)) / square.dx);
                        double diff_hat_z_p = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_p(2)) / square.dx) / square.dx;

                        double w_p_2 = hat_x_p * diff_hat_y_p * hat_z_p;
                        double w_p_3 = hat_x_p * hat_y_p * diff_hat_z_p;

                        for (int a = 0; a < dimensions; a++) {
                            WeightNPXi(a) += Phi_NP(3 * n + a, p)
                                * (w_n_2 * w_p_3 * w_xi_1
                                    - w_n_1 * w_p_3 * w_xi_2
                                    + w_n_1 * w_p_2 * w_xi_3);
                        }

                    }
                }


                for (int i = 0; i < NumberOfParticles; i++) {
                    Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                    if (!allElementsWithinOne(i_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                    // 内挿関数の計算
                    double hat_x_i = HatFunction((cal_point(0) - grid_point_coordinates_i(0)) / square.dx);
                    double hat_y_i = HatFunction((cal_point(1) - grid_point_coordinates_i(1)) / square.dx);
                    double hat_z_i = HatFunction((cal_point(2) - grid_point_coordinates_i(2)) / square.dx);
                    WeightI += power(i) * hat_x_i * hat_y_i * hat_z_i;
                }

                // HessianXi1の計算
                for (int col = 0; col < dimensions; col++) { // 列数
                    for (int row = 0; row < dimensions; row++) { // 行数
                        double term = detF * WeightNPXi(row) * WeightLOTau(col) * WeightI * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianXi1(3 * xi + row, 3 * tau + col) += term;
                    }
                }

            }

        }

    }

    return  HessianXi1;
}


Eigen::MatrixXd calHessianXi2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd  HessianXi2 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

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
    Eigen::MatrixXd Phi_LN = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // 係数の計算
    // 現在座標phiの計算
    for (int l = 0; l < NumberOfParticles; l++) {
        for (int n = 0; n < NumberOfParticles; n++) {
            double Phi1 = phi(3 * l + 1) * phi(3 * n + 2) - phi(3 * l + 2) * phi(3 * n + 1);
            double Phi2 = -(phi(3 * l) * phi(3 * n + 2) - phi(3 * l + 2) * phi(3 * n));
            double Phi3 = phi(3 * l) * phi(3 * n + 1) - phi(3 * l + 1) * phi(3 * n);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_LN(l, 3 * n + a) = Phi(a);
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
        // std::cout << "----------------------------" << std::endl;

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
            // std::cout << detF << std::endl;

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

                Eigen::Vector3d WeightLNTau = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightJXi = Eigen::Vector3d::Zero();

                for (int l = 0; l < NumberOfParticles; l++) {
                    Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                    if (!allElementsWithinOne(l_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_l = { re_phi(3 * l), re_phi(3 * l + 1), re_phi(3 * l + 2) };

                    // l関連の内挿関数の計算
                    double hat_x_l = HatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx);
                    double diff_hat_x_l = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx) / square.dx;
                    double hat_y_l = HatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx);
                    double diff_hat_y_l = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx) / square.dx;
                    double hat_z_l = HatFunction((cal_point(2) - grid_point_coordinates_l(2)) / square.dx);

                    for (int n = 0; n < NumberOfParticles; n++) {
                        Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                        if (!allElementsWithinOne(n_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_n = { re_phi(3 * n), re_phi(3 * n + 1), re_phi(3 * n + 2) };

                        // n関連の内挿関数の計算
                        double hat_x_n = HatFunction((cal_point(0) - grid_point_coordinates_n(0)) / square.dx);
                        double hat_y_n = HatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx);
                        double diff_hat_y_n = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx) / square.dx;
                        double hat_z_n = HatFunction((cal_point(2) - grid_point_coordinates_n(2)) / square.dx);
                        double diff_hat_z_n = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_n(2)) / square.dx) / square.dx;

                        // 各項の計算
                        double w_l_2 = hat_x_l * diff_hat_y_l * hat_z_l;
                        double w_n_3 = hat_x_n * hat_y_n * diff_hat_z_n;
                        double w_tau_1 = diff_hat_x_tau * hat_y_tau * hat_z_tau;

                        double w_l_1 = diff_hat_x_l * hat_y_l * hat_z_l;
                        double w_tau_2 = hat_x_tau * diff_hat_y_tau * hat_z_tau;

                        double w_n_2 = hat_x_n * diff_hat_y_n * hat_z_n;
                        double w_tau_3 = hat_x_tau * hat_y_tau * diff_hat_z_tau;

                        for (int a = 0; a < dimensions; a++) {
                            WeightLNTau(a) += Phi_LN(l, 3 * n + a) *
                                (w_l_2 * w_n_3 * w_tau_1
                                    - w_l_1 * w_n_3 * w_tau_2
                                    + w_l_1 * w_n_2 * w_tau_3);
                        }
                    }
                }

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

                    // 各項の計算
                    double w_j_1 = diff_hat_x_j * hat_y_j * hat_z_j;
                    double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;
                    double w_j_3 = hat_x_j * hat_y_j * diff_hat_z_j;

                    double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;
                    double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;
                    double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                    for (int a = 0; a < dimensions; a++) {
                        WeightJXi(a) += phi(3 * j + a) * (w_j_1 * w_xi_1 + w_j_2 * w_xi_2 + w_j_3 * w_xi_3);
                    }
                }

                for (int col = 0; col < dimensions; col++) { // 列数
                    for (int row = 0; row < dimensions; row++) { // 行数
                        double term = (2.0 / 3.0) * mu * detF * WeightJXi(row) * WeightLNTau(col) * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianXi2(3 * xi + row, 3 * tau + col) += term;
                    }
                }
            }

        }

    }

    return  HessianXi2;
}


Eigen::MatrixXd calHessianXi3(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd  HessianXi3 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

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
    Eigen::MatrixXd Phi_MO = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_LN = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);

    // 係数の計算
    // 現在座標phiの計算
    for (int m = 0; m < NumberOfParticles; m++) {
        for (int o = 0; o < NumberOfParticles; o++) {
            double Phi1 = phi(3 * m + 1) * phi(3 * o + 2) - phi(3 * m + 2) * phi(3 * o + 1);
            double Phi2 = -(phi(3 * m) * phi(3 * o + 2) - phi(3 * m + 2) * phi(3 * o));
            double Phi3 = phi(3 * m) * phi(3 * o + 1) - phi(3 * m + 1) * phi(3 * o);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_MO(3 * m + a, o) = Phi(a);
            }

        }
    }

    for (int l = 0; l < NumberOfParticles; l++) {
        for (int n = 0; n < NumberOfParticles; n++) {
            double Phi1 = phi(3 * l + 1) * phi(3 * n + 2) - phi(3 * l + 2) * phi(3 * n + 1);
            double Phi2 = -(phi(3 * l) * phi(3 * n + 2) - phi(3 * l + 2) * phi(3 * n));
            double Phi3 = phi(3 * l) * phi(3 * n + 1) - phi(3 * l + 1) * phi(3 * n);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_LN(l, 3 * n + a) = Phi(a);
            }
        }
    }

    for (int j = 0; j < NumberOfParticles; j++) {
        for (int k = 0; k < NumberOfParticles; k++) {
            Phi_JK(j, k) = phi(3 * j) * phi(3 * k) + phi(3 * j + 1) * phi(3 * k + 1) + phi(3 * j + 2) * phi(3 * k + 2);
        }
    }

    // 内挿関数の計算
    // 区間分割
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;
        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        // std::cout << "--------------------------------------------------------" << std::endl;

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
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, square.dx, -8.0/3.0);

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

                Eigen::Vector3d WeightLNTau = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightMOXi = Eigen::Vector3d::Zero();
                double WeightJK = 0.0;

                for (int l = 0; l < NumberOfParticles; l++) {
                    Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                    if (!allElementsWithinOne(l_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_l = { re_phi(3 * l), re_phi(3 * l + 1), re_phi(3 * l + 2) };

                    // l関連の内挿関数の計算
                    double hat_x_l = HatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx);
                    double diff_hat_x_l = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx) / square.dx;
                    double hat_y_l = HatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx);
                    double diff_hat_y_l = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx) / square.dx;
                    double hat_z_l = HatFunction((cal_point(2) - grid_point_coordinates_l(2)) / square.dx);

                    for (int n = 0; n < NumberOfParticles; n++) {
                        Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                        if (!allElementsWithinOne(n_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_n = { re_phi(3 * n), re_phi(3 * n + 1), re_phi(3 * n + 2) };

                        // n関連の内挿関数の計算
                        double hat_x_n = HatFunction((cal_point(0) - grid_point_coordinates_n(0)) / square.dx);
                        double hat_y_n = HatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx);
                        double diff_hat_y_n = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx) / square.dx;
                        double hat_z_n = HatFunction((cal_point(2) - grid_point_coordinates_n(2)) / square.dx);
                        double diff_hat_z_n = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_n(2)) / square.dx) / square.dx;

                        // 各項の計算
                        double w_l_2 = hat_x_l * diff_hat_y_l * hat_z_l;
                        double w_n_3 = hat_x_n * hat_y_n * diff_hat_z_n;
                        double w_tau_1 = diff_hat_x_tau * hat_y_tau * hat_z_tau;

                        double w_l_1 = diff_hat_x_l * hat_y_l * hat_z_l;
                        double w_tau_2 = hat_x_tau * diff_hat_y_tau * hat_z_tau;

                        double w_n_2 = hat_x_n * diff_hat_y_n * hat_z_n;
                        double w_tau_3 = hat_x_tau * hat_y_tau * diff_hat_z_tau;

                        for (int a = 0; a < dimensions; a++) {
                            WeightLNTau(a) += Phi_LN(l, 3 * n + a) *
                                (w_l_2 * w_n_3 * w_tau_1 - w_l_1 * w_n_3 * w_tau_2 + w_l_1 * w_n_2 * w_tau_3);
                        }
                    }
                }

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

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_k = { re_phi(3 * k), re_phi(3 * k + 1), re_phi(3 * k + 2) };

                        // k関連の内挿関数の計算
                        double diff_hat_x_k = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_k(0)) / square.dx) / square.dx;
                        double diff_hat_y_k = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_k(1)) / square.dx) / square.dx;
                        double diff_hat_z_k = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_k(2)) / square.dx) / square.dx;
                        double hat_x_k = HatFunction((cal_point(0) - grid_point_coordinates_k(0)) / square.dx);
                        double hat_y_k = HatFunction((cal_point(1) - grid_point_coordinates_k(1)) / square.dx);
                        double hat_z_k = HatFunction((cal_point(2) - grid_point_coordinates_k(2)) / square.dx);

                        double w_k_1 = diff_hat_x_k * hat_y_k * hat_z_k;
                        double w_k_2 = hat_x_k * diff_hat_y_k * hat_z_k;
                        double w_k_3 = hat_x_k * hat_y_k * diff_hat_z_k;

                        WeightJK += Phi_JK(j, k) * (w_j_1 * w_k_1 + w_j_2 * w_k_2 + w_j_3 * w_k_3);
                    }
                }


                // std::cout << WeightLM << std::endl;
                // std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;

                for (int m = 0; m < NumberOfParticles; m++) {
                    Eigen::Vector3i m_minus_xi = FlatToGrid(m) - grid_xi;
                    if (!allElementsWithinOne(m_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_m = { re_phi(3 * m), re_phi(3 * m + 1), re_phi(3 * m + 2) };

                    // m関連の内挿関数の計算
                    double hat_x_m = HatFunction((cal_point(0) - grid_point_coordinates_m(0)) / square.dx);
                    double diff_hat_x_m = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_m(0)) / square.dx) / square.dx;
                    double hat_y_m = HatFunction((cal_point(1) - grid_point_coordinates_m(1)) / square.dx);
                    double diff_hat_y_m = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_m(1)) / square.dx) / square.dx;
                    double hat_z_m = HatFunction((cal_point(2) - grid_point_coordinates_m(2)) / square.dx);

                    for (int o = 0; o < NumberOfParticles; o++) {
                        Eigen::Vector3i o_minus_xi = FlatToGrid(o) - grid_xi;
                        if (!allElementsWithinOne(o_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_o = { re_phi(3 * o), re_phi(3 * o + 1), re_phi(3 * o + 2) };

                        // o関連の内挿関数の計算
                        double hat_x_o = HatFunction((cal_point(0) - grid_point_coordinates_o(0)) / square.dx);
                        double hat_y_o = HatFunction((cal_point(1) - grid_point_coordinates_o(1)) / square.dx);
                        double diff_hat_y_o = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_o(1)) / square.dx) / square.dx;
                        double hat_z_o = HatFunction((cal_point(2) - grid_point_coordinates_o(2)) / square.dx);
                        double diff_hat_z_o = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_o(2)) / square.dx) / square.dx;

                        // 各項の計算
                        double w_m_2 = hat_x_m * diff_hat_y_m * hat_z_m;
                        double w_o_3 = hat_x_o * hat_y_o * diff_hat_z_o;
                        double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                        double w_m_1 = diff_hat_x_m * hat_y_m * hat_z_m;
                        double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                        double w_o_2 = hat_x_o * diff_hat_y_o * hat_z_o;
                        double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int a = 0; a < dimensions; a++) {
                            WeightMOXi(a) += Phi_MO(3 * m + a, o)
                                * (w_m_2 * w_o_3 * w_xi_1 - w_m_1 * w_o_3 * w_xi_2 + w_m_1 * w_o_2 * w_xi_3);
                        }

                    }
                }

                for (int col = 0; col < dimensions; col++) { // 列数
                    for (int row = 0; row < dimensions; row++) { // 行数
                        double term = -(2.0 / 9.0) * mu * detF * WeightMOXi(row) * WeightLNTau(col) * WeightJK * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianXi3(3 * xi + row, 3 * tau + col) += term;
                    }
                }

            }

        }

    }

    return  HessianXi3;
}
