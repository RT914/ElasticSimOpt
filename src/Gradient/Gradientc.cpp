#include <Eigen/Dense>
#include <iostream>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Gradient/Gradientc.h"


// Calculate Gradient c
Eigen::VectorXd calGradientc(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power, const Eigen::VectorXd& theta) {
    Eigen::VectorXd Gradientc = Eigen::VectorXd::Zero(NumberOfParticles);

    Eigen::VectorXd Gradientc1 = calGradientc1(square, re_phi, phi, power, theta);
    Eigen::VectorXd Gradientc2 = calGradientc2(square, re_phi, theta);

    Gradientc = Gradientc1 - Gradientc2;

    /*std::cout << "Gradientc1" << std::endl;
    std::cout << Gradientc1 << std::endl;
    std::cout  << std::endl;
    std::cout << "Gradientc2" << std::endl;
    std::cout << Gradientc2 << std::endl;*/

    /*std::cout << "Gradientc" << std::endl;
    std::cout << Gradientc << std::endl;*/

    return Gradientc;
}


// Calculate Gradient c1
Eigen::VectorXd calGradientc1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power, const Eigen::VectorXd& theta) {
    Eigen::VectorXd Gradientc1 = Eigen::VectorXd::Zero(NumberOfParticles);

    const int kNumSection = 3; // 各区間の分割数
    const double kWidth = square.dx / kNumSection; // 分割の正規化
    const int kNum = square.SideNumber * kNumSection; // 全区間の分割数
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
    Eigen::VectorXd K = Eigen::VectorXd::Zero(NumberOfParticles);

    // 係数の計算
    // 体積変化率thetaと圧力pの計算
    for (int i = 0; i < NumberOfParticles; i++) {
        K(i) = (kappa / 2) * theta(i) + power(i);
    }

    // 内挿関数の計算
    // 区間分割
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                // Stencil Baseの計算
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
                    double hat_y_xi = HatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx);
                    double hat_z_xi = HatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx);
                    double f_xi_0 = hat_x_xi * hat_y_xi * hat_z_xi;

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                        // i関連の内挿関数の計算
                        double hat_x_i = HatFunction((cal_point(0) - grid_point_coordinates_i(0)) / square.dx);
                        double hat_y_i = HatFunction((cal_point(1) - grid_point_coordinates_i(1)) / square.dx);
                        double hat_z_i = HatFunction((cal_point(2) - grid_point_coordinates_i(2)) / square.dx);
                        double f_i_0 = hat_x_i * hat_y_i * hat_z_i;

                        double WeightIXi = K(i) * f_i_0 * f_xi_0;
                        double f_ixi = WeightIXi; // 柱の高さ

                        Gradientc1(xi) += f_ixi * volume_element;
                    }
                }

            }
        }
    }

    return Gradientc1;
}

// Calculate Gradient c2
Eigen::VectorXd calGradientc2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& theta) {
    Eigen::VectorXd Gradientc2 = Eigen::VectorXd::Zero(NumberOfParticles);

    const int kNumSection = 3; // 各区間の分割数
    const double kWidth = square.dx / kNumSection; // 分割の正規化
    const int kNum = square.SideNumber * kNumSection; // 全区間の分割数
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
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
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
                    double hat_y_xi = HatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx);
                    double hat_z_xi = HatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx);
                    double f_xi_0 = hat_x_xi * hat_y_xi * hat_z_xi;

                    double f_i_0 = 0.0;

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                        // i関連の内挿関数の計算
                        double hat_x_i = HatFunction((cal_point(0) - grid_point_coordinates_i(0)) / square.dx);
                        double hat_y_i = HatFunction((cal_point(1) - grid_point_coordinates_i(1)) / square.dx);
                        double hat_z_i = HatFunction((cal_point(2) - grid_point_coordinates_i(2)) / square.dx);
                        f_i_0 += theta(i) * hat_x_i * hat_y_i * hat_z_i;
                    }

                    if (abs(f_i_0) > 1e-6) {
                        double f_ixi = (kappa / 2) * (1 / f_i_0) * f_xi_0;
                        Gradientc2(xi) += f_ixi * volume_element;
                    }
                }

            }
        }
    }

    return Gradientc2;
}

