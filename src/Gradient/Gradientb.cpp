#include <Eigen/Dense>
#include <iostream>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/Gradient/Gradientb.h"


// Calculate Gradient b
Eigen::VectorXd calGradientb(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& theta) {
    Eigen::VectorXd Gradientb = Eigen::VectorXd::Zero(NumberOfParticles);

    Eigen::VectorXd Gradientb1 = calGradientb1(square, re_phi, theta);
    Eigen::VectorXd Gradientb2 = calGradientb2(square, re_phi, phi);

    Gradientb = Gradientb1 - Gradientb2;

    /*std::cout << "Gradientb1" << std::endl;
    std::cout << Gradientb1 << std::endl;
    std::cout  << std::endl;
    std::cout << "Gradientb2" << std::endl;
    std::cout << Gradientb2 << std::endl;*/

    return Gradientb;
}

Eigen::VectorXd calGradientb1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& theta)
{
    Eigen::VectorXd Gradientb1 = Eigen::VectorXd::Zero(NumberOfParticles);

    const int kNumSection = 3; // 各区間の分割数
    const double kWidth = square.dx / kNumSection; // 分割の正規化
    const int kNum = square.SideNumber * kNumSection; // 全区間の分割数
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -1; offset <= 0; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
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
                Eigen::Vector3d stencil_base = calculateStencilBase(cal_point);

                // Stencil行列とstencil_numの生成
                Eigen::MatrixXi stencil;
                std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

                    double f_ixi = 0.0;
                    double WeightIXi;

                    // xi関連の内挿関数の計算
                    double hat_x_xi = HatFunction(cal_point(0) - grid_point_coordinates_xi(0));
                    double hat_y_xi = HatFunction(cal_point(1) - grid_point_coordinates_xi(1));
                    double hat_z_xi = HatFunction(cal_point(2) - grid_point_coordinates_xi(2));
                    double f_xi_0 = hat_x_xi * hat_y_xi * hat_z_xi;


                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                        // 内挿関数の計算
                        // i関連の内挿関数の計算
                        double hat_x_i = HatFunction(cal_point(0) - grid_point_coordinates_i(0));
                        double hat_y_i = HatFunction(cal_point(1) - grid_point_coordinates_i(1));
                        double hat_z_i = HatFunction(cal_point(2) - grid_point_coordinates_i(2));
                        double f_i_0 = hat_x_i * hat_y_i * hat_z_i;

                        WeightIXi = theta(i) * f_i_0 * f_xi_0;
                        f_ixi += WeightIXi;
                    }

                    Gradientb1(xi) += f_ixi * volume_element;
                }

            }
        }
    }

    return Gradientb1;
}

Eigen::VectorXd calGradientb2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi)
{
    Eigen::VectorXd Gradientb2 = Eigen::VectorXd::Zero(NumberOfParticles);

    const int kNumSection = 3; // 各区間の分割数
    const double kWidth = square.dx / kNumSection; // 分割の正規化
    const int kNum = 2 * kNumSection; // 全区間の分割数
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -1; offset <= 0; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    for (int x = 0; x < kNum; x++) {
        for (int y = 0; y < kNum; y++) {
            for (int z = 0; z < kNum; z++) {
                Eigen::Vector3d cal_point(cal_points(x), cal_points(y), cal_points(z));

                // Stencil Baseの計算
                Eigen::Vector3d stencil_base = calculateStencilBase(cal_point);

                // Stencil行列とstencil_numの生成
                Eigen::MatrixXi stencil;
                std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

                    // xi関連の内挿関数の計算
                    double WeightXi 
                        = HatFunction(cal_point(0) - grid_point_coordinates_xi(0)) 
                        * HatFunction(cal_point(1) - grid_point_coordinates_xi(1)) 
                        * HatFunction(cal_point(2) - grid_point_coordinates_xi(2));

                    // 体積変化率の計算
                    double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, 1.0);
                    // std::cout << detF << std::endl;

                    Gradientb2(xi) += volume_element * detF * WeightXi;
                }
            }
        }
    }

    return Gradientb2;
}
