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

    // std::cout << "Vectorb1 size : " << Gradientb1.size() << std::endl;
    // std::cout << Gradientb1(31) << std::endl;
    // std::cout << Gradientb2(31) << std::endl;

    // std::cout << Gradientb1(13) << std::endl;
    // std::cout << Gradientb2(13) << std::endl;

    Gradientb = Gradientb1 - Gradientb2;

    // std::cout << Gradientb << std::endl;

    return Gradientb;
}

Eigen::VectorXd calGradientb1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& theta)
{
    Eigen::VectorXd Gradientb1 = Eigen::VectorXd::Zero(NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = square.SideNumber * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = 0; offset < square.SideNumber; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = (static_cast<double>(offset) + 1.0 / (2.0 * kNumSection)) * square.dx + divIndex * kWidth - 1.0;
            index++;
        }
    }

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                // Stencil Base‚ÌŒvZ
                Eigen::Vector3d stencil_base = calculateStencilBase(cal_point, square.dx);

                // Stencils—ñ‚Æstencil_num‚Ì¶¬
                Eigen::MatrixXi stencil;
                std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

                    double f_ixi = 0.0;
                    double WeightIXi;

                    // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_xi = HatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx);
                    double hat_y_xi = HatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx);
                    double hat_z_xi = HatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx);
                    double f_xi_0 = hat_x_xi * hat_y_xi * hat_z_xi;

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                        // “à‘}ŠÖ”‚ÌŒvZ
                        // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_i = HatFunction((cal_point(0) - grid_point_coordinates_i(0)) / square.dx);
                        double hat_y_i = HatFunction((cal_point(1) - grid_point_coordinates_i(1)) / square.dx);
                        double hat_z_i = HatFunction((cal_point(2) - grid_point_coordinates_i(2)) / square.dx);
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

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = square.SideNumber * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = 0; offset < square.SideNumber; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = (static_cast<double>(offset) + 1.0 / (2.0 * kNumSection)) * square.dx + divIndex * kWidth - 1.0;
            index++;
        }
    }

    // std::cout << cal_points.size() << std::endl;

    for (int x = 0; x < kNum; x++) {
        for (int y = 0; y < kNum; y++) {
            for (int z = 0; z < kNum; z++) {
                Eigen::Vector3d cal_point(cal_points(x), cal_points(y), cal_points(z));

                // Stencil Base‚ÌŒvZ
                Eigen::Vector3d stencil_base = calculateStencilBase(cal_point, square.dx);
                // std::cout << stencil_base << std::endl;

                // Stencils—ñ‚Æstencil_num‚Ì¶¬
                Eigen::MatrixXi stencil;
                std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

                // std::cout << "-----------------------" << std::endl;
                /*for (int s = 0; s < stencil_num.size(); s++) {
                    std::cout << stencil_num[s] << std::endl;
                }*/

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

                    // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double WeightXi 
                        = HatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx) 
                        * HatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx) 
                        * HatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx);

                    // std::cout << WeightXi << std::endl;

                    // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
                    double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, square.dx, 1.0);
                    // std::cout << detF << std::endl;

                    // std::cout << "xi = " << xi << std::endl;

                    Gradientb2(xi) += detF * WeightXi * volume_element;
                    /*if (xi == 31) {
                        std::cout << Gradientb2(xi) << std::endl;
                    }*/
                }
            }
        }
    }

    return Gradientb2;
}
