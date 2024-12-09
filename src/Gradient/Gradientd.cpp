#include <Eigen/Dense>
#include <iostream>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Gradient/Gradientd.h"


Eigen::VectorXd calGradientd(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& phi_previous, const Eigen::VectorXd& power) {
    Eigen::VectorXd Gradientd(3 * NumberOfParticles);
    Gradientd.setZero();

    Eigen::VectorXd GradientG1 = calGradientG1(square, re_phi, phi, phi_previous);

    Eigen::VectorXd GradientG2 = calGradientG2(square, re_phi, phi, power);

    // std::cout << "GradientG2 : " << std::endl;
    // std::cout << GradientG2 << std::endl;

    // Gradientd = - GradientG1 - GradientG2;
    Gradientd = - GradientG2;

    // std::cout << "Gradientd" << std::endl;
    // std::cout << Gradientd << std::endl;

    return Gradientd;
}

Eigen::VectorXd calGradientG1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& phi_previous) {
    Eigen::VectorXd GradientG1 = Eigen::VectorXd::Zero(3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 2 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -1; offset <= 0; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                Eigen::VectorXd WeightIXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);

                // Stencil Base‚ÌŒvZ
                Eigen::Vector3d stencil_base = calculateStencilBase(cal_point);

                // Stencils—ñ‚Æstencil_num‚Ì¶¬
                Eigen::MatrixXi stencil;
                std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

                    // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_xi = HatFunction(cal_point(0) - grid_point_coordinates_xi(0));
                    double hat_y_xi = HatFunction(cal_point(1) - grid_point_coordinates_xi(1));
                    double hat_z_xi = HatFunction(cal_point(2) - grid_point_coordinates_xi(2));

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                        // “à‘}ŠÖ”‚ÌŒvZ
                        // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_i = HatFunction(cal_point(0) - grid_point_coordinates_i(0));
                        double hat_y_i = HatFunction(cal_point(1) - grid_point_coordinates_i(1));
                        double hat_z_i = HatFunction(cal_point(2) - grid_point_coordinates_i(2));

                        // Še€‚ÌŒvZ
                        double term1 = hat_x_i * hat_y_i * hat_z_i;
                        double term2 = hat_x_xi * hat_y_xi * hat_z_xi;

                        for (int p = 0; p < dimensions; p++) {
                            WeightIXi(3 * xi + p)
                                += ( phi(3 * i + p) - phi_previous(3 * i + p) ) * term1 * term2;
                        }

                        // GradientG1 += rho / (dt * dt) * WeightIXi.transpose()) ‚ÌŒvZ
                        for (int col = 0; col < dimensions; col++) { // —ñ”
                            GradientG1(3 * xi + col)
                                += rho / (dt * dt) * WeightIXi(3 * xi + col) * volume_element;
                        }

                    }
                }

            }
        }
    }

    return GradientG1;
}


Eigen::VectorXd calGradientG2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power) {


    Eigen::VectorXd GradientG2_1 = calGradientG2_1(square, re_phi, phi);

    Eigen::VectorXd GradientG2_2 = calGradientG2_2(square, re_phi, phi);

    Eigen::VectorXd GradientG2_3 = calGradientG2_3(square, re_phi, phi, power);

    // std::cout << "GradientG2_1 + GradientG2_2" << std::endl;
    // std::cout << GradientG2_1 + GradientG2_2 << std::endl;

    // std::cout << std::endl;

    // std::cout << "GradientG2_3" << std::endl;
    // std::cout << GradientG2_3 << std::endl;

    Eigen::VectorXd GradientG2 = GradientG2_1 + GradientG2_2 + GradientG2_3;

    return GradientG2;
}


Eigen::VectorXd calGradientG2_1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::VectorXd GradientG2_1 = Eigen::VectorXd::Zero(3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 2 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -1; offset <= 0; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));
                Eigen::VectorXd WeightIXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);

                // Stencil Base‚ÌŒvZ
                Eigen::Vector3d stencil_base = calculateStencilBase(cal_point);

                // Stencils—ñ‚Æstencil_num‚Ì¶¬
                Eigen::MatrixXi stencil;
                std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;

                    Eigen::Vector3i grid_xi = FlatToGrid(xi);
                    Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

                    // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_xi = HatFunction(cal_point(0) - grid_point_coordinates_xi(0));
                    double diff_hat_x_xi = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_xi(0));
                    double hat_y_xi = HatFunction(cal_point(1) - grid_point_coordinates_xi(1));
                    double diff_hat_y_xi = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_xi(1));
                    double hat_z_xi = HatFunction(cal_point(2) - grid_point_coordinates_xi(2));
                    double diff_hat_z_xi = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_xi(2));

                    // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
                    double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, -2.0 / 3.0);

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                        // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_i = HatFunction(cal_point(0) - grid_point_coordinates_i(0));
                        double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_i(0));
                        double hat_y_i = HatFunction(cal_point(1) - grid_point_coordinates_i(1));
                        double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_i(1));
                        double hat_z_i = HatFunction(cal_point(2) - grid_point_coordinates_i(2));
                        double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_i(2));

                        // Še€‚ÌŒvZ
                        double w_i_1 = diff_hat_x_i * hat_y_i * hat_z_i;
                        double w_i_2 = hat_x_i * diff_hat_y_i * hat_z_i;
                        double w_i_3 = hat_x_i * hat_y_i * diff_hat_z_i;

                        double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;
                        double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;
                        double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int p = 0; p < dimensions; p++) {
                            WeightIXi(3 * xi + p) += phi(3 * i + p) * (w_i_1 * w_xi_1 + w_i_2 * w_xi_2 + w_i_3 * w_xi_3);
                        }

                    }

                    for (int row = 0; row < dimensions; row++) { // s”
                        GradientG2_1(3 * xi + row) += - mu * volume_element * WeightIXi(3 * xi + row) * detF;
                    }

                }

            }
        }
    }

    return GradientG2_1;
}


Eigen::VectorXd calGradientG2_2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::VectorXd GradientG2_2 = Eigen::VectorXd::Zero(3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 2 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -1; offset <= 0; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::MatrixXd Phi_IJ = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_KL = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int i = 0; i < NumberOfParticles; i++) {
        for (int j = 0; j < NumberOfParticles; j++) {
            Phi_IJ(i, j)
                = phi(3 * i) * phi(3 * j) + phi(3 * i + 1) * phi(3 * j + 1) + phi(3 * i + 2) * phi(3 * j + 2);
        }
    }

    for (int k = 0; k < NumberOfParticles; k++) {
        for (int l = 0; l < NumberOfParticles; l++) {
            double Phi1 = phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1);
            double Phi2 = -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l));
            double Phi3 = phi(3 * k) * phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_KL(k, 3 * l + p) = Phi(p);
            }

        }
    }

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));
                Eigen::VectorXd WeightIJ = Eigen::VectorXd::Zero(NumberOfParticles);
                Eigen::VectorXd WeightKLXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);

                // Stencil Base‚ÌŒvZ
                Eigen::Vector3d stencil_base = calculateStencilBase(cal_point);

                // Stencils—ñ‚Æstencil_num‚Ì¶¬
                Eigen::MatrixXi stencil;
                std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

                    // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_xi = HatFunction(cal_point(0) - grid_point_coordinates_xi(0));
                    double diff_hat_x_xi = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_xi(0));
                    double hat_y_xi = HatFunction(cal_point(1) - grid_point_coordinates_xi(1));
                    double diff_hat_y_xi = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_xi(1));
                    double hat_z_xi = HatFunction(cal_point(2) - grid_point_coordinates_xi(2));
                    double diff_hat_z_xi = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_xi(2));

                    // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
                    double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, -5.0 / 3.0);

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                        // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_i = HatFunction(cal_point(0) - grid_point_coordinates_i(0));
                        double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_i(0));
                        double hat_y_i = HatFunction(cal_point(1) - grid_point_coordinates_i(1));
                        double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_i(1));
                        double hat_z_i = HatFunction(cal_point(2) - grid_point_coordinates_i(2));
                        double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_i(2));

                        double w_i_1 = diff_hat_x_i * hat_y_i * hat_z_i;
                        double w_i_2 = hat_x_i * diff_hat_y_i * hat_z_i;
                        double w_i_3 = hat_x_i * hat_y_i * diff_hat_z_i;

                        for (int j = 0; j < NumberOfParticles; j++) {
                            Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                            if (!allElementsWithinOne(j_minus_xi)) continue;

                            Eigen::Vector3d grid_point_coordinates_j = { re_phi(3 * j), re_phi(3 * j + 1), re_phi(3 * j + 2) };

                            // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_j = HatFunction(cal_point(0) - grid_point_coordinates_j(0));
                            double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_j(0));
                            double hat_y_j = HatFunction(cal_point(1) - grid_point_coordinates_j(1));
                            double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_j(1));
                            double hat_z_j = HatFunction(cal_point(2) - grid_point_coordinates_j(2));
                            double diff_hat_z_j = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_j(2));

                            double w_j_1 = diff_hat_x_j * hat_y_j * hat_z_j;
                            double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;
                            double w_j_3 = hat_x_j * hat_y_j * diff_hat_z_j;

                            WeightIJ(xi)
                                += Phi_IJ(i, j) * (w_i_1 * w_j_1 + w_i_2 * w_j_2 + w_i_3 * w_j_3);

                        }
                    }

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_k = { re_phi(3 * k), re_phi(3 * k + 1), re_phi(3 * k + 2) };

                        // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_k = HatFunction(cal_point(0) - grid_point_coordinates_k(0));
                        double diff_hat_x_k = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_k(0));
                        double hat_y_k = HatFunction(cal_point(1) - grid_point_coordinates_k(1));
                        double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_k(1));
                        double hat_z_k = HatFunction(cal_point(2) - grid_point_coordinates_k(2));

                        for (int l = 0; l < NumberOfParticles; l++) {
                            Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                            if (!allElementsWithinOne(l_minus_xi)) continue;

                            Eigen::Vector3d grid_point_coordinates_l = { re_phi(3 * l), re_phi(3 * l + 1), re_phi(3 * l + 2) };

                            // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_l = HatFunction(cal_point(0) - grid_point_coordinates_l(0));
                            double hat_y_l = HatFunction(cal_point(1) - grid_point_coordinates_l(1));
                            double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_l(1));
                            double hat_z_l = HatFunction(cal_point(2) - grid_point_coordinates_l(2));
                            double diff_hat_z_l = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_l(2));

                            // Še€‚ÌŒvZ
                            double w_k_2 = hat_x_k * diff_hat_y_k * hat_z_k;
                            double w_l_3 = hat_x_l * hat_y_l * diff_hat_z_l;
                            double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                            double w_k_1 = diff_hat_x_k * hat_y_k * hat_z_k;
                            double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                            double w_l_2 = hat_x_l * diff_hat_y_l * hat_z_l;
                            double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                            for (int p = 0; p < dimensions; p++) {
                                WeightKLXi(3 * xi + p)
                                    += Phi_KL(k, 3 * l + p)
                                    * (w_k_2 * w_l_3 * w_xi_1 - w_k_1 * w_l_3 * w_xi_2 + w_k_1 * w_l_2 * w_xi_3);
                            }

                        }
                    }

                    // GradientG2_2 += 1/3 * mu * J^(-5.0/3.0)) * WeightIJ * WeightKLXi.transpose() ‚ÌŒvZ
                    for (int col = 0; col < dimensions; col++) { // —ñ”
                        GradientG2_2(3 * xi + col) += (1.0 / 3.0) * mu * detF * WeightIJ(xi) * WeightKLXi(3 * xi + col) * volume_element;
                    }

                }

            }
        }
    }

    return GradientG2_2;
}


Eigen::VectorXd calGradientG2_3(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power) {
    Eigen::VectorXd GradientG2_3 = Eigen::VectorXd::Zero(3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 2 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -1; offset <= 0; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::MatrixXd Phi_IJ = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int i = 0; i < NumberOfParticles; i++) {
        for (int j = 0; j < NumberOfParticles; j++) {
            double Phi1 = phi(3 * i + 1) * phi(3 * j + 2) - phi(3 * i + 2) * phi(3 * j + 1);
            double Phi2 = -(phi(3 * i) * phi(3 * j + 2) - phi(3 * i + 2) * phi(3 * j));
            double Phi3 = phi(3 * i) * phi(3 * j + 1) - phi(3 * i + 1) * phi(3 * j);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_IJ(i, 3 * j + p) = Phi(p);
            }
        }
    }

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));
                Eigen::VectorXd WeightIJXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);

                // Stencil Base‚ÌŒvZ
                Eigen::Vector3d stencil_base = calculateStencilBase(cal_point);

                // Stencils—ñ‚Æstencil_num‚Ì¶¬
                Eigen::MatrixXi stencil;
                std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

                    // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_xi = HatFunction(cal_point(0) - grid_point_coordinates_xi(0));
                    double diff_hat_x_xi = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_xi(0));
                    double hat_y_xi = HatFunction(cal_point(1) - grid_point_coordinates_xi(1));
                    double diff_hat_y_xi = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_xi(1));
                    double hat_z_xi = HatFunction(cal_point(2) - grid_point_coordinates_xi(2));
                    double diff_hat_z_xi = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_xi(2));

                    double WeightK = 0.0;

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                        // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_i = HatFunction(cal_point(0) - grid_point_coordinates_i(0));
                        double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_i(0));
                        double hat_y_i = HatFunction(cal_point(1) - grid_point_coordinates_i(1));
                        double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_i(1));
                        double hat_z_i = HatFunction(cal_point(2) - grid_point_coordinates_i(2));

                        for (int j = 0; j < NumberOfParticles; j++) {
                            Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                            if (!allElementsWithinOne(j_minus_xi)) continue;

                            Eigen::Vector3d grid_point_coordinates_j = { re_phi(3 * j), re_phi(3 * j + 1), re_phi(3 * j + 2) };

                            // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_j = HatFunction(cal_point(0) - grid_point_coordinates_j(0));
                            double hat_y_j = HatFunction(cal_point(1) - grid_point_coordinates_j(1));
                            double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_j(1));
                            double hat_z_j = HatFunction(cal_point(2) - grid_point_coordinates_j(2));
                            double diff_hat_z_j = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_j(2));

                            // Še€‚ÌŒvZ
                            double w_i_2 = hat_x_i * diff_hat_y_i * hat_z_i;
                            double w_j_3 = hat_x_j * hat_y_j * diff_hat_z_j;
                            double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                            double w_i_1 = diff_hat_x_i * hat_y_i * hat_z_i;
                            double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                            double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;
                            double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                            for (int p = 0; p < dimensions; p++) {
                                WeightIJXi(3 * xi + p)
                                    += Phi_IJ(i, 3 * j + p)
                                    * (w_i_2 * w_j_3 * w_xi_1 - w_i_1 * w_j_3 * w_xi_2 + w_i_1 * w_j_2 * w_xi_3);
                            }

                        }
                    }

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_k = { re_phi(3 * k), re_phi(3 * k + 1), re_phi(3 * k + 2) };

                        // “à‘}ŠÖ”‚ÌŒvZ
                        // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_k = HatFunction(cal_point(0) - grid_point_coordinates_k(0));
                        double hat_y_k = HatFunction(cal_point(1) - grid_point_coordinates_k(1));
                        double hat_z_k = HatFunction(cal_point(2) - grid_point_coordinates_k(2));

                        // Še€‚ÌŒvZ
                        double w_k = hat_x_k * hat_y_k * hat_z_k;

                        WeightK += power(k) * w_k;

                    }

                    // GradientG2_3 += WeightIJXi.transpose() * WeightK ‚ÌŒvZ
                    for (int col = 0; col < dimensions; col++) { // —ñ”
                        GradientG2_3(3 * xi + col) += WeightIJXi(3 * xi + col) * WeightK * volume_element;
                    }

                }

            }
        }
    }

    return GradientG2_3;
}
