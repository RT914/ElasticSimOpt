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
    // exportMatrix_CSV(HessianXi1, "csv/HessianXi1.csv");

    Eigen::MatrixXd HessianXi2 = calHessianXi2(square, re_phi, phi);
    if (HessianXi2.array().isNaN().any()) {
        std::cerr << "NaN detected HessianXi2" << std::endl;
    }

    Eigen::MatrixXd HessianXi3 = calHessianXi3(square, re_phi, phi);
    if (HessianXi3.array().isNaN().any()) {
        std::cerr << "NaN detected HessianXi3" << std::endl;
    }

    HessianXi = HessianXi1 + HessianXi2 + HessianXi3;

    /*std::cout << "HessianXi2 + HessianXi3" << std::endl;
    for (int i = 0; i < 3 * NumberOfParticles; i++) {
        for (int j = 0; j < 3 * NumberOfParticles; j++) {
            if(abs(HessianXi2(i,j)+HessianXi3(i,j))>1e-10)
                std::cout << HessianXi2(i, j) + HessianXi3(i, j) << std::endl;
        }
    }*/

    return HessianXi;
}


Eigen::MatrixXd calHessianXi1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power)
{
    Eigen::MatrixXd HessianXi1 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 2 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const int AllkNum = pow(kNum, 3);// ‘SŸŒ³‚Ì‘S‹æŠÔ•ªŠ„”
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
    Eigen::MatrixXd Phi_NP = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_LO = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
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

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;
        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

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

            double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;
            double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;
            double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

            // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, -1.0);

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

                double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;
                double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;
                double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
                double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, -1.0);

                for (int tau = 0; tau < NumberOfParticles; tau++) {
                    Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                    if (!allElementsWithinOne(tau_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                    // tauŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_tau = HatFunction(cal_point(0) - grid_point_coordinates_tau(0));
                    double diff_hat_x_tau = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_tau(0));
                    double hat_y_tau = HatFunction(cal_point(1) - grid_point_coordinates_tau(1));
                    double diff_hat_y_tau = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_tau(1));
                    double hat_z_tau = HatFunction(cal_point(2) - grid_point_coordinates_tau(2));
                    double diff_hat_z_tau = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_tau(2));

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

                        // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_l = HatFunction(cal_point(0) - grid_point_coordinates_l(0));
                        double diff_hat_x_l = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_l(0));
                        double hat_y_l = HatFunction(cal_point(1) - grid_point_coordinates_l(1));
                        double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_l(1));
                        double hat_z_l = HatFunction(cal_point(2) - grid_point_coordinates_l(2));

                        double w_l_1 = diff_hat_x_l * hat_y_l * hat_z_l;
                        double w_l_2 = hat_x_l * diff_hat_y_l * hat_z_l;

                        for (int o = 0; o < NumberOfParticles; o++) {
                            Eigen::Vector3i o_minus_xi = FlatToGrid(o) - grid_xi;
                            if (!allElementsWithinOne(o_minus_xi)) continue;

                            Eigen::Vector3d grid_point_coordinates_o = { re_phi(3 * o), re_phi(3 * o + 1), re_phi(3 * o + 2) };

                            // oŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_o = HatFunction(cal_point(0) - grid_point_coordinates_o(0));
                            double hat_y_o = HatFunction(cal_point(1) - grid_point_coordinates_o(1));
                            double diff_hat_y_o = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_o(1));
                            double hat_z_o = HatFunction(cal_point(2) - grid_point_coordinates_o(2));
                            double diff_hat_z_o = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_o(2));

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

                        // nŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_n = HatFunction(cal_point(0) - grid_point_coordinates_n(0));
                        double diff_hat_x_n = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_n(0));
                        double hat_y_n = HatFunction(cal_point(1) - grid_point_coordinates_n(1));
                        double diff_hat_y_n = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_n(1));
                        double hat_z_n = HatFunction(cal_point(2) - grid_point_coordinates_n(2));

                        double w_n_1 = diff_hat_x_n * hat_y_n * hat_z_n;
                        double w_n_2 = hat_x_n * diff_hat_y_n * hat_z_n;

                        for (int p = 0; p < NumberOfParticles; p++) {
                            Eigen::Vector3i p_minus_xi = FlatToGrid(p) - grid_xi;
                            if (!allElementsWithinOne(p_minus_xi)) continue;

                            Eigen::Vector3d grid_point_coordinates_p = { re_phi(3 * p), re_phi(3 * p + 1), re_phi(3 * p + 2) };

                            // pŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_p = HatFunction(cal_point(0) - grid_point_coordinates_p(0));
                            double hat_y_p = HatFunction(cal_point(1) - grid_point_coordinates_p(1));
                            double diff_hat_y_p = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_p(1));
                            double hat_z_p = HatFunction(cal_point(2) - grid_point_coordinates_p(2));
                            double diff_hat_z_p = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_p(2));

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

                        // “à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_i = HatFunction(cal_point(0) - grid_point_coordinates_i(0));
                        double hat_y_i = HatFunction(cal_point(1) - grid_point_coordinates_i(1));
                        double hat_z_i = HatFunction(cal_point(2) - grid_point_coordinates_i(2));
                        WeightI += power(i) * hat_x_i * hat_y_i * hat_z_i;
                    }

                    // HessianXi1‚ÌŒvZ
                    for (int col = 0; col < dimensions; col++) { // —ñ”
                        for (int row = 0; row < dimensions; row++) { // s”
                            double term = detF * WeightNPXi(row) * WeightLOTau(col) * WeightI * volume_element;
                            if (abs(term) < 1e-10) continue;
                            HessianXi1(3 * xi + row, 3 * tau + col) += term;
                        }
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

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 2 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const int AllkNum = pow(kNum, 3);// ‘SŸŒ³‚Ì‘S‹æŠÔ•ªŠ„”
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
    Eigen::MatrixXd Phi_LN = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
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


    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;
        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));


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

            for (int tau = 0; tau < NumberOfParticles; tau++) {
                Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                if (!allElementsWithinOne(tau_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                // tauŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                double hat_x_tau = HatFunction(cal_point(0) - grid_point_coordinates_tau(0));
                double diff_hat_x_tau = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_tau(0));
                double hat_y_tau = HatFunction(cal_point(1) - grid_point_coordinates_tau(1));
                double diff_hat_y_tau = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_tau(1));
                double hat_z_tau = HatFunction(cal_point(2) - grid_point_coordinates_tau(2));
                double diff_hat_z_tau = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_tau(2));

                Eigen::Vector3d WeightLNTau = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightJXi = Eigen::Vector3d::Zero();

                for (int l = 0; l < NumberOfParticles; l++) {
                    Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                    if (!allElementsWithinOne(l_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_l = { re_phi(3 * l), re_phi(3 * l + 1), re_phi(3 * l + 2) };

                    // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_l = HatFunction(cal_point(0) - grid_point_coordinates_l(0));
                    double diff_hat_x_l = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_l(0));
                    double hat_y_l = HatFunction(cal_point(1) - grid_point_coordinates_l(1));
                    double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_l(1));
                    double hat_z_l = HatFunction(cal_point(2) - grid_point_coordinates_l(2));

                    for (int n = 0; n < NumberOfParticles; n++) {
                        Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                        if (!allElementsWithinOne(n_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_n = { re_phi(3 * n), re_phi(3 * n + 1), re_phi(3 * n + 2) };

                        // nŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_n = HatFunction(cal_point(0) - grid_point_coordinates_n(0));
                        double hat_y_n = HatFunction(cal_point(1) - grid_point_coordinates_n(1));
                        double diff_hat_y_n = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_n(1));
                        double hat_z_n = HatFunction(cal_point(2) - grid_point_coordinates_n(2));
                        double diff_hat_z_n = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_n(2));

                        // Še€‚ÌŒvZ
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

                    // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_j = HatFunction(cal_point(0) - grid_point_coordinates_j(0));
                    double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_j(0));
                    double hat_y_j = HatFunction(cal_point(1) - grid_point_coordinates_j(1));
                    double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_j(1));
                    double hat_z_j = HatFunction(cal_point(2) - grid_point_coordinates_j(2));
                    double diff_hat_z_j = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_j(2));

                    // Še€‚ÌŒvZ
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

                for (int col = 0; col < dimensions; col++) { // —ñ”
                    for (int row = 0; row < dimensions; row++) { // s”
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

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 2 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const int AllkNum = pow(kNum, 3);// ‘SŸŒ³‚Ì‘S‹æŠÔ•ªŠ„”
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
    Eigen::MatrixXd Phi_MO = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_LN = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
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

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;
        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        // std::cout << "--------------------------------------------------------" << std::endl;


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
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, -8.0/3.0);
           

            for (int tau = 0; tau < NumberOfParticles; tau++) {
                Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                if (!allElementsWithinOne(tau_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                // tauŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                double hat_x_tau = HatFunction(cal_point(0) - grid_point_coordinates_tau(0));
                double diff_hat_x_tau = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_tau(0));
                double hat_y_tau = HatFunction(cal_point(1) - grid_point_coordinates_tau(1));
                double diff_hat_y_tau = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_tau(1));
                double hat_z_tau = HatFunction(cal_point(2) - grid_point_coordinates_tau(2));
                double diff_hat_z_tau = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_tau(2));

                Eigen::Vector3d WeightLNTau = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightMOXi = Eigen::Vector3d::Zero();
                double WeightJK = 0.0;

                for (int l = 0; l < NumberOfParticles; l++) {
                    Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                    if (!allElementsWithinOne(l_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_l = { re_phi(3 * l), re_phi(3 * l + 1), re_phi(3 * l + 2) };

                    // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_l = HatFunction(cal_point(0) - grid_point_coordinates_l(0));
                    double diff_hat_x_l = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_l(0));
                    double hat_y_l = HatFunction(cal_point(1) - grid_point_coordinates_l(1));
                    double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_l(1));
                    double hat_z_l = HatFunction(cal_point(2) - grid_point_coordinates_l(2));

                    for (int n = 0; n < NumberOfParticles; n++) {
                        Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                        if (!allElementsWithinOne(n_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_n = { re_phi(3 * n), re_phi(3 * n + 1), re_phi(3 * n + 2) };

                        // nŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_n = HatFunction(cal_point(0) - grid_point_coordinates_n(0));
                        double hat_y_n = HatFunction(cal_point(1) - grid_point_coordinates_n(1));
                        double diff_hat_y_n = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_n(1));
                        double hat_z_n = HatFunction(cal_point(2) - grid_point_coordinates_n(2));
                        double diff_hat_z_n = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_n(2));

                        // Še€‚ÌŒvZ
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

                    // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_j(0));
                    double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_j(1));
                    double diff_hat_z_j = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_j(2));
                    double hat_x_j = HatFunction(cal_point(0) - grid_point_coordinates_j(0));
                    double hat_y_j = HatFunction(cal_point(1) - grid_point_coordinates_j(1));
                    double hat_z_j = HatFunction(cal_point(2) - grid_point_coordinates_j(2));

                    double w_j_1 = diff_hat_x_j * hat_y_j * hat_z_j;
                    double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;
                    double w_j_3 = hat_x_j * hat_y_j * diff_hat_z_j;

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_k = { re_phi(3 * k), re_phi(3 * k + 1), re_phi(3 * k + 2) };

                        // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double diff_hat_x_k = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_k(0));
                        double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_k(1));
                        double diff_hat_z_k = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_k(2));
                        double hat_x_k = HatFunction(cal_point(0) - grid_point_coordinates_k(0));
                        double hat_y_k = HatFunction(cal_point(1) - grid_point_coordinates_k(1));
                        double hat_z_k = HatFunction(cal_point(2) - grid_point_coordinates_k(2));

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

                    // mŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_m = HatFunction(cal_point(0) - grid_point_coordinates_m(0));
                    double diff_hat_x_m = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_m(0));
                    double hat_y_m = HatFunction(cal_point(1) - grid_point_coordinates_m(1));
                    double diff_hat_y_m = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_m(1));
                    double hat_z_m = HatFunction(cal_point(2) - grid_point_coordinates_m(2));

                    for (int o = 0; o < NumberOfParticles; o++) {
                        Eigen::Vector3i o_minus_xi = FlatToGrid(o) - grid_xi;
                        if (!allElementsWithinOne(o_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_o = { re_phi(3 * o), re_phi(3 * o + 1), re_phi(3 * o + 2) };

                        // oŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_o = HatFunction(cal_point(0) - grid_point_coordinates_o(0));
                        double hat_y_o = HatFunction(cal_point(1) - grid_point_coordinates_o(1));
                        double diff_hat_y_o = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_o(1));
                        double hat_z_o = HatFunction(cal_point(2) - grid_point_coordinates_o(2));
                        double diff_hat_z_o = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_o(2));

                        // Še€‚ÌŒvZ
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



                for (int col = 0; col < dimensions; col++) { // —ñ”
                    for (int row = 0; row < dimensions; row++) { // s”
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
