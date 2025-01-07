#include <iostream>
#include <omp.h>
#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Hessian/HessianUpsilon.h"


// Caluculate HessianUpsilon
Eigen::MatrixXd calHessianUpsilon(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power) {
    Eigen::MatrixXd HessianUpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianUpsilon.setZero();
    Eigen::MatrixXd HessianUpsilon1 = calHessianUpsilon1(square, re_phi, phi);
    Eigen::MatrixXd HessianUpsilon2 = calHessianUpsilon2(square, re_phi, phi);
    Eigen::MatrixXd HessianUpsilon3 = calHessianUpsilon3(square, re_phi, phi, power);

    HessianUpsilon = HessianUpsilon1 + HessianUpsilon2 + HessianUpsilon3;

    return HessianUpsilon;
}

Eigen::MatrixXd calHessianUpsilon1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianUpsilon1 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = square.SideNumber * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const int AllkNum = pow(kNum, 3);// ‘SŸŒ³‚Ì‘S‹æŠÔ•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = 0; offset < square.SideNumber; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = (static_cast<double>(offset) + 1.0 / (2.0 * kNumSection)) * square.dx + divIndex * kWidth - 1.0;
            index++;
        }
    }


    // ŒW”‚Ì‰Šú‰»
    Eigen::MatrixXd Phi_MN = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int m = 0; m < NumberOfParticles; m++) {
        for (int n = 0; n < NumberOfParticles; n++) {
            double Phi1 = phi(3 * m + 1) * phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n + 1);
            double Phi2 = -(phi(3 * m) * phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n));
            double Phi3 = phi(3 * m) * phi(3 * n + 1) - phi(3 * m + 1) * phi(3 * n);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_MN(m, 3 * n + a) = Phi(a);
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
        Eigen::Vector3d stencil_base = calculateStencilBase(cal_point, square.dx);

        // Stencils—ñ‚Æstencil_num‚Ì¶¬
        Eigen::MatrixXi stencil;
        std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

        for (int xi = 0; xi < NumberOfParticles; xi++) {
            if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
            Eigen::Vector3i grid_xi = FlatToGrid(xi);
            Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
            double hat_x_xi = HatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx);
            double diff_hat_x_xi = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx) / square.dx;
            double hat_y_xi = HatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx);
            double diff_hat_y_xi = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx) / square.dx;
            double hat_z_xi = HatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx);
            double diff_hat_z_xi = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx) / square.dx;

            // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, square.dx, -5.0/3.0);

            for (int tau = 0; tau < NumberOfParticles; tau++) {
                Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                if (!allElementsWithinOne(tau_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                // tauŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
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
                Eigen::Vector3d WeightMNXi = Eigen::Vector3d::Zero();

                for (int j = 0; j < NumberOfParticles; j++) {
                    Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                    if (!allElementsWithinOne(j_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_j = { re_phi(3 * j), re_phi(3 * j + 1), re_phi(3 * j + 2) };

                    // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
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

                for (int m = 0; m < NumberOfParticles; m++) {
                    Eigen::Vector3i m_minus_xi = FlatToGrid(m) - grid_xi;
                    if (!allElementsWithinOne(m_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_m = { re_phi(3 * m), re_phi(3 * m + 1), re_phi(3 * m + 2) };

                    // mŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_m = HatFunction((cal_point(0) - grid_point_coordinates_m(0)) / square.dx);
                    double diff_hat_x_m = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_m(0)) / square.dx) / square.dx;
                    double hat_y_m = HatFunction((cal_point(1) - grid_point_coordinates_m(1)) / square.dx);
                    double diff_hat_y_m = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_m(1)) / square.dx) / square.dx;
                    double hat_z_m = HatFunction((cal_point(2) - grid_point_coordinates_m(2)) / square.dx);

                    for (int n = 0; n < NumberOfParticles; n++) {
                        Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                        if (!allElementsWithinOne(n_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_n = { re_phi(3 * n), re_phi(3 * n + 1), re_phi(3 * n + 2) };

                        // nŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_n = HatFunction((cal_point(0) - grid_point_coordinates_n(0)) / square.dx);
                        double hat_y_n = HatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx);
                        double diff_hat_y_n = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx) / square.dx;
                        double hat_z_n = HatFunction((cal_point(2) - grid_point_coordinates_n(2)) / square.dx);
                        double diff_hat_z_n = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_n(2)) / square.dx) / square.dx;

                        // Še€‚ÌŒvZ
                        double w_m_2 = hat_x_m * diff_hat_y_m * hat_z_m;
                        double w_n_3 = hat_x_n * hat_y_n * diff_hat_z_n;
                        double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                        double w_m_1 = diff_hat_x_m * hat_y_m * hat_z_m;
                        double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                        double w_n_2 = hat_x_n * diff_hat_y_n * hat_z_n;
                        double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int a = 0; a < dimensions; a++) {
                            WeightMNXi(a) += Phi_MN(m, 3 * n + a) *
                                (w_m_2 * w_n_3 * w_xi_1 - w_m_1 * w_n_3 * w_xi_2 + w_m_1 * w_n_2 * w_xi_3);
                        }

                    }
                }


                // HessianUpsilon1 += mu * pow(J, (-5/3)) * WeightIJ * WeightKLXi.transpose() ‚ÌŒvZ
                for (int col = 0; col < dimensions; col++) { // —ñ”
                    for (int row = 0; row < dimensions; row++) { // s”
                        double term = mu * detF * WeightJTau(row) * WeightMNXi(col) * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianUpsilon1(3 * xi + row, 3 * tau + col) += term;
                    }
                }

            }
        }


    }

    return HessianUpsilon1;
}

Eigen::MatrixXd calHessianUpsilon2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd HessianUpsilon2 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = square.SideNumber * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const int AllkNum = pow(kNum, 3);// ‘SŸŒ³‚Ì‘S‹æŠÔ•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = 0; offset < square.SideNumber; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = (static_cast<double>(offset) + 1.0 / (2.0 * kNumSection)) * square.dx + divIndex * kWidth - 1.0;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::MatrixXd Phi_LN = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_MO = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int l = 0; l < NumberOfParticles; l++) {
        for (int n = 0; n < NumberOfParticles; n++) {
            double Phi1 = phi(3 * l + 1) * phi(3 * n + 2) - phi(3 * l + 2) * phi(3 * n + 1);
            double Phi2 = -(phi(3 * l) * phi(3 * n + 2) - phi(3 * l + 2) * phi(3 * n));
            double Phi3 = phi(3 * l) * phi(3 * n + 1) - phi(3 * l + 1) * phi(3 * n);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_LN(3 * l + a, n) = Phi(a);
            }
        }
    }

    for (int m = 0; m < NumberOfParticles; m++) {
        for (int o = 0; o < NumberOfParticles; o++) {
            double Phi1 = phi(3 * m + 1) * phi(3 * o + 2) - phi(3 * m + 2) * phi(3 * o + 1);
            double Phi2 = -(phi(3 * m) * phi(3 * o + 2) - phi(3 * m + 2) * phi(3 * o));
            double Phi3 = phi(3 * m) * phi(3 * o + 1) - phi(3 * m + 1) * phi(3 * o);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_MO(m, 3 * o + a) = Phi(a);
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

        // Stencil Base‚ÌŒvZ
        Eigen::Vector3d stencil_base = calculateStencilBase(cal_point, square.dx);

        // Stencils—ñ‚Æstencil_num‚Ì¶¬
        Eigen::MatrixXi stencil;
        std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

        for (int xi = 0; xi < NumberOfParticles; xi++) {
            if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
            Eigen::Vector3i grid_xi = FlatToGrid(xi);

            Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
            double hat_x_xi = HatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx);
            double diff_hat_x_xi = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx) / square.dx;
            double hat_y_xi = HatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx);
            double diff_hat_y_xi = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx) / square.dx;
            double hat_z_xi = HatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx);
            double diff_hat_z_xi = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx) / square.dx;

            // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, square.dx, -8.0 / 3.0);

            for (int tau = 0; tau < NumberOfParticles; tau++) {
                Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                if (!allElementsWithinOne(tau_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                // tauŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                double diff_hat_x_tau = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx) / square.dx;
                double diff_hat_y_tau = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx) / square.dx;
                double diff_hat_z_tau = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx) / square.dx;
                double hat_x_tau = HatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx);
                double hat_y_tau = HatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx);
                double hat_z_tau = HatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx);

                Eigen::Vector3d WeightLNTau = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightMOXi = Eigen::Vector3d::Zero();
                double WeightJK = 0.0;

                for (int l = 0; l < NumberOfParticles; l++) {
                    Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                    if (!allElementsWithinOne(l_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_l = { re_phi(3 * l), re_phi(3 * l + 1), re_phi(3 * l + 2) };

                    // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double diff_hat_x_l = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx) / square.dx;
                    double diff_hat_y_l = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx) / square.dx;
                    double hat_x_l = HatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx);
                    double hat_y_l = HatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx);
                    double hat_z_l = HatFunction((cal_point(2) - grid_point_coordinates_l(2)) / square.dx);

                    for (int n = 0; n < NumberOfParticles; n++) {
                        Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                        if (!allElementsWithinOne(n_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_n = { re_phi(3 * n), re_phi(3 * n + 1), re_phi(3 * n + 2) };

                        // nŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double diff_hat_y_n = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx) / square.dx;
                        double diff_hat_z_n = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_n(2)) / square.dx) / square.dx;
                        double hat_x_n = HatFunction((cal_point(0) - grid_point_coordinates_n(0)) / square.dx);
                        double hat_y_n = HatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx);
                        double hat_z_n = HatFunction((cal_point(2) - grid_point_coordinates_n(2)) / square.dx);

                        // Še€‚ÌŒvZ
                        double w_l_2 = hat_x_l * diff_hat_y_l * hat_z_l;
                        double w_n_3 = hat_x_n * hat_y_n * diff_hat_z_n;
                        double w_tau_1 = diff_hat_x_tau * hat_y_tau * hat_z_tau;

                        double w_l_1 = diff_hat_x_l * hat_y_l * hat_z_l;
                        double w_tau_2 = hat_x_tau * diff_hat_y_tau * hat_z_tau;

                        double w_n_2 = hat_x_n * diff_hat_y_n * hat_z_n;
                        double w_tau_3 = hat_x_tau * hat_y_tau * diff_hat_z_tau;

                        for (int a = 0; a < dimensions; a++) {
                            WeightLNTau(a) += Phi_LN(3 * l + a, n) *
                                (w_l_2 * w_n_3 * w_tau_1 - w_l_1 * w_n_3 * w_tau_2 + w_l_1 * w_n_2 * w_tau_3);
                        }

                    }
                }

                for (int m = 0; m < NumberOfParticles; m++) {
                    Eigen::Vector3i m_minus_xi = FlatToGrid(m) - grid_xi;
                    if (!allElementsWithinOne(m_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_m = { re_phi(3 * m), re_phi(3 * m + 1), re_phi(3 * m + 2) };

                    // mŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_m = HatFunction((cal_point(0) - grid_point_coordinates_m(0)) / square.dx);
                    double diff_hat_x_m = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_m(0)) / square.dx) / square.dx;
                    double hat_y_m = HatFunction((cal_point(1) - grid_point_coordinates_m(1)) / square.dx);
                    double diff_hat_y_m = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_m(1)) / square.dx) / square.dx;
                    double hat_z_m = HatFunction((cal_point(2) - grid_point_coordinates_m(2)) / square.dx);

                    for (int o = 0; o < NumberOfParticles; o++) {
                        Eigen::Vector3i o_minus_xi = FlatToGrid(o) - grid_xi;
                        if (!allElementsWithinOne(o_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_o = { re_phi(3 * o), re_phi(3 * o + 1), re_phi(3 * o + 2) };

                        // oŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_o = HatFunction((cal_point(0) - grid_point_coordinates_o(0)) / square.dx);
                        double hat_y_o = HatFunction((cal_point(1) - grid_point_coordinates_o(1)) / square.dx);
                        double diff_hat_y_o = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_o(1)) / square.dx) / square.dx;
                        double hat_z_o = HatFunction((cal_point(2) - grid_point_coordinates_o(2)) / square.dx);
                        double diff_hat_z_o = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_o(2)) / square.dx) / square.dx;

                        // Še€‚ÌŒvZ
                        double w_m_2 = hat_x_m * diff_hat_y_m * hat_z_m;
                        double w_o_3 = hat_x_o * hat_y_o * diff_hat_z_o;
                        double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                        double w_m_1 = diff_hat_x_m * hat_y_m * hat_z_m;
                        double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                        double w_o_2 = hat_x_o * diff_hat_y_o * hat_z_o;
                        double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int a = 0; a < dimensions; a++) {
                            WeightMOXi(a) += Phi_MO(m, 3 * o + a)
                                * (w_m_2 * w_o_3 * w_xi_1 - w_m_1 * w_o_3 * w_xi_2 + w_m_1 * w_o_2 * w_xi_3);
                        }

                    }
                }

                for (int j = 0; j < NumberOfParticles; j++) {
                    Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                    if (!allElementsWithinOne(j_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_j = { re_phi(3 * j), re_phi(3 * j + 1), re_phi(3 * j + 2) };

                    // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
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

                // HessianUpsilon2 += detF * WeightIJ * WeightKLXi.transpose() ‚ÌŒvZ
                for (int col = 0; col < dimensions; col++) { // —ñ”
                    for (int row = 0; row < dimensions; row++) { // s”
                        double term = -(1.0 / 3.0) * mu * detF * WeightLNTau(row) * WeightMOXi(col) *  WeightJK * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianUpsilon2(3 * xi + row, 3 * tau + col) += term;
                    }
                }

            }
        }

    }

    return HessianUpsilon2;
}

Eigen::MatrixXd calHessianUpsilon3(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power)
{
    Eigen::MatrixXd HessianUpsilon3 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = square.SideNumber * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const int AllkNum = pow(kNum, 3);// ‘SŸŒ³‚Ì‘S‹æŠÔ•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = 0; offset < square.SideNumber; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = (static_cast<double>(offset) + 1.0 / (2.0 * kNumSection)) * square.dx + divIndex * kWidth - 1.0;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::MatrixXd Phi_LN = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_MO = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int l = 0; l < NumberOfParticles; l++) {
        for (int n = 0; n < NumberOfParticles; n++) {
            double Phi1 = phi(3 * l + 1) * phi(3 * n + 2) - phi(3 * l + 2) * phi(3 * n + 1);
            double Phi2 = -(phi(3 * l) * phi(3 * n + 2) - phi(3 * l + 2) * phi(3 * n));
            double Phi3 = phi(3 * l) * phi(3 * n + 1) - phi(3 * l + 1) * phi(3 * n);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_LN(3 * l + a, n) = Phi(a);
            }
        }
    }

    for (int m = 0; m < NumberOfParticles; m++) {
        for (int o = 0; o < NumberOfParticles; o++) {
            double Phi1 = phi(3 * m + 1) * phi(3 * o + 2) - phi(3 * m + 2) * phi(3 * o + 1);
            double Phi2 = -(phi(3 * m) * phi(3 * o + 2) - phi(3 * m + 2) * phi(3 * o));
            double Phi3 = phi(3 * m) * phi(3 * o + 1) - phi(3 * m + 1) * phi(3 * o);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_MO(m, 3 * o + a) = Phi(a);
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
        Eigen::Vector3d stencil_base = calculateStencilBase(cal_point, square.dx);

        // Stencils—ñ‚Æstencil_num‚Ì¶¬
        Eigen::MatrixXi stencil;
        std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

        for (int xi = 0; xi < NumberOfParticles; xi++) {
            if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
            Eigen::Vector3i grid_xi = FlatToGrid(xi);

            Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
            double hat_x_xi = HatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx);
            double diff_hat_x_xi = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx) / square.dx;
            double hat_y_xi = HatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx);
            double diff_hat_y_xi = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx) / square.dx;
            double hat_z_xi = HatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx);
            double diff_hat_z_xi = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx) / square.dx;

            // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, square.dx, -1.0);

            for (int tau = 0; tau < NumberOfParticles; tau++) {
                Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                if (!allElementsWithinOne(tau_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                // tauŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                double hat_x_tau = HatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx);
                double diff_hat_x_tau = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx) / square.dx;
                double hat_y_tau = HatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx);
                double diff_hat_y_tau = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx) / square.dx;
                double hat_z_tau = HatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx);
                double diff_hat_z_tau = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx) / square.dx;

                Eigen::Vector3d WeightLNTau = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightMOXi = Eigen::Vector3d::Zero();
                double WeightI = 0.0;

                for (int l = 0; l < NumberOfParticles; l++) {
                    Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                    if (!allElementsWithinOne(l_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_l = { re_phi(3 * l), re_phi(3 * l + 1), re_phi(3 * l + 2) };

                    // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_l = HatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx);
                    double diff_hat_x_l = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_l(0)) / square.dx) / square.dx;
                    double hat_y_l = HatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx);
                    double diff_hat_y_l = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_l(1)) / square.dx) / square.dx;
                    double hat_z_l = HatFunction((cal_point(2) - grid_point_coordinates_l(2)) / square.dx);

                    for (int n = 0; n < NumberOfParticles; n++) {
                        Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                        if (!allElementsWithinOne(n_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_n = { re_phi(3 * n), re_phi(3 * n + 1), re_phi(3 * n + 2) };

                        // nŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_n = HatFunction((cal_point(0) - grid_point_coordinates_n(0)) / square.dx);
                        double hat_y_n = HatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx);
                        double diff_hat_y_n = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_n(1)) / square.dx) / square.dx;
                        double hat_z_n = HatFunction((cal_point(2) - grid_point_coordinates_n(2)) / square.dx);
                        double diff_hat_z_n = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_n(2)) / square.dx) / square.dx;

                        // Še€‚ÌŒvZ
                        double w_l_2 = hat_x_l * diff_hat_y_l * hat_z_l;
                        double w_n_3 = hat_x_n * hat_y_n * diff_hat_z_n;
                        double w_tau_1 = diff_hat_x_tau * hat_y_tau * hat_z_tau;

                        double w_l_1 = diff_hat_x_l * hat_y_l * hat_z_l;
                        double w_tau_2 = hat_x_tau * diff_hat_y_tau * hat_z_tau;

                        double w_n_2 = hat_x_n * diff_hat_y_n * hat_z_n;
                        double w_tau_3 = hat_x_tau * hat_y_tau * diff_hat_z_tau;

                        for (int a = 0; a < dimensions; a++) {
                            WeightLNTau(a) += Phi_LN(3 * l + a, n) *
                                (w_l_2 * w_n_3 * w_tau_1 - w_l_1 * w_n_3 * w_tau_2 + w_l_1 * w_n_2 * w_tau_3);
                        }

                    }
                }


                for (int m = 0; m < NumberOfParticles; m++) {
                    Eigen::Vector3i m_minus_xi = FlatToGrid(m) - grid_xi;
                    if (!allElementsWithinOne(m_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_m = { re_phi(3 * m), re_phi(3 * m + 1), re_phi(3 * m + 2) };

                    // mŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_m = HatFunction((cal_point(0) - grid_point_coordinates_m(0)) / square.dx);
                    double diff_hat_x_m = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_m(0)) / square.dx) / square.dx;
                    double hat_y_m = HatFunction((cal_point(1) - grid_point_coordinates_m(1)) / square.dx);
                    double diff_hat_y_m = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_m(1)) / square.dx) / square.dx;
                    double hat_z_m = HatFunction((cal_point(2) - grid_point_coordinates_m(2)) / square.dx);

                    for (int o = 0; o < NumberOfParticles; o++) {
                        Eigen::Vector3i o_minus_xi = FlatToGrid(o) - grid_xi;
                        if (!allElementsWithinOne(o_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_o = { re_phi(3 * o), re_phi(3 * o + 1), re_phi(3 * o + 2) };

                        // oŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_o = HatFunction((cal_point(0) - grid_point_coordinates_o(0)) / square.dx);
                        double hat_y_o = HatFunction((cal_point(1) - grid_point_coordinates_o(1)) / square.dx);
                        double diff_hat_y_o = DifferentialHatFunction((cal_point(1) - grid_point_coordinates_o(1)) / square.dx) / square.dx;
                        double hat_z_o = HatFunction((cal_point(2) - grid_point_coordinates_o(2)) / square.dx);
                        double diff_hat_z_o = DifferentialHatFunction((cal_point(2) - grid_point_coordinates_o(2)) / square.dx) / square.dx;

                        // Še€‚ÌŒvZ
                        double w_m_2 = hat_x_m * diff_hat_y_m * hat_z_m;
                        double w_o_3 = hat_x_o * hat_y_o * diff_hat_z_o;
                        double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                        double w_m_1 = diff_hat_x_m * hat_y_m * hat_z_m;
                        double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                        double w_o_2 = hat_x_o * diff_hat_y_o * hat_z_o;
                        double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int a = 0; a < dimensions; a++) {
                            WeightMOXi(a) += Phi_MO(m, 3 * o + a)
                                * (w_m_2 * w_o_3 * w_xi_1 - w_m_1 * w_o_3 * w_xi_2 + w_m_1 * w_o_2 * w_xi_3);
                        }

                    }
                }


                for (int i = 0; i < NumberOfParticles; i++) {
                    Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                    if (!allElementsWithinOne(i_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                    // “à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_i = HatFunction((cal_point(0) - grid_point_coordinates_i(0)) / square.dx);
                    double hat_y_i = HatFunction((cal_point(1) - grid_point_coordinates_i(1)) / square.dx);
                    double hat_z_i = HatFunction((cal_point(2) - grid_point_coordinates_i(2)) / square.dx);
                    WeightI += power(i) * hat_x_i * hat_y_i * hat_z_i;
                }

                // HessianUpsilon3 += - pow(J, (-1)) * WeightIJ.transpose() * WeightKLXi ‚ÌŒvZ
                for (int col = 0; col < dimensions; col++) { // —ñ”
                    for (int row = 0; row < dimensions; row++) { // s”
                        double term = - detF * WeightLNTau(row) * WeightMOXi(col) * WeightI * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianUpsilon3(3 * xi + row, 3 * tau + col) += term;
                    }
                }


            }
        }

    }

    return HessianUpsilon3;
}

