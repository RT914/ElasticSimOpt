#include <Eigen/Dense>
#include <iostream>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/Hessian/HessianP.h"


// Calculate Hessian P
Eigen::MatrixXd calHessianP(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd HessianP = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = square.SideNumber * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
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
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int j = 0; j < NumberOfParticles; j++) {
        for (int k = 0; k < NumberOfParticles; k++) {
            double Phi1 = phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1);
            double Phi2 = -(phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k));
            double Phi3 = phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int a = 0; a < dimensions; a++) {
                Phi_JK(j, 3 * k + a) = Phi(a);
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
            double hat_y_xi = HatFunction(cal_point(1) - grid_point_coordinates_xi(1));
            double hat_z_xi = HatFunction(cal_point(2) - grid_point_coordinates_xi(2));

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

                Eigen::Matrix3d WeightJKTau = Eigen::Matrix3d::Zero();

                double WeightXi = hat_x_xi * hat_y_xi * hat_z_xi;

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

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_k = { re_phi(3 * k), re_phi(3 * k + 1), re_phi(3 * k + 2) };

                        // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_k = HatFunction(cal_point(0) - grid_point_coordinates_k(0));
                        double hat_y_k = HatFunction(cal_point(1) - grid_point_coordinates_k(1));
                        double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_k(1));
                        double hat_z_k = HatFunction(cal_point(2) - grid_point_coordinates_k(2));
                        double diff_hat_z_k = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_k(2));

                        // Še€‚ÌŒvZ
                        double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;
                        double w_k_3 = hat_x_k * hat_y_k * diff_hat_z_k;
                        double w_tau_1 = diff_hat_x_tau * hat_y_tau * hat_z_tau;

                        double w_j_1 = diff_hat_x_j * hat_y_j * hat_z_j;
                        double w_tau_2 = hat_x_tau * diff_hat_y_tau * hat_z_tau;

                        double w_k_2 = hat_x_k * diff_hat_y_k * hat_z_k;
                        double w_tau_3 = hat_x_tau * hat_y_tau * diff_hat_z_tau;

                        for (int a = 0; a < dimensions; a++) {
                            WeightJKTau(a) += Phi_JK(j, 3 * k + a)
                                * (w_j_2 * w_k_3 * w_tau_1 - w_j_1 * w_k_3 * w_tau_2 + w_j_1 * w_k_2 * w_tau_3);
                        }

                    }
                }

                // HessianP = WeightXi * WeightJKI ‚ÌŒvZ
                for (int col = 0; col < dimensions; col++) { // —ñ”
                    double term = WeightXi * WeightJKTau(col) * volume_element;
                    if (abs(term) < 1e-10) continue;
                    HessianP(xi, 3 * tau + col) += term;
                }


            }
        }

    }

    return HessianP;
}
