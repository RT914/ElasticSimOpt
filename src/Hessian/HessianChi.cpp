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

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;

        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        /*------  ®ŒvZ   ------*/
        // Stencil Base‚ÌŒvZ
        Eigen::Vector3d stencil_base = calculateStencilBase(cal_point);

        // Stencils—ñ‚Æstencil_num‚Ì¶¬
        Eigen::MatrixXi stencil;
        std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

        for (int xi = 0; xi < NumberOfParticles; xi++) {
            if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
            Eigen::Vector3i grid_xi = FlatToGrid(xi);
            Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

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
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, -2.0/3.0);

            for (int tau = 0; tau < NumberOfParticles; tau++) {
                Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                if (!allElementsWithinOne(tau_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                // “à‘}ŠÖ”‚ÌŒvZ
                double hat_x_tau = HatFunction(cal_point(0) - grid_point_coordinates_tau(0));
                double diff_hat_x_tau = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_tau(0));
                double hat_y_tau = HatFunction(cal_point(1) - grid_point_coordinates_tau(1));
                double diff_hat_y_tau = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_tau(1));
                double hat_z_tau = HatFunction(cal_point(2) - grid_point_coordinates_tau(2));
                double diff_hat_z_tau = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_tau(2));

                double w_tau_1 = diff_hat_x_tau * hat_y_tau * hat_z_tau;
                double w_tau_2 = hat_x_tau * diff_hat_y_tau * hat_z_tau;
                double w_tau_3 = hat_x_tau * hat_y_tau * diff_hat_z_tau;

                double WeightTauXi = w_tau_1 * w_xi_1 + w_tau_2 * w_xi_2 + w_tau_3 * w_xi_3;

                // ’PˆÊs—ñ
                Eigen::Matrix3d identityMatrix = Eigen::Matrix3d::Identity();

                for (int col = 0; col < dimensions; col++) { // —ñ”i‰¡‚Ì”j
                    for (int row = 0; row < dimensions; row++) { // s”ic‚Ì”j
                        double term = mu * detF * WeightTauXi * identityMatrix(row, col) * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianChi1(3 * xi + row, 3 * tau + col) += term;
                    }
                }

            }
        }

        /*------  ®ŒvZI—¹   ------*/

    }

    return HessianChi1;
}

Eigen::MatrixXd calHessianChi2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianChi2 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

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
    Eigen::MatrixXd Phi_KL = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
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

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;

        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        /*------  ®ŒvZ   ------*/
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

                // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                double diff_hat_x_tau = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_tau(0));
                double diff_hat_y_tau = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_tau(1));
                double diff_hat_z_tau = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_tau(2));
                double hat_x_tau = HatFunction(cal_point(0) - grid_point_coordinates_tau(0));
                double hat_y_tau = HatFunction(cal_point(1) - grid_point_coordinates_tau(1));
                double hat_z_tau = HatFunction(cal_point(2) - grid_point_coordinates_tau(2));

                double w_tau_1 = diff_hat_x_tau * hat_y_tau * hat_z_tau;
                double w_tau_2 = hat_x_tau * diff_hat_y_tau * hat_z_tau;
                double w_tau_3 = hat_x_tau * hat_y_tau * diff_hat_z_tau;

                Eigen::Vector3d WeightJTau = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightKLXi = Eigen::Vector3d::Zero();

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

                    for (int a = 0; a < dimensions; a++) {
                        WeightJTau(a) += phi(3 * j + a) * (w_tau_1 * w_j_1 + w_tau_2 * w_j_2 + w_tau_3 * w_j_3);
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

                        for (int a = 0; a < dimensions; a++) {
                            WeightKLXi(a) += Phi_KL(k, 3 * l + a)
                                * (w_k_2 * w_l_3 * w_xi_1 - w_k_1 * w_l_3 * w_xi_2 + w_k_1 * w_l_2 * w_xi_3);
                        }

                    }
                }

                for (int col = 0; col < dimensions; col++) { // —ñ”i‰¡‚Ì”j
                    for (int row = 0; row < dimensions; row++) { // s”ic‚Ì”j
                        double term = mu * detF * WeightJTau(row) * WeightKLXi(col) * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianChi2(3 * xi + row, 3 * tau + col) += term;
                    }
                }

            }
        }

        /*------  ®ŒvZI—¹   ------*/

    }

    return HessianChi2;
}

Eigen::MatrixXd calHessianChi3(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianChi3 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

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

    Eigen::MatrixXd Phi_KL = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
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

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;
    
        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        /*------  ®ŒvZ   ------*/
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

                double w_tau_1 = diff_hat_x_tau * hat_y_tau * hat_z_tau;
                double w_tau_2 = hat_x_tau * diff_hat_y_tau * hat_z_tau;
                double w_tau_3 = hat_x_tau * hat_y_tau * diff_hat_z_tau;

                Eigen::Vector3d WeightJTau = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightKLXi = Eigen::Vector3d::Zero();

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

                    for (int a = 0; a < dimensions; a++) {
                        WeightJTau(a) += phi(3 * j + a) * (w_tau_1 * w_j_1 + w_tau_2 * w_j_2 + w_tau_3 * w_j_3);
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

                        for (int a = 0; a < dimensions; a++) {
                            WeightKLXi(a) += Phi_KL(3 * k + a, l) 
                                * (w_k_2 * w_l_3 * w_xi_1 - w_k_1 * w_l_3 * w_xi_2 + w_k_1 * w_l_2 * w_xi_3);
                        }

                    }
                }

                // ’PˆÊs—ñ
                Eigen::Matrix3d identityMatrix = Eigen::Matrix3d::Identity();

                for (int col = 0; col < dimensions; col++) { // —ñ”i‰¡‚Ì”j
                    for (int row = 0; row < dimensions; row++) { // s”ic‚Ì”j
                        double term = -2.0 / 3.0 * mu * detF * WeightKLXi(row) * WeightJTau(col) * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianChi3(3 * xi + row, 3 * tau + col) += term;
                    }
                }

            }
        }

        /*------  ®ŒvZI—¹   ------*/
    }

    return HessianChi3;
}

