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
    // Eigen::MatrixXd HessianXi1 = calHessianXi1(square, re_phi, phi, power);
    // Eigen::MatrixXd HessianXi2 = calHessianXi2(square, re_phi, phi);
    Eigen::MatrixXd HessianXi3 = calHessianXi3(square, re_phi, phi);

    // HessianXi = HessianXi1 + HessianXi2 + HessianXi3;

    return HessianXi3;
}

Eigen::MatrixXd calHessianXi1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power)
{
    Eigen::MatrixXd HessianXi1 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 4 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const int AllkNum = pow(kNum, 3);// ‘SŸŒ³‚Ì‘S‹æŠÔ•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_LM = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int j = 0; j < NumberOfParticles; j++) {
        for (int k = 0; k < NumberOfParticles; k++) {
            double Phi1 = phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1);
            double Phi2 = -(phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k));
            double Phi3 = phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_JK(3 * j + p, k) = Phi(p);
            }
        }
    }

    for (int l = 0; l < NumberOfParticles; l++) {
        for (int m = 0; m < NumberOfParticles; m++) {
            double Phi1 = phi(3 * l + 1) * phi(3 * m + 2) - phi(3 * l + 2) * phi(3 * m + 1);
            double Phi2 = -(phi(3 * l) * phi(3 * m + 2) - phi(3 * l + 2) * phi(3 * m));
            double Phi3 = phi(3 * l) * phi(3 * m + 1) - phi(3 * l + 1) * phi(3 * m);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_LM(l, 3 * m + p) = Phi(p);
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
        for (int xi = 0; xi < NumberOfParticles; xi++) {
            Eigen::Vector3i grid_xi = FlatToGrid(xi);

            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
            double hat_x_xi = HatFunction(cal_point(0));
            double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
            double hat_y_xi = HatFunction(cal_point(1));
            double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
            double hat_z_xi = HatFunction(cal_point(2));
            double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));

            double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;
            double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;
            double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

            // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, -1.0);

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - i_minus_xi(0));
                double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - i_minus_xi(1));
                double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));
                double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - i_minus_xi(2));

                double w_i_1 = diff_hat_x_i * hat_y_i * hat_z_i;
                double w_i_2 = hat_x_i * diff_hat_y_i * hat_z_i;
                double w_i_3 = hat_x_i * hat_y_i * diff_hat_z_i;

                Eigen::Vector3d WeightIJK = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightLMXi = Eigen::Vector3d::Zero();
                double WeightN = 0.0;

                for (int j = 0; j < NumberOfParticles; j++) {
                    Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                    if (!allElementsWithinOne(j_minus_xi)) continue;

                    // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_j = HatFunction(cal_point(0) - j_minus_xi(0));
                    double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - j_minus_xi(0));
                    double hat_y_j = HatFunction(cal_point(1) - j_minus_xi(1));
                    double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - j_minus_xi(1));
                    double hat_z_j = HatFunction(cal_point(2) - j_minus_xi(2));

                    double w_j_1 = diff_hat_x_j * hat_y_j * hat_z_j;
                    double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_k = HatFunction(cal_point(0) - k_minus_xi(0));
                        double hat_y_k = HatFunction(cal_point(1) - k_minus_xi(1));
                        double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - k_minus_xi(1));
                        double hat_z_k = HatFunction(cal_point(2) - k_minus_xi(2));
                        double diff_hat_z_k = DifferentialHatFunction(cal_point(2) - k_minus_xi(2));

                        double w_k_2 = hat_x_k * diff_hat_y_k * hat_z_k;
                        double w_k_3 = hat_x_k * hat_y_k * diff_hat_z_k;

                        for (int p = 0; p < dimensions; p++) {
                            WeightIJK(p) += Phi_JK(3 * j + p, k) *
                                (w_j_2 * w_k_3 * w_i_1
                                    - w_j_1 * w_k_3 * w_i_2
                                    + w_j_1 * w_k_2 * w_i_3);
                        }

                    }
                }

                for (int l = 0; l < NumberOfParticles; l++) {
                    Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                    if (!allElementsWithinOne(l_minus_xi)) continue;

                    // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_l = HatFunction(cal_point(0) - l_minus_xi(0));
                    double diff_hat_x_l = DifferentialHatFunction(cal_point(0) - l_minus_xi(0));
                    double hat_y_l = HatFunction(cal_point(1) - l_minus_xi(1));
                    double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - l_minus_xi(1));
                    double hat_z_l = HatFunction(cal_point(2) - l_minus_xi(2));

                    double w_l_1 = diff_hat_x_l * hat_y_l * hat_z_l;
                    double w_l_2 = hat_x_l * diff_hat_y_l * hat_z_l;

                    for (int m = 0; m < NumberOfParticles; m++) {
                        Eigen::Vector3i m_minus_xi = FlatToGrid(m) - grid_xi;
                        if (!allElementsWithinOne(m_minus_xi)) continue;

                        // mŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_m = HatFunction(cal_point(0) - m_minus_xi(0));
                        double hat_y_m = HatFunction(cal_point(1) - m_minus_xi(1));
                        double diff_hat_y_m = DifferentialHatFunction(cal_point(1) - m_minus_xi(1));
                        double hat_z_m = HatFunction(cal_point(2) - m_minus_xi(2));
                        double diff_hat_z_m = DifferentialHatFunction(cal_point(2) - m_minus_xi(2));

                        double w_m_2 = hat_x_m * diff_hat_y_m * hat_z_m;
                        double w_m_3 = hat_x_m * hat_y_m * diff_hat_z_m;

                        for (int p = 0; p < dimensions; p++) {
                            WeightLMXi(p) += Phi_LM(l, 3 * m + p)
                                * (w_l_2 * w_m_3 * w_xi_1
                                - w_l_1 * w_m_3 * w_xi_2
                                + w_l_1 * w_m_2 * w_xi_3);
                        }

                    }
                }

                for (int n = 0; n < NumberOfParticles; n++) {
                    Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                    if (!allElementsWithinOne(n_minus_xi)) continue;

                    // “à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_n = HatFunction(cal_point(0) - n_minus_xi(0));
                    double hat_y_n = HatFunction(cal_point(1) - n_minus_xi(1));
                    double hat_z_n = HatFunction(cal_point(2) - n_minus_xi(2));
                    WeightN += power(n) * hat_x_n * hat_y_n * hat_z_n;
                }

                // HessianXi1‚ÌŒvZ
                for (int col = 0; col < dimensions; col++) { // —ñ”
                    for (int row = 0; row < dimensions; row++) { // s”
                        double term = detF * WeightIJK(row) * WeightLMXi(col) * WeightN * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianXi1(3 * i + row, 3 * xi + col) += term;
                    }
                }

            }
        }

        /*------  ®ŒvZI—¹   ------*/   
    }

    return  HessianXi1;
}

// ”ñ‘ÎÌs—ñ
Eigen::MatrixXd calHessianXi2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd  HessianXi2 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 4 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const int AllkNum = pow(kNum, 3);// ‘SŸŒ³‚Ì‘S‹æŠÔ•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_LM = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int j = 0; j < NumberOfParticles; j++) {
        for (int k = 0; k < NumberOfParticles; k++) {
            double Phi1 = phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1);
            double Phi2 = -(phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k));
            double Phi3 = phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_JK(3 * j + p, k) = Phi(p);
            }
        }
    }

    for (int l = 0; l < NumberOfParticles; l++) {
        for (int m = 0; m < NumberOfParticles; m++) {
            double Phi1 = phi(3 * l + 1) * phi(3 * m + 2) - phi(3 * l + 2) * phi(3 * m + 1);
            double Phi2 = -(phi(3 * l) * phi(3 * m + 2) - phi(3 * l + 2) * phi(3 * m));
            double Phi3 = phi(3 * l) * phi(3 * m + 1) - phi(3 * l + 1) * phi(3 * m);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_LM(l, 3 * m + p) = Phi(p);
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
        for (int xi = 0; xi < NumberOfParticles; xi++) {
            Eigen::Vector3i grid_xi = FlatToGrid(xi);

            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
            double hat_x_xi = HatFunction(cal_point(0));
            double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
            double hat_y_xi = HatFunction(cal_point(1));
            double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
            double hat_z_xi = HatFunction(cal_point(2));
            double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));

            // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, -3.0);

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - i_minus_xi(0));
                double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - i_minus_xi(1));
                double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));
                double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - i_minus_xi(2));

                Eigen::Vector3d WeightIJK = Eigen::Vector3d::Zero();
                Eigen::Vector3d WeightLMXi = Eigen::Vector3d::Zero();
                Eigen::Matrix3d WeightN = Eigen::Matrix3d::Zero();

                for (int j = 0; j < NumberOfParticles; j++) {
                    Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                    if (!allElementsWithinOne(j_minus_xi)) continue;

                    // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_j = HatFunction(cal_point(0) - j_minus_xi(0));
                    double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - j_minus_xi(0));
                    double hat_y_j = HatFunction(cal_point(1) - j_minus_xi(1));
                    double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - j_minus_xi(1));
                    double hat_z_j = HatFunction(cal_point(2) - j_minus_xi(2));

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_k = HatFunction(cal_point(0) - k_minus_xi(0));
                        double hat_y_k = HatFunction(cal_point(1) - k_minus_xi(1));
                        double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - k_minus_xi(1));
                        double hat_z_k = HatFunction(cal_point(2) - k_minus_xi(2));
                        double diff_hat_z_k = DifferentialHatFunction(cal_point(2) - k_minus_xi(2));

                        // Še€‚ÌŒvZ
                        double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;
                        double w_k_3 = hat_x_k * hat_y_k * diff_hat_z_k;
                        double w_i_1 = diff_hat_x_i * hat_y_i * hat_z_i;

                        double w_j_1 = diff_hat_x_j * hat_y_j * hat_z_j;
                        double w_i_2 = hat_x_i * diff_hat_y_i * hat_z_i;

                        double w_k_2 = hat_x_k * diff_hat_y_k * hat_z_k;
                        double w_i_3 = hat_x_i * hat_y_i * diff_hat_z_i;

                        for (int p = 0; p < dimensions; p++) {
                            WeightIJK(p) += Phi_JK(3 * j + p, k) *
                                (w_j_2 * w_k_3 * w_i_1
                                    - w_j_1 * w_k_3 * w_i_2
                                    + w_j_1 * w_k_2 * w_i_3);
                        }

                    }
                }

                for (int l = 0; l < NumberOfParticles; l++) {
                    Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                    if (!allElementsWithinOne(l_minus_xi)) continue;

                    // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_l = HatFunction(cal_point(0) - l_minus_xi(0));
                    double diff_hat_x_l = DifferentialHatFunction(cal_point(0) - l_minus_xi(0));
                    double hat_y_l = HatFunction(cal_point(1) - l_minus_xi(1));
                    double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - l_minus_xi(1));
                    double hat_z_l = HatFunction(cal_point(2) - l_minus_xi(2));

                    for (int m = 0; m < NumberOfParticles; m++) {
                        Eigen::Vector3i m_minus_xi = FlatToGrid(m) - grid_xi;
                        if (!allElementsWithinOne(m_minus_xi)) continue;

                        // mŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_m = HatFunction(cal_point(0) - m_minus_xi(0));
                        double hat_y_m = HatFunction(cal_point(1) - m_minus_xi(1));
                        double diff_hat_y_m = DifferentialHatFunction(cal_point(1) - m_minus_xi(1));
                        double hat_z_m = HatFunction(cal_point(2) - m_minus_xi(2));
                        double diff_hat_z_m = DifferentialHatFunction(cal_point(2) - m_minus_xi(2));

                        // Še€‚ÌŒvZ
                        double w_l_2 = hat_x_l * diff_hat_y_l * hat_z_l;
                        double w_m_3 = hat_x_m * hat_y_m * diff_hat_z_m;
                        double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                        double w_l_1 = diff_hat_x_l * hat_y_l * hat_z_l;
                        double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                        double w_m_2 = hat_x_m * diff_hat_y_m * hat_z_m;
                        double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int p = 0; p < dimensions; p++) {
                            WeightLMXi(p) += Phi_LM(l, 3 * m + p) *
                                (w_l_2 * w_m_3 * w_xi_1
                                    - w_l_1 * w_m_3 * w_xi_2
                                    + w_l_1 * w_m_2 * w_xi_3);
                        }

                    }
                }

                for (int n = 0; n < NumberOfParticles; n++) {
                    Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                    if (!allElementsWithinOne(n_minus_xi)) continue;

                    // “à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_n = HatFunction(cal_point(0) - n_minus_xi(0));
                    double hat_y_n = HatFunction(cal_point(1) - n_minus_xi(1));
                    double hat_z_n = HatFunction(cal_point(2) - n_minus_xi(2));

                    double diff_hat_x_n = DifferentialHatFunction(cal_point(0) - n_minus_xi(0));
                    double diff_hat_y_n = DifferentialHatFunction(cal_point(1) - n_minus_xi(1));
                    double diff_hat_z_n = DifferentialHatFunction(cal_point(2) - n_minus_xi(2));

                    for (int p = 0; p < dimensions; p++) { // ‰¡‚Ì¬•ª‚Ì‚½‚ß‚Ìfor•¶
                        WeightN(0, p) += phi(3 * n + p) * diff_hat_x_n * hat_y_n * hat_z_n;
                        WeightN(1, p) += phi(3 * n + p) * hat_x_n * diff_hat_y_n * hat_z_n;
                        WeightN(2, p) += phi(3 * n + p) * hat_x_n * hat_y_n * diff_hat_z_n;
                    }

                }

                for (int col = 0; col < dimensions; col++) { // —ñ”
                    for (int row = 0; row < dimensions; row++) { // s”
                        double term = (2.0 / 3.0) * mu * WeightIJK(row) * WeightLMXi(col)
                            * WeightN(row, col) * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianXi2(3 * i + row, 3 * xi + col) += term;
                    }
                }
                
            }
        }
        /*------  ®ŒvZI—¹   ------*/
    }

    return  HessianXi2;
}

// ”ñ‘ÎÌs—ñ
Eigen::MatrixXd calHessianXi3(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd  HessianXi3 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 4 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const int AllkNum = pow(kNum, 3);// ‘SŸŒ³‚Ì‘S‹æŠÔ•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_LM = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_NO = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int j = 0; j < NumberOfParticles; j++) {
        for (int k = 0; k < NumberOfParticles; k++) {
            double Phi1 = phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1);
            double Phi2 = -(phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k));
            double Phi3 = phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_JK(3 * j + p, k) = Phi(p);
            }

        }
    }

    for (int l = 0; l < NumberOfParticles; l++) {
        for (int m = 0; m < NumberOfParticles; m++) {
            Phi_LM(l, m) = phi(3 * l) * phi(3 * m) + phi(3 * l + 1) * phi(3 * m + 1) + phi(3 * l + 2) * phi(3 * m + 2);
        }
    }

    for (int n = 0; n < NumberOfParticles; n++) {
        for (int o = 0; o < NumberOfParticles; o++) {
            double Phi1 = phi(3 * n + 1) * phi(3 * o + 2) - phi(3 * n + 2) * phi(3 * o + 1);
            double Phi2 = -(phi(3 * n) * phi(3 * o + 2) - phi(3 * n + 2) * phi(3 * o));
            double Phi3 = phi(3 * n) * phi(3 * o + 1) - phi(3 * n + 1) * phi(3 * o);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_NO(n, 3 * o + p) = Phi(p);
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
        for (int xi = 0; xi < NumberOfParticles; xi++) {
            Eigen::Vector3i grid_xi = FlatToGrid(xi);

            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
            double hat_x_xi = HatFunction(cal_point(0));
            double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
            double hat_y_xi = HatFunction(cal_point(1));
            double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
            double hat_z_xi = HatFunction(cal_point(2));
            double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));

            // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, -8.0);

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - i_minus_xi(0));
                double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - i_minus_xi(1));
                double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));
                double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - i_minus_xi(2));

                Eigen::Vector3d WeightIJK = Eigen::Vector3d::Zero();
                double WeightLM = 0.0;
                Eigen::Vector3d WeightNOXi = Eigen::Vector3d::Zero();

                for (int j = 0; j < NumberOfParticles; j++) {
                    Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                    if (!allElementsWithinOne(j_minus_xi)) continue;

                    // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_j = HatFunction(cal_point(0) - j_minus_xi(0));
                    double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - j_minus_xi(0));
                    double hat_y_j = HatFunction(cal_point(1) - j_minus_xi(1));
                    double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - j_minus_xi(1));
                    double hat_z_j = HatFunction(cal_point(2) - j_minus_xi(2));

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_k = HatFunction(cal_point(0) - k_minus_xi(0));
                        double hat_y_k = HatFunction(cal_point(1) - k_minus_xi(1));
                        double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - k_minus_xi(1));
                        double hat_z_k = HatFunction(cal_point(2) - k_minus_xi(2));
                        double diff_hat_z_k = DifferentialHatFunction(cal_point(2) - k_minus_xi(2));

                        // Še€‚ÌŒvZ
                        double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;
                        double w_k_3 = hat_x_k * hat_y_k * diff_hat_z_k;
                        double w_i_1 = diff_hat_x_i * hat_y_i * hat_z_i;

                        double w_j_1 = diff_hat_x_j * hat_y_j * hat_z_j;
                        double w_i_2 = hat_x_i * diff_hat_y_i * hat_z_i;

                        double w_k_2 = hat_x_k * diff_hat_y_k * hat_z_k;
                        double w_i_3 = hat_x_i * hat_y_i * diff_hat_z_i;

                        for (int p = 0; p < dimensions; p++) {
                            WeightIJK(p) += Phi_JK(3 * j + p, k) *
                                (w_j_2 * w_k_3 * w_i_1
                                    - w_j_1 * w_k_3 * w_i_2
                                    + w_j_1 * w_k_2 * w_i_3);
                        }

                    }
                }

                for (int l = 0; l < NumberOfParticles; l++) {
                    Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                    if (!allElementsWithinOne(l_minus_xi)) continue;

                    // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double diff_hat_x_l = DifferentialHatFunction(cal_point(0) - l_minus_xi(0));
                    double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - l_minus_xi(1));
                    double diff_hat_z_l = DifferentialHatFunction(cal_point(2) - l_minus_xi(2));
                    double hat_x_l = HatFunction(cal_point(0) - l_minus_xi(0));
                    double hat_y_l = HatFunction(cal_point(1) - l_minus_xi(1));
                    double hat_z_l = HatFunction(cal_point(2) - l_minus_xi(2));

                    for (int m = 0; m < NumberOfParticles; m++) {
                        Eigen::Vector3i m_minus_xi = FlatToGrid(m) - grid_xi;
                        if (!allElementsWithinOne(m_minus_xi)) continue;

                        // mŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double diff_hat_x_m = DifferentialHatFunction(cal_point(0) - m_minus_xi(0));
                        double diff_hat_y_m = DifferentialHatFunction(cal_point(1) - m_minus_xi(1));
                        double diff_hat_z_m = DifferentialHatFunction(cal_point(2) - m_minus_xi(2));
                        double hat_x_m = HatFunction(cal_point(0) - m_minus_xi(0));
                        double hat_y_m = HatFunction(cal_point(1) - m_minus_xi(1));
                        double hat_z_m = HatFunction(cal_point(2) - m_minus_xi(2));

                        WeightLM += Phi_LM(l, m) *
                            (diff_hat_x_l * hat_y_l * hat_z_l * diff_hat_x_m * hat_y_m * hat_z_m
                                + hat_x_l * diff_hat_y_l * hat_z_l * hat_x_m * diff_hat_y_m * hat_z_m
                                + hat_x_l * hat_y_l * diff_hat_z_l * hat_x_m * hat_y_m * diff_hat_z_m);

                    }
                }

                for (int n = 0; n < NumberOfParticles; n++) {
                    Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                    if (!allElementsWithinOne(n_minus_xi)) continue;

                    // nŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_n = HatFunction(cal_point(0) - n_minus_xi(0));
                    double diff_hat_x_n = DifferentialHatFunction(cal_point(0) - n_minus_xi(0));
                    double hat_y_n = HatFunction(cal_point(1) - n_minus_xi(1));
                    double diff_hat_y_n = DifferentialHatFunction(cal_point(1) - n_minus_xi(1));
                    double hat_z_n = HatFunction(cal_point(2) - n_minus_xi(2));

                    for (int o = 0; o < NumberOfParticles; o++) {
                        Eigen::Vector3i o_minus_xi = FlatToGrid(o) - grid_xi;
                        if (!allElementsWithinOne(o_minus_xi)) continue;

                        // oŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_o = HatFunction(cal_point(0) - o_minus_xi(0));
                        double hat_y_o = HatFunction(cal_point(1) - o_minus_xi(1));
                        double diff_hat_y_o = DifferentialHatFunction(cal_point(1) - o_minus_xi(1));
                        double hat_z_o = HatFunction(cal_point(2) - o_minus_xi(2));
                        double diff_hat_z_o = DifferentialHatFunction(cal_point(2) - o_minus_xi(2));

                        // Še€‚ÌŒvZ
                        double w_n_2 = hat_x_n * diff_hat_y_n * hat_z_n;
                        double w_o_3 = hat_x_o * hat_y_o * diff_hat_z_o;
                        double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                        double w_n_1 = diff_hat_x_n * hat_y_n * hat_z_n;
                        double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                        double w_o_2 = hat_x_o * diff_hat_y_o * hat_z_o;
                        double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int p = 0; p < dimensions; p++) {
                            WeightNOXi(p) += Phi_NO(3 * n + p, o) *
                                (w_n_2 * w_o_3 * w_xi_1
                                    - w_n_1 * w_o_3 * w_xi_2
                                    + w_n_1 * w_o_2 * w_xi_3);
                        }

                    }
                }

                for (int col = 0; col < dimensions; col++) { // —ñ”
                    for (int row = 0; row < dimensions; row++) { // s”
                        double term = (2.0 / 9.0) * mu * detF * WeightIJK(row) * WeightNOXi(col) * WeightLM * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianXi3(3 * i + row, 3 * xi + col) -= term;
                    }
                }

            }      
        }
        /*------  ®ŒvZI—¹   ------*/

    }

    return  HessianXi3;
}


