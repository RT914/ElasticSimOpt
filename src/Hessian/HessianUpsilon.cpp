#include <iostream>
#include <omp.h>
#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Hessian/HessianUpsilon.h"


// Caluculate HessianXi
Eigen::MatrixXd calHessianUpsilon(const Square& square, const Eigen::VectorXd& phi, const Eigen::VectorXd& power) {
    Eigen::MatrixXd HessianUpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianUpsilon.setZero();
    Eigen::MatrixXd HessianUpsilon1 = calHessianUpsilon1(square, phi);
    Eigen::MatrixXd HessianUpsilon2 = calHessianUpsilon2(square, phi);
    Eigen::MatrixXd HessianUpsilon3 = calHessianUpsilon3(square, phi, power);
    HessianUpsilon = HessianUpsilon1 + HessianUpsilon2 + HessianUpsilon3;

    return HessianUpsilon;
}

Eigen::MatrixXd calHessianUpsilon1(const Square& square, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianUpsilon1 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 4 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::VectorXd J = Eigen::VectorXd::Zero(NumberOfParticles);
    Eigen::MatrixXd Phi_KL = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // ‘ÌÏ•Ï‰»—¦J‚ÌŒvZ
    for (int xi = 0; xi < NumberOfParticles; xi++) {
        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        J(xi) = RiemannSumJ(phi, grid_xi, square.dx, cal_points, (-5.0 / 3.0));
    }

    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int k = 0; k < NumberOfParticles; k++) {
        for (int l = 0; l < NumberOfParticles; l++) {
            double Phi1 = phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1);
            double Phi2 = -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l));
            double Phi3 = phi(3 * k) * phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_KL(3 * k + p, l) = Phi(p);
            }
        }
    }

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                Eigen::MatrixXd WeightIJ = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);
                Eigen::VectorXd WeightKLXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);

                /*------  ®ŒvZ   ------*/
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        for (int j = 0; j < NumberOfParticles; j++) {
                            Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                            if (!allElementsWithinOne(j_minus_xi)) continue;

                            // “à‘}ŠÖ”‚ÌŒvZ
                            double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - i_minus_xi(0));
                            double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - i_minus_xi(1));
                            double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - i_minus_xi(2));
                            double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                            double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                            double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));

                            double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - j_minus_xi(0));
                            double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - j_minus_xi(1));
                            double diff_hat_z_j = DifferentialHatFunction(cal_point(2) - j_minus_xi(2));
                            double hat_x_j = HatFunction(cal_point(0) - j_minus_xi(0));
                            double hat_y_j = HatFunction(cal_point(1) - j_minus_xi(1));
                            double hat_z_j = HatFunction(cal_point(2) - j_minus_xi(2));

                            for (int p = 0; p < dimensions; p++) {
                                WeightIJ(i, 3 * xi + p) += phi(3 * j + p) *
                                    (diff_hat_x_i * hat_y_i * hat_z_i * diff_hat_x_j * hat_y_j * hat_z_j
                                        + hat_x_i * diff_hat_y_i * hat_z_i * hat_x_j * diff_hat_y_j * hat_z_j
                                        + hat_x_i * hat_y_i * diff_hat_z_i * hat_x_j * hat_y_j * diff_hat_z_j);
                            }

                        }
                    }

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        for (int l = 0; l < NumberOfParticles; l++) {
                            Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                            if (!allElementsWithinOne(l_minus_xi)) continue;

                            // ŒvZ“_‚ÌŒˆ’è‚Æ“à‘}ŠÖ”‚ÌŒvZ
                            // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_k = HatFunction(cal_point(0) - k_minus_xi(0));
                            double diff_hat_x_k = DifferentialHatFunction(cal_point(0) - k_minus_xi(0));
                            double hat_y_k = HatFunction(cal_point(1) - k_minus_xi(1));
                            double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - k_minus_xi(1));
                            double hat_z_k = HatFunction(cal_point(2) - k_minus_xi(2));

                            // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_l = HatFunction(cal_point(0) - l_minus_xi(0));
                            double hat_y_l = HatFunction(cal_point(1) - l_minus_xi(1));
                            double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - l_minus_xi(1));
                            double hat_z_l = HatFunction(cal_point(2) - l_minus_xi(2)); // C³: cal_point(1) ‚©‚ç cal_point(2)
                            double diff_hat_z_l = DifferentialHatFunction(cal_point(2) - l_minus_xi(2));

                            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_xi = HatFunction(cal_point(0));
                            double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
                            double hat_y_xi = HatFunction(cal_point(1));
                            double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
                            double hat_z_xi = HatFunction(cal_point(2));
                            double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));

                            // Še€‚ÌŒvZ
                            double term1 = hat_x_k * diff_hat_y_k * hat_z_k;
                            double term2 = hat_x_l * hat_y_l * diff_hat_z_l;
                            double term3 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                            double term4 = diff_hat_x_k * hat_y_k * hat_z_k;
                            double term5 = hat_x_l * hat_y_l * diff_hat_z_l;
                            double term6 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                            double term7 = diff_hat_x_k * hat_y_k * hat_z_k;
                            double term8 = hat_x_l * diff_hat_y_l * hat_z_l;
                            double term9 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                            for (int p = 0; p < dimensions; p++) {
                                WeightKLXi(3 * xi + p) += Phi_KL(3 * k + p, l) *
                                    (term1 * term2 * term3
                                        - term4 * term5 * term6
                                        + term7 * term8 * term9);
                            }

                        }
                    }
                }

                // HessianUpsilon1 += mu * pow(J, (-5/3)) * WeightIJ * WeightKLXi.transpose() ‚ÌŒvZ
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    for (int i = 0; i < NumberOfParticles; i++) {

                        for (int col = 0; col < dimensions; col++) { // —ñ”
                            for (int row = 0; row < dimensions; row++) { // s”
                                HessianUpsilon1(3 * i + row, 3 * xi + col) += mu * J(xi) * WeightIJ(i, 3 * xi + col) * WeightKLXi(3 * xi + row) * pow(kWidth, 3);
                            }
                        }

                    }
                }

                /*------  ®ŒvZI—¹   ------*/

            }
        }
    }

    return HessianUpsilon1;
}

Eigen::MatrixXd calHessianUpsilon2(const Square& square, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd HessianUpsilon2 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 4 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::VectorXd J = Eigen::VectorXd::Zero(NumberOfParticles);
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);
    Eigen::MatrixXd Phi_LM = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd Phi_NO = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);

    // ŒW”‚ÌŒvZ

    // ‘ÌÏ•Ï‰»—¦J‚ÌŒvZ
    for (int xi = 0; xi < NumberOfParticles; xi++) {
        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        J(xi) = RiemannSumJ(phi, grid_xi, square.dx, cal_points, (-8.0 / 3.0));
    }

    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int j = 0; j < NumberOfParticles; j++) {
        for (int k = 0; k < NumberOfParticles; k++) {
            double Phi1 = phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1);
            double Phi2 = -(phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k));
            double Phi3 = phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_JK(j, 3 * k + p) = Phi(p);
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
                Phi_NO(3 * n + p, o) = Phi(p);
            }
        }
    }

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                Eigen::MatrixXd WeightSum = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);
                Eigen::VectorXd WeightIJK = Eigen::VectorXd::Zero(NumberOfParticles, 3 * NumberOfParticles);
                Eigen::VectorXd WeightLM = Eigen::VectorXd::Zero(NumberOfParticles);
                Eigen::VectorXd WeightNOXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);

                /*------  ®ŒvZ   ------*/
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        for (int j = 0; j < NumberOfParticles; j++) {
                            Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                            if (!allElementsWithinOne(j_minus_xi)) continue;

                            for (int k = 0; k < NumberOfParticles; k++) {
                                Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                                if (!allElementsWithinOne(k_minus_xi)) continue;

                                // “à‘}ŠÖ”‚ÌŒvZ
                                // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                                double hat_x_j = HatFunction(cal_point(0) - j_minus_xi(0));
                                double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - j_minus_xi(0));
                                double hat_y_j = HatFunction(cal_point(1) - j_minus_xi(1));
                                double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - j_minus_xi(1));
                                double hat_z_j = HatFunction(cal_point(2) - j_minus_xi(2));

                                // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                                double hat_x_k = HatFunction(cal_point(0) - k_minus_xi(0));
                                double hat_y_k = HatFunction(cal_point(1) - k_minus_xi(1));
                                double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - k_minus_xi(1));
                                double hat_z_k = HatFunction(cal_point(2) - k_minus_xi(2));
                                double diff_hat_z_k = DifferentialHatFunction(cal_point(2) - k_minus_xi(2));

                                // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                                double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                                double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - i_minus_xi(0));
                                double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                                double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - i_minus_xi(1));
                                double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));
                                double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - i_minus_xi(2));

                                // Še€‚ÌŒvZ
                                double term1 = hat_x_j * diff_hat_y_j * hat_z_j;
                                double term2 = hat_x_k * hat_y_k * diff_hat_z_k;
                                double term3 = diff_hat_x_i * hat_y_i * hat_z_i;

                                double term4 = diff_hat_x_j * hat_y_j * hat_z_j;
                                double term5 = hat_x_k * hat_y_k * diff_hat_z_k;
                                double term6 = hat_x_i * diff_hat_y_i * hat_z_i;

                                double term7 = diff_hat_x_j * hat_y_j * hat_z_j;
                                double term8 = hat_x_k * diff_hat_y_k * hat_z_k;
                                double term9 = hat_x_i * hat_y_i * diff_hat_z_i;

                                for (int p = 0; p < dimensions; p++) {
                                    WeightIJK(i, 3 * xi + p) += Phi_JK(j, 3 * k + p) *
                                        (term1 * term2 * term3
                                            - term4 * term5 * term6
                                            + term7 * term8 * term9);
                                }

                            }
                        }
                    }

                    for (int l = 0; l < NumberOfParticles; l++) {
                        Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                        if (!allElementsWithinOne(l_minus_xi)) continue;

                        for (int m = 0; m < NumberOfParticles; m++) {
                            Eigen::Vector3i m_minus_xi = FlatToGrid(m) - grid_xi;
                            if (!allElementsWithinOne(m_minus_xi)) continue;

                            // “à‘}ŠÖ”‚ÌŒvZ
                            double diff_hat_x_l = DifferentialHatFunction(cal_point(0) - l_minus_xi(0));
                            double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - l_minus_xi(1));
                            double diff_hat_z_l = DifferentialHatFunction(cal_point(2) - l_minus_xi(2));
                            double hat_x_l = HatFunction(cal_point(0) - l_minus_xi(0));
                            double hat_y_l = HatFunction(cal_point(1) - l_minus_xi(1));
                            double hat_z_l = HatFunction(cal_point(2) - l_minus_xi(2));

                            double diff_hat_x_m = DifferentialHatFunction(cal_point(0) - m_minus_xi(0));
                            double diff_hat_y_m = DifferentialHatFunction(cal_point(1) - m_minus_xi(1));
                            double diff_hat_z_m = DifferentialHatFunction(cal_point(2) - m_minus_xi(2));
                            double hat_x_m = HatFunction(cal_point(0) - m_minus_xi(0));
                            double hat_y_m = HatFunction(cal_point(1) - m_minus_xi(1));
                            double hat_z_m = HatFunction(cal_point(2) - m_minus_xi(2));

                            WeightLM(xi) += Phi_LM(l, m) *
                                (diff_hat_x_l * hat_y_l * hat_z_l * diff_hat_x_m * hat_y_m * hat_z_m
                                    + hat_x_l * diff_hat_y_l * hat_z_l * hat_x_m * diff_hat_y_m * hat_z_m
                                    + hat_x_l * hat_y_l * diff_hat_z_l * hat_x_m * hat_y_m * diff_hat_z_m);

                        }
                    }

                    for (int n = 0; n < NumberOfParticles; n++) {
                        Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                        if (!allElementsWithinOne(n_minus_xi)) continue;

                        for (int o = 0; o < NumberOfParticles; o++) {
                            Eigen::Vector3i o_minus_xi = FlatToGrid(o) - grid_xi;
                            if (!allElementsWithinOne(o_minus_xi)) continue;

                            // “à‘}ŠÖ”‚ÌŒvZ
                            // nŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_n = HatFunction(cal_point(0) - n_minus_xi(0));
                            double diff_hat_x_n = DifferentialHatFunction(cal_point(0) - n_minus_xi(0));
                            double hat_y_n = HatFunction(cal_point(1) - n_minus_xi(1));
                            double diff_hat_y_n = DifferentialHatFunction(cal_point(1) - n_minus_xi(1));
                            double hat_z_n = HatFunction(cal_point(2) - n_minus_xi(2));

                            // oŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_o = HatFunction(cal_point(0) - o_minus_xi(0));
                            double hat_y_o = HatFunction(cal_point(1) - o_minus_xi(1));
                            double diff_hat_y_o = DifferentialHatFunction(cal_point(1) - o_minus_xi(1));
                            double hat_z_o = HatFunction(cal_point(2) - o_minus_xi(2));
                            double diff_hat_z_o = DifferentialHatFunction(cal_point(2) - o_minus_xi(2));

                            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_xi = HatFunction(cal_point(0));
                            double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
                            double hat_y_xi = HatFunction(cal_point(1));
                            double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
                            double hat_z_xi = HatFunction(cal_point(2));
                            double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));

                            // Še€‚ÌŒvZ
                            double term1 = hat_x_n * diff_hat_y_n * hat_z_n;
                            double term2 = hat_x_o * hat_y_o * diff_hat_z_o;
                            double term3 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                            double term4 = diff_hat_x_n * hat_y_n * hat_z_n;
                            double term5 = hat_x_o * hat_y_o * diff_hat_z_o;
                            double term6 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                            double term7 = diff_hat_x_n * hat_y_n * hat_z_n;
                            double term8 = hat_x_o * diff_hat_y_o * hat_z_o;
                            double term9 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                            for (int p = 0; p < dimensions; p++) {
                                WeightNOXi(3 * xi + p) += Phi_NO(3 * n + p, o) *
                                    (term1 * term2 * term3
                                        - term4 * term5 * term6
                                        + term7 * term8 * term9);
                            }

                        }
                    }

                }

                // HessianUpsilon2 += mu * pow(J, (-5/3)) * WeightIJ * WeightKLXi.transpose() ‚ÌŒvZ
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    for (int i = 0; i < NumberOfParticles; i++) {

                        for (int col = 0; col < dimensions; col++) { // —ñ”
                            for (int row = 0; row < dimensions; row++) { // s”
                                HessianUpsilon2(3 * i + row, 3 * xi + col) 
                                    -= (1.0/3.0) * mu * J(xi) * WeightIJK(i, 3 * xi + col) * WeightNOXi(3 * xi + row) * WeightLM(xi) * pow(kWidth, 3);
                            }
                        }

                    }
                }

                /*------  ®ŒvZI—¹   ------*/
            }
        }
    }

    return HessianUpsilon2;
}

Eigen::MatrixXd calHessianUpsilon3(const Square& square, const Eigen::VectorXd& phi, const Eigen::VectorXd& power)
{
    Eigen::MatrixXd HessianUpsilon3 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 4 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::VectorXd J = Eigen::VectorXd::Zero(NumberOfParticles);
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);
    Eigen::MatrixXd Phi_LM = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);

    // ŒW”‚ÌŒvZ

    // ‘ÌÏ•Ï‰»—¦J‚ÌŒvZ
    for (int xi = 0; xi < NumberOfParticles; xi++) {
        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        J(xi) = RiemannSumJ(phi, grid_xi, square.dx, cal_points, (-1.0));
    }

    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int j = 0; j < NumberOfParticles; j++) {
        for (int k = 0; k < NumberOfParticles; k++) {
            double Phi1 = phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1);
            double Phi2 = -(phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k));
            double Phi3 = phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_JK(j, 3 * k + p) = Phi(p);
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
                Phi_LM(3 * l + p, m) = Phi(p);
            }
        }
    }

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                Eigen::MatrixXd WeightSum = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);
                Eigen::VectorXd WeightIJK = Eigen::VectorXd::Zero(NumberOfParticles, 3 * NumberOfParticles);
                Eigen::VectorXd WeightLMXi = Eigen::VectorXd::Zero(3 * NumberOfParticles, NumberOfParticles);
                Eigen::VectorXd WeightN = Eigen::VectorXd::Zero(NumberOfParticles);

                /*------  ®ŒvZ   ------*/
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        for (int j = 0; j < NumberOfParticles; j++) {
                            Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                            if (!allElementsWithinOne(j_minus_xi)) continue;

                            for (int k = 0; k < NumberOfParticles; k++) {
                                Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                                if (!allElementsWithinOne(k_minus_xi)) continue;

                                // “à‘}ŠÖ”‚ÌŒvZ
                                // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                                double hat_x_j = HatFunction(cal_point(0) - j_minus_xi(0));
                                double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - j_minus_xi(0));
                                double hat_y_j = HatFunction(cal_point(1) - j_minus_xi(1));
                                double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - j_minus_xi(1));
                                double hat_z_j = HatFunction(cal_point(2) - j_minus_xi(2));

                                // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                                double hat_x_k = HatFunction(cal_point(0) - k_minus_xi(0));
                                double hat_y_k = HatFunction(cal_point(1) - k_minus_xi(1));
                                double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - k_minus_xi(1));
                                double hat_z_k = HatFunction(cal_point(2) - k_minus_xi(2));
                                double diff_hat_z_k = DifferentialHatFunction(cal_point(2) - k_minus_xi(2));

                                // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                                double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                                double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - i_minus_xi(0));
                                double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                                double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - i_minus_xi(1));
                                double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));
                                double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - i_minus_xi(2));

                                // Še€‚ÌŒvZ
                                double term1 = hat_x_j * diff_hat_y_j * hat_z_j;
                                double term2 = hat_x_k * hat_y_k * diff_hat_z_k;
                                double term3 = diff_hat_x_i * hat_y_i * hat_z_i;

                                double term4 = diff_hat_x_j * hat_y_j * hat_z_j;
                                double term5 = hat_x_k * hat_y_k * diff_hat_z_k;
                                double term6 = hat_x_i * diff_hat_y_i * hat_z_i;

                                double term7 = diff_hat_x_j * hat_y_j * hat_z_j;
                                double term8 = hat_x_k * diff_hat_y_k * hat_z_k;
                                double term9 = hat_x_i * hat_y_i * diff_hat_z_i;

                                for (int p = 0; p < dimensions; p++) {
                                    WeightIJK(i, 3 * xi + p) += Phi_JK(j, 3 * k + p) *
                                        (term1 * term2 * term3
                                            - term4 * term5 * term6
                                            + term7 * term8 * term9);
                                }

                            }
                        }
                    }

                    for (int l = 0; l < NumberOfParticles; l++) {
                        Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                        if (!allElementsWithinOne(l_minus_xi)) continue;

                        for (int m = 0; m < NumberOfParticles; m++) {
                            Eigen::Vector3i m_minus_xi = FlatToGrid(m) - grid_xi;
                            if (!allElementsWithinOne(m_minus_xi)) continue;

                            // “à‘}ŠÖ”‚ÌŒvZ
                            // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_l = HatFunction(cal_point(0) - l_minus_xi(0));
                            double diff_hat_x_l = DifferentialHatFunction(cal_point(0) - l_minus_xi(0));
                            double hat_y_l = HatFunction(cal_point(1) - l_minus_xi(1));
                            double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - l_minus_xi(1));
                            double hat_z_l = HatFunction(cal_point(2) - l_minus_xi(2));

                            // mŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_m = HatFunction(cal_point(0) - m_minus_xi(0));
                            double hat_y_m = HatFunction(cal_point(1) - m_minus_xi(1));
                            double diff_hat_y_m = DifferentialHatFunction(cal_point(1) - m_minus_xi(1));
                            double hat_z_m = HatFunction(cal_point(2) - m_minus_xi(2));
                            double diff_hat_z_m = DifferentialHatFunction(cal_point(2) - m_minus_xi(2));

                            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_xi = HatFunction(cal_point(0));
                            double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
                            double hat_y_xi = HatFunction(cal_point(1));
                            double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
                            double hat_z_xi = HatFunction(cal_point(2));
                            double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));

                            // Še€‚ÌŒvZ
                            double term1 = hat_x_l * diff_hat_y_l * hat_z_l;
                            double term2 = hat_x_m * hat_y_m * diff_hat_z_m;
                            double term3 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                            double term4 = diff_hat_x_l * hat_y_l * hat_z_l;
                            double term5 = hat_x_m * hat_y_m * diff_hat_z_m;
                            double term6 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                            double term7 = diff_hat_x_l * hat_y_l * hat_z_l;
                            double term8 = hat_x_m * diff_hat_y_m * hat_z_m;
                            double term9 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                            for (int p = 0; p < dimensions; p++) {
                                WeightLMXi(3 * xi + p) += Phi_LM(3 * l + p, m) *
                                    (term1 * term2 * term3
                                        - term4 * term5 * term6
                                        + term7 * term8 * term9);
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
                        WeightN(xi) += power(n) * hat_x_n * hat_y_n * hat_z_n;
                    }

                }

                // HessianUpsilon3 -= pow(J, (-1)) * WeightIJ.transpose() * WeightKLXi ‚ÌŒvZ
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    for (int i = 0; i < NumberOfParticles; i++) {

                        for (int col = 0; col < dimensions; col++) { // —ñ”
                            for (int row = 0; row < dimensions; row++) { // s”
                                HessianUpsilon3(3 * i + row, 3 * xi + col) 
                                    -= J(xi) * WeightIJK(i, 3 * xi + col) * WeightLMXi(3 * xi + row) * WeightN(xi) * pow(kWidth, 3);
                            }
                        }

                    }
                }

                /*------  ®ŒvZI—¹   ------*/
            }
        }
    }

    return HessianUpsilon3;
}
