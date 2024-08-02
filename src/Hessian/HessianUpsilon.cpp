#include <iostream>
#include <omp.h>
#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Hessian/HessianUpsilon.h"


// Caluculate HessianXi
Eigen::MatrixXd calHessianUpsilon(Square square, Eigen::VectorXd phi, Eigen::VectorXd power) {
    Eigen::MatrixXd HessianUpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianUpsilon.setZero();
    Eigen::MatrixXd HessianUpsilon1 = calHessianUpsilon1(square, phi);
    // Eigen::MatrixXd HessianUpsilon2 = calHessianUpsilon2(square, phi);
    // Eigen::MatrixXd HessianUpsilon3 = calHessianUpsilon3(square, phi, power);
    // HessianUpsilon = HessianUpsilon1 + HessianUpsilon2 + HessianUpsilon3;
    HessianUpsilon = HessianUpsilon1;

    return HessianUpsilon;
}

Eigen::MatrixXd calHessianUpsilon1(const Square& square, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianUpsilon1 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // äeãÊä‘ÇÃï™äÑêî
    const double kWidth = square.dx / kNumSection; // ï™äÑÇÃê≥ãKâª
    const int kNum = 4 * kNumSection; // ëSãÊä‘ÇÃï™äÑêî

    Eigen::VectorXd cal_points(kNum);
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(divIndex) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
        }
    }

    // ãÊä‘ï™äÑ
    for (int d = 0; d < kNum; d++) {
 
        Eigen::MatrixXd WeightIJ = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);
        Eigen::VectorXd WeightKLXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);
        double J = 0.0; // ëÃêœïœâªó¶ÇÃèâä˙âª

        /*------  éÆåvéZ   ------*/
        for (int xi = 0; xi < NumberOfParticles; xi++) {
            Eigen::Vector3i grid_xi = FlatToGrid(xi);

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                for (int j = 0; j < NumberOfParticles; j++) {
                    Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                    if (!allElementsWithinOne(j_minus_xi)) continue;

                    // phiÇÃåvéZ
                    double phi_j_x = phi(3 * j);
                    double phi_j_y = phi(3 * j + 1);
                    double phi_j_z = phi(3 * j + 2);
                    Eigen::Vector3d phi_j = { phi_j_x, phi_j_y, phi_j_z };

                    // åvéZì_ÇÃåàíËÇ∆ì‡ë}ä÷êîÇÃåvéZ
                    for (int x = 0; x < kNum; x++) {
                        for (int y = 0; y < kNum; y++) {
                            for (int z = 0; z < kNum; z++) {
                                Eigen::Vector3d cal_point(cal_points(x), cal_points(y), cal_points(z));

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
                                    WeightIJ(i, 3 * xi + p) += phi_j(p) *
                                        (diff_hat_x_i * hat_y_i * hat_z_i * diff_hat_x_j * hat_y_j * hat_z_j
                                            + hat_x_i * diff_hat_y_i * hat_z_i * hat_x_j * diff_hat_y_j * hat_z_j
                                            + hat_x_i * hat_y_i * diff_hat_z_i * hat_x_j * hat_y_j * diff_hat_z_j);
                                }
                            }
                        }
                    }
 
                }
            }

            for (int k = 0; k < NumberOfParticles; k++) {
                Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                if (!allElementsWithinOne(k_minus_xi)) continue;

                for (int l = 0; l < NumberOfParticles; l++) {
                    Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                    if (!allElementsWithinOne(l_minus_xi)) continue;

                    // phiÇÃåàíË
                    double Phi1 = phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1);
                    double Phi2 = -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l));
                    double Phi3 = phi(3 * k) * phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);
                    Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

                    // åvéZì_ÇÃåàíËÇ∆ì‡ë}ä÷êîÇÃåvéZ
                    for (int x = 0; x < kNum; x++) {
                        for (int y = 0; y < kNum; y++) {
                            for (int z = 0; z < kNum; z++) {
                                Eigen::Vector3d cal_point(cal_points(x), cal_points(y), cal_points(z));

                                // kä÷òAÇÃì‡ë}ä÷êîÇÃåvéZ
                                double hat_x_k = HatFunction(cal_point(0) - k_minus_xi(0));
                                double diff_hat_x_k = DifferentialHatFunction(cal_point(0) - k_minus_xi(0));
                                double hat_y_k = HatFunction(cal_point(1) - k_minus_xi(1));
                                double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - k_minus_xi(1));
                                double hat_z_k = HatFunction(cal_point(2) - k_minus_xi(2));

                                // lä÷òAÇÃì‡ë}ä÷êîÇÃåvéZ
                                double hat_x_l = HatFunction(cal_point(0) - l_minus_xi(0));
                                double hat_y_l = HatFunction(cal_point(1) - l_minus_xi(1));
                                double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - l_minus_xi(1));
                                double hat_z_l = HatFunction(cal_point(2) - l_minus_xi(2)); // èCê≥: cal_point(1) Ç©ÇÁ cal_point(2)
                                double diff_hat_z_l = DifferentialHatFunction(cal_point(2) - l_minus_xi(2));

                                // xiä÷òAÇÃì‡ë}ä÷êîÇÃåvéZ
                                double hat_x_xi = HatFunction(cal_point(0));
                                double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
                                double hat_y_xi = HatFunction(cal_point(1));
                                double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
                                double hat_z_xi = HatFunction(cal_point(2));
                                double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));

                                // äeçÄÇÃåvéZ
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
                                    WeightKLXi(3 * xi + p) +=  Phi(p) *
                                        (term1 * term2 * term3
                                            - term4 * term5 * term6
                                            + term7 * term8 * term9);
                                }

                            }
                        }
                    }
                    
                }

            }

            // ëÃêœïœâªó¶JÇÃåvéZ
            J = RiemannSumJ(phi, grid_xi, square.dx, cal_points, (-5.0 / 3.0));

        }

        // HessianUpsilon1 += mu * pow(J, (-5/3)) * WeightIJ * WeightKLXi.transpose() ÇÃåvéZ
        for (int xi = 0; xi < NumberOfParticles; xi++) {
            for (int i = 0; i < NumberOfParticles; i++) {

                for (int col = 0; col < dimensions; col++) { // óÒêî
                    for (int row = 0; row < dimensions; row++) { // çsêî
                        HessianUpsilon1(3 * i + row, 3 * xi + col) += mu * J * WeightIJ(i, 3 * xi + col) * WeightKLXi(3 * xi + row) * pow(kWidth, 3);
                    }
                }
                
            }
        }

        /*------  éÆåvéZèIóπ   ------*/
    }

    

    return HessianUpsilon1;
}


Eigen::MatrixXd calHessianUpsilon2(const Square& square, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd HessianUpsilon2 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // äeãÊä‘ÇÃï™äÑêî
    const double kWidth = square.dx / kNumSection; // ï™äÑÇÃê≥ãKâª
    const int kNum = 4 * kNumSection; // ëSãÊä‘ÇÃï™äÑêî

    Eigen::VectorXd cal_points(kNum);
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(divIndex) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
        }
    }

    // ãÊä‘ï™äÑ
    for (int d = 0; d < kNum; d++) {

        Eigen::MatrixXd WeightSum = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);
        Eigen::VectorXd WeightIJK = Eigen::VectorXd::Zero(NumberOfParticles, 3 * NumberOfParticles);
        Eigen::VectorXd WeightLM = Eigen::VectorXd::Zero(NumberOfParticles);
        Eigen::VectorXd WeightNOXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);
        double J = 0.0; // ëÃêœïœâªó¶ÇÃèâä˙âª

        /*------  éÆåvéZ   ------*/
        for (int xi = 0; xi < NumberOfParticles; xi++) {
            Eigen::Vector3i grid_xi = FlatToGrid(xi);

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                for (int j = 0; j < NumberOfParticles; j++) {
                    Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                    if (!allElementsWithinOne(j_minus_xi)) continue;

                    double phi_j_x = phi(3 * j);
                    double phi_j_y = phi(3 * j + 1);
                    double phi_j_z = phi(3 * j + 2);

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        double phi_k_x = phi(3 * k);
                        double phi_k_y = phi(3 * k + 1);
                        double phi_k_z = phi(3 * k + 2);

                        // phiÇÃåvéZ
                        double phi1 = phi_j_y * phi_k_z - phi_j_z * phi_k_y;
                        double phi2 = -(phi_j_x * phi_k_z - phi_j_z * phi_k_x);
                        double phi3 = phi_j_x * phi_k_y - phi_j_y * phi_k_x;
                        Eigen::Vector3d phi_jk = { phi1, phi2, phi3 };

                        // åvéZì_ÇÃåàíËÇ∆ì‡ë}ä÷êîÇÃåvéZ
                        for (int x = 0; x < kNum; x++) {
                            for (int y = 0; y < kNum; y++) {
                                for (int z = 0; z < kNum; z++) {
                                    Eigen::Vector3d cal_point(cal_points(x), cal_points(y), cal_points(z));

                                    // jä÷òAÇÃì‡ë}ä÷êîÇÃåvéZ
                                    double hat_x_j = HatFunction(cal_point(0) - j_minus_xi(0));
                                    double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - j_minus_xi(0));
                                    double hat_y_j = HatFunction(cal_point(1) - j_minus_xi(1));
                                    double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - j_minus_xi(1));
                                    double hat_z_j = HatFunction(cal_point(2) - j_minus_xi(2));

                                    // kä÷òAÇÃì‡ë}ä÷êîÇÃåvéZ
                                    double hat_x_k = HatFunction(cal_point(0) - k_minus_xi(0));
                                    double hat_y_k = HatFunction(cal_point(1) - k_minus_xi(1));
                                    double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - k_minus_xi(1));
                                    double hat_z_k = HatFunction(cal_point(2) - k_minus_xi(2));
                                    double diff_hat_z_k = DifferentialHatFunction(cal_point(2) - k_minus_xi(2));

                                    // iä÷òAÇÃì‡ë}ä÷êîÇÃåvéZ
                                    double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                                    double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - i_minus_xi(0));
                                    double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                                    double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - i_minus_xi(1));
                                    double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));
                                    double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - i_minus_xi(2));

                                    // äeçÄÇÃåvéZ
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
                                        WeightIJK(i, 3 * xi + p) += phi_jk(p) *
                                            (term1 * term2 * term3
                                                - term4 * term5 * term6
                                                + term7 * term8 * term9);
                                    }
                                }
                            }
                        }

                    }
                }
            }

            for (int l = 0; l < NumberOfParticles; l++) {
                Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                if (!allElementsWithinOne(l_minus_xi)) continue;

                double phi_l_x = phi(3 * l);
                double phi_l_y = phi(3 * l + 1);
                double phi_l_z = phi(3 * l + 2);

                for (int m = 0; m < NumberOfParticles; m++) {
                    Eigen::Vector3i m_minus_xi = FlatToGrid(m) - grid_xi;
                    if (!allElementsWithinOne(m_minus_xi)) continue;
                    
                    double phi_m_x = phi(3 * m);
                    double phi_m_y = phi(3 * m + 1);
                    double phi_m_z = phi(3 * m + 2);

                    // phiÇÃåvéZ
                    double phi_lm = phi_l_x * phi_m_x + phi_l_y * phi_m_y + phi_l_z * phi_m_z;

                    // åvéZì_ÇÃåàíËÇ∆ì‡ë}ä÷êîÇÃåvéZ
                    for (int x = 0; x < kNum; x++) {
                        for (int y = 0; y < kNum; y++) {
                            for (int z = 0; z < kNum; z++) {
                                Eigen::Vector3d cal_point(cal_points(x), cal_points(y), cal_points(z));

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

                                WeightLM(xi) += phi_lm * 
                                    (diff_hat_x_l * hat_y_l * hat_z_l * diff_hat_x_m * hat_y_m * hat_z_m
                                        + hat_x_l * diff_hat_y_l * hat_z_l * hat_x_m * diff_hat_y_m * hat_z_m
                                        + hat_x_l * hat_y_l * diff_hat_z_l * hat_x_m * hat_y_m * diff_hat_z_m);
                            }
                        }
                    }

                }
            }

            for (int n = 0; n < NumberOfParticles; n++) {
                Eigen::Vector3i n_minus_xi = FlatToGrid(n) - grid_xi;
                if (!allElementsWithinOne(n_minus_xi)) continue;

                double phi_n_x = phi(3 * n);
                double phi_n_y = phi(3 * n + 1);
                double phi_n_z = phi(3 * n + 2);

                for (int o = 0; o < NumberOfParticles; o++) {
                    Eigen::Vector3i o_minus_xi = FlatToGrid(o) - grid_xi;
                    if (!allElementsWithinOne(o_minus_xi)) continue;

                    double phi_o_x = phi(3 * o);
                    double phi_o_y = phi(3 * o + 1);
                    double phi_o_z = phi(3 * o + 2);

                    // phiÇÃåvéZ
                    double phi1 = phi_n_y * phi_o_z - phi_n_z * phi_o_y;
                    double phi2 = -(phi_n_x * phi_o_z - phi_n_z * phi_o_x);
                    double phi3 = phi_n_x * phi_o_y - phi_n_y * phi_o_x;
                    Eigen::Vector3d phi_no = { phi1, phi2, phi3 };

                    // åvéZì_ÇÃåàíËÇ∆ì‡ë}ä÷êîÇÃåvéZ
                    for (int x = 0; x < kNum; x++) {
                        for (int y = 0; y < kNum; y++) {
                            for (int z = 0; z < kNum; z++) {
                                Eigen::Vector3d cal_point(cal_points(x), cal_points(y), cal_points(z));

                                // nä÷òAÇÃì‡ë}ä÷êîÇÃåvéZ
                                double hat_x_n = HatFunction(cal_point(0) - n_minus_xi(0));
                                double diff_hat_x_n = DifferentialHatFunction(cal_point(0) - n_minus_xi(0));
                                double hat_y_n = HatFunction(cal_point(1) - n_minus_xi(1));
                                double diff_hat_y_n = DifferentialHatFunction(cal_point(1) - n_minus_xi(1));
                                double hat_z_n = HatFunction(cal_point(2) - n_minus_xi(2));

                                // oä÷òAÇÃì‡ë}ä÷êîÇÃåvéZ
                                double hat_x_o = HatFunction(cal_point(0) - o_minus_xi(0));
                                double hat_y_o = HatFunction(cal_point(1) - o_minus_xi(1));
                                double diff_hat_y_o = DifferentialHatFunction(cal_point(1) - o_minus_xi(1));
                                double hat_z_o = HatFunction(cal_point(2) - o_minus_xi(2)); // èCê≥: cal_point(1) Ç©ÇÁ cal_point(2)
                                double diff_hat_z_o = DifferentialHatFunction(cal_point(2) - o_minus_xi(2));

                                // xiä÷òAÇÃì‡ë}ä÷êîÇÃåvéZ
                                double hat_x_xi = HatFunction(cal_point(0));
                                double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
                                double hat_y_xi = HatFunction(cal_point(1));
                                double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
                                double hat_z_xi = HatFunction(cal_point(2));
                                double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));

                                // äeçÄÇÃåvéZ
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
                                    WeightNOXi(3 * xi + p) += phi_no(p) *
                                        (term1 * term2 * term3
                                            - term4 * term5 * term6
                                            + term7 * term8 * term9);
                                }
                            }
                        }
                    }

                }
            }

            // ëÃêœïœâªó¶JÇÃåvéZ
            J = RiemannSumJ(phi, grid_xi, square.dx, cal_points, (-8.0 / 3.0));

        }

        // HessianUpsilon2 += mu * pow(J, (-5/3)) * WeightIJ * WeightKLXi.transpose() ÇÃåvéZ
        for (int xi = 0; xi < NumberOfParticles; xi++) {
            for (int i = 0; i < NumberOfParticles; i++) {

                for (int col = 0; col < dimensions; col++) { // óÒêî
                    for (int row = 0; row < dimensions; row++) { // çsêî
                        HessianUpsilon2(3 * i + row, 3 * xi + col) += mu * J * WeightIJK(i, 3 * xi + col) * WeightNOXi(3 * xi + row) * WeightLM(xi) * pow(kWidth, 3);
                    }
                }

            }
        }

        /*------  éÆåvéZèIóπ   ------*/
    }

    return HessianUpsilon2;
}

/*

Eigen::MatrixXd calHessianUpsilon3(const Square& square, const Eigen::VectorXd& phi, const Eigen::VectorXd& power)
{
    Eigen::MatrixXd HessianUpsilon3 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    auto processIndices = [&](const std::vector<int>& indices) {
        int xi = indices[0], n = indices[1], m = indices[2], l = indices[3], k = indices[4], j = indices[5], i = indices[6];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        std::vector<Eigen::Vector3i> diff_vectors = {
            FlatToGrid(i) - grid_xi,
            FlatToGrid(j) - grid_xi,
            FlatToGrid(k) - grid_xi,
            FlatToGrid(l) - grid_xi,
            FlatToGrid(m) - grid_xi,
            FlatToGrid(n) - grid_xi
        };

        if (std::all_of(diff_vectors.begin(), diff_vectors.end(), allElementsWithinOne)) {
            Eigen::Matrix<int, 3, 6> matrix;
            for (int col = 0; col < 6; ++col) {
                matrix.col(col) = diff_vectors[col];
            }

            auto calculateW = [&](const Eigen::VectorXi& axis) {
                return RiemannSum7(matrix, axis, square.dx);
                };

            std::vector<Eigen::VectorXi> axes = {
                (Eigen::VectorXi(6) << 1,2,0,1,2,0).finished(),
                (Eigen::VectorXi(6) << 1,2,0,0,2,1).finished(),
                (Eigen::VectorXi(6) << 1,2,0,0,1,2).finished(),
                (Eigen::VectorXi(6) << 0,2,1,1,2,0).finished(),
                (Eigen::VectorXi(6) << 0,2,1,0,2,1).finished(),
                (Eigen::VectorXi(6) << 0,2,1,0,1,2).finished(),
                (Eigen::VectorXi(6) << 0,1,2,1,2,0).finished(),
                (Eigen::VectorXi(6) << 0,1,2,0,2,1).finished(),
                (Eigen::VectorXi(6) << 0,1,2,0,1,2).finished()
            };

            std::vector<double> W_values;
            for (const auto& axis : axes) {
                W_values.push_back(calculateW(axis));
            }

            double J = RiemannSumForDetF(phi, grid_xi, square.dx);

            double W = W_values[0] - W_values[1] + W_values[2] - W_values[3] + W_values[4] - W_values[5] + W_values[6] - W_values[7] + W_values[8];

            Eigen::Vector3d VectorPhi1;
            VectorPhi1 <<
                phi(3 * m + 1)* phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n + 1),
                -(phi(3 * m) * phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n)),
                phi(3 * m)* phi(3 * n + 1) - phi(3 * m + 1) * phi(3 * n);

            Eigen::Vector3d VectorPhi2;
            VectorPhi2 <<
                phi(3 * j + 1)* phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1),
                -(phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k)),
                phi(3 * j)* phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);

            Eigen::Matrix3d PhiMatrix = VectorPhi1 * VectorPhi2.transpose();

            double factor = power[i] * W / J;
            HessianUpsilon3.block<3, 3>(3 * l, 3 * xi) += factor * PhiMatrix;
        }
        };

    std::vector<int> indices(7);
    recursiveLoopForHesUpsilon(0, 7, processIndices, indices);

    return HessianUpsilon3;
}
*/