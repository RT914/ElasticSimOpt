#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/Hessian/HessianO.h"


// Calculate Hessian O
Eigen::MatrixXd calHessianO(Square square, Eigen::VectorXd phi){
    Eigen::MatrixXd HessianO = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

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
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
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

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                Eigen::MatrixXd WeightI = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);
                Eigen::VectorXd WeightJKXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                        double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                        double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));

                        WeightI(i, xi) = hat_x_i * hat_y_i * hat_z_i;
                    }

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

                            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_xi = HatFunction(cal_point(0));
                            double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
                            double hat_y_xi = HatFunction(cal_point(1));
                            double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
                            double hat_z_xi = HatFunction(cal_point(2));
                            double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));

                            // Še€‚ÌŒvZ
                            double term1 = hat_x_j * diff_hat_y_j * hat_z_j;
                            double term2 = hat_x_k * hat_y_k * diff_hat_z_k;
                            double term3 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                            double term4 = diff_hat_x_j * hat_y_j * hat_z_j;
                            double term5 = hat_x_k * hat_y_k * diff_hat_z_k;
                            double term6 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                            double term7 = diff_hat_x_j * hat_y_j * hat_z_j;
                            double term8 = hat_x_k * diff_hat_y_k * hat_z_k;
                            double term9 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                            for (int p = 0; p < dimensions; p++) {
                                WeightJKXi(3 * xi + p) += Phi_JK(j, 3 * k + p) *
                                    (term1 * term2 * term3
                                        - term4 * term5 * term6
                                        + term7 * term8 * term9);
                            }

                        }
                    }

                }

                // HessianO += Phi_JK * WeightI * WeightJKXi.transpose() ‚ÌŒvZ
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    for (int i = 0; i < NumberOfParticles; i++) {

                        for (int col = 0; col < dimensions; col++) { // —ñ”
                            HessianO(i, 3 * xi + col) 
                                += WeightI(i, xi) * WeightJKXi(3 * xi + col) * pow(kWidth, 3);
                        }

                    }
                }


            }
        }
    }

    return (-1) * HessianO;
}
