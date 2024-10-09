#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Hessian/HessianR.h"


// Calculate Hessian R
Eigen::MatrixXd calHessianR(const Square& square, const Eigen::VectorXd& theta){
    Eigen::MatrixXd HessianR = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 4 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
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

                Eigen::MatrixXd WeightIXi = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_xi = HatFunction(cal_point(0));
                    double hat_y_xi = HatFunction(cal_point(1));
                    double hat_z_xi = HatFunction(cal_point(2));

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        // “à‘}ŠÖ”‚ÌŒvZ
                        // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                        double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                        double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));

                        // Še€‚ÌŒvZ
                        double term1 = hat_x_i * hat_y_i * hat_z_i;
                        double term2 = hat_x_xi * hat_y_xi * hat_z_xi;

                        double WeightIXi = term1 * term2;

                        double WeightJ = 0.0;

                        for (int j = 0; j < NumberOfParticles; j++) {
                            Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                            if (!allElementsWithinOne(j_minus_xi)) continue;

                            // “à‘}ŠÖ”‚ÌŒvZ
                            // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_j = HatFunction(cal_point(0) - j_minus_xi(0));
                            double hat_y_j = HatFunction(cal_point(1) - j_minus_xi(1));
                            double hat_z_j = HatFunction(cal_point(2) - j_minus_xi(2));

                            // Še€‚ÌŒvZ
                            double term = hat_x_j * hat_y_j * hat_z_j;

                            WeightJ += theta(j) * term;
                        }

                        // HessianR += (kappa / 2) * (1 + 1 / pow(WeightJ, 2)) * WeightIXi ‚ÌŒvZ
                        if (WeightJ < 1e-6) {
                            HessianR(i, xi) -= (kappa / 2) * WeightIXi * volume_element;
                        }
                        else {
                            HessianR(i, xi) -= (kappa / 2) * (1 + 1 / pow(WeightJ, 2)) * WeightIXi * volume_element;
                        }

                    }
                }

            }
        }
    }

    return HessianR;
}
