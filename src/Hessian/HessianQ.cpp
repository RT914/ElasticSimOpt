#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/Hessian/HessianQ.h"


// Calculate Hessian Q
Eigen::MatrixXd calHessianQ(const Square& square){
	Eigen::MatrixXd HessianQ(NumberOfParticles, NumberOfParticles);

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

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_xi = HatFunction(cal_point(0));
                    double hat_y_xi = HatFunction(cal_point(1));
                    double hat_z_xi = HatFunction(cal_point(2));

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        double WeightIXi = 0.0;

                        // “à‘}ŠÖ”‚ÌŒvZ
                        // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                        double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                        double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));

                        // Še€‚ÌŒvZ
                        double term1 = hat_x_i * hat_y_i * hat_z_i;
                        double term2 = hat_x_xi * hat_y_xi * hat_z_xi;

                        WeightIXi = term1 * term2;

                        // HessianQ -= WeightIXi ‚ÌŒvZ
                        HessianQ(i, xi) -= WeightIXi * volume_element;

                    }
                }

            }
        }
    }

    return HessianQ;
}
