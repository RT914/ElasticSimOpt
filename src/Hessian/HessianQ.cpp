#include <Eigen/Dense>
#include <iostream>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/Hessian/HessianQ.h"


// Calculate Hessian Q
Eigen::MatrixXd calHessianQ(const Square& square, const Eigen::VectorXd& re_phi){
    Eigen::MatrixXd HessianQ = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);

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

                // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                double hat_x_tau = HatFunction(cal_point(0) - grid_point_coordinates_tau(0));
                double hat_y_tau = HatFunction(cal_point(1) - grid_point_coordinates_tau(1));
                double hat_z_tau = HatFunction(cal_point(2) - grid_point_coordinates_tau(2));

                // Še€‚ÌŒvZ
                double w_tau = hat_x_tau * hat_y_tau * hat_z_tau;
                double w_xi = hat_x_xi * hat_y_xi * hat_z_xi;

                double WeightTauXi = w_tau * w_xi;

                double term = WeightTauXi * volume_element;
                if (abs(term) > 1e-10) {
                    HessianQ(xi, tau) -= term;
                }

            }
        }

    }

    return HessianQ;
}
