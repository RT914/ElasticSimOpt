#include <Eigen/Dense>
#include <iostream>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/Hessian/HessianQ.h"


// Calculate Hessian Q
Eigen::MatrixXd calHessianQ(const Square& square, const Eigen::VectorXd& re_phi){
    Eigen::MatrixXd HessianQ = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);

    const int kNumSection = 3; // �e��Ԃ̕�����
    const double kWidth = square.dx / kNumSection; // �����̐��K��
    const int kNum = 2 * kNumSection; // �S��Ԃ̕�����
    const int AllkNum = pow(kNum, 3);// �S�����̑S��ԕ�����
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -1; offset <= 0; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ���}�֐��̌v�Z
    // ��ԕ���
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;
        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        // Stencil Base�̌v�Z
        Eigen::Vector3d stencil_base = calculateStencilBase(cal_point);

        // Stencil�s���stencil_num�̐���
        Eigen::MatrixXi stencil;
        std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

        for (int xi = 0; xi < NumberOfParticles; xi++) {
            if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
            Eigen::Vector3i grid_xi = FlatToGrid(xi);
            Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

            // xi�֘A�̓��}�֐��̌v�Z
            double hat_x_xi = HatFunction(cal_point(0) - grid_point_coordinates_xi(0));
            double hat_y_xi = HatFunction(cal_point(1) - grid_point_coordinates_xi(1));
            double hat_z_xi = HatFunction(cal_point(2) - grid_point_coordinates_xi(2));

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                // i�֘A�̓��}�֐��̌v�Z
                double hat_x_i = HatFunction(cal_point(0) - grid_point_coordinates_i(0));
                double hat_y_i = HatFunction(cal_point(1) - grid_point_coordinates_i(1));
                double hat_z_i = HatFunction(cal_point(2) - grid_point_coordinates_i(2));

                // �e���̌v�Z
                double w_i = hat_x_i * hat_y_i * hat_z_i;
                double w_xi = hat_x_xi * hat_y_xi * hat_z_xi;

                double WeightIXi = w_i * w_xi;

                // HessianQ -= WeightIXi �̌v�Z
                double term = WeightIXi * volume_element;
                if (abs(term) > 1e-10) {
                    HessianQ(i, xi) -= term;
                }

            }
        }


    }

    return HessianQ;
}
