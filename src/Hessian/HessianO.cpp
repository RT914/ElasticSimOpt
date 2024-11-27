#include <Eigen/Dense>
#include <iostream>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/Hessian/HessianO.h"


// Calculate Hessian O
Eigen::MatrixXd calHessianO(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd HessianO = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

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

    // �W���̏�����
    Eigen::MatrixXd Phi_JK = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // �W���̌v�Z
    // ���ݍ��Wphi�̌v�Z
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
            double diff_hat_x_xi = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_xi(0));
            double hat_y_xi = HatFunction(cal_point(1) - grid_point_coordinates_xi(1));
            double diff_hat_y_xi = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_xi(1));
            double hat_z_xi = HatFunction(cal_point(2) - grid_point_coordinates_xi(2));
            double diff_hat_z_xi = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_xi(2));

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                // i�֘A�̓��}�֐��̌v�Z
                double hat_x_i = HatFunction(cal_point(0) - grid_point_coordinates_i(0));
                double hat_y_i = HatFunction(cal_point(1) - grid_point_coordinates_i(1));
                double hat_z_i = HatFunction(cal_point(2) - grid_point_coordinates_i(2));

                double WeightI = hat_x_i * hat_y_i * hat_z_i;

                Eigen::Vector3d WeightJKXi = Eigen::Vector3d::Zero();

                for (int j = 0; j < NumberOfParticles; j++) {
                    Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                    if (!allElementsWithinOne(j_minus_xi)) continue;

                    Eigen::Vector3d grid_point_coordinates_j = { re_phi(3 * j), re_phi(3 * j + 1), re_phi(3 * j + 2) };

                    // j�֘A�̓��}�֐��̌v�Z
                    double hat_x_j = HatFunction(cal_point(0) - grid_point_coordinates_j(0));
                    double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - grid_point_coordinates_j(0));
                    double hat_y_j = HatFunction(cal_point(1) - grid_point_coordinates_j(1));
                    double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_j(1));
                    double hat_z_j = HatFunction(cal_point(2) - grid_point_coordinates_j(2));

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        Eigen::Vector3d grid_point_coordinates_k = { re_phi(3 * k), re_phi(3 * k + 1), re_phi(3 * k + 2) };

                        // k�֘A�̓��}�֐��̌v�Z
                        double hat_x_k = HatFunction(cal_point(0) - grid_point_coordinates_k(0));
                        double hat_y_k = HatFunction(cal_point(1) - grid_point_coordinates_k(1));
                        double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - grid_point_coordinates_k(1));
                        double hat_z_k = HatFunction(cal_point(2) - grid_point_coordinates_k(2));
                        double diff_hat_z_k = DifferentialHatFunction(cal_point(2) - grid_point_coordinates_k(2));

                        // �e���̌v�Z
                        double w_j_2 = hat_x_j * diff_hat_y_j * hat_z_j;
                        double w_k_3 = hat_x_k * hat_y_k * diff_hat_z_k;
                        double w_xi_1 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                        double w_j_1 = diff_hat_x_j * hat_y_j * hat_z_j;
                        double w_xi_2 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                        double w_k_2 = hat_x_k * diff_hat_y_k * hat_z_k;
                        double w_xi_3 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int p = 0; p < dimensions; p++) {

                            WeightJKXi(p) += Phi_JK(j, 3 * k + p)
                                * (w_j_2 * w_k_3 * w_xi_1 - w_j_1 * w_k_3 * w_xi_2 + w_j_1 * w_k_2 * w_xi_3);

                        }

                    }
                }

                for (int col = 0; col < dimensions; col++) { // ��
                    double term = WeightI * WeightJKXi(col) * volume_element;
                    if (abs(term) < 1e-10) continue;
                    HessianO(i, 3 * xi + col) += term;
                }

            }
        }

    }

    return HessianO;
}

