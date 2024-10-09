#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"

// Caluculate HessianEpsilon
Eigen::MatrixXd calHessianEpsilon(const Square& square) {
    Eigen::MatrixXd HessianEpsilon = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // �e��Ԃ̕�����
    const double kWidth = square.dx / kNumSection; // �����̐��K��
    const int kNum = 4 * kNumSection; // �S��Ԃ̕�����

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ���}�֐��̌v�Z
    // ��ԕ���
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                Eigen::MatrixXd WeightIXi = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);

                /*------  ���v�Z   ------*/
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        // ���}�֐��̌v�Z
                        // i�֘A�̓��}�֐��̌v�Z
                        double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                        double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                        double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));

                        // xi�֘A�̓��}�֐��̌v�Z
                        double hat_x_xi = HatFunction(cal_point(0));
                        double hat_y_xi = HatFunction(cal_point(1));
                        double hat_z_xi = HatFunction(cal_point(2));

                        // �e���̌v�Z
                        double term1 = hat_x_i * hat_y_i * hat_z_i;
                        double term2 = hat_x_xi * hat_y_xi * hat_z_xi;

                        WeightIXi(i, xi) += term1 * term2;
                    }

                }

                // HessianEpsilon += I(�P�ʍs��) * rho / (dt * dt) *  WeightIXi �̌v�Z
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    for (int i = 0; i < NumberOfParticles; i++) {

                        for (int col = 0; col < dimensions; col++) { // ��
                            for (int row = 0; row < dimensions; row++) { // �s��
                                // �Ίp�����ɂ̂ݒl��}��
                                if (row != col) continue;
                                HessianEpsilon(3 * i + row, 3 * xi + col)
                                    = rho / (dt * dt) * WeightIXi(i, xi) * pow(kWidth, 3);
                            }
                        }

                    }
                }

                /*------  ���v�Z�I��   ------*/
            }
        }
    }

    return HessianEpsilon;
}
