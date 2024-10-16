#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"

// Caluculate HessianEpsilon
Eigen::MatrixXd calHessianEpsilon(const Square& square) {
    Eigen::MatrixXd HessianEpsilon = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // 各区間の分割数
    const double kWidth = square.dx / kNumSection; // 分割の正規化
    const int kNum = 4 * kNumSection; // 全区間の分割数

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // 内挿関数の計算
    // 区間分割
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                Eigen::MatrixXd WeightIXi = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);

                /*------  式計算   ------*/
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        // 内挿関数の計算
                        // i関連の内挿関数の計算
                        double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                        double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                        double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));

                        // xi関連の内挿関数の計算
                        double hat_x_xi = HatFunction(cal_point(0));
                        double hat_y_xi = HatFunction(cal_point(1));
                        double hat_z_xi = HatFunction(cal_point(2));

                        // 各項の計算
                        double term1 = hat_x_i * hat_y_i * hat_z_i;
                        double term2 = hat_x_xi * hat_y_xi * hat_z_xi;

                        WeightIXi(i, xi) += term1 * term2;
                    }

                }

                // HessianEpsilon += I(単位行列) * rho / (dt * dt) *  WeightIXi の計算
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    for (int i = 0; i < NumberOfParticles; i++) {

                        for (int col = 0; col < dimensions; col++) { // 列数
                            for (int row = 0; row < dimensions; row++) { // 行数
                                // 対角成分にのみ値を挿入
                                if (row != col) continue;
                                HessianEpsilon(3 * i + row, 3 * xi + col)
                                    = rho / (dt * dt) * WeightIXi(i, xi) * pow(kWidth, 3);
                            }
                        }

                    }
                }

                /*------  式計算終了   ------*/
            }
        }
    }

    return HessianEpsilon;
}
