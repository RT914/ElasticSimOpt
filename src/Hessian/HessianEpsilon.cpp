#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"

// Caluculate HessianEpsilon
Eigen::MatrixXd calHessianEpsilon(const Square& square, const Eigen::VectorXd& re_phi) {
    Eigen::MatrixXd HessianEpsilon = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // 各区間の分割数
    const double kWidth = square.dx / kNumSection; // 分割の正規化
    const int kNum = 2 * kNumSection; // 全区間の分割数
    const int AllkNum = pow(kNum, 3);// 全次元の全区間分割数
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -1; offset <= 0; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // 内挿関数の計算
    // 区間分割
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;
        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        // Stencil Baseの計算
        Eigen::Vector3d stencil_base = calculateStencilBase(cal_point);

        // Stencil行列とstencil_numの生成
        Eigen::MatrixXi stencil;
        std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

        for (int xi = 0; xi < NumberOfParticles; xi++) {
            if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
            Eigen::Vector3i grid_xi = FlatToGrid(xi);

            Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

            // xi関連の内挿関数の計算
            double hat_x_xi = HatFunction(cal_point(0) - grid_point_coordinates_xi(0));
            double hat_y_xi = HatFunction(cal_point(1) - grid_point_coordinates_xi(1));
            double hat_z_xi = HatFunction(cal_point(2) - grid_point_coordinates_xi(2));

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                // i関連の内挿関数の計算
                double hat_x_i = HatFunction(cal_point(0) - grid_point_coordinates_i(0));
                double hat_y_i = HatFunction(cal_point(1) - grid_point_coordinates_i(1));
                double hat_z_i = HatFunction(cal_point(2) - grid_point_coordinates_i(2));

                // 各項の計算
                double w_i = hat_x_i * hat_y_i * hat_z_i;
                double w_xi = hat_x_xi * hat_y_xi * hat_z_xi;

                double WeightIXi = w_i * w_xi;

                // 単位行列
                Eigen::Matrix3d identityMatrix = Eigen::Matrix3d::Identity();

                // HessianEpsilon += I(単位行列) * rho / (dt * dt) *  WeightIXi の計算
                for (int col = 0; col < dimensions; col++) { // 列数
                    for (int row = 0; row < dimensions; row++) { // 行数
                        double term = rho / pow(dt, 2) * identityMatrix(row, col) * WeightIXi * volume_element;
                        if (abs(term) < 1e-10) continue;
                        HessianEpsilon(3 * i + row, 3 * xi + col) += term;
                    }
                }

            }
        }

    }

    return HessianEpsilon;
}
