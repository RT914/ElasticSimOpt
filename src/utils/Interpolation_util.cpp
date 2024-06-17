#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <stdio.h>
#include "../../include/FEM.h"


double HatFunction(double x)
{
	double x_num = (abs(x) < 1.0) ? (1.0 - abs(x)) : 0.0;

	return x_num;
}

double DifferentialHatFunction(double x)
{
	if (x >= -1.0 && x < 0.0) {
		return 1.0;
	}
	else if (x <= 1.0 && x > 0.0) {
		return -1.0;
	}
	else {
		return 0.0;
	}
}



int GridToFlat(Eigen::Vector3i grid_index)
{
	int flat_index;
	int grid_x = grid_index[0];
	int grid_y = grid_index[1];
	int grid_z = grid_index[2];

	flat_index = grid_x * int(pow(NumberOfOneDemensionParticles, 2)) + grid_y * NumberOfOneDemensionParticles + grid_z;

	return flat_index;
};

Eigen::Vector3i FlatToGrid(int flat_index)
{
	Eigen::Vector3i grid_index = {};

	grid_index[0] = flat_index / int(pow(NumberOfOneDemensionParticles, 2));
	grid_index[1] = (flat_index % int(pow(NumberOfOneDemensionParticles, 2))) / NumberOfOneDemensionParticles;
	grid_index[2] = ((flat_index % int(pow(NumberOfOneDemensionParticles, 2))) % NumberOfOneDemensionParticles);

	return grid_index;
};


double RiemannSum1(const Eigen::Vector3i& v, double h) {
    const int kNumDivisions = 3; // ひと範囲の分割数
    const double kDivisionWidth = 1.0 / kNumDivisions; // 分割幅
    std::vector<double> cal_points(kNumDivisions);

    for (int a = 0; a < kNumDivisions; a++) {
        cal_points[a] = 1.0 / (2.0 * kNumDivisions) + a * kDivisionWidth;
    }

    double totalSum = 1.0;

    for (int axisIndex = 0; axisIndex < dimensions; axisIndex++) {
        double axisSum = 0.0;

        for (int offset = -2; offset <= 1; offset++) {
            double sum = 0.0;

            for (double cal_point : cal_points) {
                double cal = static_cast<double>(offset) + cal_point;
                double interior = HatFunction(cal - v[axisIndex]);

                // ゼロ除算を避けるためのチェック
                if (std::abs(interior) >= 1e-10) {
                    sum += kDivisionWidth * h * interior * HatFunction(cal);
                }
            }

            axisSum += sum;
        }

        totalSum *= axisSum;
    }

    return totalSum;
}


double RiemannSum2(const Eigen::Matrix3i& m, const Eigen::Vector3i& axis, double h) {
    // m << x1, y1, z1, x2, y2, z2, x3, y3, z3 としたい．
    const int kNumDivisions = 3;
    const double kDivisionWidth = 1.0 / kNumDivisions;

    Eigen::VectorXd cal_points(kNumDivisions);
    for (int divIndex = 0; divIndex < kNumDivisions; divIndex++) {
        cal_points(divIndex) = 1.0 / (2.0 * kNumDivisions) + divIndex * kDivisionWidth;
    }

    double totalSum = 1.0;
    // axisIndex = 軸数
    for (int axisIndex = 0; axisIndex < dimensions; axisIndex++) {
        double axisSum = 0.0;

        // 区間分割
        for (int offset = -2; offset <= 1; offset++) {
            double sum = 0.0;

            // 各区間での分割領域計算
            for (int d = 0; d < cal_points.size(); d++) {
                double cal = static_cast<double>(offset) + cal_points(d);
                double interior = 1.0;
                if (axisIndex == axis(0)) {
                    interior *= (1 / h) * DifferentialHatFunction(cal - m(0, axisIndex)) * HatFunction(cal - m(1, axisIndex)) * HatFunction(cal - m(2, axisIndex));
                }
                else if (axisIndex == axis(1)) {
                    interior *= (1 / h) * HatFunction(cal - m(0, axisIndex)) * DifferentialHatFunction(cal - m(1, axisIndex)) * HatFunction(cal - m(2, axisIndex));
                }
                else if (axisIndex == axis(2)) {
                    interior *= (1 / h) * HatFunction(cal - m(0, axisIndex)) * HatFunction(cal - m(1, axisIndex)) * DifferentialHatFunction(cal - m(2, axisIndex));
                }
                else {
                    interior = 0.0;
                }

                if (std::abs(interior) > 1e-10) {
                    sum += kDivisionWidth * h * interior * HatFunction(cal);
                }
                
            }
            // std::cout << sum << std::endl;
            axisSum += sum;
        }
        // std::cout << axisSum << std::endl;
        totalSum *= abs(axisSum);
    }
    return totalSum;
}


double RiemannSum3(const Eigen::Vector3i& v, const Eigen::Vector3i& grid_xi, const Eigen::Vector3d& theta, double h) {
    // m << x1, y1, z1, x2, y2, z2, x3, y3, z3 としたい．
    const int kNumDivisions = 3;
    const double kDivisionWidth = 1.0 / kNumDivisions;

    Eigen::VectorXd cal_points(kNumDivisions);
    for (int divIndex = 0; divIndex < kNumDivisions; divIndex++) {
        cal_points(divIndex) = 1.0 / (2.0 * kNumDivisions) + divIndex * kDivisionWidth;
    }

    double totalSum = 1.0;

    // axisIndex = 軸数
    for (int axisIndex = 0; axisIndex < dimensions; axisIndex++) {
        double axisSum = 0.0;
        for (int offset = -2; offset <= 1; offset++) {
            double sum = 0.0;

            for (int d = 0; d < kNumDivisions; d++) {
                double cal = static_cast<double>(offset) + cal_points(d);
                double interior = HatFunction(cal - v[axisIndex]);

                // ゼロ除算を避けるためのチェック
                if (std::abs(interior) >= 1e-10) {

                    // 1/\Thetaの計算
                    double sum_j = 0.0;
                    for (int j = 0; j < NumberOfParticles; j++) {
                        Eigen::Vector3i grid_j = FlatToGrid(j);
                        Eigen::Vector3i j_minus_xi = grid_j - grid_xi;

                        // axisIndex = 軸数
                        double totalSum_deno = 1.0;
                        for (int axisIndex_deno = 0; axisIndex_deno < dimensions; axisIndex_deno++) {
                            double axisSum_deno = 0.0;

                            for (int offset_deno = -2; offset_deno <= 1; offset_deno++) {
                                double sum_deno = 0.0;

                                for (int d_deno = 0; d_deno < kNumDivisions; d_deno++) {
                                    double cal_deno = static_cast<double>(offset_deno) + cal_points(d_deno);
                                    double denominator = HatFunction(cal - j_minus_xi[axisIndex]);

                                    // ゼロ除算を避けるためのチェック
                                    if (std::abs(denominator) >= 1e-10) {
                                        sum_deno += kDivisionWidth * h * denominator;
                                    }
                                }

                                axisSum_deno += sum_deno;
                            }
                            totalSum_deno *= axisSum_deno;
                        }

                        sum_j += theta[j] * totalSum_deno;
                    }

                    sum += kDivisionWidth * h * interior * HatFunction(cal) / pow(sum_j, 2);
                }
            }

            axisSum += sum;
        }
        totalSum *= axisSum;
    }

    return totalSum;
}
