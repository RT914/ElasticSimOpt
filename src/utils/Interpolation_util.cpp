#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <stdio.h>
#include "../../include/FEM.h"
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

// ww
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

// 1/(w^2) * w
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

// 1/w * w
double RiemannSum4(const Eigen::Vector3i& grid_xi, const Eigen::Vector3d& theta, double h) {
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

                // 1/\Thetaの計算
                double sum_i = 0.0;
                for (int i = 0; i < NumberOfParticles; i++) {
                    Eigen::Vector3i grid_i = FlatToGrid(i);
                    Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

                    // axisIndex = 軸数
                    double totalSum_deno = 1.0;
                    for (int axisIndex_deno = 0; axisIndex_deno < dimensions; axisIndex_deno++) {
                        double axisSum_deno = 0.0;

                        for (int offset_deno = -2; offset_deno <= 1; offset_deno++) {
                            double sum_deno = 0.0;

                            for (int d_deno = 0; d_deno < kNumDivisions; d_deno++) {
                                double cal_deno = static_cast<double>(offset_deno) + cal_points(d_deno);
                                double denominator = HatFunction(cal - i_minus_xi[axisIndex]);

                                // ゼロ除算を避けるためのチェック
                                if (std::abs(denominator) >= 1e-10) {
                                    sum_deno += kDivisionWidth * h * denominator;
                                }
                            }

                            axisSum_deno += sum_deno;
                        }
                        totalSum_deno *= axisSum_deno;
                    }

                    sum_i += theta[i] * totalSum_deno;
                }

                sum += kDivisionWidth * h * HatFunction(cal) / sum_i;
            }

            axisSum += sum;
        }
        totalSum *= axisSum;
    }

    return totalSum;
}

// WWWWWWWW(n)多分できた
// 格子間距離が1以外の処理が未完
double RiemannSum5(const Eigen::MatrixXi& m, const Eigen::VectorXi& axis, double h) {
    // m << x1, y1, z1, x2, y2, z2, x3, y3, z3 ... としたい．
    // axis << 0,0,1,2,0,0,2,1,...
    const int kNumDivisions = 3;
    const double kDivisionWidth = 1.0 / kNumDivisions;

    Eigen::VectorXd cal_points(kNumDivisions);
    for (int divIndex = 0; divIndex < kNumDivisions; divIndex++) {
        cal_points(divIndex) = 1.0 / (2.0 * kNumDivisions) + divIndex * kDivisionWidth;
    }

    double totalSum = 1.0;
    // axisIndex = 次元数
    for (int axisIndex = 0; axisIndex < dimensions; axisIndex++) {
        double axisSum = 0.0;

        // 区間分割
        for (int offset = -2; offset <= 1; offset++) {
            double sum = 0.0;

            // 各区間での分割領域計算
            for (int d = 0; d < cal_points.size(); d++) {
                double cal = static_cast<double>(offset) + cal_points(d);
                double interior = 1.0;

                for (int axis_vector_num = 0; axis_vector_num < axis.size(); axis_vector_num++) {
                    if (axisIndex == axis[axis_vector_num]) {
                        interior *= (1 / h) * DifferentialHatFunction(cal - m(axisIndex, axis_vector_num));
                    }
                    else {
                        interior *= HatFunction(cal - m(axisIndex, axis_vector_num));
                    }
                }

                if (std::abs(interior) > 1e-10) { 
                    sum += kDivisionWidth * h * interior;
                }
            }
            axisSum += sum;
        }
        // std::cout << axisSum << std::endl;
        totalSum *= axisSum;
    }
    return totalSum;
}

// WWW...W(n)w
double RiemannSum6(const Eigen::MatrixXi& m, const Eigen::VectorXi& axis, double h) {
    // m << x1, y1, z1, x2, y2, z2, x3, y3, z3, ..... , xn, yn, zn としたい．
    // axis << 0,0,1,... (n)
    const int kNumDivisions = 30;
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
                
                for (int axis_vector_num = 0; axis_vector_num < axis.size(); axis_vector_num++) {
                    if (axisIndex == axis[axis_vector_num]) {
                        interior *= (1 / h) * DifferentialHatFunction(cal - m(axisIndex, axis_vector_num));
                    }
                    else {
                        interior *= HatFunction(cal - m(axisIndex, axis_vector_num));
                    }
                }

                if (std::abs(interior) > 1e-10) {
                    sum += kDivisionWidth * h * interior * HatFunction(cal);
                }

            }
            axisSum += sum;
        }
        std::cout << axisSum << std::endl;
        totalSum *= abs(axisSum);
    }
    return totalSum;
}


// wWWW...W(n)
double RiemannSum7(const Eigen::MatrixXi& m, const Eigen::VectorXi& axis, double h) {
    // m << x1, y1, z1, x2, y2, z2, x3, y3, z3, ..... , xn, yn, zn としたい．
    // axis << 0,0,1,... (n)

    const int kNumDivisions = 30;
    const double kDivisionWidth = 1.0 / kNumDivisions;

    Eigen::VectorXd cal_points(kNumDivisions);
    for (int divIndex = 0; divIndex < kNumDivisions; divIndex++) {
        cal_points(divIndex) = 1.0 / (2.0 * kNumDivisions) + divIndex * kDivisionWidth;
    }

    double totalSum = 1.0;
    // axisIndex = 軸数 XYZ
    for (int axisIndex = 0; axisIndex < dimensions; axisIndex++) {
        double axisSum = 0.0;

        // 区間分割
        for (int offset = -2; offset <= 1; offset++) {
            double sum = 0.0;

            // 各区間での分割領域計算
            for (int d = 0; d < cal_points.size(); d++) {
                double cal = static_cast<double>(offset) + cal_points(d);
                double interior = HatFunction(cal - m(axisIndex, 0));

                for (int axis_vector_num = 1; axis_vector_num <= axis.size(); axis_vector_num++) {
                    if (axis_vector_num < axis.size()) {
                        if (axisIndex == axis[axis_vector_num - 1]) {
                            interior *= (1 / h) * DifferentialHatFunction(cal - m(axisIndex, axis_vector_num));
                        }
                        else {
                            interior *= HatFunction(cal - m(axisIndex, axis_vector_num));
                        }
                    }
                    else {
                        if (axisIndex == axis[axis_vector_num - 1]) {
                            interior *= (1 / h) * DifferentialHatFunction(cal);
                        }
                        else {
                            interior *= HatFunction(cal);
                        }
                    }
                    
                }

                if (std::abs(interior) > 1e-10) {
                    sum += kDivisionWidth * h * interior;
                }

            }
            axisSum += sum;
        }
        std::cout << axisSum << std::endl;
        totalSum *= abs(axisSum);
    }
    return totalSum;
}


void recursiveLoopForDetF(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices) {
    if (level == maxLevel) {
        process(indices);
        return;
    }

    for (int i = 0; i < NumberOfParticles; i++) {
        indices[level] = i;
        recursiveLoopForDetF(level + 1, maxLevel, process, indices);
    }
}

double RiemannSumForDetF(const Eigen::VectorXd& phi, Eigen::Vector3i grid_xi, double h) {
    double VolumeChangeRate = 0.0;

    auto processIndices = [&](std::vector<int>& indices) {
        Eigen::Vector3i grid_k = FlatToGrid(indices[0]);
        Eigen::Vector3i k_minus_xi = grid_k - grid_xi;

        Eigen::Vector3i grid_j = FlatToGrid(indices[1]);
        Eigen::Vector3i j_minus_xi = grid_j - grid_xi;

        Eigen::Vector3i grid_i = FlatToGrid(indices[2]);
        Eigen::Vector3i i_minus_xi = grid_i - grid_xi;

        Eigen::Matrix3i matrix;
        matrix << i_minus_xi, j_minus_xi, k_minus_xi;

        Eigen::Vector3i axis(0, 1, 2); // 各次元の数から-1した値を挿入
        double w1w2w3 = RiemannSum5(matrix, axis, h);

        double Lphi1 = phi(3 * indices[1] + 1) * phi(3 * indices[0] + 2) - phi(3 * indices[1] + 2) * phi(3 * indices[0] + 1);
        double Lphi2 = phi(3 * indices[1]) * phi(3 * indices[0] + 2) - phi(3 * indices[1] + 2) * phi(3 * indices[0]);
        double Lphi3 = phi(3 * indices[1]) * phi(3 * indices[0] + 1) - phi(3 * indices[1] + 1) * phi(3 * indices[0]);
        double Lphi = phi(3 * indices[2]) * Lphi1 - phi(3 * indices[2] + 1) * Lphi2 + phi(3 * indices[2] + 2) * Lphi3;

        VolumeChangeRate += Lphi * w1w2w3;
        };

    std::vector<int> indices(3); // 3つのインデックス用のベクター
    recursiveLoopForDetF(0, 3, processIndices, indices); // 3重ループを再帰で実行

    return VolumeChangeRate;
}