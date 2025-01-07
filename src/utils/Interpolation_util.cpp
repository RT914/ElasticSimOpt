#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <stdio.h>
#include "../../include/FEM.h"
#include "../../include/Square.h"
#include "../../include/utils/Interpolation_util.h"



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

// Stencil Baseの計算関数
Eigen::Vector3d calculateStencilBase(const Eigen::Vector3d& cal_point, const double& distance) {
    // 負の方向への丸め込み（格子の始点が原点のため補正）
    return Eigen::Vector3d(
        std::floor((cal_point(0) + 1) / distance),
        std::floor((cal_point(1) + 1) / distance),
        std::floor((cal_point(2) + 1) / distance)
    );
}

// Stencil行列とstencil_numの生成関数
std::vector<int> generateStencil(const Eigen::Vector3d& stencil_base, Eigen::MatrixXi& stencil) {
    // Stencilの初期化（3x8の行列として設定）
    stencil.resize(3, 8);
    stencil.col(0) << stencil_base(0), stencil_base(1), stencil_base(2);
    stencil.col(1) << stencil_base(0) + 1, stencil_base(1), stencil_base(2);
    stencil.col(2) << stencil_base(0), stencil_base(1) + 1, stencil_base(2);
    stencil.col(3) << stencil_base(0), stencil_base(1), stencil_base(2) + 1;
    stencil.col(4) << stencil_base(0) + 1, stencil_base(1) + 1, stencil_base(2);
    stencil.col(5) << stencil_base(0), stencil_base(1) + 1, stencil_base(2) + 1;
    stencil.col(6) << stencil_base(0) + 1, stencil_base(1), stencil_base(2) + 1;
    stencil.col(7) << stencil_base(0) + 1, stencil_base(1) + 1, stencil_base(2) + 1;

    // stencil_num の計算
    std::vector<int> stencil_num(8);
    for (int s = 0; s < 8; s++) {
        stencil_num[s] = GridToFlat(stencil.col(s));
    }
    return stencil_num;
}

double calRiemannJ(const Eigen::Vector3d& cal_point, const Eigen::Vector3i& grid_xi, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const int NumberOfParticles, const double& distance, const double exp)
{
    double f_ijk = 0.0;

    for (int k = 0; k < NumberOfParticles; k++) {
        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
        if (!allElementsWithinOne(k_minus_xi)) continue;

        Eigen::Vector3d grid_point_coordinates_k = { re_phi(3 * k), re_phi(3 * k + 1), re_phi(3 * k + 2) };

        // k関連の内挿関数の計算
        double f_k_3 = HatFunction((cal_point(0) - grid_point_coordinates_k(0)) / distance) *
            HatFunction((cal_point(1) - grid_point_coordinates_k(1)) / distance) *
            DifferentialHatFunction((cal_point(2) - grid_point_coordinates_k(2)) / distance);

        for (int j = 0; j < NumberOfParticles; j++) {
            Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
            if (!allElementsWithinOne(j_minus_xi)) continue;

            Eigen::Vector3d grid_point_coordinates_j = { re_phi(3 * j), re_phi(3 * j + 1), re_phi(3 * j + 2) };

            // j関連の内挿関数の計算
            double f_j_2 = HatFunction((cal_point(0) - grid_point_coordinates_j(0)) / distance) *
                DifferentialHatFunction((cal_point(1) - grid_point_coordinates_j(1)) / distance) *
                HatFunction((cal_point(2) - grid_point_coordinates_j(2)) / distance);

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                // i関連の内挿関数の計算
                double f_i_1 = DifferentialHatFunction((cal_point(0) - grid_point_coordinates_i(0)) / distance) *
                    HatFunction((cal_point(1) - grid_point_coordinates_i(1)) / distance) *
                    HatFunction((cal_point(2) - grid_point_coordinates_i(2)) / distance);

                double Phi0 = phi(3 * i) * (phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1))
                    + phi(3 * i + 1) * (phi(3 * j + 2) * phi(3 * k) - phi(3 * j) * phi(3 * k + 2))
                    + phi(3 * i + 2) * (phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k));

                f_ijk += Phi0 * f_i_1 * f_j_2 * f_k_3 * (1.0 / pow(distance, 3));
            }

        }
        
    }

    if (abs(f_ijk) < 1e-6) {
        return 0.0;
    }
    else {
        return pow(f_ijk, exp);
    }
}

bool allElementsWithinOne(const Eigen::Vector3i& vec) {
    return (vec.cwiseAbs().array() <= 1).all();
}
