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

double RiemannSumJ(const Eigen::VectorXd& phi, const Eigen::Vector3i& grid_xi, double h, Eigen::VectorXd cal_points, double exp) {
    double VolumeChangeRate = 0.0;
    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = h / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 4 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”


    for (int k = 0; k < NumberOfParticles; k++) {
        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
        if (!allElementsWithinOne(k_minus_xi)) continue;
        
        for (int j = 0; j < NumberOfParticles; j++) {
            Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
            if (!allElementsWithinOne(j_minus_xi)) continue;

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                // Še phi ‚ÌÏ‚ğŒvZ‚µ‚ÄŒ‹‰Ê‚ğ‡¬
                
                double Phi
                    = phi(3 * i) * (phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1))
                    + phi(3 * i + 1) * (phi(3 * j + 2) * phi(3 * k) - phi(3 * j) * phi(3 * k + 2))
                    + phi(3 * i + 2) * (phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k));

                for (int x = 0; x < kNum; x++) {
                    for (int y = 0; y < kNum; y++) {
                        for (int z = 0; z < kNum; z++) {
                            Eigen::Vector3d cal_point(cal_points(x), cal_points(y), cal_points(z));

                            // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - i_minus_xi(0));
                            double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                            double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));

                            // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_j = HatFunction(cal_point(0) - j_minus_xi(0));
                            double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - j_minus_xi(1));
                            double hat_z_j = HatFunction(cal_point(2) - j_minus_xi(2));
                         
                            // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_k = HatFunction(cal_point(0) - k_minus_xi(0));
                            double hat_y_k = HatFunction(cal_point(1) - k_minus_xi(1));
                            double diff_hat_z_k = DifferentialHatFunction(cal_point(2) - k_minus_xi(2));

                            // Še€‚ÌŒvZ
                            double term1 = diff_hat_x_i * hat_y_i * hat_z_i;
                            double term2 = hat_x_j * diff_hat_y_j * hat_z_j;
                            double term3 = hat_x_k * hat_y_k * diff_hat_z_k;

                            double term = Phi * term1 * term2 * term3 * pow(kWidth, 3);
                            
                            if (abs(term) > 1e-10) {
                                // std::cout << term << std::endl;
                                VolumeChangeRate += pow(term, exp);
                            }
                            
                        }
                    }
                }

            }
        }
        
    }

    return VolumeChangeRate;
}

double calRiemannJ(const Eigen::Vector3d& cal_point, const Eigen::Vector3i& grid_xi, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, int NumberOfParticles, double exp)
{
    double f_ijk = 0.0;

    for (int k = 0; k < NumberOfParticles; k++) {
        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
        if (!allElementsWithinOne(k_minus_xi)) continue;

        Eigen::Vector3d P_k = { re_phi(3 * k), re_phi(3 * k + 1), re_phi(3 * k + 2) };

        // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
        double f_k_3 = HatFunction(cal_point(0) - P_k(0)) *
            HatFunction(cal_point(1) - P_k(1)) *
            DifferentialHatFunction(cal_point(2) - P_k(2));

        for (int j = 0; j < NumberOfParticles; j++) {
            Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
            if (!allElementsWithinOne(j_minus_xi)) continue;

            Eigen::Vector3d P_j = { re_phi(3 * j), re_phi(3 * j + 1), re_phi(3 * j + 2) };

            // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
            double f_j_2 = HatFunction(cal_point(0) - P_j(0)) *
                DifferentialHatFunction(cal_point(1) - P_j(1)) *
                HatFunction(cal_point(2) - P_j(2));

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                Eigen::Vector3d P_i = { re_phi(3 * i), re_phi(3 * i + 1), re_phi(3 * i + 2) };

                // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                double f_i_1 = DifferentialHatFunction(cal_point(0) - P_i(0)) *
                    HatFunction(cal_point(1) - P_i(1)) *
                    HatFunction(cal_point(2) - P_i(2));

                double Phi0 = phi(3 * i) * (phi(3 * j + 1) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1))
                    + phi(3 * i + 1) * (phi(3 * j + 2) * phi(3 * k) - phi(3 * j) * phi(3 * k + 2))
                    + phi(3 * i + 2) * (phi(3 * j) * phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k));

                f_ijk += Phi0 * f_i_1 * f_j_2 * f_k_3;
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
