#include <Eigen/Dense>
#include <iostream>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"

#include "../../include/Hessian/HessianXi.h"
#include "../../include/Hessian/HessianChi.h"
#include "../../include/Hessian/HessianUpsilon.h"
#include "../../include/Hessian/HessianN.h"

// Calculate Hessian N
Eigen::MatrixXd calHessianN(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power)
{
	Eigen::MatrixXd HessianN(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianXi(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianChi(3 * NumberOfParticles, 3 * NumberOfParticles);
	Eigen::MatrixXd HessianUpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);
    Eigen::MatrixXd HessianEpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);

	HessianXi = calHessianXi(square, re_phi, phi, power) * pow(dt, 2);
	exportMatrix_CSV(HessianXi, "csv/HessianXi.csv");
	if (HessianXi.array().isNaN().any()) {
		std::cerr << "NaN detected HessianXi" << std::endl;
	}

	HessianChi = calHessianChi(square, re_phi, phi) * pow(dt, 2);
	exportMatrix_CSV(HessianChi, "csv/HessianChi.csv");
	if (HessianChi.array().isNaN().any()) {
		std::cerr << "NaN detected HessianChi" << std::endl;
	}

	HessianUpsilon = calHessianUpsilon(square, re_phi, phi, power) * pow(dt, 2);
	exportMatrix_CSV(HessianUpsilon, "csv/HessianUpsilon.csv");
	if (HessianUpsilon.array().isNaN().any()) {
		std::cerr << "NaN detected HessianUpsilon" << std::endl;
	}

    HessianEpsilon = calHessianEpsilon(square, re_phi, phi);
    exportMatrix_CSV(HessianEpsilon, "csv/HessianEpsilon.csv");
    if (HessianUpsilon.array().isNaN().any()) {
        std::cerr << "NaN detected HessianEpsilon" << std::endl;
    }

	// HessianN = HessianXi - HessianChi + HessianUpsilon - HessianEpsilon;
    HessianN = HessianXi - HessianChi + HessianUpsilon;

	return HessianN;
}

// Calculate Hessian N
Eigen::MatrixXd calHessianEpsilon(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi)
{
	Eigen::MatrixXd HessianEpsilon =  Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);
	
    const int kNumSection = 3; // äeãÊä‘ÇÃï™äÑêî
    const double kWidth = square.dx / kNumSection; // ï™äÑÇÃê≥ãKâª
    const int kNum = square.SideNumber * kNumSection; // ëSãÊä‘ÇÃï™äÑêî
    const int AllkNum = pow(kNum, 3);// ëSéüå≥ÇÃëSãÊä‘ï™äÑêî
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = 0; offset < square.SideNumber; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = (static_cast<double>(offset) + 1.0 / (2.0 * kNumSection)) * square.dx + divIndex * kWidth - 1.0;
            index++;
        }
    }

    // ì‡ë}ä÷êîÇÃåvéZ
    // ãÊä‘ï™äÑ
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;
        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        // Stencil BaseÇÃåvéZ
        Eigen::Vector3d stencil_base = calculateStencilBase(cal_point, square.dx);

        // StencilçsóÒÇ∆stencil_numÇÃê∂ê¨
        Eigen::MatrixXi stencil;
        std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

        for (int xi = 0; xi < NumberOfParticles; xi++) {
            if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
            Eigen::Vector3i grid_xi = FlatToGrid(xi);
            Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

            // xiä÷òAÇÃì‡ë}ä÷êîÇÃåvéZ
            double hat_x_xi = HatFunction((cal_point(0) - grid_point_coordinates_xi(0)) / square.dx);
            double hat_y_xi = HatFunction((cal_point(1) - grid_point_coordinates_xi(1)) / square.dx);
            double hat_z_xi = HatFunction((cal_point(2) - grid_point_coordinates_xi(2)) / square.dx);

            for (int tau = 0; tau < NumberOfParticles; tau++) {
                Eigen::Vector3i tau_minus_xi = FlatToGrid(tau) - grid_xi;
                if (!allElementsWithinOne(tau_minus_xi)) continue;

                Eigen::Vector3d grid_point_coordinates_tau = { re_phi(3 * tau), re_phi(3 * tau + 1), re_phi(3 * tau + 2) };

                // iä÷òAÇÃì‡ë}ä÷êîÇÃåvéZ
                double hat_x_tau = HatFunction((cal_point(0) - grid_point_coordinates_tau(0)) / square.dx);
                double hat_y_tau = HatFunction((cal_point(1) - grid_point_coordinates_tau(1)) / square.dx);
                double hat_z_tau = HatFunction((cal_point(2) - grid_point_coordinates_tau(2)) / square.dx);

                // äeçÄÇÃåvéZ
                double w_tau = hat_x_tau * hat_y_tau * hat_z_tau;
                double w_xi = hat_x_xi * hat_y_xi * hat_z_xi;

                double WeightTauXi = w_tau * w_xi;

                for (int col = 0; col < dimensions; col++) { // óÒêî
                    for (int row = 0; row < dimensions; row++) { // çsêî
                        // double term = rho * WeightTauXi * (1 / pow(dt, 2)) * volume_element;
                        double term = rho * WeightTauXi * volume_element;
                        if (abs(term) > 1e-10) {
                            HessianEpsilon(3 * xi + row, 3 * tau + col) += term;
                        }
                    }
                }

            }
        }

    }

	return HessianEpsilon;
}
