#include <iostream>
#include <omp.h>
#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Hessian/HessianUpsilon.h"


// Caluculate HessianXi
Eigen::MatrixXd calHessianUpsilon(Square square, Eigen::VectorXd phi, Eigen::VectorXd power) {
    Eigen::MatrixXd HessianUpsilon(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianUpsilon.setZero();
    Eigen::MatrixXd HessianUpsilon1 = calHessianUpsilon1(square, phi);
    // Eigen::MatrixXd HessianUpsilon2 = calHessianUpsilon2(square, phi);
    // Eigen::MatrixXd HessianUpsilon3 = calHessianUpsilon3(square, phi, power);
    // HessianUpsilon = HessianUpsilon1 + HessianUpsilon2 + HessianUpsilon3;
    HessianUpsilon = HessianUpsilon1;

    return HessianUpsilon;
}

Eigen::MatrixXd calHessianUpsilon1(const Square& square, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianUpsilon1 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumDivisions = 3; // äeãÊä‘ÇÃï™äÑêî
    const double kDivisionWidth = 1.0 / kNumDivisions; // ï™äÑÇÃê≥ãKâª

    Eigen::VectorXd cal_points(kNumDivisions);
    for (int divIndex = 0; divIndex < kNumDivisions; divIndex++) {
        cal_points(divIndex) = 1.0 / (2.0 * kNumDivisions) + divIndex * kDivisionWidth;
    }

    Eigen::MatrixXd AllAxisWeight = Eigen::MatrixXd::Ones(NumberOfParticles, NumberOfParticles);

    // axisIndex = éüå≥êî
    for (int axisIndex = 0; axisIndex < dimensions; axisIndex++) {
        Eigen::MatrixXd WeightSum = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

        // ãÊä‘ï™äÑ
        for (int offset = -2; offset <= 1; offset++) {
            Eigen::MatrixXd Weight = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);
            
            // äeãÊä‘Ç≈ÇÃï™äÑóÃàÊåvéZ
            for (int d = 0; d < cal_points.size(); d++) {
                double cal = static_cast<double>(offset) + cal_points(d);
                Eigen::VectorXd WeightIJ = Eigen::VectorXd::Zero(3 * NumberOfParticles);
                Eigen::VectorXd WeightKLXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);

                /*------  éÆåvéZ   ------*/
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);
                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (allElementsWithinOne(i_minus_xi)) {
                            for (int j = 0; j < NumberOfParticles; j++) {
                                Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                                if (allElementsWithinOne(j_minus_xi)) {

                                    // weightIndex = 0, 1, 2
                                    for (int weightIndex = 0; weightIndex < 3; weightIndex++) {
                                        // ì‡ë}ä÷êîW_{i=weightIndex}{j=weightIndex}ÇÃåvéZ
                                        if (axisIndex == weightIndex) {
                                            for (int p = 0; p < dimensions; p++) {
                                                WeightIJ(3 * i + p) += phi(3 * j + p)
                                                    * DifferentialHatFunction(cal - i_minus_xi(axisIndex))
                                                    * DifferentialHatFunction(cal - j_minus_xi(axisIndex));
                                            }
                                        }
                                        else {
                                            for (int p = 0; p < dimensions; p++) {
                                                WeightIJ(3 * i + p) += phi(3 * j + p)
                                                    * HatFunction(cal - i_minus_xi(axisIndex))
                                                    * HatFunction(cal - j_minus_xi(axisIndex));
                                            }
                                        }
                                    }

                                }

                            }
                        }
                    }


                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (allElementsWithinOne(k_minus_xi)) {
                            for (int l = 0; l < NumberOfParticles; l++) {
                                Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                                if (allElementsWithinOne(l_minus_xi)) {

                                    double Phi1 = phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1);
                                    double Phi2 = - (phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l));
                                    double Phi3 = phi(3 * k) * phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);
                                    Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

                                    // ì‡ë}ä÷êîW_{k=2}{l=3}{xi=1}ÇÃåvéZ
                                    if (axisIndex == 0) {
                                        for (int p = 0; p < dimensions; p++) {
                                            WeightKLXi(3 * xi + p) += Phi(p)
                                                * HatFunction(cal - k_minus_xi(axisIndex))
                                                * HatFunction(cal - l_minus_xi(axisIndex))
                                                * DifferentialHatFunction(cal);
                                        }
                                    }
                                    else if(axisIndex == 1){
                                        for (int p = 0; p < dimensions; p++) {
                                            WeightKLXi(3 * xi + p) += Phi(p)
                                                * DifferentialHatFunction(cal - k_minus_xi(axisIndex))
                                                * HatFunction(cal - l_minus_xi(axisIndex))
                                                * HatFunction(cal);
                                        }
                                    }
                                    else if (axisIndex == 2) {
                                        for (int p = 0; p < dimensions; p++) {
                                            WeightKLXi(3 * xi + p) += Phi(p)
                                                * HatFunction(cal - k_minus_xi(axisIndex))
                                                * DifferentialHatFunction(cal - l_minus_xi(axisIndex))
                                                * HatFunction(cal);
                                        }
                                    }

                                    // ì‡ë}ä÷êîW_{k=1}{l=3}{xi=2}ÇÃåvéZ
                                    if (axisIndex == 0) {
                                        for (int p = 0; p < dimensions; p++) {
                                            WeightKLXi(3 * xi + p) -= Phi(p)
                                                * DifferentialHatFunction(cal - k_minus_xi(axisIndex))
                                                * HatFunction(cal - l_minus_xi(axisIndex))
                                                * HatFunction(cal);
                                        }
                                    }
                                    else if (axisIndex == 1) {
                                        for (int p = 0; p < dimensions; p++) {
                                            WeightKLXi(3 * xi + p) -= Phi(p)
                                                * HatFunction(cal - k_minus_xi(axisIndex))
                                                * HatFunction(cal - l_minus_xi(axisIndex))
                                                * DifferentialHatFunction(cal);;
                                        }
                                    }
                                    else if (axisIndex == 2) {
                                        for (int p = 0; p < dimensions; p++) {
                                            WeightKLXi(3 * xi + p) -= Phi(p)
                                                * HatFunction(cal - k_minus_xi(axisIndex))
                                                * DifferentialHatFunction(cal - l_minus_xi(axisIndex))
                                                * HatFunction(cal);
                                        }
                                    }

                                    // ì‡ë}ä÷êîW_{k=1}{l=2}{xi=3}ÇÃåvéZ
                                    if (axisIndex == 0) {
                                        for (int p = 0; p < dimensions; p++) {
                                            WeightKLXi(3 * xi + p) += Phi(p)
                                                * DifferentialHatFunction(cal - k_minus_xi(axisIndex))
                                                * HatFunction(cal - l_minus_xi(axisIndex))
                                                * HatFunction(cal);
                                        }
                                    }
                                    else if (axisIndex == 1) {
                                        for (int p = 0; p < dimensions; p++) {
                                            WeightKLXi(3 * xi + p) += Phi(p)
                                                * HatFunction(cal - k_minus_xi(axisIndex))
                                                * DifferentialHatFunction(cal - l_minus_xi(axisIndex))
                                                * HatFunction(cal);
                                        }
                                    }
                                    else if (axisIndex == 2) {
                                        for (int p = 0; p < dimensions; p++) {
                                            WeightKLXi(3 * xi + p) += Phi(p)
                                                * HatFunction(cal - k_minus_xi(axisIndex))
                                                * HatFunction(cal - l_minus_xi(axisIndex))
                                                * DifferentialHatFunction(cal);
                                        }
                                    }

                                }
                            }
                        }

                    }

                }
                
                for (int xi = 0; xi < 3 * NumberOfParticles; xi++) {
                    for (int i = 0; i < 3 * NumberOfParticles; i++) {
                        Weight(i, xi) = WeightIJ(i) * WeightKLXi(xi);
                    }
                }

                WeightSum += Weight;
                /*------  éÆåvéZèIóπ   ------*/
            }
            
        }

        // äeóvëfÇ≤Ç∆ÇÃêœÇåvéZ
        AllAxisWeight = AllAxisWeight.cwiseProduct(WeightSum);

        std::cout << "finish axis : " << axisIndex << std::endl;
    }

    HessianUpsilon1 = AllAxisWeight;
    return HessianUpsilon1;
}

/*
Eigen::MatrixXd calHessianUpsilon2(const Square& square, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd HessianUpsilon2 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    auto processIndices = [&](const std::vector<int>& indices) {
        int xi = indices[0], o = indices[1], n = indices[2], m = indices[3], l = indices[4], k = indices[5], j = indices[6], i = indices[7];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        std::vector<Eigen::Vector3i> diff_vectors = {
            FlatToGrid(i) - grid_xi,
            FlatToGrid(j) - grid_xi,
            FlatToGrid(k) - grid_xi,
            FlatToGrid(l) - grid_xi,
            FlatToGrid(m) - grid_xi,
            FlatToGrid(n) - grid_xi,
            FlatToGrid(o) - grid_xi
        };

        if (std::all_of(diff_vectors.begin(), diff_vectors.end(), allElementsWithinOne)) {
            Eigen::Matrix<int, 3, 7> matrix;
            for (int col = 0; col < 7; ++col) {
                matrix.col(col) = diff_vectors[col];
            }

            auto calculateW = [&](const Eigen::VectorXi& axis) {
                return RiemannSum6(matrix, axis, square.dx);
                };

            std::vector<Eigen::VectorXi> axes = {
                (Eigen::VectorXi(8) << 0,0,1,2,0,1,2,0).finished(),
                (Eigen::VectorXi(8) << 0,0,1,2,0,0,2,1).finished(),
                (Eigen::VectorXi(8) << 0,0,1,2,0,0,1,2).finished(),
                (Eigen::VectorXi(8) << 0,0,0,2,1,1,2,0).finished(),
                (Eigen::VectorXi(8) << 0,0,0,2,1,0,2,1).finished(),
                (Eigen::VectorXi(8) << 0,0,0,2,1,0,1,2).finished(),
                (Eigen::VectorXi(8) << 0,0,0,1,2,1,2,0).finished(),
                (Eigen::VectorXi(8) << 0,0,0,1,2,0,2,1).finished(),
                (Eigen::VectorXi(8) << 0,0,0,1,2,0,1,2).finished(),
                (Eigen::VectorXi(8) << 1,1,1,2,0,1,2,0).finished(),
                (Eigen::VectorXi(8) << 1,1,1,2,0,0,2,1).finished(),
                (Eigen::VectorXi(8) << 1,1,1,2,0,0,1,2).finished(),
                (Eigen::VectorXi(8) << 1,1,0,2,1,1,2,0).finished(),
                (Eigen::VectorXi(8) << 1,1,0,2,1,0,2,1).finished(),
                (Eigen::VectorXi(8) << 1,1,0,2,1,0,1,2).finished(),
                (Eigen::VectorXi(8) << 1,1,0,1,2,1,2,0).finished(),
                (Eigen::VectorXi(8) << 1,1,0,1,2,0,2,1).finished(),
                (Eigen::VectorXi(8) << 1,1,0,1,2,0,1,2).finished(),
                (Eigen::VectorXi(8) << 2,2,1,2,0,1,2,0).finished(),
                (Eigen::VectorXi(8) << 2,2,1,2,0,0,2,1).finished(),
                (Eigen::VectorXi(8) << 2,2,1,2,0,0,1,2).finished(),
                (Eigen::VectorXi(8) << 2,2,0,2,1,1,2,0).finished(),
                (Eigen::VectorXi(8) << 2,2,0,2,1,0,2,1).finished(),
                (Eigen::VectorXi(8) << 2,2,0,2,1,0,1,2).finished(),
                (Eigen::VectorXi(8) << 2,2,0,1,2,1,2,0).finished(),
                (Eigen::VectorXi(8) << 2,2,0,1,2,0,2,1).finished(),
                (Eigen::VectorXi(8) << 2,2,0,1,2,0,1,2).finished()
            };

            std::vector<double> W_values;
            for (const auto& axis : axes) {
                W_values.push_back(calculateW(axis));
            }

            double J = RiemannSumForDetF(phi, grid_xi, square.dx);

            double W1 = W_values[0] - W_values[1] + W_values[2] - W_values[3] + W_values[4] - W_values[5] + W_values[6] - W_values[7] + W_values[8];
            double W2 = W_values[9] - W_values[10] + W_values[11] - W_values[12] + W_values[13] - W_values[14] + W_values[15] - W_values[16] + W_values[17];
            double W3 = W_values[18] - W_values[19] + W_values[20] - W_values[21] + W_values[22] - W_values[23] + W_values[24] - W_values[25] + W_values[8];  // Note: W_values[8] is used here instead of W_values[26]
            double W = W1 + W2 + W3;

            Eigen::Vector3d VectorPhi1;
            VectorPhi1 <<
                phi(3 * n + 1)* phi(3 * o + 2) - phi(3 * n + 2) * phi(3 * o + 1),
                -(phi(3 * n) * phi(3 * o + 2) - phi(3 * n + 2) * phi(3 * o)),
                phi(3 * n)* phi(3 * o + 1) - phi(3 * n + 1) * phi(3 * o);

            Eigen::Vector3d VectorPhi2;
            VectorPhi2 <<
                phi(3 * k + 1)* phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1),
                -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l)),
                phi(3 * k)* phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);

            double Phi = phi.segment<3>(3 * i).dot(phi.segment<3>(3 * j));

            Eigen::Matrix3d PhiMatrix = VectorPhi1 * VectorPhi2.transpose() * Phi;

            double factor = (1.0 / 3.0) * mu * std::pow(J, -8.0 / 3.0) * W;
            HessianUpsilon2.block<3, 3>(3 * m, 3 * xi) += factor * PhiMatrix;
        }
        };

    std::vector<int> indices(8);
    recursiveLoopForHesUpsilon(0, 8, processIndices, indices);

    return HessianUpsilon2;
}

Eigen::MatrixXd calHessianUpsilon3(const Square& square, const Eigen::VectorXd& phi, const Eigen::VectorXd& power)
{
    Eigen::MatrixXd HessianUpsilon3 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    auto processIndices = [&](const std::vector<int>& indices) {
        int xi = indices[0], n = indices[1], m = indices[2], l = indices[3], k = indices[4], j = indices[5], i = indices[6];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        std::vector<Eigen::Vector3i> diff_vectors = {
            FlatToGrid(i) - grid_xi,
            FlatToGrid(j) - grid_xi,
            FlatToGrid(k) - grid_xi,
            FlatToGrid(l) - grid_xi,
            FlatToGrid(m) - grid_xi,
            FlatToGrid(n) - grid_xi
        };

        if (std::all_of(diff_vectors.begin(), diff_vectors.end(), allElementsWithinOne)) {
            Eigen::Matrix<int, 3, 6> matrix;
            for (int col = 0; col < 6; ++col) {
                matrix.col(col) = diff_vectors[col];
            }

            auto calculateW = [&](const Eigen::VectorXi& axis) {
                return RiemannSum7(matrix, axis, square.dx);
                };

            std::vector<Eigen::VectorXi> axes = {
                (Eigen::VectorXi(6) << 1,2,0,1,2,0).finished(),
                (Eigen::VectorXi(6) << 1,2,0,0,2,1).finished(),
                (Eigen::VectorXi(6) << 1,2,0,0,1,2).finished(),
                (Eigen::VectorXi(6) << 0,2,1,1,2,0).finished(),
                (Eigen::VectorXi(6) << 0,2,1,0,2,1).finished(),
                (Eigen::VectorXi(6) << 0,2,1,0,1,2).finished(),
                (Eigen::VectorXi(6) << 0,1,2,1,2,0).finished(),
                (Eigen::VectorXi(6) << 0,1,2,0,2,1).finished(),
                (Eigen::VectorXi(6) << 0,1,2,0,1,2).finished()
            };

            std::vector<double> W_values;
            for (const auto& axis : axes) {
                W_values.push_back(calculateW(axis));
            }

            double J = RiemannSumForDetF(phi, grid_xi, square.dx);

            double W = W_values[0] - W_values[1] + W_values[2] - W_values[3] + W_values[4] - W_values[5] + W_values[6] - W_values[7] + W_values[8];

            Eigen::Vector3d VectorPhi1;
            VectorPhi1 <<
                phi(3 * m + 1)* phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n + 1),
                -(phi(3 * m) * phi(3 * n + 2) - phi(3 * m + 2) * phi(3 * n)),
                phi(3 * m)* phi(3 * n + 1) - phi(3 * m + 1) * phi(3 * n);

            Eigen::Vector3d VectorPhi2;
            VectorPhi2 <<
                phi(3 * j + 1)* phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k + 1),
                -(phi(3 * j) * phi(3 * k + 2) - phi(3 * j + 2) * phi(3 * k)),
                phi(3 * j)* phi(3 * k + 1) - phi(3 * j + 1) * phi(3 * k);

            Eigen::Matrix3d PhiMatrix = VectorPhi1 * VectorPhi2.transpose();

            double factor = power[i] * W / J;
            HessianUpsilon3.block<3, 3>(3 * l, 3 * xi) += factor * PhiMatrix;
        }
        };

    std::vector<int> indices(7);
    recursiveLoopForHesUpsilon(0, 7, processIndices, indices);

    return HessianUpsilon3;
}
*/