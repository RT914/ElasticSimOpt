#include <iostream>
#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Hessian/HessianChi.h"

Eigen::MatrixXd calHessianChi(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
	Eigen::MatrixXd HessianChi(3 * NumberOfParticles, 3 * NumberOfParticles);
    HessianChi.setZero();
    // Eigen::MatrixXd HessianChi1 = calHessianChi1(square, re_phi, phi);
    Eigen::MatrixXd HessianChi2 = calHessianChi2(square, re_phi, phi);
    // Eigen::MatrixXd HessianChi3 = calHessianChi3(square, re_phi, phi);

    // HessianChi = HessianChi1 + HessianChi2 + HessianChi3;

    return HessianChi2;
}

Eigen::MatrixXd calHessianChi1(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianChi1 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 4 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const int AllkNum = pow(kNum, 3);// ‘SŸŒ³‚Ì‘S‹æŠÔ•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;

        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        /*------  ®ŒvZ   ------*/
        for (int xi = 0; xi < NumberOfParticles; xi++) {
            Eigen::Vector3i grid_xi = FlatToGrid(xi);

            double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
            double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
            double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));
            double hat_x_xi = HatFunction(cal_point(0));
            double hat_y_xi = HatFunction(cal_point(1));
            double hat_z_xi = HatFunction(cal_point(2));

            // ‘ÌÏ•Ï‰»—¦‚ÌŒvZ
            double detF = calRiemannJ(cal_point, grid_xi, re_phi, phi, NumberOfParticles, -2.0/3.0);

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                double WeightIXi = 0.0;

                // “à‘}ŠÖ”‚ÌŒvZ
                double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - i_minus_xi(0));
                double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - i_minus_xi(1));
                double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - i_minus_xi(2));
                double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));

                WeightIXi
                    += diff_hat_x_i * hat_y_i * hat_z_i * diff_hat_x_xi * hat_y_xi * hat_z_xi
                    + hat_x_i * diff_hat_y_i * hat_z_i * hat_x_xi * diff_hat_y_xi * hat_z_xi
                    + hat_x_i * hat_y_i * diff_hat_z_i * hat_x_xi * hat_y_xi * diff_hat_z_xi;

                for (int col = 0; col < dimensions; col++) { // —ñ”i‰¡‚Ì”j
                    for (int row = 0; row < dimensions; row++) { // s”ic‚Ì”j
                        // ‘ÎŠp¬•ª‚É‚Ì‚İ’l‚ğ‘}“ü
                        // ŒvZ•s‰Â‚Å‚ ‚é‚½‚ßCdetF‚Í–¢g—p
                        if (row != col) continue;
                        HessianChi1(3 * i + row, 3 * xi + col)
                            += mu * WeightIXi * volume_element;
                    }
                }

            }
        }

        /*------  ®ŒvZI—¹   ------*/

    }

    return HessianChi1;
}

Eigen::MatrixXd calHessianChi2(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianChi2 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 4 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”
    const int AllkNum = pow(kNum, 3);// ‘SŸŒ³‚Ì‘S‹æŠÔ•ªŠ„”
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::MatrixXd Phi_KL = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int k = 0; k < NumberOfParticles; k++) {
        for (int l = 0; l < NumberOfParticles; l++) {
            double Phi1 = phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1);
            double Phi2 = -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l));
            double Phi3 = phi(3 * k) * phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_KL(3 + k + p, l) = Phi(p);
            }
        }
    }

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int d = 0; d < AllkNum; d++) {
        int xd = d / (kNum * kNum);
        int yd = (d / kNum) % kNum;
        int zd = d % kNum;

        Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

        
        Eigen::VectorXd WeightKLXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);

        /*------  ®ŒvZ   ------*/
        for (int xi = 0; xi < NumberOfParticles; xi++) {
            Eigen::Vector3i grid_xi = FlatToGrid(xi);

            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
            double hat_x_xi = HatFunction(cal_point(0));
            double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
            double hat_y_xi = HatFunction(cal_point(1));
            double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
            double hat_z_xi = HatFunction(cal_point(2));
            double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));

            for (int i = 0; i < NumberOfParticles; i++) {
                Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                if (!allElementsWithinOne(i_minus_xi)) continue;

                // iŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - i_minus_xi(0));
                double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - i_minus_xi(1));
                double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - i_minus_xi(2));
                double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));

                Eigen::Vector3d WeightIJ = Eigen::Vector3d::Zero();

                for (int j = 0; j < NumberOfParticles; j++) {
                    Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                    if (!allElementsWithinOne(j_minus_xi)) continue;
                    
                    // jŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - j_minus_xi(0));
                    double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - j_minus_xi(1));
                    double diff_hat_z_j = DifferentialHatFunction(cal_point(2) - j_minus_xi(2));
                    double hat_x_j = HatFunction(cal_point(0) - j_minus_xi(0));
                    double hat_y_j = HatFunction(cal_point(1) - j_minus_xi(1));
                    double hat_z_j = HatFunction(cal_point(2) - j_minus_xi(2));

                    for (int p = 0; p < dimensions; p++) {
                        WeightIJ(p) += phi(3 * j + p) *
                            (diff_hat_x_i * hat_y_i * hat_z_i * diff_hat_x_j * hat_y_j * hat_z_j
                                + hat_x_i * diff_hat_y_i * hat_z_i * hat_x_j * diff_hat_y_j * hat_z_j
                                + hat_x_i * hat_y_i * diff_hat_z_i * hat_x_j * hat_y_j * diff_hat_z_j);
                    }

                }

                for (int k = 0; k < NumberOfParticles; k++) {
                    Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                    if (!allElementsWithinOne(k_minus_xi)) continue;

                    // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                    double hat_x_k = HatFunction(cal_point(0) - k_minus_xi(0));
                    double diff_hat_x_k = DifferentialHatFunction(cal_point(0) - k_minus_xi(0));
                    double hat_y_k = HatFunction(cal_point(1) - k_minus_xi(1));
                    double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - k_minus_xi(1));
                    double hat_z_k = HatFunction(cal_point(2) - k_minus_xi(2));

                    for (int l = 0; l < NumberOfParticles; l++) {
                        Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                        if (!allElementsWithinOne(l_minus_xi)) continue;

                        // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                        double hat_x_l = HatFunction(cal_point(0) - l_minus_xi(0));
                        double hat_y_l = HatFunction(cal_point(1) - l_minus_xi(1));
                        double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - l_minus_xi(1));
                        double hat_z_l = HatFunction(cal_point(2) - l_minus_xi(2));
                        double diff_hat_z_l = DifferentialHatFunction(cal_point(2) - l_minus_xi(2));

                        // Še€‚ÌŒvZ
                        double term1 = hat_x_k * diff_hat_y_k * hat_z_k;
                        double term2 = hat_x_l * hat_y_l * diff_hat_z_l;
                        double term3 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                        double term4 = diff_hat_x_k * hat_y_k * hat_z_k;
                        double term5 = hat_x_l * hat_y_l * diff_hat_z_l;
                        double term6 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                        double term7 = diff_hat_x_k * hat_y_k * hat_z_k;
                        double term8 = hat_x_l * diff_hat_y_l * hat_z_l;
                        double term9 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                        for (int p = 0; p < dimensions; p++) {
                            WeightKLXi(3 * xi + p) += Phi_KL(k, 3 * l + p) *
                                (term1 * term2 * term3
                                    - term4 * term5 * term6
                                    + term7 * term8 * term9);
                        }

                    }
                }

                for (int col = 0; col < dimensions; col++) { // —ñ”i‰¡‚Ì”j
                    for (int row = 0; row < dimensions; row++) { // s”ic‚Ì”j
                        HessianChi2(3 * i + row, 3 * xi + col)
                            += mu * WeightIJ(i, 3 * xi + col) * WeightKLXi(3 * xi + row) * pow(kWidth, 3);
                    }
                }

            }
        }

        /*------  ®ŒvZI—¹   ------*/

    }

    return HessianChi2;
}

Eigen::MatrixXd calHessianChi3(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi) {
    Eigen::MatrixXd HessianChi3 = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    const int kNumSection = 3; // Še‹æŠÔ‚Ì•ªŠ„”
    const double kWidth = square.dx / kNumSection; // •ªŠ„‚Ì³‹K‰»
    const int kNum = 4 * kNumSection; // ‘S‹æŠÔ‚Ì•ªŠ„”

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -2; offset <= 1; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }

    // ŒW”‚Ì‰Šú‰»
    Eigen::VectorXd J = Eigen::VectorXd::Zero(NumberOfParticles);
    Eigen::MatrixXd Phi_KL = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);

    // ŒW”‚ÌŒvZ
    // ‘ÌÏ•Ï‰»—¦J‚ÌŒvZ
    for (int xi = 0; xi < NumberOfParticles; xi++) {
        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        J(xi) = RiemannSumJ(phi, grid_xi, square.dx, cal_points, (-5.0 / 3.0));
    }

    // Œ»İÀ•Wphi‚ÌŒvZ
    for (int k = 0; k < NumberOfParticles; k++) {
        for (int l = 0; l < NumberOfParticles; l++) {
            double Phi1 = phi(3 * k + 1) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l + 1);
            double Phi2 = -(phi(3 * k) * phi(3 * l + 2) - phi(3 * k + 2) * phi(3 * l));
            double Phi3 = phi(3 * k) * phi(3 * l + 1) - phi(3 * k + 1) * phi(3 * l);
            Eigen::Vector3d Phi = { Phi1, Phi2, Phi3 };

            for (int p = 0; p < dimensions; p++) {
                Phi_KL(k, 3 * l + p) = Phi(p);
            }
        }
    }

    // “à‘}ŠÖ”‚ÌŒvZ
    // ‹æŠÔ•ªŠ„
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                Eigen::MatrixXd WeightIJ = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);
                Eigen::VectorXd WeightKLXi = Eigen::VectorXd::Zero(3 * NumberOfParticles);

                /*------  ®ŒvZ   ------*/
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    for (int i = 0; i < NumberOfParticles; i++) {
                        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;
                        if (!allElementsWithinOne(i_minus_xi)) continue;

                        for (int j = 0; j < NumberOfParticles; j++) {
                            Eigen::Vector3i j_minus_xi = FlatToGrid(j) - grid_xi;
                            if (!allElementsWithinOne(j_minus_xi)) continue;

                            // “à‘}ŠÖ”‚ÌŒvZ
                            double diff_hat_x_i = DifferentialHatFunction(cal_point(0) - i_minus_xi(0));
                            double diff_hat_y_i = DifferentialHatFunction(cal_point(1) - i_minus_xi(1));
                            double diff_hat_z_i = DifferentialHatFunction(cal_point(2) - i_minus_xi(2));
                            double hat_x_i = HatFunction(cal_point(0) - i_minus_xi(0));
                            double hat_y_i = HatFunction(cal_point(1) - i_minus_xi(1));
                            double hat_z_i = HatFunction(cal_point(2) - i_minus_xi(2));

                            double diff_hat_x_j = DifferentialHatFunction(cal_point(0) - j_minus_xi(0));
                            double diff_hat_y_j = DifferentialHatFunction(cal_point(1) - j_minus_xi(1));
                            double diff_hat_z_j = DifferentialHatFunction(cal_point(2) - j_minus_xi(2));
                            double hat_x_j = HatFunction(cal_point(0) - j_minus_xi(0));
                            double hat_y_j = HatFunction(cal_point(1) - j_minus_xi(1));
                            double hat_z_j = HatFunction(cal_point(2) - j_minus_xi(2));

                            for (int p = 0; p < dimensions; p++) {
                                WeightIJ(3 * i + p, xi) += phi(3 * j + p) *
                                    (diff_hat_x_i * hat_y_i * hat_z_i * diff_hat_x_j * hat_y_j * hat_z_j
                                        + hat_x_i * diff_hat_y_i * hat_z_i * hat_x_j * diff_hat_y_j * hat_z_j
                                        + hat_x_i * hat_y_i * diff_hat_z_i * hat_x_j * hat_y_j * diff_hat_z_j);
                            }


                        }
                    }

                    for (int k = 0; k < NumberOfParticles; k++) {
                        Eigen::Vector3i k_minus_xi = FlatToGrid(k) - grid_xi;
                        if (!allElementsWithinOne(k_minus_xi)) continue;

                        for (int l = 0; l < NumberOfParticles; l++) {
                            Eigen::Vector3i l_minus_xi = FlatToGrid(l) - grid_xi;
                            if (!allElementsWithinOne(l_minus_xi)) continue;

                            // ŒvZ“_‚ÌŒˆ’è‚Æ“à‘}ŠÖ”‚ÌŒvZ
                            // kŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_k = HatFunction(cal_point(0) - k_minus_xi(0));
                            double diff_hat_x_k = DifferentialHatFunction(cal_point(0) - k_minus_xi(0));
                            double hat_y_k = HatFunction(cal_point(1) - k_minus_xi(1));
                            double diff_hat_y_k = DifferentialHatFunction(cal_point(1) - k_minus_xi(1));
                            double hat_z_k = HatFunction(cal_point(2) - k_minus_xi(2));

                            // lŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_l = HatFunction(cal_point(0) - l_minus_xi(0));
                            double hat_y_l = HatFunction(cal_point(1) - l_minus_xi(1));
                            double diff_hat_y_l = DifferentialHatFunction(cal_point(1) - l_minus_xi(1));
                            double hat_z_l = HatFunction(cal_point(2) - l_minus_xi(2));
                            double diff_hat_z_l = DifferentialHatFunction(cal_point(2) - l_minus_xi(2));

                            // xiŠÖ˜A‚Ì“à‘}ŠÖ”‚ÌŒvZ
                            double hat_x_xi = HatFunction(cal_point(0));
                            double diff_hat_x_xi = DifferentialHatFunction(cal_point(0));
                            double hat_y_xi = HatFunction(cal_point(1));
                            double diff_hat_y_xi = DifferentialHatFunction(cal_point(1));
                            double hat_z_xi = HatFunction(cal_point(2));
                            double diff_hat_z_xi = DifferentialHatFunction(cal_point(2));

                            // Še€‚ÌŒvZ
                            double term1 = hat_x_k * diff_hat_y_k * hat_z_k;
                            double term2 = hat_x_l * hat_y_l * diff_hat_z_l;
                            double term3 = diff_hat_x_xi * hat_y_xi * hat_z_xi;

                            double term4 = diff_hat_x_k * hat_y_k * hat_z_k;
                            double term5 = hat_x_l * hat_y_l * diff_hat_z_l;
                            double term6 = hat_x_xi * diff_hat_y_xi * hat_z_xi;

                            double term7 = diff_hat_x_k * hat_y_k * hat_z_k;
                            double term8 = hat_x_l * diff_hat_y_l * hat_z_l;
                            double term9 = hat_x_xi * hat_y_xi * diff_hat_z_xi;

                            for (int p = 0; p < dimensions; p++) {
                                WeightKLXi(3 * xi + p) += Phi_KL(k, 3 * l + p) *
                                    (term1 * term2 * term3
                                        - term4 * term5 * term6
                                        + term7 * term8 * term9);
                            }

                        }
                    }
                }

                // HessianChi3 += - mu * pow(J, (-5/3)) * WeightIJ * WeightKLXi.transpose() ‚ÌŒvZ
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    for (int i = 0; i < NumberOfParticles; i++) {

                        for (int col = 0; col < dimensions; col++) { // —ñ”i‰¡‚Ì”j
                            for (int row = 0; row < dimensions; row++) { // s”ic‚Ì”j
                                HessianChi3(3 * i + row, 3 * xi + col)
                                    += (-1) * mu * J(xi) * WeightIJ(3 * i + row, xi) * WeightKLXi(3 * xi + col) * pow(kWidth, 3);
                            }
                        }

                    }
                }

                /*------  ®ŒvZI—¹   ------*/

            }
        }
    }

    return HessianChi3;
}

