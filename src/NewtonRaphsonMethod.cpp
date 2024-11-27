#include <GL/freeglut.h>
#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <iomanip> 
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>

#include "../include/NewtonRaphsonMethod.h"
#include "../include/Square.h"
#include "../include/FEM.h"
#include "../Window/Window.h"
#include "../Camera/Camera.h"
// �o�͊m�F�̂��߂�include
#include "../include/utils/Interpolation_util.h"

// Hessain
#include "../include/Hessian/HessianN.h"
#include "../include/Hessian/HessianO.h"
#include "../include/Hessian/HessianP.h"
#include "../include/Hessian/HessianQ.h"
#include "../include/Hessian/HessianR.h"

// Gradient
#include "../include/Gradient/Gradientb.h"
#include "../include/Gradient/Gradientc.h"
#include "../include/Gradient/Gradientd.h"

extern Camera g_Camera;
int looptimes = 0;

Eigen::MatrixXd calMatrixS(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& power, const Eigen::VectorXd& theta) 
{
    // Hessian
    Eigen::MatrixXd MatrixS = Eigen::MatrixXd::Zero(5 * NumberOfParticles, 5 * NumberOfParticles);

    // Calculation Hessian
    // MatrixN(3 * NumberOfParticles, 3 * NumberOfParticles)
    Eigen::MatrixXd MatrixN = calHessianN(square, re_phi, phi, power);
    exportMatrix_CSV(MatrixN, "csv/MatrixN.csv");
    if (MatrixN.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix N" << std::endl;
    }

    // MatrixO(NumberOfParticles, 3 * NumberOfParticles)
    Eigen::MatrixXd MatrixO = calHessianO(square, re_phi, phi);
    exportMatrix_CSV(MatrixO, "csv/MatrixO.csv");
    if (MatrixO.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix O" << std::endl;
    }

    // MatrixP(3 * NumberOfParticles, NumberOfParticles)
    Eigen::MatrixXd MatrixP = calHessianP(square, re_phi, phi);
    exportMatrix_CSV(MatrixP, "csv/MatrixP.csv");
    if (MatrixP.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix P" << std::endl;
    }

    // MatrixQ(NumberOfParticles, NumberOfParticles)
    Eigen::MatrixXd MatrixQ = calHessianQ(square, re_phi);
    exportMatrix_CSV(MatrixQ, "csv/MatrixQ.csv");
    if (MatrixQ.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix Q" << std::endl;
    }

    //MatrixR(NumberOfParticles, NumberOfParticles)
    Eigen::MatrixXd MatrixR = calHessianR(square, re_phi, theta);
    exportMatrix_CSV(MatrixR, "csv/MatrixR.csv");
    if (MatrixR.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix R" << std::endl;
    }

    // Calculation Hessian
    Eigen::MatrixXd MatrixN_t = MatrixN.transpose();
    Eigen::MatrixXd MatrixO_t = MatrixO.transpose();
    Eigen::MatrixXd MatrixP_t = MatrixP.transpose();
    Eigen::MatrixXd MatrixQ_t = MatrixQ.transpose();
    Eigen::MatrixXd MatrixR_t = MatrixR.transpose();

    // Set HessianS using blocks
    MatrixS.block(0, 0, 3 * NumberOfParticles, 3 * NumberOfParticles) = MatrixN_t;
    MatrixS.block(0, 4 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles) = MatrixO_t;
    MatrixS.block(3 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixR_t;
    MatrixS.block(3 * NumberOfParticles, 4 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixQ_t;
    MatrixS.block(4 * NumberOfParticles, 0, NumberOfParticles, 3 * NumberOfParticles) = MatrixP_t;
    MatrixS.block(4 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixQ_t;

    exportMatrix_CSV(MatrixS, "csv/MatrixS.csv");

    return MatrixS;
}

Eigen::VectorXd calVectore(const Square& square, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const Eigen::VectorXd& phi_previous, const Eigen::VectorXd& power, const Eigen::VectorXd& theta)
{
    // Gradient
    Eigen::VectorXd Vectore = Eigen::VectorXd::Zero(5 * NumberOfParticles);

    // Calculation Gradient
    // Vectorb(NumberOfParticles)
    Eigen::VectorXd Vectorb = calGradientb(square, re_phi, phi, theta);
    exportVector_CSV(Vectorb, "csv/Vectorb.csv");
    if (Vectorb.array().isNaN().any()) {
        std::cerr << "NaN detected Vector b" << std::endl;
    }

    // Vectorc(NumberOfParticles)
    Eigen::VectorXd Vectorc = calGradientc(square, re_phi, phi, power, theta);
    exportVector_CSV(Vectorc, "csv/Vectorc.csv");
    if (Vectorc.array().isNaN().any()) {
        std::cerr << "NaN detected Vector c" << std::endl;
    }

    // Vectord(NumberOfParticles)
    Eigen::VectorXd Vectord = calGradientd(square, re_phi, phi, phi_previous, power);
    exportVector_CSV(Vectord, "csv/Vectord.csv");
    if (Vectord.array().isNaN().any()) {
        std::cerr << "NaN detected Vector d" << std::endl;
    }

    // Set Vectore
    Vectore.head(3 * NumberOfParticles) = Vectord;
    Vectore.segment(3 * NumberOfParticles, NumberOfParticles) = Vectorc;
    Vectore.tail(NumberOfParticles) = Vectorb;

    exportVector_CSV(Vectore, "csv/Vectore.csv");

    return Vectore;
}

Eigen::VectorXd Newton(Square square) {

    // ���s���Ԃ̑���J�n
    auto start = std::chrono::high_resolution_clock::now();

    double lambda = 0.2;

    // �œK���v�Z�̏�����
    double NormVectorDeltaPhi = 1.0;
    double NormVectorDeltaTheta = 1.0;
    double NormVectorDeltaPower = 1.0;
    double NormVectorDelta = 1.0;
    int SquarePointsNumber = square.points.size();


    // �ψ�
    Eigen::VectorXd barphi(3 * NumberOfParticles);
    Eigen::VectorXd doublebarphi(3 * NumberOfParticles);

    // �i�q�_���W
    Eigen::VectorXd re_phi(3 * NumberOfParticles);

    // ����
    Eigen::VectorXd barpower(NumberOfParticles);
    // �̐ϕω���
    Eigen::VectorXd bartheta(NumberOfParticles);

    // ���W�̎擾
    for (int i = 0; i < SquarePointsNumber; i++) {
        barphi.segment<3>(3 * i) = square.points[i].position;
        doublebarphi.segment<3>(3 * i) = square.points[i].position;
        re_phi.segment<3>(3 * i) = square.points[i].reference_position;
        barpower(i) = square.points[i].power;
        bartheta(i) = square.points[i].theta;
    }

    // Hessian
    Eigen::MatrixXd MatrixS;

    // Gradient
    Eigen::VectorXd Vectore;
    
    while (NormVectorDelta > 1.0e-5) {

        // Hessian
        MatrixS = calMatrixS(square, re_phi, barphi, barpower, bartheta);

        // checkHessian(MatrixS);

        // Gradient
        Vectore = calVectore(square, re_phi, barphi, doublebarphi, barpower, bartheta);

        Eigen::FullPivLU<Eigen::MatrixXd> LU(MatrixS);
        Eigen::VectorXd VectorDelta = LU.solve(Vectore);

        VectorDelta = - VectorDelta;

        VectorDelta *= lambda;

        Eigen::VectorXd VectorDeltaPhi(3 * NumberOfParticles);
        Eigen::VectorXd VectorDeltaTheta(NumberOfParticles);
        Eigen::VectorXd VectorDeltaPower(NumberOfParticles);


        // Set VectorDeltaPhi
        for (int i = 0; i < NumberOfParticles; i++) {
            VectorDeltaPhi.segment<3>(3 * i) = VectorDelta.segment<3>(3 * i);
        }

        // Set VectorDeltaTheta
        VectorDeltaTheta = VectorDelta.segment(NumberOfParticles * 3, NumberOfParticles);

        // Set VectorDeltaPower
        VectorDeltaPower = VectorDelta.segment(NumberOfParticles * 4, NumberOfParticles);

        NormVectorDelta = VectorDelta.norm();
        NormVectorDeltaPhi = VectorDeltaPhi.norm();
        NormVectorDeltaTheta = VectorDeltaTheta.norm();
        NormVectorDeltaPower = VectorDeltaPower.norm();

        std::cout << "Norm : " << NormVectorDelta << std::endl;

        //Line Search with Wolfe
        double sigma = 0.5;
        double eps1 = 1.0e-4;
        double eps2 = 0.9;
        
        int line_search_times = 1;

        /*

        // CSV�t�@�C���̃I�[�v��
        std::ofstream csv_file("csv/line_search_results.csv");
        csv_file << "Iteration,lambda,f_prime.norm(),f.norm() + eps1 * lambda * nabla_f_p.norm(),nabla_f_prime_p.norm(),eps2 * nabla_f_p.norm()\n";

        // ������
        Eigen::VectorXd barphi_prime = VectorDeltaPhi * lambda + barphi;
        Eigen::VectorXd bartheta_prime = VectorDeltaTheta * lambda + bartheta;
        Eigen::VectorXd barpower_prime = VectorDeltaPower * lambda + barpower;

        Eigen::MatrixXd MatrixS_prime = calMatrixS(square, re_phi, barphi_prime, barpower_prime, bartheta_prime);
        Eigen::VectorXd Vectore_prime = calVectore(square, re_phi, barphi_prime, doublebarphi, barpower_prime, bartheta_prime);

        // Armijo & curvature �����̌v�Z
        Eigen::VectorXd f_prime = Vectore_prime;
        Eigen::VectorXd f = Vectore;
        Eigen::VectorXd nabla_f_p = MatrixS * VectorDeltaPhi;
        Eigen::VectorXd nabla_f_prime_p = MatrixS_prime * VectorDeltaPhi;

        // Armijo & curvature
        while (!(f_prime.norm() <= f.norm() + eps1 * lambda * nabla_f_p.norm() &&
            nabla_f_prime_p.norm() >= eps2 * nabla_f_p.norm())) {

            // CSV�ւ̏o��
            exportLineSearch_CSV(csv_file, line_search_times, lambda, f_prime.norm(),
                f.norm() + eps1 * lambda * nabla_f_p.norm(),
                nabla_f_prime_p.norm(), eps2 * nabla_f_p.norm());

            std::cout << "f_prime.norm() : " << f_prime.norm() << std::endl;
            std::cout << "f.norm() + eps1 * lambda * nabla_f_p.norm() : " << f.norm() + eps1 * lambda * nabla_f_p.norm() << std::endl;
            std::cout << "nabla_f_prime_p.norm() : " << nabla_f_prime_p.norm() << std::endl;
            std::cout << "eps2 * nabla_f_p.norm() : " << eps2 * nabla_f_p.norm() << std::endl;
            std::cout << std::endl;

            lambda *= sigma;

            // �X�V
            barphi_prime = VectorDeltaPhi * lambda + barphi;
            bartheta_prime = VectorDeltaTheta * lambda + bartheta;
            barpower_prime = VectorDeltaPower * lambda + barpower;

            MatrixS_prime = calMatrixS(square, re_phi, barphi_prime, barpower_prime, bartheta_prime);
            Vectore_prime = calVectore(square, re_phi, barphi_prime, doublebarphi, barpower_prime, bartheta_prime);

            // Armijo & curvature �����̍Čv�Z
            f_prime = Vectore_prime;
            f = Vectore;
            nabla_f_p = MatrixS * VectorDeltaPhi;
            nabla_f_prime_p = MatrixS_prime * VectorDeltaPhi;

            line_search_times++;
            std::cout << "lambda : " << lambda << "\n";
        }


        exportLineSearch_CSV(csv_file, line_search_times, lambda, f_prime.norm(),
            f.norm() + eps1 * lambda * nabla_f_p.norm(),
            nabla_f_prime_p.norm(), eps2 * nabla_f_p.norm());
        csv_file.close();

        std::cout << "result:" << std::endl;
        std::cout << "f_prime.norm() : " << f_prime.norm() << std::endl;
        std::cout << "f.norm() : " << f.norm() << std::endl;
        std::cout << "nabla_f_prime_p.norm() : " << nabla_f_prime_p.norm() << std::endl;
        std::cout << "eps2 * nabla_f_p.norm() : " << eps2 * nabla_f_p.norm() << std::endl;
        std::cout << std::endl;
        */


        // Update
        // doublebarphi = barphi;
        barphi += VectorDeltaPhi * lambda;
        bartheta += VectorDeltaTheta * lambda;
        barpower += VectorDeltaPower * lambda;

        // barpower = Eigen::VectorXd::Zero(NumberOfParticles);

        // std::cout << "barphi : " << barphi << std::endl;
        // std::cout << "bartheta : " << bartheta << std::endl;

        looptimes++;

        std::cout << "�����񐔁F�@" << looptimes << "��" << std::endl;

        // if (looptimes > 5) break;
    }
    

    // ���s���Ԃ̑���I��
    auto end = std::chrono::high_resolution_clock::now();
    // ���s���Ԃ̌v�Z
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;


    return barphi;
}

Eigen::VectorXd NewtonIteration(Square square) {

    // ���s���Ԃ̑���J�n
    auto start = std::chrono::high_resolution_clock::now();

    // �œK���v�Z�̏�����
    double NormVectorDeltaPhi = 1.0;
    double NormVectorDeltaTheta = 1.0;
    double NormVectorDeltaPower = 1.0;
    double NormVectorDelta = 1.0;
    int SquarePointsNumber = square.points.size();

    // �ψ�
    Eigen::VectorXd barphi(3 * NumberOfParticles);
    Eigen::VectorXd doublebarphi(3 * NumberOfParticles);

    // �i�q�_���W
    Eigen::VectorXd re_phi(3 * NumberOfParticles);

    // ����
    Eigen::VectorXd barpower(NumberOfParticles);

    // �̐ϕω���
    Eigen::VectorXd bartheta(NumberOfParticles);

    // ���W�̎擾
    for (int i = 0; i < SquarePointsNumber; i++) {
        barphi.segment<3>(3 * i) = square.points[i].position;
        doublebarphi.segment<3>(3 * i) = square.points[i].position;
        re_phi.segment<3>(3 * i) = square.points[i].reference_position;
        barpower(i) = square.points[i].power;
        bartheta(i) = square.points[i].theta;
    }

    /* -- test -- */
    /*
    const int kNumSection = 3; // �e��Ԃ̕�����
    const double kWidth = square.dx / kNumSection; // �����̐��K��
    const int kNum = 2 * kNumSection; // �S��Ԃ̕�����
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -1; offset <= 0; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            // std::cout << cal_points(index) << std::endl;
            index++;
        }
    }


    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                // Stencil Base�̌v�Z
                Eigen::Vector3d stencil_base = calculateStencilBase(cal_point);
                std::cout << "Stencil Base: " << stencil_base.transpose() << std::endl;

                // Stencil�s���stencil_num�̐���
                Eigen::MatrixXi stencil;
                std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

                // �̐ϕω����̌v�Z
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);
                    Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

                    // �̐ϕω����̌v�Z
                    double detF = calRiemannJ(cal_point, grid_xi, re_phi, barphi, NumberOfParticles, -2.0 / 3.0);
                    std::cout << "detF : " << detF << std::endl;
                }
            }
        }
    }
    */    

    /* -- test -- */

    // Hessian
    Eigen::MatrixXd MatrixS = Eigen::MatrixXd::Zero(5 * NumberOfParticles, 5 * NumberOfParticles);

    // Gradient
    Eigen::VectorXd Vectore = Eigen::VectorXd::Zero(5 * NumberOfParticles);

    // Calculation Hessian
    // MatrixN(3 * NumberOfParticles, 3 * NumberOfParticles)
    Eigen::MatrixXd MatrixN = calHessianN(square, re_phi, barphi, barpower);
    exportMatrix_CSV(MatrixN, "csv/MatrixN.csv");
    if (MatrixN.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix N" << std::endl;
    }

    // MatrixO(NumberOfParticles, 3 * NumberOfParticles)
    Eigen::MatrixXd MatrixO = calHessianO(square, re_phi, barphi);
    exportMatrix_CSV(MatrixO, "csv/MatrixO.csv");
    if (MatrixO.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix O" << std::endl;
    }

    // MatrixP(3 * NumberOfParticles, NumberOfParticles)
    Eigen::MatrixXd MatrixP = calHessianP(square, re_phi, barphi);
    exportMatrix_CSV(MatrixP, "csv/MatrixP.csv");
    if (MatrixP.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix P" << std::endl;
    }

    // MatrixQ(NumberOfParticles, NumberOfParticles)
    Eigen::MatrixXd MatrixQ = calHessianQ(square, re_phi);
    exportMatrix_CSV(MatrixQ, "csv/MatrixQ.csv");
    if (MatrixQ.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix Q" << std::endl;
    }

    //MatrixR(NumberOfParticles, NumberOfParticles)
    Eigen::MatrixXd MatrixR = calHessianR(square, re_phi, bartheta);
    exportMatrix_CSV(MatrixR, "csv/MatrixR.csv");
    if (MatrixR.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix R" << std::endl;
    }

    // std::cout << MatrixN << std::endl;

    // Calculation Hessian
    Eigen::MatrixXd MatrixN_t = MatrixN.transpose();
    Eigen::MatrixXd MatrixO_t = MatrixO.transpose();
    Eigen::MatrixXd MatrixP_t = MatrixP.transpose();
    Eigen::MatrixXd MatrixQ_t = MatrixQ.transpose();
    Eigen::MatrixXd MatrixR_t = MatrixR.transpose();

    // Set HessianS using blocks
    MatrixS.block(0, 0, 3 * NumberOfParticles, 3 * NumberOfParticles) = MatrixN_t;
    MatrixS.block(0, 4 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles) = MatrixO_t;
    MatrixS.block(3 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixR_t;
    MatrixS.block(3 * NumberOfParticles, 4 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixQ_t;
    MatrixS.block(4 * NumberOfParticles, 0, NumberOfParticles, 3 * NumberOfParticles) = MatrixP_t;
    MatrixS.block(4 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixQ_t;

    exportMatrix_CSV(MatrixS, "csv/MatrixS.csv");

    // Calculation Gradient
    // Vectorb(NumberOfParticles)
    Eigen::VectorXd Vectorb = calGradientb(square, re_phi, barphi, bartheta);
    exportVector_CSV(Vectorb, "csv/Vectorb.csv");
    if (Vectorb.array().isNaN().any()) {
        std::cerr << "NaN detected Vector b" << std::endl;
    }

    // Vectorc(NumberOfParticles)
    Eigen::VectorXd Vectorc = calGradientc(square, re_phi, barphi, barpower, bartheta);
    exportVector_CSV(Vectorc, "csv/Vectorc.csv");
    if (Vectorc.array().isNaN().any()) {
        std::cerr << "NaN detected Vector c" << std::endl;
    }

    // Vectord(NumberOfParticles)
    Eigen::VectorXd Vectord = calGradientd(square, re_phi, barphi, doublebarphi, barpower);
    exportVector_CSV(Vectord, "csv/Vectord.csv");
    if (Vectord.array().isNaN().any()) {
        std::cerr << "NaN detected Vector d" << std::endl;
    }

    // std::cout << Vectord << std::endl;

    // Set Vectore
    Vectore.head(3 * NumberOfParticles) = Vectord;
    Vectore.segment(3 * NumberOfParticles, NumberOfParticles) = Vectorc;
    Vectore.tail(NumberOfParticles) = Vectorb;

    exportVector_CSV(Vectore, "csv/Vectore.csv");

    Eigen::FullPivLU<Eigen::MatrixXd> LU(MatrixS);
    Eigen::VectorXd VectorDelta = LU.solve(Vectore);

    Eigen::VectorXd VectorDeltaPhi(3 * NumberOfParticles);
    Eigen::VectorXd VectorDeltaTheta(NumberOfParticles);
    Eigen::VectorXd VectorDeltaPower(NumberOfParticles);

    // Set VectorDeltaPhi
    for (int i = 0; i < NumberOfParticles; i++) {
        VectorDeltaPhi.segment<3>(3 * i) = VectorDelta.segment<3>(3 * i);
    }

    // Set VectorDeltaTheta
    VectorDeltaTheta = VectorDelta.segment(NumberOfParticles * 3, NumberOfParticles);

    // Set VectorDeltaPower
    VectorDeltaPower = VectorDelta.segment(NumberOfParticles * 4, NumberOfParticles);

    exportVector_CSV(VectorDeltaPhi, "csv/VectorDeltaPhi.csv");
    exportVector_CSV(VectorDeltaTheta, "csv/VectorDeltaTheta.csv");
    exportVector_CSV(VectorDeltaPower, "csv/VectorDeltaPower.csv");

    NormVectorDelta = VectorDelta.norm();
    NormVectorDeltaPhi = VectorDeltaPhi.norm();
    NormVectorDeltaTheta = VectorDeltaTheta.norm();
    NormVectorDeltaPower = VectorDeltaPower.norm();

    std::cout << "Vector Norm: " << NormVectorDelta << std::endl;

    Eigen::VectorXd update_phi(3 * NumberOfParticles);
    Eigen::VectorXd update_theta(NumberOfParticles);
    Eigen::VectorXd update_power(NumberOfParticles);

    // Update
    update_phi = barphi + VectorDeltaPhi;
    exportVector_CSV(update_phi, "csv/update_phi.csv");
    update_theta = bartheta + VectorDeltaTheta;
    exportVector_CSV(update_theta, "csv/update_theta.csv");
    update_power = barpower + VectorDeltaPower;
    exportVector_CSV(update_power, "csv/update_power.csv");

    // calculation equation H
    Eigen::VectorXd Equation_H = MatrixP_t * VectorDeltaPhi + MatrixQ_t * VectorDeltaTheta - Vectorb;
    exportVector_CSV(Equation_H, "csv/Equation_H.csv");
    Eigen::VectorXd Equation_Gamma = MatrixQ_t * VectorDeltaPower + MatrixR_t * VectorDeltaTheta - Vectorc;
    exportVector_CSV(Equation_Gamma, "csv/Equation_Gamma.csv");

    // ���s���Ԃ̑���I��
    auto end = std::chrono::high_resolution_clock::now();
    // ���s���Ԃ̌v�Z
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return update_phi;
}

void exportMatrix_CSV(Eigen::MatrixXd M,  std::string file_name) {
    std::string output_csv_file_name = file_name;
    std::ofstream data_file(output_csv_file_name);

    // Header
    data_file << "" << ",";
    for (int i = 0; i < M.rows(); i++) {
        data_file << i + 1 << ",";
    }
    data_file << std::endl;

    for (int i = 0; i < M.cols(); i++) {
        data_file << i + 1 << ",";
        for (int j = 0; j < M.rows(); j++) {
            data_file << M(j, i) << ",";
        }
        data_file << std::endl;
    }

    data_file.close();
}

void exportVector_CSV(Eigen::VectorXd V, std::string file_name) {
    std::ofstream data_file(file_name);

    for (int i = 0; i < V.size(); i++) {
        data_file << i + 1 << "," << V(i) << std::endl;
    }
    data_file.close();
}

void exportLineSearch_CSV(std::ofstream& file, int line_search_times, double lambda,
    double f_prime_norm, double armijo_rhs,
    double nabla_f_prime_p_norm, double curvature_rhs) {
    file << line_search_times << ","
        << lambda << ","
        << f_prime_norm << ","
        << armijo_rhs << ","
        << nabla_f_prime_p_norm << ","
        << curvature_rhs << "\n";
}

// ����l�s�񂩔���
void checkHessian(const Eigen::MatrixXd& H) {
    // Eigen��SelfAdjointEigenSolver���g�p���ČŗL�l���v�Z
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(H);

    if (eigenSolver.info() != Eigen::Success) {
        std::cerr << "�ŗL�l�����Ɏ��s���܂����B" << std::endl;
        return;
    }

    // �ŗL�l���擾
    Eigen::VectorXd eigenValues = eigenSolver.eigenvalues();

    std::cout << "�ŗL�l:\n" << eigenValues << std::endl;

    // �T�����������������m�F
    bool isPositiveDefinite = true;
    for (int i = 0; i < eigenValues.size(); ++i) {
        if (eigenValues[i] <= 0) {
            isPositiveDefinite = false;
            break;
        }
    }

    if (isPositiveDefinite) {
        std::cout << "�w�b�Z�s��͐���l�ł��B�T�������͐������ł��B" << std::endl;
    }
    else {
        std::cout << "�w�b�Z�s��͐���l�ł͂���܂���B�T���������ĕ]������K�v������܂��B" << std::endl;
    }
}


// �`��������I�ɍs���֐�
void renderOnce() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // �J�����̍X�V�𔽉f
    projection_and_modelview(g_Camera);

    // �`�揈���i��: ���_�ɐԂ��_��`��j
    glColor3f(1.0, 0.0, 0.0);
    glPointSize(5.0);
    glBegin(GL_POINTS);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glEnd();

    glFlush();
    glutSwapBuffers();
}

// �摜��ۑ�����֐�
void saveImageAsPPM(const std::string& filename) {
    std::vector<unsigned char> pixels(3 * WindowWidth * WindowHeight);

    // OpenGL ����s�N�Z���f�[�^���擾
    glReadPixels(0, 0, WindowWidth, WindowHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // �t�@�C���ɏ����o��
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    // PPM �w�b�_�[�̏����o��
    file << "P6\n" << WindowWidth << " " << WindowHeight << "\n255\n";

    // �s�N�Z���f�[�^�̏����o���i�㉺���]���ĕۑ��j
    for (int y = WindowHeight - 1; y >= 0; --y) {
        file.write(reinterpret_cast<const char*>(pixels.data() + y * WindowWidth * 3), WindowWidth * 3);
    }

    file.close();
    std::cout << "Image saved as " << filename << std::endl;
}
