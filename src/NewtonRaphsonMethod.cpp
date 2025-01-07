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
#include "../Draw/Draw.h"

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
double EPSILON = 1e-2;

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

    // Set HessianS using blocks
    MatrixS.block(0, 0, 3 * NumberOfParticles, 3 * NumberOfParticles) = MatrixN;
    MatrixS.block(0, 4 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles) = MatrixO;
    MatrixS.block(3 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixR;
    MatrixS.block(3 * NumberOfParticles, 4 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixQ;
    MatrixS.block(4 * NumberOfParticles, 0, NumberOfParticles, 3 * NumberOfParticles) = MatrixP;
    MatrixS.block(4 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixQ;

    // exportMatrix_CSV(MatrixS, "csv/MatrixS.csv");

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

    // exportVector_CSV(Vectore, "csv/Vectore.csv");

    return Vectore;
}

Eigen::VectorXd Newton(Square square) {

    // ���s���Ԃ̑���J�n
    auto start = std::chrono::high_resolution_clock::now();

    // �œK���v�Z�̏�����
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

    // �덷
    std::vector<double> NormVec;

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

    // �����l�̃����_�����O
    renderAndSave(square, looptimes);
    
    while (NormVectorDelta > 1.0e-2) {

        std::cout << "�����񐔁F�@" << ++looptimes << "��" << std::endl;

        // Hessian
        MatrixS = calMatrixS(square, re_phi, barphi, barpower, bartheta);

        // Gradient
        Vectore = calVectore(square, re_phi, barphi, doublebarphi, barpower, bartheta);

        Eigen::FullPivLU<Eigen::MatrixXd> LU;
        LU.compute(MatrixS);
        Eigen::VectorXd VectorDelta = LU.solve(Vectore);

        // �T�������x�N�g���̒�`
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

        std::cout << "Norm : " << NormVectorDelta << std::endl;
        NormVec.push_back(NormVectorDelta);
        
        //Line Search with Wolfe
        double sigma_armijo = 0.5;
        double sigma_curvature = 1.2;
        double eps1 = 1e-2;
        double eps2 = 0.9;
        double lambda = 1.0;
        int line_search_times = 0;
        
        // ������
        Eigen::VectorXd barphi_prime = VectorDeltaPhi * lambda + barphi;
        Eigen::VectorXd bartheta_prime = VectorDeltaTheta * lambda + bartheta;
        Eigen::VectorXd barpower_prime = VectorDeltaPower * lambda + barpower;

        Eigen::MatrixXd MatrixS_prime = calMatrixS(square, re_phi, barphi_prime, barpower_prime, bartheta_prime);
        Eigen::VectorXd Vectore_prime = calVectore(square, re_phi, barphi_prime, doublebarphi, barpower_prime, bartheta_prime);

        // Armijo���� & �ȗ������̌v�Z
        Eigen::VectorXd f_prime = Vectore_prime;
        Eigen::VectorXd f = Vectore;
        Eigen::VectorXd nabla_f_p = MatrixS * VectorDelta;
        Eigen::VectorXd nabla_f_prime_p = MatrixS_prime * VectorDelta;
        
        // Armijo����
        while ( !( f_prime.norm() <= f.norm() + (eps1 * lambda * nabla_f_p).norm())) {
            // �ȗ�����
            while ( !(nabla_f_prime_p.norm() >= eps2 * nabla_f_p.norm()) ) {

                std::cout << "  �����T�� �����񐔁F�@" << ++line_search_times << "��" << std::endl;

                // �X�V���̍X�V
                lambda *= sigma_curvature;

                std::cout << "  lambda : " << lambda << "\n";

                // �e�v�f�̍X�V
                barphi_prime = VectorDeltaPhi * lambda + barphi;
                bartheta_prime = VectorDeltaTheta * lambda + bartheta;
                barpower_prime = VectorDeltaPower * lambda + barpower;

                MatrixS_prime = calMatrixS(square, re_phi, barphi_prime, barpower_prime, bartheta_prime);

                // �ȗ������̍Čv�Z
                nabla_f_prime_p = MatrixS_prime * VectorDelta;
                nabla_f_p = MatrixS * VectorDelta;
            }
            
            std::cout << "  �����T�� �����񐔁F�@" << ++line_search_times << "��" << std::endl;

            // �X�V���̍X�V
            lambda *= sigma_armijo;

            std::cout << "  lambda : " << lambda << "\n";

            // �e�v�f�̍X�V
            barphi_prime = VectorDeltaPhi * lambda + barphi;
            bartheta_prime = VectorDeltaTheta * lambda + bartheta;
            barpower_prime = VectorDeltaPower * lambda + barpower;   

            Vectore_prime = calVectore(square, re_phi, barphi_prime, doublebarphi, barpower_prime, bartheta_prime);

            // Armijo �����̍Čv�Z
            f_prime = Vectore_prime;
            nabla_f_p = MatrixS * VectorDelta;
        }
        std::cout << std::endl;
        

        // Update
        barphi += lambda * VectorDeltaPhi;
        bartheta += lambda * VectorDeltaTheta;
        barpower += lambda * VectorDeltaPower;

        // �e�����ł̉摜�o��
        for (int i = 0; i < NumberOfParticles; i++) {
            square.points[i].position[0] = barphi(3 * i); 
            square.points[i].position[1] = barphi(3 * i + 1);
            square.points[i].position[2] = barphi(3 * i + 2);
        }
        renderAndSave(square, looptimes);

        // �w�蔽�����ŏI��
        // if (looptimes > 2) break;
    }

    Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(NormVec.data(), NormVec.size());
    exportVector_CSV(vec, "csv/NormVec.csv");

    // 1�ӂ̊i�q��
    int n = square.SideNumber + 1;

    // 3�~3�~3�̊e�p���_���̒��o
    std::vector<Eigen::Vector3d> vertices;
    for (int i = 0; i < SquarePointsNumber; i++) {
        if (i == 0 || i == n - 1 || i == pow(n, 2) - n || i == pow(n, 2) - 1 || i == pow(n, 3) - pow(n, 2) || i == pow(n, 3) - pow(n, 2) + n - 1 
            || i == pow(n, 3) - n || i == pow(n, 3) - 1) {
            Eigen::Vector3d vector = { barphi(3 * i), barphi(3 * i + 1), barphi(3 * i + 2) };
            vertices.emplace_back(vector);
        }
    }

    double edgeLength = 0.0;
    if (isCube(vertices, edgeLength)) {
        std::cout << "����͗����̂ł��B" << std::endl;
        std::cout << "�ӂ̒���: " << edgeLength << std::endl;
    }
    else {
        std::cout << "����͗����̂ł͂���܂���B" << std::endl;
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

    // �����l�̃����_�����O
    renderAndSave(square, looptimes);
    
    // Hessian
    Eigen::MatrixXd MatrixS = Eigen::MatrixXd::Zero(5 * NumberOfParticles, 5 * NumberOfParticles);

    // Gradient
    Eigen::VectorXd Vectore = Eigen::VectorXd::Zero(5 * NumberOfParticles);

    // Calculation Hessian
    // MatrixN(3 * NumberOfParticles, 3 * NumberOfParticles)
    Eigen::MatrixXd MatrixN = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);
    // Eigen::MatrixXd MatrixN = calHessianN(square, re_phi, barphi, barpower);
    exportMatrix_CSV(MatrixN, "csv/MatrixN.csv");
    if (MatrixN.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix N" << std::endl;
    }

    // MatrixO(3 * NumberOfParticles, NumberOfParticles)
    Eigen::MatrixXd MatrixO = Eigen::MatrixXd::Zero(3 * NumberOfParticles, NumberOfParticles);
    // Eigen::MatrixXd MatrixO = calHessianO(square, re_phi, barphi);
    exportMatrix_CSV(MatrixO, "csv/MatrixO.csv");
    if (MatrixO.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix O" << std::endl;
    }

    // MatrixP(NumberOfParticles, 3 * NumberOfParticles)
    Eigen::MatrixXd MatrixP = Eigen::MatrixXd::Zero(NumberOfParticles, 3 * NumberOfParticles);
    // Eigen::MatrixXd MatrixP = calHessianP(square, re_phi, barphi);
    exportMatrix_CSV(MatrixP, "csv/MatrixP.csv");
    if (MatrixP.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix P" << std::endl;
    }

    // MatrixQ(NumberOfParticles, NumberOfParticles)
    Eigen::MatrixXd MatrixQ = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);
    // Eigen::MatrixXd MatrixQ = calHessianQ(square, re_phi);
    exportMatrix_CSV(MatrixQ, "csv/MatrixQ.csv");
    if (MatrixQ.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix Q" << std::endl;
    }

    //MatrixR(NumberOfParticles, NumberOfParticles)
    Eigen::MatrixXd MatrixR = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);
    // Eigen::MatrixXd MatrixR = calHessianR(square, re_phi, bartheta);
    exportMatrix_CSV(MatrixR, "csv/MatrixR.csv");
    if (MatrixR.array().isNaN().any()) {
        std::cerr << "NaN detected Matrix R" << std::endl;
    }

    // std::cout << MatrixN << std::endl;

    // Set HessianS using blocks
    MatrixS.block(0, 0, 3 * NumberOfParticles, 3 * NumberOfParticles) = MatrixN;
    MatrixS.block(0, 4 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles) = MatrixO;
    MatrixS.block(3 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixR;
    MatrixS.block(3 * NumberOfParticles, 4 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixQ;
    MatrixS.block(4 * NumberOfParticles, 0, NumberOfParticles, 3 * NumberOfParticles) = MatrixP;
    MatrixS.block(4 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixQ;

    exportMatrix_CSV(MatrixS, "csv/MatrixS.csv");

    // Calculation Gradient
    // Vectorb(NumberOfParticles)
    // Eigen::VectorXd Vectorb = Eigen::VectorXd::Zero(NumberOfParticles);
    Eigen::VectorXd Vectorb = calGradientb(square, re_phi, barphi, bartheta);
    // exportVector_CSV(Vectorb, "csv/Vectorb.csv");
    if (Vectorb.array().isNaN().any()) {
        std::cerr << "NaN detected Vector b" << std::endl;
    }

    // std::cout << "Vectorb : " << std::endl;
    // std::cout << Vectorb << std::endl;

    // Vectorc(NumberOfParticles)
    Eigen::VectorXd Vectorc = Eigen::VectorXd::Zero(NumberOfParticles);
    // Eigen::VectorXd Vectorc = calGradientc(square, re_phi, barphi, barpower, bartheta);
    // exportVector_CSV(Vectorc, "csv/Vectorc.csv");
    if (Vectorc.array().isNaN().any()) {
        std::cerr << "NaN detected Vector c" << std::endl;
    }

    // std::cout << "Vectorc : " << std::endl;
    // std::cout << Vectorc << std::endl;

    // Vectord(3 * NumberOfParticles)
    Eigen::VectorXd Vectord = Eigen::VectorXd::Zero(3 * NumberOfParticles);
    // Eigen::VectorXd Vectord = calGradientd(square, re_phi, barphi, doublebarphi, barpower);
    // exportVector_CSV(Vectord, "csv/Vectord.csv");
    if (Vectord.array().isNaN().any()) {
        std::cerr << "NaN detected Vector d" << std::endl;
    }

    // std::cout << "Vectord : " << std::endl;
    // std::cout << Vectord << std::endl;

    // Set Vectore
    Vectore.head(3 * NumberOfParticles) = Vectord;
    Vectore.segment(3 * NumberOfParticles, NumberOfParticles) = Vectorc;
    Vectore.tail(NumberOfParticles) = Vectorb;

    // std::cout << "-----------------------------------------" << std::endl;
    // std::cout << "Vectore : " << std::endl;
    // std::cout << Vectore << std::endl;

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

    /*std::cout << "VectorDeltaTheta : " << std::endl;
    std::cout << VectorDeltaTheta << std::endl;

    std::cout << "VectorDeltaPower : " << std::endl;
    std::cout << VectorDeltaPower << std::endl;*/

    // std::cout << "VectorDelta : " << std::endl;
    // std::cout << VectorDelta << std::endl;

    exportVector_CSV(VectorDeltaPhi, "csv/VectorDeltaPhi.csv");
    exportVector_CSV(VectorDeltaTheta, "csv/VectorDeltaTheta.csv");
    exportVector_CSV(VectorDeltaPower, "csv/VectorDeltaPower.csv");

    NormVectorDelta = VectorDelta.norm();
    NormVectorDeltaPhi = VectorDeltaPhi.norm();
    NormVectorDeltaTheta = VectorDeltaTheta.norm();
    NormVectorDeltaPower = VectorDeltaPower.norm();

    std::cout << "Vector Norm: " << NormVectorDelta << std::endl;
    
    Eigen::VectorXd update_phi = Eigen::VectorXd::Zero(3 * NumberOfParticles);
    Eigen::VectorXd update_theta(NumberOfParticles);
    Eigen::VectorXd update_power(NumberOfParticles);
    
    // Update
    update_phi = barphi + VectorDeltaPhi;
    exportVector_CSV(update_phi, "csv/update_phi.csv");
    update_theta = bartheta + VectorDeltaTheta;
    exportVector_CSV(update_theta, "csv/update_theta.csv");
    update_power = barpower + VectorDeltaPower;
    exportVector_CSV(update_power, "csv/update_power.csv");

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

    data_file << "num" << "," << "value" << std::endl;

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

bool Armijo(Eigen::VectorXd V1, Eigen::VectorXd V2) {

    for (int i = 0; i < V1.size(); i++) {
        if (V1(i) > V2(i)) return false;
    }
    return true;
}

// �`��Ɖ摜�ۑ��𓝍������֐�
void renderAndSave(Square square, int repetitionTime) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // �o�b�t�@�̏���

    // �J�����̍X�V�𔽉f
    projection_and_modelview(g_Camera);

    // �`�揈���i��: ���_�ɐԂ��_��`��j
    glColor3f(0.5, 0.0, 0.0);
    drawSquare(square, 10.0);
    Ground();

    // �`��������I�ɍs��
    glFlush();
    glutSwapBuffers();

    // �摜�ۑ�
    std::vector<unsigned char> pixels(3 * WindowWidth * WindowHeight);

    // OpenGL ����s�N�Z���f�[�^���擾
    glReadPixels(0, 0, WindowWidth, WindowHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // �t�@�C�����̐���
    char filename[100];
    snprintf(filename, sizeof(filename), "result/output_image_%d.ppm", repetitionTime);

    // PPM�`���ŕۑ�
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

// �x�N�g���̒������v�Z
double calculateDistance(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return (a - b).norm();
}

// �x�N�g�����������Ă��邩�𔻒�
bool isOrthogonal(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return std::abs(a.dot(b)) < EPSILON;
}

// �ӂ̒�������ɒ��_�������̂��ǂ�������
bool isCube(const std::vector<Eigen::Vector3d>& vertices, double& edgeLength) {
    if (vertices.size() != 8) {
        std::cerr << "���_����8�ł͂���܂���B" << std::endl;
        return false;
    }

    // ���_�Ԃ̑S�Ă̋������v�Z
    std::vector<double> distances;
    for (size_t i = 0; i < vertices.size(); ++i) {
        for (size_t j = i + 1; j < vertices.size(); ++j) {
            distances.push_back(calculateDistance(vertices[i], vertices[j]));
        }
    }

    // ���������������Ƀ\�[�g
    std::sort(distances.begin(), distances.end());

    // �ӂ̒����A�ʑΊp���̒����A��ԑΊp���̒����𕪗�
    for (int i = 0; i < 12; i++) {
        edgeLength += distances[i];
    }
    // std::cout << edgeLength/ distances.size() << std::endl;
    // edgeLength = distances[0];
    edgeLength = edgeLength / 12;

    double faceDiagonal = edgeLength * std::sqrt(2);
    double spaceDiagonal = edgeLength * std::sqrt(3);

    // �ӂ̒����A�ʑΊp���A��ԑΊp���̖{�����J�E���g
    int edgeCount = std::count_if(distances.begin(), distances.end(),
        [edgeLength](double d) { return std::abs(d - edgeLength) < EPSILON; });
    int faceDiagonalCount = std::count_if(distances.begin(), distances.end(),
        [faceDiagonal](double d) { return std::abs(d - faceDiagonal) < EPSILON; });
    int spaceDiagonalCount = std::count_if(distances.begin(), distances.end(),
        [spaceDiagonal](double d) { return std::abs(d - spaceDiagonal) < EPSILON; });

    // �{�������ꂼ�ꐳ���������m�F
    if (edgeCount != 12 || faceDiagonalCount != 12 || spaceDiagonalCount != 4) {
        return false;
    }

    // �e���_���璼������3�̃x�N�g�����m�F
    for (size_t i = 0; i < vertices.size(); ++i) {
        std::vector<Eigen::Vector3d> edges;
        for (size_t j = 0; j < vertices.size(); ++j) {
            if (i != j && std::abs(calculateDistance(vertices[i], vertices[j]) - edgeLength) < EPSILON) {
                edges.push_back(vertices[j] - vertices[i]);
            }
        }

        // �����������m�F
        if (edges.size() != 3 ||
            !isOrthogonal(edges[0], edges[1]) ||
            !isOrthogonal(edges[0], edges[2]) ||
            !isOrthogonal(edges[1], edges[2])) {
            return false;
        }
    }

    return true;
}