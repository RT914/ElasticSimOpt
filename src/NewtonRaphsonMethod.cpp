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

// 出力確認のためのinclude
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

    // 実行時間の測定開始
    auto start = std::chrono::high_resolution_clock::now();

    double lambda = 0.2;

    // 最適化計算の初期化
    double NormVectorDeltaPhi = 1.0;
    double NormVectorDeltaTheta = 1.0;
    double NormVectorDeltaPower = 1.0;
    double NormVectorDelta = 1.0;
    int SquarePointsNumber = square.points.size();

    // 変位
    Eigen::VectorXd barphi(3 * NumberOfParticles);
    Eigen::VectorXd doublebarphi(3 * NumberOfParticles);

    // 格子点座標
    Eigen::VectorXd re_phi(3 * NumberOfParticles);

    // 圧力
    Eigen::VectorXd barpower(NumberOfParticles);
    // 体積変化率
    Eigen::VectorXd bartheta(NumberOfParticles);

    // 座標の取得
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

    
    while (NormVectorDelta > 1.0e-2) {

        std::cout << "反復回数：　" << looptimes << "回" << std::endl;

        /*for (int i = 0; i < NumberOfParticles; i++) {
            std::cout << i << "----------------------" << std::endl;
            std::cout << barphi(3 * i) << std::endl;
            std::cout << barphi(3 * i + 1) << std::endl;
            std::cout << barphi(3 * i + 2) << std::endl;
        }
        std::cout << std::endl;*/

        // Hessian
        MatrixS = calMatrixS(square, re_phi, barphi, barpower, bartheta);

        // checkHessian(MatrixS);

        // Gradient
        Vectore = calVectore(square, re_phi, barphi, doublebarphi, barpower, bartheta);
        // std::cout << "Vectore : " << std::endl;
        // std::cout << Vectore << std::endl;

        /*Eigen::FullPivLU<Eigen::MatrixXd> LU(MatrixS);
        Eigen::VectorXd VectorDelta = LU.solve(Vectore);*/

        Eigen::FullPivLU<Eigen::MatrixXd> LU;
        LU.compute(MatrixS);
        Eigen::VectorXd VectorDelta = LU.solve(Vectore);

        /*Eigen::JacobiSVD<Eigen::MatrixXd> svd(MatrixS, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd VectorDelta = svd.solve(Vectore);*/

        // std::cout << "VectorDelta : " << std::endl;
        // std::cout << VectorDelta << std::endl;

        // VectorDelta = - VectorDelta;

        // VectorDelta *= 0.2;

        // std::cout << VectorDelta << std::endl;

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

        /*std::cout << "DeltaPhi : " << std::endl;
        for (int i = 0; i < NumberOfParticles; i++) {
            std::cout << i << "----------------------" << std::endl;
            std::cout << VectorDeltaPhi(3 * i) << std::endl;
            std::cout << VectorDeltaPhi(3 * i + 1) << std::endl;
            std::cout << VectorDeltaPhi(3 * i + 2) << std::endl;
        }
        std::cout << std::endl;*/
        
        /*std::cout << "DeltaTheta : " << std::endl;
        std::cout << VectorDeltaTheta << std::endl;
        std::cout << "DeltaPower : " << std::endl;
        std::cout << VectorDeltaPower << std::endl;*/

        NormVectorDelta = VectorDelta.norm();
        NormVectorDeltaPhi = VectorDeltaPhi.norm();
        NormVectorDeltaTheta = VectorDeltaTheta.norm();
        NormVectorDeltaPower = VectorDeltaPower.norm();

        std::cout << "Norm : " << NormVectorDelta << std::endl;

        /*
        //Line Search with Wolfe
        double sigma = 0.5;
        double eps1 = 0.5;
        double lambda = 1.0;
        int line_search_times = 0;

        // CSVファイルのオープン
        // std::ofstream csv_file("csv/line_search_results.csv");
        // csv_file << "Iteration,lambda,f_prime.norm(),f.norm() + eps1 * lambda * nabla_f_p.norm(),nabla_f_prime_p.norm(),eps2 * nabla_f_p.norm()\n";

        // 初期化
        Eigen::VectorXd barphi_prime = VectorDeltaPhi * lambda + barphi;
        Eigen::VectorXd bartheta_prime = VectorDeltaTheta * lambda + bartheta;
        Eigen::VectorXd barpower_prime = VectorDeltaPower * lambda + barpower;

        Eigen::VectorXd Vectore_prime = calVectore(square, re_phi, barphi_prime, doublebarphi, barpower_prime, bartheta_prime);

        // Armijo 条件の計算
        Eigen::VectorXd f_prime = Vectore_prime;
        Eigen::VectorXd f = Vectore;
        Eigen::VectorXd nabla_f_p = MatrixS * VectorDelta;

        // Armijo 条件
        // while ( !(checkArmijo(f_prime, f + eps1 * lambda * nabla_f_p)) ) {
        while ( !( f_prime.norm() <= (f + eps1 * lambda * nabla_f_p).norm()) ) {

            // CSVへの出力
            // exportLineSearch_CSV(csv_file, line_search_times, lambda, f_prime.norm(), f.norm() + eps1 * lambda * nabla_f_p.norm(), nabla_f_prime_p.norm(), eps2 * nabla_f_p.norm());

            // std::cout << "line search time : " << line_search_times++ << std::endl;
            // std::cout << "Armijo : " << std::endl;
            // std::cout << f_prime - (f + eps1 * lambda * nabla_f_p) << std::endl;
            // std::cout << std::endl;

            // std::cout << f_prime.norm() << std::endl;
            // std::cout << (f + eps1 * lambda * nabla_f_p).norm() << std::endl;

            // 更新幅の更新
            lambda *= sigma;

            // 各要素の更新
            barphi_prime = VectorDeltaPhi * lambda + barphi;
            bartheta_prime = VectorDeltaTheta * lambda + bartheta;
            barpower_prime = VectorDeltaPower * lambda + barpower;

            Vectore_prime = calVectore(square, re_phi, barphi_prime, doublebarphi, barpower_prime, bartheta_prime);

            // Armijo 条件の再計算
            f_prime = Vectore_prime;
            f = Vectore;
            nabla_f_p = MatrixS * VectorDelta;

            std::cout << "lambda : " << lambda << "\n";
        }
        */

        // Update
        // doublebarphi = barphi;
        barphi += lambda * VectorDeltaPhi;
        bartheta += lambda * VectorDeltaTheta;
        barpower += lambda * VectorDeltaPower;

        // Update
        // doublebarphi = barphi;
        barphi += VectorDeltaPhi;
        bartheta += VectorDeltaTheta;
        barpower += VectorDeltaPower;

        // barpower = Eigen::VectorXd::Zero(NumberOfParticles);

        // std::cout << "barphi : " << barphi << std::endl;
        // std::cout << "bartheta : " << bartheta << std::endl;

        // 各反復での画像出力
        /*for (int i = 0; i < NumberOfParticles; i++) {
            square.points[i].position[0] = barphi(3 * i);
            square.points[i].position[1] = barphi(3 * i + 1);
            square.points[i].position[2] = barphi(3 * i + 2);
        }
        renderAndSave(square, looptimes);*/

        looptimes++;
        std::cout << std::endl;

        // if (looptimes > 2) break;
    }

    // 3×3×3の各角頂点情報の抽出
    std::vector<Eigen::Vector3d> vertices;
    for (int i = 0; i < SquarePointsNumber; i++) {
        if (i == 0 || i ==2 || i == 6 || i == 8 || i == 18 || i == 20 || i == 24 || i == 26 ) {
            Eigen::Vector3d vector = { barphi(3 * i), barphi(3 * i + 1), barphi(3 * i + 2) };
            vertices.emplace_back(vector);
        }
    }

    double edgeLength = 0.0;
    if (isCube(vertices, edgeLength)) {
        std::cout << "これは立方体です。" << std::endl;
        std::cout << "辺の長さ: " << edgeLength << std::endl;
    }
    else {
        std::cout << "これは立方体ではありません。" << std::endl;
    }

    // 実行時間の測定終了
    auto end = std::chrono::high_resolution_clock::now();
    // 実行時間の計算
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return barphi;
}

Eigen::VectorXd NewtonIteration(Square square) {

    // 実行時間の測定開始
    auto start = std::chrono::high_resolution_clock::now();

    // 最適化計算の初期化
    double NormVectorDeltaPhi = 1.0;
    double NormVectorDeltaTheta = 1.0;
    double NormVectorDeltaPower = 1.0;
    double NormVectorDelta = 1.0;
    int SquarePointsNumber = square.points.size();

    // 変位
    Eigen::VectorXd barphi(3 * NumberOfParticles);
    Eigen::VectorXd doublebarphi(3 * NumberOfParticles);

    // 格子点座標
    Eigen::VectorXd re_phi(3 * NumberOfParticles);

    // 圧力
    Eigen::VectorXd barpower(NumberOfParticles);

    // 体積変化率
    Eigen::VectorXd bartheta(NumberOfParticles);

    // 座標の取得
    for (int i = 0; i < SquarePointsNumber; i++) {
        barphi.segment<3>(3 * i) = square.points[i].position;
        doublebarphi.segment<3>(3 * i) = square.points[i].position;
        re_phi.segment<3>(3 * i) = square.points[i].reference_position;
        barpower(i) = square.points[i].power;
        bartheta(i) = square.points[i].theta;
    }

    /* -- test -- */
    /*
    const int kNumSection = 3; // 各区間の分割数
    const double kWidth = square.dx / kNumSection; // 分割の正規化
    const int kNum = 2 * kNumSection; // 全区間の分割数
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

                // Stencil Baseの計算
                Eigen::Vector3d stencil_base = calculateStencilBase(cal_point);
                std::cout << "Stencil Base: " << stencil_base.transpose() << std::endl;

                // Stencil行列とstencil_numの生成
                Eigen::MatrixXi stencil;
                std::vector<int> stencil_num = generateStencil(stencil_base, stencil);

                // 体積変化率の計算
                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    if (std::find(stencil_num.begin(), stencil_num.end(), xi) == stencil_num.end()) continue;
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);
                    Eigen::Vector3d grid_point_coordinates_xi = { re_phi(3 * xi), re_phi(3 * xi + 1), re_phi(3 * xi + 2) };

                    // 体積変化率の計算
                    double detF = calRiemannJ(cal_point, grid_xi, re_phi, barphi, NumberOfParticles, 1.0);
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
    Eigen::VectorXd Vectorb = calGradientb(square, re_phi, barphi, bartheta);
    exportVector_CSV(Vectorb, "csv/Vectorb.csv");
    if (Vectorb.array().isNaN().any()) {
        std::cerr << "NaN detected Vector b" << std::endl;
    }

    // std::cout << "Vectorb : " << std::endl;
    // std::cout << Vectorb << std::endl;

    // Vectorc(NumberOfParticles)
    Eigen::VectorXd Vectorc = calGradientc(square, re_phi, barphi, barpower, bartheta);
    exportVector_CSV(Vectorc, "csv/Vectorc.csv");
    if (Vectorc.array().isNaN().any()) {
        std::cerr << "NaN detected Vector c" << std::endl;
    }

    // std::cout << "Vectorc : " << std::endl;
    // std::cout << Vectorc << std::endl;

    // Vectord(NumberOfParticles)
    Eigen::VectorXd Vectord = calGradientd(square, re_phi, barphi, doublebarphi, barpower);
    exportVector_CSV(Vectord, "csv/Vectord.csv");
    if (Vectord.array().isNaN().any()) {
        std::cerr << "NaN detected Vector d" << std::endl;
    }

    // std::cout << "Vectord : " << std::endl;
    // std::cout << Vectord << std::endl;

    // Set Vectore
    Vectore.head(3 * NumberOfParticles) = Vectord;
    Vectore.segment(3 * NumberOfParticles, NumberOfParticles) = Vectorc;
    Vectore.tail(NumberOfParticles) = Vectorb;

    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Vectore : " << std::endl;
    std::cout << Vectore << std::endl;

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

    // std::cout << "VectorDeltaTheta : " << std::endl;
    // std::cout << VectorDeltaTheta << std::endl;

    // std::cout << "VectorDeltaPower : " << std::endl;
    // std::cout << VectorDeltaPower << std::endl;

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

    // 実行時間の測定終了
    auto end = std::chrono::high_resolution_clock::now();
    // 実行時間の計算
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

// 正定値行列か判定
void checkHessian(const Eigen::MatrixXd& H) {
    // EigenのSelfAdjointEigenSolverを使用して固有値を計算
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(H);

    if (eigenSolver.info() != Eigen::Success) {
        std::cerr << "固有値分解に失敗しました。" << std::endl;
        return;
    }

    // 固有値を取得
    Eigen::VectorXd eigenValues = eigenSolver.eigenvalues();

    std::cout << "固有値:\n" << eigenValues << std::endl;

    // 探索方向が正しいか確認
    bool isPositiveDefinite = true;
    for (int i = 0; i < eigenValues.size(); ++i) {
        if (eigenValues[i] <= 0) {
            isPositiveDefinite = false;
            break;
        }
    }

    if (isPositiveDefinite) {
        std::cout << "ヘッセ行列は正定値です。探索方向は正しいです。" << std::endl;
    }
    else {
        std::cout << "ヘッセ行列は正定値ではありません。探索方向を再評価する必要があります。" << std::endl;
    }
}

// Line Search 条件式
bool checkArmijo(const Eigen::VectorXd VectorA, const Eigen::VectorXd VectorB) {
    for (int i = 0; i < VectorA.size(); i++) {
        if (VectorA(i) > VectorB(i)) {
            return false;
        }
    }

    return true;
}

// 描画と画像保存を統合した関数
void renderAndSave(Square square, int repetitionTime) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // バッファの消去

    // カメラの更新を反映
    projection_and_modelview(g_Camera);

    // 描画処理（例: 原点に赤い点を描画）
    glColor3f(0.5, 0.0, 0.0);
    drawSquare(square, 10.0);
    Ground();

    // 描画を強制的に行う
    glFlush();
    glutSwapBuffers();

    // 画像保存
    std::vector<unsigned char> pixels(3 * WindowWidth * WindowHeight);

    // OpenGL からピクセルデータを取得
    glReadPixels(0, 0, WindowWidth, WindowHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // ファイル名の生成
    char filename[100];
    snprintf(filename, sizeof(filename), "result/output_image_%d.ppm", repetitionTime);

    // PPM形式で保存
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    // PPM ヘッダーの書き出し
    file << "P6\n" << WindowWidth << " " << WindowHeight << "\n255\n";

    // ピクセルデータの書き出し（上下反転して保存）
    for (int y = WindowHeight - 1; y >= 0; --y) {
        file.write(reinterpret_cast<const char*>(pixels.data() + y * WindowWidth * 3), WindowWidth * 3);
    }

    file.close();
    std::cout << "Image saved as " << filename << std::endl;
}

// ベクトルの長さを計算
double calculateDistance(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return (a - b).norm();
}

// ベクトルが直交しているかを判定
bool isOrthogonal(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return std::abs(a.dot(b)) < EPSILON;
}

// 辺の長さを基準に頂点が立方体かどうか判定
bool isCube(const std::vector<Eigen::Vector3d>& vertices, double& edgeLength) {
    if (vertices.size() != 8) {
        std::cerr << "頂点数が8ではありません。" << std::endl;
        return false;
    }

    // 頂点間の全ての距離を計算
    std::vector<double> distances;
    for (size_t i = 0; i < vertices.size(); ++i) {
        for (size_t j = i + 1; j < vertices.size(); ++j) {
            distances.push_back(calculateDistance(vertices[i], vertices[j]));
        }
    }

    // 距離を小さい順にソート
    std::sort(distances.begin(), distances.end());

    // 辺の長さ、面対角線の長さ、空間対角線の長さを分類
    for (int i = 0; i < 12; i++) {
        edgeLength += distances[i];
    }
    // std::cout << edgeLength/ distances.size() << std::endl;
    // edgeLength = distances[0];
    edgeLength = edgeLength / 12;

    double faceDiagonal = edgeLength * std::sqrt(2);
    double spaceDiagonal = edgeLength * std::sqrt(3);

    // 辺の長さ、面対角線、空間対角線の本数をカウント
    int edgeCount = std::count_if(distances.begin(), distances.end(),
        [edgeLength](double d) { return std::abs(d - edgeLength) < EPSILON; });
    int faceDiagonalCount = std::count_if(distances.begin(), distances.end(),
        [faceDiagonal](double d) { return std::abs(d - faceDiagonal) < EPSILON; });
    int spaceDiagonalCount = std::count_if(distances.begin(), distances.end(),
        [spaceDiagonal](double d) { return std::abs(d - spaceDiagonal) < EPSILON; });

    // 本数がそれぞれ正しいかを確認
    if (edgeCount != 12 || faceDiagonalCount != 12 || spaceDiagonalCount != 4) {
        return false;
    }

    // 各頂点から直交する3つのベクトルを確認
    for (size_t i = 0; i < vertices.size(); ++i) {
        std::vector<Eigen::Vector3d> edges;
        for (size_t j = 0; j < vertices.size(); ++j) {
            if (i != j && std::abs(calculateDistance(vertices[i], vertices[j]) - edgeLength) < EPSILON) {
                edges.push_back(vertices[j] - vertices[i]);
            }
        }

        // 直交条件を確認
        if (edges.size() != 3 ||
            !isOrthogonal(edges[0], edges[1]) ||
            !isOrthogonal(edges[0], edges[2]) ||
            !isOrthogonal(edges[1], edges[2])) {
            return false;
        }
    }

    return true;
}