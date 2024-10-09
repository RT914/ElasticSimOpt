#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <iomanip> 
#include <vector>
#include <fstream>
#include <chrono>

#include "../include/NewtonRaphsonMethod.h"
#include "../include/Square.h"
#include "../include/FEM.h"
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

int looptimes = 0;

Eigen::VectorXd Newton(Square square) {

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


    // test
    
    /*
    // b の確認
    Eigen::VectorXd Vectorb = calGradientb(square, re_phi, barphi, bartheta);
    if (Vectorb.array().isNaN().any()) {
        std::cerr << "NaN detected" << std::endl;
    }
    std::cout << "Vectorb elements: " << std::endl;
    std::cout << std::setprecision(6);  // 6桁の精度を指定
    for (int i = 0; i < Vectorb.size(); ++i) {
        std::cout << "i = " << i << " : " << Vectorb(i) << std::endl;
    }
    std::cout << std::endl;
    */

    /*
    // c の確認
    Eigen::VectorXd Vectorc = calGradientc(square, re_phi, barphi, barpower, bartheta);
    if (Vectorc.array().isNaN().any()) {
        std::cerr << "NaN detected" << std::endl;
    }
    std::cout << "Vectorc elements: " << std::endl;
    std::cout << std::setprecision(6);  // 6桁の精度を指定
    for (int i = 0; i < Vectorc.size(); ++i) {
        std::cout << "i = " << i << " : " << Vectorc(i) << std::endl;
    }
    std::cout << std::endl;
    */
    

    
    // d の確認
    Eigen::VectorXd Vectord = calGradientd(square, re_phi, barphi, doublebarphi, barpower);
    if (Vectord.array().isNaN().any()) {
        std::cerr << "NaN detected" << std::endl;
    }
    std::cout << "Vectord elements: " << std::endl;
    std::cout << std::setprecision(6);  // 6桁の精度を指定
    for (int i = 0; i < Vectord.size(); ++i) {
        std::cout << "i = " << i << " : " << Vectord(i) << std::endl;
    }
    std::cout << std::endl;
    

    
    // Eigen::MatrixXd MatrixN = calHessianN(square, barphi, barpower);
    // exportMatrix_CSV(MatrixN, "csv/MatrixChi1_test.csv");

    /*
    const int kNumSection = 3; // 各区間の分割数
    const double kWidth = square.dx / kNumSection; // 分割の正規化
    const int kNum = 2 * kNumSection; // 全区間の分割数
    const int AllkNum = pow(kNum, 3);// 全次元の全区間分割数
    const double volume_element = pow(kWidth, 3);

    Eigen::VectorXd cal_points(kNum);
    int index = 0;
    for (int offset = -1; offset <= 0; offset++) {
        for (int divIndex = 0; divIndex < kNumSection; divIndex++) {
            cal_points(index) = static_cast<double>(offset) + 1.0 / (2.0 * kNumSection) + divIndex * kWidth;
            index++;
        }
    }
    
    
    // 体積変化率の計算
    std::vector<double> detF_vector(NumberOfParticles, 0.0);

    // 内挿関数の計算
    // 区間分割
    for (int xd = 0; xd < kNum; xd++) {
        for (int yd = 0; yd < kNum; yd++) {
            for (int zd = 0; zd < kNum; zd++) {
                Eigen::Vector3d cal_point(cal_points(xd), cal_points(yd), cal_points(zd));

                std::cout << "-------------------------------------------------------------" << std::endl;

                for (int xi = 0; xi < NumberOfParticles; xi++) {
                    Eigen::Vector3i grid_xi = FlatToGrid(xi);

                    Eigen::Vector3d P_xi = { barphi(3 * xi), barphi(3 * xi + 1), barphi(3 * xi + 2) };

                    double W = HatFunction(cal_point(0) - P_xi(0)) * HatFunction(cal_point(1) - P_xi(1)) * HatFunction(cal_point(2) - P_xi(2));

                    double J = calRiemannJ(cal_point, grid_xi, re_phi, barphi, NumberOfParticles, 1.0);
                    
                    if (W <= 0.0) continue;
                    double detF
                        = volume_element * J * W;

                    if (J <= 0.0) {
                        std::cout << J << std::endl;
                    }
                    else {
                        std::cout << "xi = " << xi << "  *  " << J << std::endl;
                    }

                    detF_vector[xi] += detF;
                                      
                }

            }
        }
    }

    for (int a = 0; a < NumberOfParticles; a++) {
        std::cout << detF_vector[a] << std::endl;
    }
    */
    
    //test end

    
    /*
    while (NormVectorDelta > 1.0e-6) {

        // Hessian
        Eigen::MatrixXd MatrixS = Eigen::MatrixXd::Zero(5 * NumberOfParticles, 5 * NumberOfParticles);

        // Gradient
        Eigen::VectorXd Vectore = Eigen::VectorXd::Zero(5 * NumberOfParticles);

        // Calculation Hessian
        // MatrixN(3 * NumberOfParticles, 3 * NumberOfParticles)
        Eigen::MatrixXd MatrixN = calHessianN(square, barphi, barpower);
        // std::cout << "MatrixN = (" << MatrixN.rows() << ", " << MatrixN.cols() << ")" << std::endl;
        if (MatrixN.array().isNaN().any()) {
            std::cerr << "NaN detected" << std::endl;
        }

        // MatrixO(NumberOfParticles, 3 * NumberOfParticles)
        Eigen::MatrixXd MatrixO = calHessianO(square, barphi);
        // std::cout << "MatrixO = (" << MatrixO.rows() << ", " << MatrixO.cols() << ")" << std::endl;
        if (MatrixO.array().isNaN().any()) {
            std::cerr << "NaN detected" << std::endl;
        }

        // MatrixP(3 * NumberOfParticles, NumberOfParticles)
        Eigen::MatrixXd MatrixP = calHessianP(square, barphi);
        // std::cout << "MatrixP = (" << MatrixP.rows() << ", " << MatrixP.cols() << ")" << std::endl;
        if (MatrixP.array().isNaN().any()) {
            std::cerr << "NaN detected" << std::endl;
        }

        // MatrixQ(NumberOfParticles, NumberOfParticles)
        Eigen::MatrixXd MatrixQ = calHessianQ(square);
        // std::cout << "MatrixQ = (" << MatrixQ.rows() << ", " << MatrixQ.cols() << ")" << std::endl;
        if (MatrixQ.array().isNaN().any()) {
            std::cerr << "NaN detected" << std::endl;
        }


        //MatrixR(NumberOfParticles, NumberOfParticles)
        Eigen::MatrixXd MatrixR = calHessianR(square, bartheta);
        // std::cout << "MatrixR = (" << MatrixR.rows() << ", " << MatrixR.cols() << ")" << std::endl;
        if (MatrixR.array().isNaN().any()) {
            std::cerr << "NaN detected" << std::endl;
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
        MatrixS.block(3 * NumberOfParticles, 0, NumberOfParticles, 3 * NumberOfParticles) = MatrixP_t;
        MatrixS.block(3 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixQ_t;
        MatrixS.block(4 * NumberOfParticles, 3 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixR_t;
        MatrixS.block(4 * NumberOfParticles, 4 * NumberOfParticles, NumberOfParticles, NumberOfParticles) = MatrixQ_t;


        // Calculation Gradient
        // Vectorb(NumberOfParticles)
        Eigen::VectorXd Vectorb = calGradientb(square, barphi, bartheta);
        std::cout << "Vectorb = (" << Vectorb << ")" << std::endl;
        // std::cout << "Vectorb = (" << Vectorb.size() << ")" << std::endl;
        if (Vectorb.array().isNaN().any()) {
            std::cerr << "NaN detected" << std::endl;
        }

        // Vectorc(NumberOfParticles)
        Eigen::VectorXd Vectorc = calGradientc(square, barpower, bartheta);
        // std::cout << "Vectorc = (" << Vectorc.size() << ")" << std::endl;
        if (Vectorc.array().isNaN().any()) {
            std::cerr << "NaN detected" << std::endl;
        }

        // Vectord(NumberOfParticles)
        Eigen::VectorXd Vectord = calGradientd(square, barphi, doublebarphi, barpower);
        // std::cout << "Vectord = (" << Vectord.size() << ")" << std::endl;
        if (Vectord.array().isNaN().any()) {
            std::cerr << "NaN detected" << std::endl;
        }


        // Set Vectore
        Vectore.head(3 * NumberOfParticles) = Vectord;
        Vectore.segment(3 * NumberOfParticles, NumberOfParticles) = Vectorb;
        Vectore.tail(NumberOfParticles) = Vectorc;


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


        NormVectorDelta = VectorDelta.norm();
        NormVectorDeltaPhi = VectorDeltaPhi.norm();
        NormVectorDeltaTheta = VectorDeltaTheta.norm();
        NormVectorDeltaPower = VectorDeltaPower.norm();

        // Update
        barphi += VectorDeltaPhi;
        bartheta += VectorDeltaTheta;
        barpower += VectorDeltaPower;

        looptimes++;

        std::cout << "norm : " << NormVectorDelta << std::endl;
        std::cout << "反復回数：　" << looptimes << "回" << std::endl;
    }
    */

    // 実行時間の測定終了
    auto end = std::chrono::high_resolution_clock::now();
    // 実行時間の計算
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    

	return barphi;
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