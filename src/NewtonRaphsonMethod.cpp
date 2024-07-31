#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
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

    // 圧力
    Eigen::VectorXd barpower(NumberOfParticles);
    // 体積変化率
    Eigen::VectorXd bartheta(NumberOfParticles);

	// 座標の取得
	for (int i = 0; i < square.points.size(); i++) {
		barphi(3 * i) = square.points[i].position[0];
		barphi(3 * i + 1) = square.points[i].position[1];
        barphi(3 * i + 2) = square.points[i].position[2];
        doublebarphi(3 * i) = square.points[i].position[0];
        doublebarphi(3 * i + 1) = square.points[i].position[1];
        doublebarphi(3 * i + 2) = square.points[i].position[2];
        barpower(i) = square.points[i].power;
        bartheta(i) = square.points[i].theta;
	}

    /* 内挿関数計算の確認 */
    /*
    Eigen::VectorXi axis(3);
    axis << 0, 1, 2;

    // VectorXi i_minus_xi, j_minus_xi, k_minus_xi, l_minus_xi initialization with size 3
    Eigen::VectorXi i_minus_xi(3), j_minus_xi(3), k_minus_xi(3), l_minus_xi(3);
    i_minus_xi << 1, 1, 1;
    j_minus_xi << 0, 0, 0;
    k_minus_xi << 0, 0, 0;
    l_minus_xi << 0, 0, 0;

    // MatrixXi m initialization with size 3x4
    Eigen::MatrixXi m(3,4);
    m << i_minus_xi, j_minus_xi, k_minus_xi, l_minus_xi;

    // Printing axis vector
    std::cout << "axis:\n" << axis << std::endl;

    // Printing matrix m
    std::cout << "matrix m:\n" << m << std::endl;

	// std::cout << "answer = " << RiemannSum5(m, axis, 1) << std::endl;
    std::cout << "answer = " << RiemannSum6(m, axis, 1) << std::endl;
    // std::cout << "answer = " << RiemannSum7(m, axis, 1) << std::endl;
    */

    /*
    for (int xi = 0; xi < NumberOfParticles; xi++) {
        Eigen::Vector3i grid_xi = FlatToGrid(xi);

        std::cout << "Xi = " << xi << std::endl;
        std::cout << "answer = " << RiemannSumForDetF(barphi, grid_xi, square.dx) << std::endl;
        std::cout << std::endl;
    }
    */

    /*
    Eigen::Vector3i v1(1, 0, 0);
    Eigen::Vector3i v2(1, -1, 0);
    Eigen::Vector3i v3(0, 1, 0);
    Eigen::Vector3i v4(0, 0, 0);

    std::vector<Eigen::Vector3i> diff_vectors = {
            v1, v2, v3, v4,
    };

    if (allElementsWithinTwo(diff_vectors)) {
        std::cout << "True" << endl;
    }else{
        std::cout << "False" << endl;
    }
    */
    
    
    // Hessian
    Eigen::MatrixXd MatrixN(3 * NumberOfParticles, 3 * NumberOfParticles);
    Eigen::MatrixXd MatrixO(NumberOfParticles, 3 * NumberOfParticles);
    Eigen::MatrixXd MatrixP(3 * NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd MatrixQ(NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd MatrixR(NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd MatrixS(5 * NumberOfParticles, 5 * NumberOfParticles);

    // Hessian Traspose
    Eigen::MatrixXd MatrixN_t(3 * NumberOfParticles, 3 * NumberOfParticles);
    Eigen::MatrixXd MatrixO_t(3 * NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd MatrixP_t(NumberOfParticles, 3 * NumberOfParticles);
    Eigen::MatrixXd MatrixQ_t(NumberOfParticles, NumberOfParticles);
    Eigen::MatrixXd MatrixR_t(NumberOfParticles, NumberOfParticles);

    // Gradient
    Eigen::VectorXd Vectorb(NumberOfParticles);
    Eigen::VectorXd Vectorc(NumberOfParticles);
    Eigen::VectorXd Vectord(3 * NumberOfParticles);
    Eigen::VectorXd Vectore(5 * NumberOfParticles);

    // Calculation Hessian
    MatrixN = calHessianN(square, barphi, barpower);
    // MatrixO = calHessianO(square, barphi);
    // MatrixP = calHessianP(square, barphi);
    // MatrixQ = calHessianQ(square);
    // MatrixR = calHessianR(square, bartheta);

    // Calculation Gradient
    // Vectorb = calGradientb(square, barphi, bartheta);
    // Vectorc = calGradientc(square, barphi, bartheta);
    // Vectord = calGradientd(square, barphi, doublebarphi, barpower);

    // All Hessian Transpose
    // MatrixN_t = MatrixN.transpose();
    // MatrixO_t = MatrixO.transpose();
    // MatrixP_t = MatrixP.transpose();
    // MatrixQ_t = MatrixQ.transpose();
    // MatrixR_t = MatrixR.transpose();

    

    /*
    while (NormVectorDelta > 1.0e-6) {

        for (int i = 0; i < 5 * NumberOfParticles; i++) {
        // Set Gradient
        if (i < 3 * NumberOfParticles) {
            Vectore(i) = Vectord(i);
        }
        else if (3 * NumberOfParticles <= i < 4 * NumberOfParticles) {
            Vectore(i) = Vectorb(i - 3);
        }
        else {
            Vectore(i) = Vectorc(i - 4);
        }


        for (int j = 0; j < 5 * NumberOfParticles; j++) {
            // Set Hessian
            if (i < 3 * NumberOfParticles && j < 3 * NumberOfParticles) {
                MatrixS(i, j) = MatrixN_t(i, j);
            }
            else if (i < 3 * NumberOfParticles && 4 * NumberOfParticles <= j < 5 * NumberOfParticles) {
                MatrixS(i, j) = MatrixO_t(i, j - 4);
            }
            else if (3 * NumberOfParticles <= i < 4 * NumberOfParticles && j < 3 * NumberOfParticles) {
                MatrixS(i, j) = MatrixP_t(i - 3, j);
            }
            else if (3 * NumberOfParticles <= i < 4 * NumberOfParticles && 3 * NumberOfParticles <= j < 4 * NumberOfParticles) {
                MatrixS(i, j) = MatrixQ_t(i - 3, j - 3);
            }
            else if (4 * NumberOfParticles <= i < 5 * NumberOfParticles && 3 * NumberOfParticles <= j < 4 * NumberOfParticles) {
                MatrixS(i, j) = MatrixQ_t(i - 4, j - 3);
            }
            else if (4 * NumberOfParticles <= i < 5 * NumberOfParticles && 4 * NumberOfParticles <= j < 5 * NumberOfParticles) {
                MatrixS(i, j) = MatrixQ_t(i - 4, j - 4);
            }
            else {
                MatrixS(i, j) = 0.0;
            }
        }
    }
        
        Eigen::FullPivLU<Eigen::MatrixXd> LU(MatrixS);
        Eigen::VectorXd VectorDelta = LU.solve(Vectore);

        Eigen::VectorXd VectorDeltaPhi(3 * NumberOfParticles);
        Eigen::VectorXd VectorDeltaTheta(NumberOfParticles);
        Eigen::VectorXd VectorDeltaPower(NumberOfParticles);

        for (int i = 0; i < NumberOfParticles; i++) {
            VectorDeltaPhi(3 * i) = VectorDelta(3 * i);
            VectorDeltaPhi(3 * i) = VectorDelta(3 * i + 1);
            VectorDeltaPhi(3 * i) = VectorDelta(3 * i + 2);
            VectorDeltaTheta(i) = VectorDelta(i + 3 * NumberOfParticles);
            VectorDeltaPower(i) = VectorDelta(i + 4 * NumberOfParticles);
        }

        NormVectorDelta = VectorDelta.norm();
        NormVectorDeltaPhi = VectorDeltaPhi.norm();
        NormVectorDeltaTheta = VectorDeltaTheta.norm();
        NormVectorDeltaPower = VectorDeltaPower.norm();

        std::cout << "norm : " << NormVectorDelta << std::endl;

        // Update
        barphi += VectorDeltaPhi;
        bartheta += VectorDeltaTheta;
        barpower += VectorDeltaPower;

        looptimes++;
    }

    */

    // 実行時間の測定終了
    auto end = std::chrono::high_resolution_clock::now();
    // 実行時間の計算
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

	return barphi;
}
