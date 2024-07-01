#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <vector>

#include "../include/NewtonRaphsonMethod.h"
#include "../include/Square.h"
#include "../include/FEM.h"
// 出力確認のためのinclude
#include "../include/utils/Interpolation_util.h"


int looptimes = 0;

Eigen::VectorXd Newton(Square square) {
	// ベクトルの大きさは [3×NumberOfParticles]　とする．
	Eigen::VectorXd barphi(3 * NumberOfParticles);

	// 座標の取得
	for (int i = 0; i < square.points.size(); i++) {
		barphi(3 * i) = square.points[i].position[0];
		barphi(3 * i + 1) = square.points[i].position[1];
		barphi(3 * i + 2) = square.points[i].position[2];
	}

	
    /* 内挿関数計算の確認 */
    
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

    /*
    for (int xi = 0; xi < NumberOfParticles; xi++) {
        Eigen::Vector3i grid_xi = FlatToGrid(xi);

        std::cout << "Xi = " << xi << std::endl;
        std::cout << "answer = " << RiemannSumForDetF(barphi, grid_xi, square.dx) << std::endl;
        std::cout << std::endl;
    }
    */

	return barphi;
}
