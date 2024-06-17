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

	return barphi;
}
