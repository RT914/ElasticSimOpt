#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <vector>

#include "../include/NewtonRaphsonMethod.h"
#include "../include/Square.h"
#include "../include/FEM.h"
// �o�͊m�F�̂��߂�include
#include "../include/utils/Interpolation_util.h"


int looptimes = 0;

Eigen::VectorXd Newton(Square square) {
	// �x�N�g���̑傫���� [3�~NumberOfParticles]�@�Ƃ���D
	Eigen::VectorXd barphi(3 * NumberOfParticles);

	// ���W�̎擾
	for (int i = 0; i < square.points.size(); i++) {
		barphi(3 * i) = square.points[i].position[0];
		barphi(3 * i + 1) = square.points[i].position[1];
		barphi(3 * i + 2) = square.points[i].position[2];
	}

	return barphi;
}
