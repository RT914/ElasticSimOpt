#include <GL/freeglut.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>

#include "../Draw/Draw.h"
#include "../include/Square.h"
#include "../include/FEM.h"
#include "../include/NewtonRaphsonMethod.h"
// #include "export.h"



// fem_for_key用
int calculation_times = 0;

Eigen::Vector3d gravity{ 0.0, 0.0, -9.81 };

Eigen::VectorXd new_phi;

Eigen::Vector3d calConflict(Eigen::Vector3d vel, Eigen::Vector3d pos)
{
	if (pos.z() <= 0.001) {
		return { vel.x(), vel.y(), 0.0 };
	}

	return vel;
};

Square fem(Square square, int SimulationTime)
{
	/*for (int i = 0; i < NumberOfParticles; i++) {
		std::cout << square.points[i].position[0] << std::endl;
		std::cout << square.points[i].position[1] << std::endl;
		std::cout << square.points[i].position[2] << std::endl;
	}*/
	
	new_phi = Newton(square, SimulationTime);
	for (int i = 0; i < NumberOfParticles; i++) {
		// 座標の更新
		square.points[i].previous_position[0] = square.points[i].position[0];
		square.points[i].previous_position[1] = square.points[i].position[1];
		square.points[i].previous_position[2] = square.points[i].position[2];

		square.points[i].position[0] = new_phi(3 * i);
		square.points[i].position[1] = new_phi(3 * i + 1);
		square.points[i].position[2] = new_phi(3 * i + 2);
	}

	glColor3f(0.5, 0.0, 0.0);
	drawSquare(square, 10.0);
	Ground();

	return square;
};

void fem_vector(Square square, int SimulationTime)
{
	if (SimulationTime == 1) {
		new_phi = NewtonIteration(square);
	}

	glColor3f(0.5, 0.0, 0.0);
	drawSquareAndVector(square, new_phi, 10.0);
	Ground();
};