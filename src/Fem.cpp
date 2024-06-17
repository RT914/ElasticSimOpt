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

Square square = createSquare(NumberOfOneDemensionParticles);

// fem_for_key—p
int calculation_times = 0;

double dt = 1.0e-3;
Eigen::Vector3d gravity{ 0.0, 0.0, -9.81 };

Eigen::VectorXd new_phi;

void calVelocity()
{
	for (int i = 0; i < pow(NumberOfOneDemensionParticles, 3); i++) {
		square.points[i].velocity = square.points[i].velocity + gravity * dt;

	}
}


Eigen::Vector3d calConflict(Eigen::Vector3d vel, Eigen::Vector3d pos)
{
	if (pos.z() <= 0.001) {
		return { vel.x(), vel.y(), 0.0 };
	}

	return vel;
};


void calPosition()
{
	for (int i = 0; i < pow(NumberOfOneDemensionParticles, 3); i++) {
		square.points[i].velocity = calConflict(square.points[i].velocity, square.points[i].position); //Õ“Ë”»’èEŒvŽZ
		square.points[i].position = square.points[i].position + square.points[i].velocity * dt;
	}
};


void fem(int SimulationTime)
{

	if (SimulationTime == 1) {
		new_phi = Newton(square);
	}

	// update position
	for (int i = 0; i < NumberOfParticles; i++) {
		square.points[i].position[0] = new_phi(3 * i);
		square.points[i].position[1] = new_phi(3 * i + 1);
		square.points[i].position[2] = new_phi(3 * i + 2);
	}

	glColor3f(0.5, 0.0, 0.0);
	drawSquare(square);
	Ground();
};