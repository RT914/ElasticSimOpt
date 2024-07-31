#include <Eigen/Dense>
#include <stdio.h>
#include "../include/Square.h"
#include <iostream>
#include <math.h>

double generation_random(double min, double max);

Square createSquare(int N)
{
	// —§•û‘Ì‚Ì’†SÀ•W
	Eigen::Vector3d pos;
	pos << 0.0, 0.0, 0.0;
	int one_d_point_num = N;
	double range = (N - 1) / 2.0;
	double dx = 2 * range / (N - 1);
	Square square(pos, dx, one_d_point_num);
	Eigen::Vector3d velocity;
	Eigen::Vector3d position;
	Eigen::Vector3d re_position;
	Eigen::Vector3i grid_node;
	double theta = 1.0;
	double power = 0.0;
	velocity << 0.0, 0.0, 0.0;

	double square_x = square.position(0);
	double square_y = square.position(1);
	double square_z = square.position(2);
	Eigen::Vector3d base_point;
	base_point << pos.x() - range, pos.y() - range, pos.z() - range;
	srand(2);

	double c = 1.0 * 0.1;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {

				/*
				double x = i * dx + base_point.x() +c * ((rand() % 2) - 1);
				double y = j * dx + base_point.y() +c * ((rand() % 2) - 1);
				double z = k * dx + base_point.z() +c * ((rand() % 2) - 1);
				*/

				double x = i * dx + base_point.x();
				double y = j * dx + base_point.y();
				double z = k * dx + base_point.z();

				double re_x = i * dx + base_point.x();
				double re_y = j * dx + base_point.y();
				double re_z = k * dx + base_point.z();

				// printf("%d‰ñ–Ú\n", i);
				position << x, y, z;
				// printf("x:%f, y:%f, z:%f\n", x, y, z);
				re_position << re_x, re_y, re_z;
				// printf("x:%f, y:%f, z:%f\n", re_x, re_y, re_z);
				grid_node << i, j, k;
				Point p = Point(position, re_position, velocity, theta, grid_node, power);
				square.points.emplace_back(p);
			}
		}
	}

	return square;
};

//----------------------------------------------------
// ”ÍˆÍ“à‚Ì—”‚Ì¶¬
//----------------------------------------------------
double generation_random(double min, double max)
{
	return((max - min) * ((float)rand() / RAND_MAX)) + min;
};