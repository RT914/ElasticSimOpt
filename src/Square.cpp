#include <Eigen/Dense>
#include <stdio.h>
#include "../include/Square.h"
#include <iostream>
#include <math.h>

double generation_random(double min, double max);

Square createSquare(int N)
{
	// 立方体の中心座標
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

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {

				// 可変の座標：mpmの粒子点と同義
				
				// 剪断変形1
				/*double x;
				if (k == 0) {
					x = i * dx + base_point.x() + 0.1;
				}
				else if (k == 2) {
					x = i * dx + base_point.x() - 0.1;
				}
				else {
					x = i * dx + base_point.x();
				}
				double y = j * dx + base_point.y();
				double z = k * dx + base_point.z();*/

				// 剪断変形2
				/*double x = i * dx + base_point.x();
				double y = j * dx + base_point.y();
				double z;
				if (i == 0) {
					z = k * dx + base_point.z() + 0.1;
				}
				else if (i == 2) {
					z = k * dx + base_point.z() - 0.1;
				}
				else {
					z = k * dx + base_point.z();
				}*/
				
				// 体積変形(膨張状態)
				double x = i * dx * 2.0 + base_point.x();
				double y = j * dx * 2.0 + base_point.y();
				double z = k * dx * 2.0 + base_point.z();

				// 体積変形(縮小状態)
				/*double x = i * dx * 0.9 + base_point.x();
				double y = j * dx * 0.9 + base_point.y();
				double z = k * dx * 0.9 + base_point.z();*/
				
				// 初期状態
				/*double x = i * dx + base_point.x();
				double y = j * dx + base_point.y();
				double z = k * dx + base_point.z();*/

				// 不変の座標：mpmの格子点と同義
				double re_x = i * dx + base_point.x();
				double re_y = j * dx + base_point.y();
				double re_z = k * dx + base_point.z();

				// printf("%d回目\n", i);
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
// 範囲内の乱数の生成
//----------------------------------------------------
double generation_random(double min, double max)
{
	return((max - min) * ((float)rand() / RAND_MAX)) + min;
};