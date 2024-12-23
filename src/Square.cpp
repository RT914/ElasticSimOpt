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
	int SideNumber = N - 1;
	double dx = 2.0 / SideNumber; // 格子間距離
	Square square(pos, dx, SideNumber);
	Eigen::Vector3d position;
	Eigen::Vector3d re_position;
	Eigen::Vector3d velocity;
	velocity << 0.0, 0.0, 0.0;
	Eigen::Vector3i grid_node;
	double theta = 1.0;
	double power = 0.0;

	// 剪断変形の変化倍率 （格子間距離×変化倍率）
	double magnification = 0.5;

	double square_x = square.position(0);
	double square_y = square.position(1);
	double square_z = square.position(2);
	Eigen::Vector3d base_point;
	// base_point << pos.x() - 1.0, pos.y() - 1.0, pos.z() - 1.0; // 参照座標と等しい
	// base_point << pos.x() - 0.5, pos.y() - 0.5, pos.z() - 0.5; // 体積変形(縮小状態)
	// base_point << pos.x() - 1.2, pos.y() - 1.2, pos.z() - 1.2; // 体積変形(膨張状態)
	base_point << pos.x() - 1.0 - magnification * dx, pos.y() - 1.0, pos.z() - 1.0; // 剪断変形
	Eigen::Vector3d base_refpoint;
	base_refpoint << pos.x() - 1.0, pos.y() - 1.0, pos.z() - 1.0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {

				// 可変の座標：mpmの粒子点と同義
				// 剪断変形
				double x = i * dx + base_point.x() + 2 * magnification * dx / SideNumber * k;
				double y = j * dx + base_point.y();
				double z = k * dx + base_point.z();
				
				// 体積変形(膨張状態)
				/*double x = i * dx * 1.2 + base_point.x();
				double y = j * dx * 1.2 + base_point.y();
				double z = k * dx * 1.2 + base_point.z();*/

				// 体積変形(縮小状態)
				/*double x = i * dx * 0.5 + base_point.x();
				double y = j * dx * 0.5 + base_point.y();
				double z = k * dx * 0.5 + base_point.z();*/
				
				// 初期状態
				/*double x = i * dx + base_point.x();
				double y = j * dx + base_point.y();
				double z = k * dx + base_point.z();*/

				// 不変の座標：mpmの格子点と同義
				double re_x = i * dx + base_refpoint.x();
				double re_y = j * dx + base_refpoint.y();
				double re_z = k * dx + base_refpoint.z();

				// printf("%d回目\n", i);
				position << x, y, z;
				// printf("x:%f, y:%f, z:%f\n", x, y, z);
				re_position << re_x, re_y, re_z;
				// printf("x:%f, y:%f, z:%f\n", re_x, re_y, re_z);
				grid_node << i, j, k;
				// power* i * 10;
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