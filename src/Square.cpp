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
	int SideNumber = N - 1;
	double dx = 2.0 / SideNumber; // ŠiqŠÔ‹——£
	Square square(pos, dx, SideNumber);
	Eigen::Vector3d position;
	Eigen::Vector3d re_position;
	Eigen::Vector3d velocity;
	velocity << 0.0, 0.0, 0.0;
	Eigen::Vector3i grid_node;
	double theta = 1.0;
	double power = 0.0;

	// ™’’f•ÏŒ`‚Ì•Ï‰»”{—¦ iŠiqŠÔ‹——£~•Ï‰»”{—¦j
	double magnification = 0.5;
	// ‘ÌÏ•ÏŒ`‚Ìk¬”{—¦
	double reduct = 0.3;
	// ‘ÌÏ•ÏŒ`‚Ìk¬”{—¦
	double expanse = 1.3;

	double square_x = square.position(0);
	double square_y = square.position(1);
	double square_z = square.position(2);
	Eigen::Vector3d base_point;
	// base_point << pos.x() - 1.0, pos.y() - 1.0, pos.z() - 1.0; // QÆÀ•W‚Æ“™‚µ‚¢
	// base_point << pos.x() - reduct, pos.y() - reduct, pos.z() - reduct; // ‘ÌÏ•ÏŒ`(k¬ó‘Ô)
	// base_point << pos.x() - expanse, pos.y() - expanse, pos.z() - expanse; // ‘ÌÏ•ÏŒ`(–c’£ó‘Ô)
	// base_point << pos.x() - 1.0 - magnification * dx, pos.y() - 1.0, pos.z() -1.0; // ™’’f•ÏŒ`
	base_point << pos.x() - reduct - magnification * dx * reduct, pos.y() - reduct, pos.z() - reduct; // ™’’f•ÏŒ`‚Æ‘ÌÏ•ÏŒ`(k¬ó‘Ô)
	// base_point << pos.x() - expanse - magnification * dx * expanse, pos.y() - expanse, pos.z() - expanse; // ™’’f•ÏŒ`‚Æ‘ÌÏ•ÏŒ`(Šg‘åó‘Ô)
	Eigen::Vector3d base_refpoint;
	base_refpoint << pos.x() - 1.0, pos.y() - 1.0, pos.z() - 1.0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {

				// ‰Â•Ï‚ÌÀ•WFmpm‚Ì—±q“_‚Æ“¯‹`
				// ™’’f•ÏŒ`
				/*double x = i * dx + base_point.x() + 2 * magnification * dx / SideNumber * k;
				double y = j * dx + base_point.y();
				double z = k * dx + base_point.z();*/
				
				// ‘ÌÏ•ÏŒ`(–c’£ó‘Ô)
				/*double x = i * dx * expanse + base_point.x();
				double y = j * dx * expanse + base_point.y();
				double z = k * dx * expanse + base_point.z();*/

				// ‘ÌÏ•ÏŒ`(k¬ó‘Ô)
				/*double x = i * dx * reduct + base_point.x();
				double y = j * dx * reduct + base_point.y();
				double z = k * dx * reduct + base_point.z();*/

				// ™’’f•ÏŒ`‚Æ‘ÌÏ•ÏŒ`(k¬ó‘Ô)
				double x = i * dx * reduct + base_point.x() + 2 * reduct * magnification * dx / SideNumber * k;
				double y = j * dx * reduct + base_point.y();
				double z = k * dx * reduct + base_point.z();

				// ™’’f•ÏŒ`‚Æ‘ÌÏ•ÏŒ`(Šg‘åó‘Ô)
				/*double x = i * dx * expanse + base_point.x() + 2 * expanse * magnification * dx / SideNumber * k;
				double y = j * dx * expanse + base_point.y();
				double z = k * dx * expanse + base_point.z();*/
				
				// ‰Šúó‘Ô
				/*double x = i * dx + base_point.x();
				double y = j * dx + base_point.y();
				double z = k * dx + base_point.z();*/

				// •s•Ï‚ÌÀ•WFmpm‚ÌŠiq“_‚Æ“¯‹`
				double re_x = i * dx + base_refpoint.x();
				double re_y = j * dx + base_refpoint.y();
				double re_z = k * dx + base_refpoint.z();

				// printf("%d‰ñ–Ú\n", i);
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
// ”ÍˆÍ“à‚Ì—”‚Ì¶¬
//----------------------------------------------------
double generation_random(double min, double max)
{
	return((max - min) * ((float)rand() / RAND_MAX)) + min;
};