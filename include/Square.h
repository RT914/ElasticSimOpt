#ifndef __SQUARE_H__
#define __SQUARE_H__

#include <Eigen/Dense>
#include <vector>

using namespace std;

struct Point
{
	Point(Eigen::Vector3d& pos, Eigen::Vector3d& re_pos, Eigen::Vector3d& v, double& t, Eigen::Vector3i& gri, double& p)
		: position(pos), reference_position(re_pos), velocity(v), theta(t), grid_node(gri), power(p) {};

	Eigen::Vector3d position;
	Eigen::Vector3d reference_position;
	Eigen::Vector3d velocity;
	double theta;
	Eigen::Vector3i grid_node;
	double power;
};

struct Square
{
	Square(Eigen::Vector3d& pos, double delta_x, int SideNumber) : position(pos), dx(delta_x), SideNumber(SideNumber) {};

	Eigen::Vector3d position;
	vector<Point> points;
	double dx;
	int SideNumber;
};

Square createSquare(int particles);

#endif
