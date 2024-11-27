#ifndef __DRAW_H__
#define __DRAW_H__

#include <Eigen/Dense>
#include <vector>

#include "../include/Square.h"

void drawPoint(double x, double y, double z, float size);
void drawVector(Eigen::Vector3d StartPoint, Eigen::Vector3d EndPoint, float width);
void drawSquare(Square square, float scale_factor);
void drawSquareAndVector(Square square, Eigen::VectorXd update_phi, float scale_factor);
void Ground();
void draw();

#endif
