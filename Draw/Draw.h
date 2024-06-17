#ifndef __DRAW_H__
#define __DRAW_H__

#include <Eigen/Dense>
#include <vector>

#include "../include/Square.h"

void drawPoint(double x, double y, double z, float size);
void drawSquare(Square square);
void Ground();
void draw();

#endif
