#include <GL/freeglut.h>
#include <stdio.h>
#include "Draw.h"
#include "../include/Square.h"

//----------------------------------------------------
// “_‚Ì•`‰æ
//----------------------------------------------------
void drawPoint(double x, double y, double z, float size) {
	glPointSize(size);
	glBegin(GL_POINTS);
	glVertex3d(x, y, z);
	glEnd();
}


//----------------------------------------------------
// —§•û‘Ì‚Ì•`‰æ
//----------------------------------------------------
void drawSquare(Square square)
{
	int N = square.one_dimension_point_number;
	float point_size = 3.0;

	for (int i = 0; i < square.points.size(); i++)
	{
		double pos_x = square.points[i].position(0);
		double pos_y = square.points[i].position(1);
		double pos_z = square.points[i].position(2);
		drawPoint(pos_x, pos_y, pos_z, point_size);
	}


	// ’¸“_ŠÔ‚Ìü‚Ì•`‰æ
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N - 1; k++) {

				int num = i * pow(N, 2) + j * (N)+k;

				glColor3d(0.0, 0.5, 0.0);
				glBegin(GL_LINES);
				glVertex3d(square.points[num].position(0), square.points[num].position(1), square.points[num].position(2));
				glVertex3d(square.points[num + 1].position(0), square.points[num + 1].position(1), square.points[num + 1].position(2));

			}
		}
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N - 1; j++) {
			for (int k = 0; k < N; k++) {

				int num = i * pow(N, 2) + j * (N)+k;

				glColor3d(0.0, 0.5, 0.0);
				glBegin(GL_LINES);
				glVertex3d(square.points[num].position(0), square.points[num].position(1), square.points[num].position(2));
				glVertex3d(square.points[num + N].position(0), square.points[num + N].position(1), square.points[num + N].position(2));

			}
		}
	}

	for (int i = 0; i < N - 1; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				int num = i * pow(N, 2) + j * (N)+k;

				glColor3d(0.0, 0.5, 0.0);
				glBegin(GL_LINES);
				glVertex3d(square.points[num].position(0), square.points[num].position(1), square.points[num].position(2));
				glVertex3d(square.points[num + pow(N, 2)].position(0), square.points[num + pow(N, 2)].position(1), square.points[num + pow(N, 2)].position(2));

			}
		}
	}



};

//----------------------------------------------------
// ’n–Ê‚Ì•`‰æ
//----------------------------------------------------
void Ground()
{
	double ground_max_x = 300.0;
	double ground_max_y = 300.0;
	glColor3d(0.2, 0.8, 0.8);  // ‘å’n‚ÌF
	glBegin(GL_LINES);
	for (double ly = -ground_max_y; ly <= ground_max_y; ly += 10.0) {
		glVertex3d(-ground_max_x, ly, 0);
		glVertex3d(ground_max_x, ly, 0);
	}
	for (double lx = -ground_max_x; lx <= ground_max_x; lx += 10.0) {
		glVertex3d(lx, ground_max_y, 0);
		glVertex3d(lx, -ground_max_y, 0);
	}
	glEnd();
}

//----------------------------------------------------
// ‘S‘Ì‚Ì•`‰æ
//----------------------------------------------------
void draw()
{
	glColor3d(0, 0, 0);  // ‘å’n‚ÌF
	Ground();
}