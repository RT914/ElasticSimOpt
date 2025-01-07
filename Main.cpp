#include <GL/freeglut.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

#include "include/Fem.h"
#include "Draw/Draw.h"
#include "Camera/Camera.h"
#include "Window/Window.h"

int SimulationTime = 0;
bool input_key = false;
int mx, my;

Camera g_Camera;

//----------------------------------------------------
// �֐��v���g�^�C�v�i��ɌĂяo���֐����ƈ����̐錾�j
//----------------------------------------------------
void Initialize(void);
void Display(void);
void Idle();
void projection_and_modelview(const Camera& in_Camera);
void mouseDrag(int x, int y);
void mouseDown(int x, int y);
void mouse(int button, int state, int x, int y);
void Keyboard(unsigned char key, int x, int y);

//----------------------------------------------------
// ���C���֐�
//----------------------------------------------------
int main(int argc, char* argv[]) {
	g_Camera.setEyePoint(Eigen::Vector3d{ 0.0, -100.0, 5.0 });
	g_Camera.lookAt(Eigen::Vector3d{ 0.0, 500.0, 0.0 }, Eigen::Vector3d{ 0.0, 0.0, 1.0 });
	g_Camera.setFocalLength(0.047);  // �Ώە��Ƃ̋���

	glutInit(&argc, argv);//���̏�����
	glutInitWindowPosition(WindowPositionX, WindowPositionY); //�E�B���h�E�̈ʒu�̎w��
	glutInitWindowSize(WindowWidth, WindowHeight); //�E�B���h�E�T�C�Y�̎w��
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH); //�f�B�X�v���C���[�h�̎w��
	glutCreateWindow(WindowTitle);  //�E�B���h�E�̍쐬
	glutIdleFunc(Idle); //�v���O�����A�C�h����Ԏ�(�ɂȎ�)�ɌĂяo�����֐�
	glutKeyboardFunc(Keyboard);//�L�[�{�[�h���͎��ɌĂяo�����֐����w�肷��
	glutDisplayFunc(Display); //�`�掞�ɌĂяo�����֐����w�肷��
	glutMouseFunc(mouse);
	glutMotionFunc(mouseDrag);
	Initialize(); //�����ݒ�̊֐����Ăяo��
	glutMainLoop();

	return 0;
}
//----------------------------------------------------
// �����ݒ�̊֐�
//----------------------------------------------------
void Initialize(void) {
	glClearColor(1.0, 1.0, 1.0, 1.0); //�w�i�F
	glEnable(GL_DEPTH_TEST);//�f�v�X�o�b�t�@���g�p�FglutInitDisplayMode() �� GLUT_DEPTH ���w�肷��

	gluPerspective(30.0, (double)WindowWidth / (double)WindowHeight, 0.1, 1000.0); //�������e�@�̎��̐�gluPerspactive(th, w/h, near, far);
}

//----------------------------------------------------
// �`��̊֐�
//----------------------------------------------------
void Display(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // �o�b�t�@�̏���

	// �J�����̍X�V���m���ɔ��f
	projection_and_modelview(g_Camera);

	glColor4f(0.5f, 0.0f, 0.0f, 1.0f);
	// printf("Simulation Time : %d���\n", SimulationTime);
	SimulationTime++;
	fem(SimulationTime);
	// fem_vector(SimulationTime);
	input_key = false;

	glFlush();
	glutSwapBuffers();  // �_�u���o�b�t�@�����O�̏ꍇ�͕K�v
}

void Idle() {
	glutPostRedisplay(); //glutDisplayFunc()���P����s����
}

void mouseDrag(int x, int y)
{
	const int _dx = x - mx;
	const int _dy = y - my;  // y�����̃h���b�O��ǉ�
	mx = x; my = y;

	const double dx = double(_dx) / double(WindowWidth);
	const double dy = double(_dy) / double(WindowHeight);
	const double scale = 2.0;

	g_Camera.rotateCameraInLocalFrameFixLookAt(-dx * scale); // �������̉�]
	g_Camera.rotateCameraInVerticalFrameFixLookAt(-dy * scale); // �c�����̉�]��ǉ�
	glutPostRedisplay();
}


void mouseDown(int x, int y)
{
	mx = x; my = y;
}

void mouse(int button, int state, int x, int y)
{
	// ���N���b�N�̏����������c��
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		mouseDown(x, y);
	}
}

void Keyboard(unsigned char key, int x, int y) {
	const double zoomSpeed = 0.05;

	switch (key)
	{
	case 'w':
		input_key = true;
		break;
	default:
		break;
	}
}