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
// 関数プロトタイプ（後に呼び出す関数名と引数の宣言）
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
// メイン関数
//----------------------------------------------------
int main(int argc, char* argv[]) {
	g_Camera.setEyePoint(Eigen::Vector3d{ 0.0, -100.0, 5.0 });
	g_Camera.lookAt(Eigen::Vector3d{ 0.0, 500.0, 0.0 }, Eigen::Vector3d{ 0.0, 0.0, 1.0 });
	g_Camera.setFocalLength(0.047);  // 対象物との距離

	glutInit(&argc, argv);//環境の初期化
	glutInitWindowPosition(WindowPositionX, WindowPositionY); //ウィンドウの位置の指定
	glutInitWindowSize(WindowWidth, WindowHeight); //ウィンドウサイズの指定
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH); //ディスプレイモードの指定
	glutCreateWindow(WindowTitle);  //ウィンドウの作成
	glutIdleFunc(Idle); //プログラムアイドル状態時(暇な時)に呼び出される関数
	glutKeyboardFunc(Keyboard);//キーボード入力時に呼び出される関数を指定する
	glutDisplayFunc(Display); //描画時に呼び出される関数を指定する
	glutMouseFunc(mouse);
	glutMotionFunc(mouseDrag);
	Initialize(); //初期設定の関数を呼び出す
	glutMainLoop();

	return 0;
}
//----------------------------------------------------
// 初期設定の関数
//----------------------------------------------------
void Initialize(void) {
	glClearColor(1.0, 1.0, 1.0, 1.0); //背景色
	glEnable(GL_DEPTH_TEST);//デプスバッファを使用：glutInitDisplayMode() で GLUT_DEPTH を指定する

	gluPerspective(30.0, (double)WindowWidth / (double)WindowHeight, 0.1, 1000.0); //透視投影法の視体積gluPerspactive(th, w/h, near, far);
}

//----------------------------------------------------
// 描画の関数
//----------------------------------------------------
void Display(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // バッファの消去

	// カメラの更新を確実に反映
	projection_and_modelview(g_Camera);

	glColor4f(0.5f, 0.0f, 0.0f, 1.0f);
	// printf("Simulation Time : %d回目\n", SimulationTime);
	SimulationTime++;
	fem(SimulationTime);
	// fem_vector(SimulationTime);
	input_key = false;

	glFlush();
	glutSwapBuffers();  // ダブルバッファリングの場合は必要
}

void Idle() {
	glutPostRedisplay(); //glutDisplayFunc()を１回実行する
}

void mouseDrag(int x, int y)
{
	const int _dx = x - mx;
	const int _dy = y - my;  // y方向のドラッグを追加
	mx = x; my = y;

	const double dx = double(_dx) / double(WindowWidth);
	const double dy = double(_dy) / double(WindowHeight);
	const double scale = 2.0;

	g_Camera.rotateCameraInLocalFrameFixLookAt(-dx * scale); // 横方向の回転
	g_Camera.rotateCameraInVerticalFrameFixLookAt(-dy * scale); // 縦方向の回転を追加
	glutPostRedisplay();
}


void mouseDown(int x, int y)
{
	mx = x; my = y;
}

void mouse(int button, int state, int x, int y)
{
	// 左クリックの処理だけを残す
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