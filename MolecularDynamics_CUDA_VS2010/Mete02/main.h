#include <GL\glut.h>
#include <windows.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <list>
#include <sstream>

#include "camera.h"

using namespace std;

struct double3{
	double x,y,z;
};

string InputFileName;

// Screen
CCamera Camera;
GLdouble AspectRatio;

// Atom variables
int TotalAtoms;
vector<double3> AtomPositions;
GLuint AtomDisplayList = 0;
bool SolidShapes = true;
double BondDistance = 4.5;
double AtomRadius = 0.75;

// Colors
vector<vector<float>> Colors(10);

// Mouse variables
int mousePrevX = 0;
int mousePrevY = 0;
bool mouseLeftButtonPressed = false;
set<int> ClickZone;
float sensitivity = 1.0;


void ReadAtoms();
void DrawAtoms(bool picking);
void InitOpenGl(int argc, char **argv);
void ReshapeEvent(int x, int y);
void KeyDown(unsigned char key, int x, int y);
void mouseMove(int x, int y);
void mouseButton(int button, int state, int x, int y);
void Display();
void BuildDisplayScene();
void IncreaseAtomSize();
void DecreaseAtomSize();
void PickAtom(int cursorX, int cursorY);
void ProcessHits(GLint hits, GLuint buffer[], int sw);
void ProcessClick(int atomNum);
void MarkClicked(int atomNum);
void LookFromTheSide();
void LookFromFront();
void RodInfo();