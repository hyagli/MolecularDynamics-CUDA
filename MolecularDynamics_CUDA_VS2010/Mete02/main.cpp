/**********************************************************************
14.01.2013
Hüseyin YAÐLI
huseyinyagli@gmail.com
***********************************************************************/

#include "main.h"


int main(int argc, char **argv)
{
	InputFileName = "input.txt";
	if(argc > 1)
		InputFileName = argv[1];

	ReadAtoms();
	InitOpenGl(argc, argv);
	LookFromTheSide();
	BuildDisplayScene();
	RodInfo();
	glutMainLoop();
	return 0;             
}

// Displays the previously prepared "AtomDisplayList"
void Display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	Camera.Render();
	//Draw atoms:
	glCallList(AtomDisplayList);
	//finish rendering
	glutSwapBuffers();
}

// OpenGL related stuff
void InitOpenGl(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(1000,700);
	glutCreateWindow("METE565");
	glutDisplayFunc(Display);
	glutReshapeFunc(ReshapeEvent);
	glutKeyboardFunc(KeyDown);

	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	
	glShadeModel(GL_SMOOTH);

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	// Colors for different type of atoms
	Colors[0].resize(3);
	Colors[0][0] = 1.0f; Colors[0][1] = 1.0f; Colors[0][2] = 0.0f; // yellow
	Colors[1].resize(3);
	Colors[1][0] = 0.0f; Colors[1][1] = 0.0f; Colors[1][2] = 1.0f; // blue
	Colors[2].resize(3);
	Colors[2][0] = 0.0f; Colors[2][1] = 1.0f; Colors[2][2] = 0.0f; // green
	Colors[3].resize(3);
	Colors[3][0] = 1.0f; Colors[3][1] = 0.5f; Colors[3][2] = 0.0f; // orange
	Colors[4].resize(3);
	Colors[4][0] = 1.0f; Colors[4][1] = 0.0f; Colors[4][2] = 1.0f; // purple
	Colors[5].resize(3);
	Colors[5][0] = 0.0f; Colors[5][1] = 1.0f; Colors[5][2] = 1.0f; // turquoise
	Colors[6].resize(3);
	Colors[6][0] = 1.0f; Colors[6][1] = 1.0f; Colors[6][2] = 1.0f; // white
}

// Read parameters and coordinates from the input file
void ReadAtoms() {
	ifstream inputFile;
	inputFile.open(InputFileName);
	if(!inputFile)
	{
		cout << "Error opening input file." << endl;
		exit(1);
	}

	string line;

	// Entire line is read
	getline(inputFile, line);
	
	// Allocate memory for atoms
	AtomPositions.resize(5000);

	// Read the atom coordinates
	TotalAtoms = 0;
	while(inputFile.eof() == false){
		getline(inputFile, line);
		if(line.length() > 0){
			istringstream(line) >> AtomPositions[TotalAtoms].x >> AtomPositions[TotalAtoms].y >> AtomPositions[TotalAtoms].z;
			TotalAtoms++;
		}
	}
	AtomPositions.resize(TotalAtoms);
}


void DrawAtoms(bool picking){
	double prevX = 0, prevY = 0, prevZ = 0;
	bool changedColor = false;
	
	// Give the first color
	double3 colorGold;
	colorGold.x = 1.0;    colorGold.y = 0.75;    colorGold.z = 0.0;
	glColor3d(colorGold.x, colorGold.y, colorGold.z);

	for(int i=0; i<TotalAtoms; i++){		
		if(picking) // If this is not a draw but right-click event
			glPushName(i);  // Needed for atom selection purposes
		else 
		{
			if(!ClickZone.empty()){ // If there is a selected atom group
				set<int>::iterator it = ClickZone.find(i);
				if(it != ClickZone.end()){ // Found in the selection group
					glColor3f(1.0, 0.0, 0.0); // give it a red color
					changedColor = true;
				}
			}
		}

		glTranslated(AtomPositions[i].x-prevX, AtomPositions[i].y-prevY, AtomPositions[i].z-prevZ);  // position of current atom
		if(SolidShapes)
			glutSolidSphere(AtomRadius, 10, 10);
		else
			glutWireSphere(AtomRadius, 10, 10);

		// Store this position for the next atom position calculation
		prevX = AtomPositions[i].x;
		prevY = AtomPositions[i].y;
		prevZ = AtomPositions[i].z;

		if(picking) // If this is not a draw but right-click event
			glPopName(); // Needed for atom selection purposes
		else
		{
			if(changedColor) // If we changed the color to red in order to draw a selected atom, revert the color back to normal atom type color
			{ 
				glColor3d(colorGold.x, colorGold.y, colorGold.z);
				changedColor = false;
			}
		}
	}
}

// Change all atom type sizes by 20%
void IncreaseAtomSize()
{
	AtomRadius += AtomRadius/5.0;
	BuildDisplayScene();
	Display();
}

void DecreaseAtomSize()
{
	AtomRadius -= AtomRadius/5.0;
	BuildDisplayScene();
	Display();
}

// Keyboard events
void KeyDown(unsigned char key, int x, int y)
{	
	switch (key) 
	{
		case 27:		//ESC
			PostQuitMessage(0);
			break;
		case 'a':		
			Camera.StrafeRight(0-sensitivity);
			Display();
			break;
		case 'd':		
			Camera.StrafeRight(sensitivity);
			Display();
			break;
		case 'w':		
			Camera.MoveForward(0-sensitivity) ;
			Display();
			break;
		case 's':		
			Camera.MoveForward(sensitivity) ;
			Display();
			break;
		case 'f':
			Camera.MoveUpward(0-sensitivity);
			Display();
			break;
		case 'r':
			Camera.MoveUpward(sensitivity);
			Display();
			break;
		case 'z':
			Camera.RotateZ(-1.0f);
			Display();
			break;
		case 'c':
			Camera.RotateZ(1.0f);
			Display();
			break;
		case '+':
			IncreaseAtomSize();
			break;
		case '-':
			DecreaseAtomSize();
			break;
		case 'x':
			// Change the atom drawing to wire or solid
			SolidShapes = !SolidShapes;
			BuildDisplayScene();   // Rebuild the atom drawings
			Display();
			break;
		case 'q':
			// Reset the atom colors and sizes
			ClickZone.clear();  // colored zone
			ReadAtoms();        // re-read all the parameters to get the original atom sizes
			BuildDisplayScene(); // Rebuild the atom drawings
			Display();
			break;
		case 'o':
			LookFromFront();
			Display();
			break;
		case 'p':
			LookFromTheSide();
			Display();
			break;
	}
}

// Mouse movements
void mouseMove(int x, int y) {
	// If the mouse is moved while left clicked, look around (rotate camera)
	if(mouseLeftButtonPressed){
		// How much the mouse position changed determines the amount of rotation
		Camera.RotateY((x - mousePrevX) / 15.0f);
		Camera.RotateX((y - mousePrevY) / 15.0f);
		// Store current position in order to use in the next event
		mousePrevX = x;
		mousePrevY = y;
		Display(); // Refresh screen
	}
}

// Mouse clicks
void mouseButton(int button, int state, int x, int y) {
	// Start motion if the left button is pressed
	if (button == GLUT_LEFT_BUTTON) {
		// when the button is pressed
		if (state == GLUT_DOWN) {
			mouseLeftButtonPressed = true;
			mousePrevX = x;
			mousePrevY = y;
		}
		// when the button is released
		else {
			mouseLeftButtonPressed = false;
		}
	}
	// Right click
	else if(button == GLUT_RIGHT_BUTTON){
		if(state == GLUT_DOWN){
			PickAtom(x, y);
		}
	}
	// Middle button scrolls
	else if(button == 3){ // scroll up
		Camera.MoveForward(-(sensitivity*2));
		Display();
	}
	else if(button == 4){ // scroll down
		Camera.MoveForward(sensitivity*2);
		Display();
	}
}

// Build the atoms scene to be drawn later in Display()
void BuildDisplayScene()
{
	// Delete the previous scene
	if(AtomDisplayList != 0)
		glDeleteLists(AtomDisplayList, 1);

	// Create new scene
	AtomDisplayList = glGenLists(1);
	glNewList(AtomDisplayList, GL_COMPILE);
	DrawAtoms(false);
	glEndList();
}

// Window size changed event
void ReshapeEvent(int x, int y)
{
	if (y == 0 || x == 0) return;  //Nothing is visible then, so return

	//Set a new projection matrix
	glMatrixMode(GL_PROJECTION);  
	glLoadIdentity();

	//Angle of view: 45 degrees
	//Near clipping plane distance: 0.01
	//Far clipping plane distance: 1000.0
	AspectRatio = (GLdouble)x/(GLdouble)y;
	gluPerspective(45.0, AspectRatio, 0.05, 1000.0);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, x, y);  //Use the whole window for rendering
}

// Atom selection after a right click
void PickAtom(int cursorX, int cursorY)
{
	// Clear current atom selection
	ClickZone.clear();

	// OpenGL stuff to create a selection environment
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	GLint viewport[4];

	GLuint selectBuf[1024];
	glSelectBuffer(1024, selectBuf);
	glRenderMode(GL_SELECT);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glGetIntegerv(GL_VIEWPORT, viewport);
	
	// Click selection area
	gluPickMatrix(cursorX-1, viewport[3]-cursorY-1, 2, 2, viewport);

	gluPerspective(45.0, AspectRatio, 0.05, 1000.0);
	glMatrixMode(GL_MODELVIEW);
	glInitNames();

	glLoadIdentity();
	Camera.Render();

	// Call draw atoms with picking parameter true
	DrawAtoms(true);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glFlush();
	GLint hits = glRenderMode(GL_RENDER); // Call render to get the hits
	
	ProcessHits(hits,selectBuf,0);
}

// Process clicked atoms (multiple atoms may have been clicked in viewing axis Z, we should get the top most one)
void ProcessHits (GLint hits, GLuint buffer[], int sw)
{
	GLint i, numberOfNames = 0;
	GLuint names, *ptr, minZ,*ptrNames;

	ptr = (GLuint *) buffer;
	minZ = 0xffffffff;
	for (i = 0; i < hits; i++) {	
		names = *ptr;
		ptr++;
		if (*ptr < minZ) {
			numberOfNames = names;
			minZ = *ptr;
			ptrNames = ptr+2;
		}

		ptr += names+2;
	}
	// If at least one atom is in the clicked area
	if (numberOfNames > 0) {
		ptr = ptrNames;
		ProcessClick(*ptr); // Process the atom click
	}
	// No atom clicked
	else{
		cout << endl << "You didn't click an atom!" << endl;
		// Rebuild the scene to reset colors
		BuildDisplayScene();
		Display();
	}
}

// Atom click
void ProcessClick(int atomNum)
{
	cout << endl << "You picked atom " << atomNum << endl;
	cout << "Coordinates: " << AtomPositions[atomNum].x << "  " << AtomPositions[atomNum].y << "  " << AtomPositions[atomNum].z << endl;
	
	// Recursively mark the selected atoms
	MarkClicked(atomNum);

	// Rebuild the scene to show the colored atoms
	BuildDisplayScene();
	Display();

	// Write selected number and percentage
	double zonePercent = double(ClickZone.size()) / double(TotalAtoms) * 100.0;
	cout << ClickZone.size() << " atoms selected ( " << zonePercent << " percent)." << endl;
}

// Recursive function to find the neighbour atoms of this atom, and find the neighbours of those atoms
void MarkClicked(int atomNum)
{
	double x, y, z, dx, dy, dz, distance;
	
	// Current atom coordinates
	x = AtomPositions[atomNum].x;
	y = AtomPositions[atomNum].y;
	z = AtomPositions[atomNum].z;

	// Search all atoms to find neighbours
	for(int i=0; i<TotalAtoms; i++){
		dx = abs(x-AtomPositions[i].x);
		if(dx > BondDistance)
			continue;
		dy = abs(y-AtomPositions[i].y);
		if(dy > BondDistance)
			continue;
		dz = abs(z-AtomPositions[i].z);
		if(dz > BondDistance)
			continue;
		
		distance = sqrt(dx*dx+dy*dy+dz*dz);
		if(distance < BondDistance){
			pair<set<int>::iterator, bool> result = ClickZone.insert(i);
			if(result.second == true) // inserted
				MarkClicked(i);
		}
	}
}

void LookFromTheSide()
{
	double3 mins, maxs;
	mins.x = AtomPositions[0].x;
	mins.y = AtomPositions[0].y;
	mins.z = AtomPositions[0].z;
	maxs.x = AtomPositions[0].x;
	maxs.y = AtomPositions[0].y;
	maxs.z = AtomPositions[0].z;

	for(int i=1; i<TotalAtoms; i++){
		if(AtomPositions[i].x < mins.x)
			mins.x = AtomPositions[i].x;
		if(AtomPositions[i].y < mins.y)
			mins.y = AtomPositions[i].y;
		if(AtomPositions[i].z < mins.z)
			mins.z = AtomPositions[i].z;
		if(AtomPositions[i].x > maxs.x)
			maxs.x = AtomPositions[i].x;
		if(AtomPositions[i].y > maxs.y)
			maxs.y = AtomPositions[i].y;
		if(AtomPositions[i].z > maxs.z)
			maxs.z = AtomPositions[i].z;
	}

	double rodLength = (maxs.z - mins.z);
	Camera.Position.z = float(mins.z + (rodLength / 2));
	Camera.Position.x = float(maxs.x + 50);
	Camera.Position.y = float(mins.y + ((maxs.y - mins.y) / 2));

	Camera.ViewDir = F3dVector(-1.0f, 0.0f, 0.0f);
	Camera.RightVector = F3dVector(0.0f, 0.0f, -1.0f);
}

void LookFromFront()
{
	double3 mins, maxs;
	mins.x = AtomPositions[0].x;
	mins.y = AtomPositions[0].y;
	mins.z = AtomPositions[0].z;
	maxs.x = AtomPositions[0].x;
	maxs.y = AtomPositions[0].y;
	maxs.z = AtomPositions[0].z;

	for(int i=1; i<TotalAtoms; i++){
		if(AtomPositions[i].x < mins.x)
			mins.x = AtomPositions[i].x;
		if(AtomPositions[i].y < mins.y)
			mins.y = AtomPositions[i].y;
		if(AtomPositions[i].z < mins.z)
			mins.z = AtomPositions[i].z;
		if(AtomPositions[i].x > maxs.x)
			maxs.x = AtomPositions[i].x;
		if(AtomPositions[i].y > maxs.y)
			maxs.y = AtomPositions[i].y;
		if(AtomPositions[i].z > maxs.z)
			maxs.z = AtomPositions[i].z;
	}

	double rodLength = (maxs.z - mins.z);
	Camera.Position.z = float(mins.z - 50);
	Camera.Position.x = float(mins.x + ((maxs.x - mins.x) / 2));
	Camera.Position.y = float(mins.y + ((maxs.y - mins.y) / 2));

	Camera.ViewDir = F3dVector(0.0f, 0.0f, 1.0f);
	Camera.RightVector = F3dVector(-1.0f, 0.0f, 0.0f);
}

void RodInfo()
{
	double3 mins, maxs;
	mins.x = AtomPositions[0].x;
	mins.y = AtomPositions[0].y;
	mins.z = AtomPositions[0].z;
	maxs.x = AtomPositions[0].x;
	maxs.y = AtomPositions[0].y;
	maxs.z = AtomPositions[0].z;

	for(int i=1; i<TotalAtoms; i++){
		if(AtomPositions[i].x < mins.x)
			mins.x = AtomPositions[i].x;
		if(AtomPositions[i].y < mins.y)
			mins.y = AtomPositions[i].y;
		if(AtomPositions[i].z < mins.z)
			mins.z = AtomPositions[i].z;
		if(AtomPositions[i].x > maxs.x)
			maxs.x = AtomPositions[i].x;
		if(AtomPositions[i].y > maxs.y)
			maxs.y = AtomPositions[i].y;
		if(AtomPositions[i].z > maxs.z)
			maxs.z = AtomPositions[i].z;
	}

	double rodLength = (maxs.z - mins.z);
	cout << "Rod length: " << rodLength << endl;
}