#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sstream>
#include <cstdlib>
#include <iomanip>

using namespace std;

bool OpenFiles();
bool Input();
bool Solve();
void Force();
void PeriodicBoundary();
bool FileExists(const string& filename);
bool WriteBsFile();
bool CloseFiles();
void ShowStatus();
void PrintInitial();
void PrintPeriodic();
void PrintFinal();
string GetTime();
void PrintElapsedTime();
double Randum(double U, double S);
void MaxWell();
void Reposition();
double Distance(int i, int j);
void SortAtoms(char sortAxis);
void GenerateLatis();
string GetValueStr();
int GetValueInt();
double GetValueDouble();
double GetValueFloat();
string SkipSpace();
string ReadLine();


struct stMd{
	string Title;	// First line of the input file
	int MDSL;		// MD-STEPS
	int IAVL;		// ROLLING AVERAGE STEPS
	int IPPL;		// PERIODIC PRINTING STEPS
	int ISCAL;		// SCALING STEP FOR TEMPERATURE
	int IPD;		// PRINT CONTROL PARAMETER, IF 0 NO PRINT FOR DISTANCES
	double TE;		// TEMPERATURE (IN KELVIN)
	int NA;			// NUMBER OF MOVING PARTICLES
	int NN;			// NUMBER OF TOTAL ATOMS
	int IPBC;		// PBC PARAMETER, IF 0 NO PBC IS APPLIED
	int LAYER;		// NUMBER OF LINES IN COORDINATE GENERATIONS

	// Periodic Boundary Properties
	double PP[3];	// PERIODIC BOUNDARY LENGTHS
	double PA[3];   // Lowest coordinates in all three axis
	double PB[3];	// Highest coordinates in all three axis (PA[i]+PP[i])
	double PL[3];	// Periodic boundary medium (PP / 2)

	// Coordinates of atoms
	double X[5000];
	double Y[5000];
	double Z[5000];

	// Force arrays
	double FF[5000][3];

	// Velocity arrays
	double VV[5000][3];
	double VT[5000];

	// Energy variables
	double EPOT,EKIN,ETOT,TEMP,SCFAC,TCALC,EKINA,TCALAV;
	int MDS;				// Current md simulation step;

	// Potential parameters
	double RM;
	double DT;
	double A1;
	double A2;
	double RL1;
	double RL2;
	double AL1;
	double AL2;
	double D21;
	double D22;

	double BK;
} Md;

// Files
ifstream fileInp;
ofstream fileOut, fileEne;

int main()
{
	if(OpenFiles() == false)
		return -1;

	if(Input() == false)
		return -1;

	if(Solve() == false)
		return -1;

	return 0;
}

bool OpenFiles()
{
	if(FileExists("mdse.out"))
	{
		cout << "mdse.out already exists. Enter 'y' to overwrite, 'n' to exit: ";
		string answer;
		cin >> answer;
		if(answer != "y"){
			cout << "Stopping." << endl;
			return false;
		}
	}

	fileInp.open("mdse.inp");
	if(fileInp.good() == false)
	{
		cout << "mdse.inp couldn't be opened for reading. Stopping." << endl;
		return false;
	}

	fileOut.open("mdse.out");
	if(fileInp.good() == false)
	{
		cout << "mdse.out couldn't be opened for writing. Stopping." << endl;
		return false;
	}
	fileOut << fixed << setprecision(6);

	fileEne.open("mdse.ene");
	if(fileEne.good() == false)
	{
		cout << "mdse.ene couldn't be opened for writing. Stopping." << endl;
		return false;
	}
	fileEne << fixed << setprecision(6);

	return true;
}

bool FileExists(const string& filename)
{
	struct stat buf;
	if (stat(filename.c_str(), &buf) != -1)
	{
		return true;
	}
	return false;
}

bool Input()
{
	// Potential parameters for Cu
	Md.RM = 63.546;
	Md.DT = 0.9E-15;
	Md.A1 = 110.766008;
	Md.A2 = -46.1649783;
	Md.RL1 = 2.09045946;
	Md.RL2 = 1.49853083;
	Md.AL1 = 0.394142248;
	Md.AL2 = 0.207225507;
	Md.D21 = 0.436092895;
	Md.D22 = 0.245082238;

	double FACM = 0.103655772E-27;
	Md.BK = 8.617385E-05;
	Md.RM = Md.RM * FACM;

	try
	{
		// Read the title
		Md.Title = ReadLine();

		// Skip the second line
		ReadLine();
		
		// Read MDSL, IAVL, IPPL, ISCAL, IPD, TE, NA, LAYER, IPBC, PP(1), PP(2), PP(3)
		Md.MDSL = GetValueInt();
		Md.IAVL = GetValueInt();
		Md.IPPL = GetValueInt();
		Md.ISCAL = GetValueInt();
		Md.IPD = GetValueInt();
		Md.TE = GetValueDouble();
		Md.NA = GetValueInt();
		Md.LAYER = GetValueInt();
		Md.IPBC = GetValueInt();
		Md.PP[0] = GetValueDouble();
		Md.PP[1] = GetValueDouble();
		Md.PP[2] = GetValueDouble();

		// Generate atom coordinates
		GenerateLatis();
		// Sort atoms by the z axis
		SortAtoms('Z');
		// Find the periodic boundary limits if PBC is applied
		PeriodicBoundary();
	}
	catch(exception& e)
	{
		cout << "Error in Input(): " << e.what() << endl;
		return false;
	}

	return true;
}

bool Solve()
{
	// INITIALIZE SOME VARIABLES AND DEFINE SOME FACTORS
	Md.MDS = 0;				// Current md simulation step;
	int IPP=0;				// Print counter
	double EPAV = 0;		// Average potential energy
	double EKAV = 0;		// Average kinetic energy
	double ETAV = 0;		// Average total energy
	double SCFAV = 0;		// Average scaling factor
	Md.TCALAV = 0;			// System temperature
	int IAV = 0;			// Average counter
	int ISCA = 0;			// Scaling counter

	double FFPR[5000][3];   // Array to store forces from previous step

	// CALCULATE THE INITIAL POTENTIAL ENERGY OF EACH ATOM
	// AND THE INITIAL FORCE THAT EACH ATOM EXPERIENCES
	Force();

	// SET INITIAL VELOCITIES ACC. TO MAXWELL VEL. DISTRIBUTION
	MaxWell();

	fileOut << "# **********   MD STEPS STARTED   **********" << endl;
	fileEne << "# **********   MD STEPS STARTED   **********" << endl;
	fileOut << "#  MDS       EPAV          EKAV          ETAV         TCALAV" << endl;
	fileEne << "#  MDS       EPAV          EKAV          ETAV         TCALAV" << endl;
	fileOut << "# -----  ------------  ------------  ------------  ------------" << endl << "#" << endl;
	fileEne << "# -----  ------------  ------------  ------------  ------------" << endl << "#" << endl;

	// Start Md Steps
	while(Md.MDS < Md.MDSL){
		Md.MDS++;
		IPP++;
		ISCA++;

		// Show status at each 50 steps
		if((Md.MDS % 50) == 0)
			ShowStatus();

		// REPOSITION THE PARTICLES IF PBC IS APPLIED
		if(Md.IPBC != 0)
			Reposition();

		// CALCULATE VELOCITY AND POSITION OF THE PARTICLES
		// USING THE VELOCITY SUMMED FORM OF VERLET ALGORITHM
		// (NVE MD VELOCITY FORM)
		Force();

		// COMPUTE THE POSITIONS AT TIME STEP n+1 AS
		// ri(n+1)=ri(n)+hvi(n)+(1/2m)h2Fi(n)
		for(int i=0; i<Md.NA; i++){
			Md.X[i] = Md.X[i] + Md.DT*Md.VV[i][0] + (pow(Md.DT,2)*Md.FF[i][0]) / (2*Md.RM);
			Md.Y[i] = Md.Y[i] + Md.DT*Md.VV[i][1] + (pow(Md.DT,2)*Md.FF[i][1]) / (2*Md.RM);
			Md.Z[i] = Md.Z[i] + Md.DT*Md.VV[i][2] + (pow(Md.DT,2)*Md.FF[i][2]) / (2*Md.RM);
		}

		// STORE THE FORCES AT TIME STEP Fi(n)
		//memcpy(FFPR, Md.FF, Md.NA*3*sizeof(double));
		for(int i=0; i<Md.NA; i++){
			for(int j=0; j<3; j++){
				FFPR[i][j] = Md.FF[i][j];
			}
		}

		Force();

		// COMPUTE THE VELOCITIES AT TIME STEP n+1 AS
		// vi(n+1)=vi(n)+(h/2m)(Fi(n+1)+Fi(n))
		for(int i=0; i<Md.NA; i++){
			Md.VV[i][0] = Md.VV[i][0] +  Md.DT * (Md.FF[i][0]+FFPR[i][0]) / (2*Md.RM);
			Md.VV[i][1] = Md.VV[i][1] +  Md.DT * (Md.FF[i][1]+FFPR[i][1]) / (2*Md.RM);
			Md.VV[i][2] = Md.VV[i][2] +  Md.DT * (Md.FF[i][2]+FFPR[i][2]) / (2*Md.RM);
			Md.VT[i] = pow(Md.VV[i][0],2) + pow(Md.VV[i][1],2) + pow(Md.VV[i][2],2);
		}

		// CALCULATE THE TEMPERATURE THAT SYSTEM REACHED
		// BY CALCULATING THE KINETIC ENERGY OF EACH ATOM
		Md.EKINA = 0;
		for(int i=0; i<Md.NA; i++)
			Md.EKINA += Md.VT[i];
		Md.EKINA *= Md.RM;
		Md.TCALC = Md.EKINA / (3*Md.NA*Md.BK);

		// CALCULATE THE SCALING FACTOR AND SCALE THE VELOCITIES
		Md.SCFAC = sqrt(Md.TE/Md.TCALC);
		if(ISCA == Md.ISCAL)
		{
			Md.EKIN = 0;
			for(int i=0; i<Md.NA; i++){
				for(int j=0; j<3; j++){
					Md.VV[i][j] *= Md.SCFAC;
				}
				Md.VT[i] = pow(Md.VV[i][0],2) + pow(Md.VV[i][1],2) + pow(Md.VV[i][2],2);
				Md.EKIN += Md.VT[i];
			}
			ISCA = 0;
			Md.EKIN *= Md.RM;
			Md.TCALC = Md.EKIN / (3 * Md.NA * Md.BK);
		}

		// CALCULATE TOTAL ENERGY
		Md.ETOT = Md.EPOT + Md.EKINA;

		// CALCULATE THE AVERAGES OF EPOT, EKINA, ETOT, SCFAC AND TCALC
		EPAV += Md.EPOT;
		EKAV += Md.EKINA;
		ETAV += Md.ETOT;
		SCFAV += Md.SCFAC;
		Md.TCALAV += Md.TCALC;
		IAV++;

		if(IAV < Md.IAVL)
			continue;

		EPAV /= Md.IAVL;
		EKAV /= Md.IAVL;
		ETAV /= Md.IAVL;
		SCFAV /= Md.IAVL;
		Md.TCALAV /= Md.IAVL;

		// PRINT THE AVERAGES

		// Periodic printing of coordinates
		if(IPP == Md.IPPL){
			PrintPeriodic();
			IPP = 0;
		}

		IAV = 0;
		EPAV = 0;
		EKAV = 0;
		ETAV = 0;
		SCFAV = 0;
		Md.TCALAV = 0;

	} // Md Steps Loop

	PrintFinal();

	return true;
}

bool WriteBsFile()
{
	return true;
}

bool CloseFiles()
{
	return true;
}

void ShowStatus()
{

}

string GetTime()
{
	time_t rawtime;
	struct tm * timeinfo;
	char chars[100];
	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	strftime (chars, 100, "%Y.%m.%d %H:%M:%S", timeinfo);
	string final = " DATE AND TIME: ";
	final += chars;
	return final;
}

void PrintElapsedTime()
{
}

void Force()
{
}

void PeriodicBoundary()
{
	if(Md.IPBC == 0)
		return;

	for(int i=0; i<3; i++)
		Md.PL[i] = Md.PP[i] / 2;
	
	// Find smallest coordinates for X, Y and Z coordinates
	Md.PA[0] = Md.X[0];
	Md.PA[1] = Md.Y[0];
	Md.PA[2] = Md.Z[0];
	for(int i=1; i<Md.NN; i++)
	{
		if(Md.PA[0] > Md.X[i])
			Md.PA[0] = Md.X[i];
		if(Md.PA[1] > Md.Y[i])
			Md.PA[1] = Md.Y[i];
		if(Md.PA[2] > Md.Z[i])
			Md.PA[2] = Md.Z[i];
	}

	// Find ending coordinates of working system
	Md.PB[0] = Md.PA[0] + Md.PP[0];
	Md.PB[1] = Md.PA[1] + Md.PP[1];
	Md.PB[2] = Md.PA[2] + Md.PP[2];
}

void PrintInitial()
{
	fileOut << "******************************************************************************************" << endl;
	fileOut << Md.Title;
	fileOut << "******************************************************************************************" << endl << endl;
	fileOut << GetTime() << endl;
}

void PrintPeriodic()
{
}

void PrintFinal()
{
}

double Randum(double U, double S)
{
	return 0.5;
}

// DISTRUBUTES THE VELOCITIES FOR THE ATOMS FOR THE SPECIFIED 
// TEMPERATURE TE ACCORDING TO THE MAXWELL VELOCITY DISTRIBUTION
void MaxWell()
{
	double FAC1 = sqrt(3*Md.BK*Md.TE/Md.RM);
	double U = 0;
	double S = 1;
	double VVX = 0;
	double VVY = 0;
	double VVZ = 0;
	double FAC2 = (2/3) * FAC1 / sqrt(3.0);

	// EQUATING Vmean TO FAC2 
	for(int i=0; i<Md.NA; i++){
		for(int j=0; j<3; j++){
			Md.VV[i][j] = (FAC2 - FAC2*Randum(U,S));
		}
	}

	// CALCULATING AVERAGES
	double VVV = 0;
	for(int i=0; i<Md.NA; i++){
		VVX = VVX + Md.VV[i][0];
		VVY = VVY + Md.VV[i][1];
		VVZ = VVZ + Md.VV[i][2];
	}
	VVX /= Md.NA;
	VVY /= Md.NA;
	VVZ /= Md.NA;
	VVV = VVX*VVX + VVY*VVY + VVZ*VVZ;
	double COSX = VVX / sqrt(VVV);
	double COSY = VVY / sqrt(VVV);
	double COSZ = VVZ / sqrt(VVV);

	// CALCULATING EKIN AND TEMPERATURE WRT THE CALCULATED Vmean
	Md.EKIN = 0.5 * Md.RM * (VVV * (9/4));
	Md.TCALC = Md.EKIN / (1.5 * Md.BK);

	// CALCULATING THE SCALING FACTOR 
	Md.SCFAC = sqrt(Md.TE / Md.TCALC);

	// REDISTRIBUTING THE INITIAL VELOCITIES WRT SCALING FACTOR
	VVV = sqrt(VVV);
	double VVXNEW = COSX * VVV * Md.SCFAC;
	double VVYNEW = COSY * VVV * Md.SCFAC;
	double VVZNEW = COSZ * VVV * Md.SCFAC;
	double XSCALE = (VVXNEW-VVX);
	double YSCALE = (VVYNEW-VVY);
	double ZSCALE = (VVZNEW-VVZ);
	for(int i=0; i<Md.NA; i++){
		Md.VV[i][0] += XSCALE;
		Md.VV[i][1] += YSCALE;
		Md.VV[i][2] += ZSCALE;
		Md.VT[i] = pow(Md.VV[i][0],2) + pow(Md.VV[i][1],2) + pow(Md.VV[i][2],2);
	}

	// CALCULATING AVERAGES  OF SCALED VELOCITIES
	VVX = 0;
	VVY = 0;
	VVZ = 0;
	for(int i=0; i<Md.NA; i++){
		VVX += Md.VV[i][0];
		VVY += Md.VV[i][1];
		VVZ += Md.VV[i][2];
	}
	VVX /= Md.NA;
	VVY /= Md.NA;
	VVZ /= Md.NA;

	// CALCULATING EKIN AND TEMPERATURE WRT THE SCALED Vmean
	VVV = VVX*VVX + VVY*VVY + VVZ*VVZ;
	Md.EKIN = 0.5 * Md.RM * (VVV * (9/4));
	Md.TCALC = Md.EKIN / (1.5 * Md.BK);
}

// REPOSITIONS COORDINATES WHEN ANY MOVING ATOM CROSSES THE BOUNDARY.
void Reposition()
{
	double PAPL, H, B;

	if(Md.PP[0] > 0){
		PAPL = Md.PA[0] + Md.PL[0];
		for(int i=0; i<Md.NA; i++){
			H = (Md.X[i]-PAPL) / Md.PL[0];
			B = H - 2*int(H);
			Md.X[i] = B*Md.PL[0] + PAPL;
		}
	}

	if(Md.PP[1] > 0){
		PAPL = Md.PA[1] + Md.PL[1];
		for(int i=0; i<Md.NA; i++){
			H = (Md.Y[i]-PAPL) / Md.PL[1];
			B = H - 2*int(H);
			Md.Y[i] = B*Md.PL[1] + PAPL;
		}
	}

	if(Md.PP[2] > 0){
		PAPL = Md.PA[2] + Md.PL[2];
		for(int i=0; i<Md.NA; i++){
			H = (Md.Z[i]-PAPL) / Md.PL[2];
			B = H - 2*int(H);
			Md.Z[i] = B*Md.PL[2] + PAPL;
		}
	}
}

// Sorts atoms by the given axis
void SortAtoms(char sortAxis)
{
	double *sortArray;
	if(sortAxis == 'X')
		sortArray = Md.X;
	else if(sortAxis == 'Y')
		sortArray = Md.Y;
	else
		sortArray = Md.Z;

	double tempX, tempY, tempZ;
	for (int i = 0; i < Md.NA; i++)
	{
		for (int j = i+1; j < Md.NA; j++)
		{
			if (sortArray[i] > sortArray[j])
			{
				tempX = Md.X[i];
				tempY = Md.Y[i];
				tempZ = Md.Z[i];

				Md.X[i] = Md.X[j];
				Md.Y[i] = Md.Y[j];
				Md.Z[i] = Md.Z[j];

				Md.X[j] = tempX;
				Md.Y[j] = tempY;
				Md.Z[j] = tempZ;
			}
		}
	}
}

// Generates the atoms according to coordinates and repeat parameters from the input
// In the input, the first 3 numbers are x,y,z coordinates, the second 3 numbers are unit cell lengths 
// and the last 3 numbers specify how many times to copy that atom in x,y,z direction
void GenerateLatis()
{
	// Skip the first line: (W(J,K),K=1,6),(NO(J,K),K=1,3) 
	ReadLine();

	Md.NN = 0;
	for(int i=0; i<Md.LAYER; i++)
	{
		double coordinateX = GetValueDouble();
		double coordinateY = GetValueDouble();
		double coordinateZ = GetValueDouble();
		double unitCellLengthX = GetValueDouble();
		double unitCellLengthY = GetValueDouble();
		double unitCellLengthZ = GetValueDouble();
		int multiplierX = GetValueInt();
		int multiplierY = GetValueInt();
		int multiplierZ = GetValueInt();

		for (int iX = 0; iX < multiplierX; iX++)
		{
			for (int iY = 0; iY < multiplierY; iY++)
			{
				for (int iZ = 0; iZ < multiplierZ; iZ++)
				{
					double newCoordinateX = coordinateX + (iX * unitCellLengthX);
					double newCoordinateY = coordinateY + (iY * unitCellLengthY);
					double newCoordinateZ = coordinateZ + (iZ * unitCellLengthZ);

					Md.X[Md.NN] = newCoordinateX;
					Md.Y[Md.NN] = newCoordinateY;
					Md.Z[Md.NN] = newCoordinateZ;
					Md.NN++;
				}
			}
		}
	}

	if (Md.NN != Md.NA)
		cout << "Warning: number of total atoms NN is different from number of moving atoms NA." << endl;
}

string GetValue()
{
	SkipSpace();
	string val = "";
	char c;
	do {
		fileInp.get(c);
		val += c;
	} while ((c != ' ') && (c != ','));
	val = val.substr(0, val.size() - 1);

	return val;
}

int GetValueInt()
{
	string str = GetValue();
	int result = 0;
	bool success = (stringstream(str) >> result);
	if(success == false)
	{
		cout << "Error converting input to integer. Stopping." << endl;
		exit(1);
	}

	return result;
}

double GetValueDouble()
{
	string str = GetValue();
	double result = 0;
	bool success = (stringstream(str) >> result);
	if(success == false)
	{
		cout << "Error converting input to double. Stopping." << endl;
		exit(1);
	}

	return result;
}

double GetValueFloat()
{
	string str = GetValue();
	float result = 0;
	bool success = (stringstream(str) >> result);
	if(success == false)
	{
		cout << "Error converting input to double. Stopping." << endl;
		exit(1);
	}

	return result;
}

string SkipSpace()
{
	string val = "";
	char c;
	do {
		fileInp.get(c);
		val += c;
	} while ((c == ' ') || (c == ',') || (c == '\n') || (c == '\r'));
	val = val.substr(0, val.size() - 1);
	fileInp.unget();

	return val;
}

string ReadLine()
{
	string line = "";
	getline(fileInp, line);
	return line;
}

// CALCULATES INTERATOMIC DISTANCE BETWEEN ATOMS I AND J
double Distance(int i, int j)
{
	double XX = Md.X[i] - Md.X[j];
	double YY = Md.Y[i] - Md.Y[j];
	double ZZ = Md.Z[i] - Md.Z[j];
	
	return XX*XX + YY*YY + ZZ*ZZ;
}