#include "mdCuda.h"


int main(int argc, char* argv[])
{
	if(CheckParameters(argc, argv) == false)
		return 0;

	if(OpenFiles() == false)
		return -1;

	if(Input() == false)
		return -1;

	Solve();

	CloseFiles();

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
	fileOut << fixed << setprecision(5);

	fileEne.open("mdse.ene");
	if(fileEne.good() == false)
	{
		cout << "mdse.ene couldn't be opened for writing. Stopping." << endl;
		return false;
	}
	fileEne << fixed << setprecision(5);

	filePic.open("mdse.pic");
	if(filePic.good() == false)
	{
		cout << "mdse.pic couldn't be opened for writing. Stopping." << endl;
		return false;
	}
	filePic << fixed << setprecision(5);

	fileBs.open("mdse.bs");
	if(fileBs.good() == false)
	{
		cout << "mdse.bs couldn't be opened for writing. Stopping." << endl;
		return false;
	}
	fileBs << fixed << setprecision(5);

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
	/*RM = 63.546;
	DT = 0.9E-15;
	A1 = 110.766008;
	A2 = -46.1649783;
	RL1 = 2.09045946;
	RL2 = 1.49853083;
	AL1 = 0.394142248;
	AL2 = 0.207225507;
	D21 = 0.436092895;
	D22 = 0.245082238;
	*/

	// Potential parameters for Au
	RM=196.9665;
	DT=1.6E-15;
	A1=345.923364;
	A2=-38.9245908;
	RL1=1.0428923;
	RL2=1.05974062;
	AL1=0.750775965;
	AL2=0.229377368;
	D21=0.888911352;
	D22=0.254280292;

	double FACM = 0.103655772E-27;
	BK = 8.617385E-05;
	RM = RM * FACM;

	try
	{
		// Read the title
		Title = ReadLine();

		// Skip the second line
		ReadLine();
		
		// Read MDSL, IAVL, IPPL, ISCAL, IPD, TE, NA, LAYER, IPBC, PP(1), PP(2), PP(3)
		MDSL = GetValueInt();
		IAVL = GetValueInt();
		IPPL = GetValueInt();
		ISCAL = GetValueInt();
		IPD = GetValueInt();
		TE = GetValueDouble();
		NA = GetValueInt();
		LAYER = GetValueInt();
		IPBC = GetValueInt();
		PP[0] = GetValueDouble();
		PP[1] = GetValueDouble();
		PP[2] = GetValueDouble();

		// Generate atom coordinates
		GenerateLatis();
		// Sort atoms by the z axis
		SortAtoms('Z');
		// Find the periodic boundary limits if PBC is applied
		FindBoundaries();
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
	// Initialize some variables and define some factors
	MDS = 0;				// Current md simulation step;
	int IPP=0;				// Print counter
	double EPAV = 0;		// Average potential energy
	double EKAV = 0;		// Average kinetic energy
	double ETAV = 0;		// Average total energy
	double SCFAV = 0;		// Average scaling factor
	TCALAV = 0;			// System temperature
	int IAV = 0;			// Average counter
	int ISCA = 0;			// Scaling counter

	double FFPR[MAX_ATOMS][3];   // Array to store forces from previous step

	// Calculate the initial potential energy of each atom and the initial force that each atom experiences
	Force();

	// SET INITIAL VELOCITIES ACC. TO MAXWELL VEL. DISTRIBUTION
	MaxWell();

	// Printing initially distributed velocities, potential energies, forces, total energy and temperature
	PrintInitial();

	fileOut << "#" << endl << "# *********************   MD STEPS STARTED   *************************" << endl << "#" << endl;
	fileOut << "#  MDS       EPAV          EKAV          ETAV         TCALAV" << endl;
	fileOut << "# -----  ------------  ------------  ------------  ------------" << endl << "#" << endl;
	fileEne << "#" << endl << "# *********************   MD STEPS STARTED   *************************" << endl << "#" << endl;
	fileEne << "#  MDS       EPAV          EKAV          ETAV         TCALAV" << endl;
	fileEne << "# -----  ------------  ------------  ------------  ------------" << endl << "#" << endl;

	// Start Md Steps
	while(MDS < MDSL){
		MDS++;
		IPP++;
		ISCA++;

		// Show status at each 100 steps
		if((MDS % 100) == 0)
			ShowStatus();

		// Reposition the particles if PBC is applied
		if(IPBC != 0)
			Reposition();

		// Calculate velocity and position of the particles using the velocity summed form of verlet algorithm (NVE MD velocity form)
		Force();
	
		// Compute the positions at time step n+1 as:
		// ri(n+1)=ri(n)+hvi(n)+(1/2m)h2Fi(n)
		for(int i=0; i<NA; i++){
			X[i] = X[i] + DT*VV[i][0] + (pow(DT,2)*FF[i][0]) / (2*RM);
			Y[i] = Y[i] + DT*VV[i][1] + (pow(DT,2)*FF[i][1]) / (2*RM);
			Z[i] = Z[i] + DT*VV[i][2] + (pow(DT,2)*FF[i][2]) / (2*RM);
		}

		// Store the forces at time step Fi(n)
		memcpy(FFPR, FF, NA*3*sizeof(double));
		//for(int i=0; i<NA; i++){
		//	for(int j=0; j<3; j++){
		//		FFPR[i][j] = FF[i][j];
		//	}
		//}

		Force();

		// Compute the velocities at time step n+1 as
		// vi(n+1)=vi(n)+(h/2m)(Fi(n+1)+Fi(n))
		for(int i=0; i<NA; i++){
			VV[i][0] = VV[i][0] +  DT * (FF[i][0]+FFPR[i][0]) / (2*RM);
			VV[i][1] = VV[i][1] +  DT * (FF[i][1]+FFPR[i][1]) / (2*RM);
			VV[i][2] = VV[i][2] +  DT * (FF[i][2]+FFPR[i][2]) / (2*RM);
			VT[i] = pow(VV[i][0],2) + pow(VV[i][1],2) + pow(VV[i][2],2);
		}

		// Calculate the temperature that system reached by calculating the kinetic energy of each atom
		EKINA = 0;
		for(int i=0; i<NA; i++)
			EKINA += VT[i];
		EKINA *= RM;
		TCALC = EKINA / (3*NA*BK);

		// Calculate the scaling factor and scale the velocities
		SCFAC = sqrt(TE/TCALC);
		if(ISCA == ISCAL)
		{
			EKIN = 0;
			for(int i=0; i<NA; i++){
				for(int j=0; j<3; j++){
					VV[i][j] *= SCFAC;
				}
				VT[i] = pow(VV[i][0],2) + pow(VV[i][1],2) + pow(VV[i][2],2);
				EKIN += VT[i];
			}
			ISCA = 0;
			EKIN *= RM;
			TCALC = EKIN / (3 * NA * BK);
		}

		// Calculate total energy
		ETOT = EPOT + EKINA;

		// Calculate the averages of EPOT, EKINA, ETOT, SCFAC AND TCALC
		EPAV += EPOT;
		EKAV += EKINA;
		ETAV += ETOT;
		SCFAV += SCFAC;
		TCALAV += TCALC;
		IAV++;

		if(IAV < IAVL)
			continue;

		EPAV /= IAVL;
		EKAV /= IAVL;
		ETAV /= IAVL;
		SCFAV /= IAVL;
		TCALAV /= IAVL;

		// Print the averages
		fileOut << setw(6) << MDS << "  " << scientific << EPAV << "  " << EKAV << "  " << ETAV << "  " << TCALAV << endl << fixed;
		fileEne << setw(6) << MDS << "  " << scientific << EPAV << "  " << EKAV << "  " << ETAV << "  " << TCALAV << endl << fixed;

		// Periodic printing of coordinates
		if(IPP == IPPL){
			PrintPeriodic();
			IPP = 0;
		}

		IAV = 0;
		EPAV = 0;
		EKAV = 0;
		ETAV = 0;
		SCFAV = 0;
		TCALAV = 0;

	} // Md Steps Loop

	PrintFinal();

	return true;
}

void WriteBsFile()
{
	for(int i=0; i<NA; i++){
		fileBs << "atom   Au " << setw(12) << X[i] << "  " << setw(12) << Y[i] << "  " << setw(12) << Z[i] << endl;
	}
	
	fileBs << "" << endl;
	fileBs << "spec   Au   0.50   1 0.75 0" << endl;  // "atom size" "color codes rgb"
	fileBs << "" << endl;
	fileBs << "bonds  Au  Au   1.0   4.05   0.03   0.5 0.7 0.9" << endl; // Bond "min length" "max length" "line width" "color codes rgb"
	fileBs << "" << endl;
	fileBs << "tmat  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  1.0" << endl;
	fileBs << "dist  100.0" << endl;
	fileBs << "inc     1.0" << endl;
	fileBs << "scale  40.0" << endl;
	fileBs << "rfac    1.0" << endl;
	fileBs << "bfac    1.0" << endl;
	fileBs << "switches 1 0 1 0 0 1 1 0 0" << endl;
	fileBs << "" << endl;
}

bool CloseFiles()
{
	fileInp.close();
	fileOut.close();
	fileEne.close();
	filePic.close();
	fileBs.close();

	CuErr( cudaFreeHost(h_FFX));
	CuErr( cudaFreeHost(h_FFY));
	CuErr( cudaFreeHost(h_FFZ));
	CuErr( cudaFreeHost(h_Params));
	CuErr( cudaFree(d_FFX));
	CuErr( cudaFree(d_FFY));
	CuErr( cudaFree(d_FFZ));
	CuErr( cudaFree(d_EE));
	CuErr( cudaFree(d_X));
	CuErr( cudaFree(d_Y));
	CuErr( cudaFree(d_Z));
	CuErr( cudaFree(d_Params));

	return true;
}

void ShowStatus()
{
	cout << "\rMDS Steps: " << MDS << " of " << MDSL;
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

void Force()
{
	if(OnlyCpu)
		ForceCpu();
	else
		ForceCuda();
}

void ForceCpu()
{
	double E2 = 0;				// Total energy

	double XIJ, YIJ, ZIJ, RIJ, RIJ2, EPP, FX2, FY2, FZ2;
	double ARG1, ARG2, EXP1, EXP2, UIJ1, UIJ2, UIJ;
	double FAC1, FAC2, FAC12, XRIJ, YRIJ, ZRIJ;

	int i, j;

#pragma omp parallel for private(i,j,EPP,FX2,FY2,FZ2,RIJ,RIJ2,XIJ,YIJ,ZIJ,ARG1,ARG2,EXP1,EXP2,UIJ1,UIJ2,UIJ,FAC1,FAC2,FAC12,XRIJ,YRIJ,ZRIJ) reduction(+:E2)
	for(i=0; i<NA; i++)
	{
		EE[i] = 0;
		EPP = 0;
		//Forces that effect atoms indexed with i in all three axes
		FX2 = 0;
		FY2 = 0;
		FZ2 = 0;

		for(j=0; j<NA; j++)
		{
			if(i == j)
				continue;

			// Apply periodic boundaries and find distances between atom I and j. RIJ2 is square of RIJ
			Period(i, j, XIJ, YIJ, ZIJ, RIJ2, RIJ);

			// Calculate potential energy U(r)
			ARG1 = AL1*RIJ2;
			ARG2 = AL2*RIJ2;
			EXP1 = exp(-ARG1);
			EXP2 = exp(-ARG2);
			UIJ1 = A1*EXP1/(pow(RIJ,RL1));
			UIJ2 = A2*EXP2/(pow(RIJ,RL2));
			UIJ = D21*UIJ1 + D22*UIJ2;
			EPP += UIJ;

			// Calculate forces
			FAC1 = -(RL1/RIJ + 2.0*AL1*RIJ);
			FAC2 = -(RL2/RIJ + 2.0*AL2*RIJ);
			FAC12 = FAC1*D21*UIJ1 + FAC2*D22*UIJ2;
			XRIJ = XIJ/RIJ;
			YRIJ = YIJ/RIJ;  
			ZRIJ = ZIJ/RIJ;
			FX2 += FAC12*XRIJ;
			FY2 += FAC12*YRIJ;
			FZ2 += FAC12*ZRIJ;
		}

		FF[i][0] = -FX2;
		FF[i][1] = -FY2;
		FF[i][2] = -FZ2;
		EE[i] = EPP;
		E2 += EPP;
		//FFF[i] = sqrt(FF[i][0]*FF[i][0] + FF[i][1]*FF[i][1] + FF[i][2]*FF[i][2]);
	}

	EPOT = E2;
}

void ForceCuda()
{
	int sizeNA = NA * sizeof(double);
	int sizeParams = 11 * sizeof(double);

	// Pointers are global, allocating once is enough
	if(h_FFX == NULL){
		CuErr( cudaMallocHost(&h_FFX, sizeNA));
		CuErr( cudaMallocHost(&h_FFY, sizeNA));
		CuErr( cudaMallocHost(&h_FFZ, sizeNA));
		CuErr( cudaMallocHost(&h_Params, sizeParams));
		CuErr( cudaMalloc(&d_FFX, sizeNA));
		CuErr( cudaMalloc(&d_FFY, sizeNA));
		CuErr( cudaMalloc(&d_FFZ, sizeNA));
		CuErr( cudaMalloc(&d_EE, sizeNA));
		CuErr( cudaMalloc(&d_X, sizeNA));
		CuErr( cudaMalloc(&d_Y, sizeNA));
		CuErr( cudaMalloc(&d_Z, sizeNA));
		CuErr( cudaMalloc(&d_Params, sizeParams));

		h_Params[0]  = PP[0];
		h_Params[1]  = PP[1];
		h_Params[2]  = PP[2];
		h_Params[3]  = AL1;
		h_Params[4]  = AL2;
		h_Params[5]  = A1 ;
		h_Params[6]  = A2 ;
		h_Params[7]  = RL1;
		h_Params[8]  = RL2;
		h_Params[9]  = D21;
		h_Params[10] = D22;
		CuErr( cudaMemcpy(d_Params, h_Params, sizeParams, cudaMemcpyHostToDevice));

		//cudaChannelFormatDesc chanDouble = cudaCreateChannelDesc<double>();
		//CuErr( cudaBindTexture(0, &texX, d_X, &chanDouble, sizeNA));
		//CuErr( cudaBindTexture(0, &texY, d_Y, &chanDouble, sizeNA));
		//CuErr( cudaBindTexture(0, &texZ, d_Z, &chanDouble, sizeNA));
	}
	
	CuErr( cudaMemcpy(d_X, X, sizeNA, cudaMemcpyHostToDevice));
	CuErr( cudaMemcpy(d_Y, Y, sizeNA, cudaMemcpyHostToDevice));
	CuErr( cudaMemcpy(d_Z, Z, sizeNA, cudaMemcpyHostToDevice));

	int blockSize = 32;
	int numBlocks = NA / blockSize + (NA % blockSize == 0 ? 0:1);
	kernelForce <<< numBlocks, blockSize >>> (NA, d_FFX, d_FFY, d_FFZ, d_EE, d_X, d_Y, d_Z, IPBC, d_Params);
	CuErrC("kernelForce kernel execution failed");

	CuErr( cudaMemcpy(X, d_X, sizeNA, cudaMemcpyDeviceToHost));
	CuErr( cudaMemcpy(Y, d_Y, sizeNA, cudaMemcpyDeviceToHost));
	CuErr( cudaMemcpy(Z, d_Z, sizeNA, cudaMemcpyDeviceToHost));
	CuErr( cudaMemcpy(h_FFX, d_FFX, sizeNA, cudaMemcpyDeviceToHost));
	CuErr( cudaMemcpy(h_FFY, d_FFY, sizeNA, cudaMemcpyDeviceToHost));
	CuErr( cudaMemcpy(h_FFZ, d_FFZ, sizeNA, cudaMemcpyDeviceToHost));
	CuErr( cudaMemcpy(EE, d_EE, sizeNA, cudaMemcpyDeviceToHost));

	EPOT = 0;
	for(int i=0; i<NA; i++){
		FF[i][0] = h_FFX[i];
		FF[i][1] = h_FFY[i];
		FF[i][2] = h_FFZ[i];
		EPOT += EE[i];
	}
}

void FindBoundaries()
{
	if(IPBC == 0)
		return;

	for(int i=0; i<3; i++)
		PL[i] = PP[i] / 2.0;
	
	// Find smallest coordinates for X, Y and Z coordinates
	PA[0] = X[0];
	PA[1] = Y[0];
	PA[2] = Z[0];
	for(int i=1; i<NN; i++)
	{
		if(PA[0] > X[i])
			PA[0] = X[i];
		if(PA[1] > Y[i])
			PA[1] = Y[i];
		if(PA[2] > Z[i])
			PA[2] = Z[i];
	}

	// Find ending coordinates of working system
	PB[0] = PA[0] + PP[0];
	PB[1] = PA[1] + PP[1];
	PB[2] = PA[2] + PP[2];
}

// PRINTING OF POSITIONS, FORCES, AND ENERGIES
void PrintCoordinatesForcesEnergy(){
	fileOut << "     I       X             Y             Z            FX             FY             FZ             EE" << endl;
	fileOut << "  ------  ---------  ------------  ------------  ------------   ------------   ------------   ------------" << endl << endl;

	for(int i=0; i<NA; i++){
		fileOut << setw(6) << i+1;
		fileOut << setw(12) << X[i] << "  " << setw(12) << Y[i] << "  " << setw(12) << Z[i] << "  ";
		fileOut << scientific << setw(13) << FF[i][0] << "  " << setw(13) << FF[i][1] << "  " << setw(13) << FF[i][2] << "  " << setw(13) << EE[i];
		fileOut << fixed << endl;
	}
}

void PrintInitial()
{
	string str;
	fileInp.clear();
	fileInp.seekg(0, ios::beg);

	if(PSilent == false)
		cout << "Simulation started" << endl;

	fileOut << "******************************************************************************************" << endl;
	fileOut << Title << endl;
	fileOut << "******************************************************************************************" << endl << endl;
	fileOut << GetTime() << endl << endl;

	tStart = clock();
	
	getline(fileInp, str);
	getline(fileInp, str);
	fileOut << str << endl;
	getline(fileInp, str);
	fileOut << str << endl << endl;
	getline(fileInp, str);

	fileOut << "  INITIAL COORDINATES:" << endl;
	for(int i=0; i<LAYER; i++){
		getline(fileInp, str);
		fileOut << str << endl;
	}
	
	fileOut << "******************************************************************************************" << endl << endl;
	fileOut << "  NUMBER OF MOVING ATOMS: NA= " << NA << endl;
	fileOut << "  NUMBER OF TOTAL ATOMS: NN= " << NN << endl << endl;
	fileOut << "  INITIAL COORDINATES OF ALL ATOMS: (X,Y,Z)" << endl << endl;
	for(int i=0; i<NN; i++){
		fileOut << setw(5) << i+1 << " " << setw(12) << X[i] << " " << setw(12) << Y[i] << " " << setw(12) << Z[i] << endl;
	}
	fileOut << "******************************************************************************************" << endl << endl;

	fileOut << endl << "  INITIAL COORDINATES, FORCES AND ENERGIES:" << endl << endl;
	PrintCoordinatesForcesEnergy();
	
	fileOut << endl << scientific;
	fileOut << " EPOT=" << EPOT << "  EKIN=" << EKIN << "  TCALC=" << TCALC << "  SCFAC=" << SCFAC << endl << endl << fixed;
}

void PrintPeriodic()
{
	fileOut << endl << endl << "  PERIODIC PRINTING OF COORDINATES, FORCES AND ENERGIES AT MDS: " << MDS << endl << endl;
	PrintCoordinatesForcesEnergy();

	fileOut << endl << scientific;
	fileOut << " EPOT=" << EPOT << "  EKIN=" << EKIN << "  TCALC=" << TCALC;
	fileOut << "  SCFAC=" << SCFAC << endl << endl << fixed;
}

void PrintFinal()
{
	if(IPBC != 0)
		Reposition();

	fileOut << endl << endl << "  FINAL COORDINATES, FORCES AND ENERGIES:" << endl << endl;
	PrintCoordinatesForcesEnergy();

	fileOut << endl << scientific;
	fileOut << " EPOT=" << EPOT << "  EKINA=" << EKINA << "  ETOT=" << ETOT << "  TCALC=" << TCALC << endl << endl << fixed;

	PrintElapsedTime();
	fileOut << "  *************** END OF THE CALCULATION ***************";

	WritePicFile();
	WriteBsFile();

	if(PSilent == false)
		cout << endl << "Simulation complete" << endl;
}

void PrintElapsedTime()
{
	// Write current time
	fileOut << endl << GetTime() << endl << endl;
	
	// Calculate and write elapsed time
	tStop = clock();
	float seconds = float(tStop - tStart)/CLOCKS_PER_SEC;
	int minutes = seconds/60;
	seconds -= minutes*60;
	int hours = minutes/60;
	minutes -= hours*60;
	fileOut << "  ELAPSED TIME: " << hours << " HOURS " << minutes << " MINUTES " << seconds << " SECONDS" << endl << endl;
}

// RANDOM NUMBER GENERATOR, GENERATES RN IN THE INTERVAL (-1,1)
double Randum(double U, double S)
{
	U = 23*U + 0.21132486579;
	if((U-1.0) >= 0)
		U = U - int(U);
	if(U > 0.5)
		S = -S;
	U = U-int(U);
	return (S * U);
}

// DISTRUBUTES THE VELOCITIES FOR THE ATOMS FOR THE SPECIFIED 
// TEMPERATURE TE ACCORDING TO THE MAXWELL VELOCITY DISTRIBUTION
void MaxWell()
{
	double FAC1 = sqrt(3.0*BK*TE/RM);
	double U = 0.0;
	double S = 1.0;
	double VVX = 0.0;
	double VVY = 0.0;
	double VVZ = 0.0;
	double FAC2 = (2.0/3.0) * FAC1;
	FAC2 /= sqrt(3.0);

	// EQUATING Vmean TO FAC2 
	for(int i=0; i<NA; i++){
		for(int j=0; j<3; j++){
			VV[i][j] = (FAC2 - FAC2*Randum(U,S));
		}
	}

	// CALCULATING AVERAGES
	double VVV = 0.0;
	for(int i=0; i<NA; i++){
		VVX = VVX + VV[i][0];
		VVY = VVY + VV[i][1];
		VVZ = VVZ + VV[i][2];
	}
	VVX /= NA;
	VVY /= NA;
	VVZ /= NA;
	VVV = VVX*VVX + VVY*VVY + VVZ*VVZ;
	double COSX = VVX / sqrt(VVV);
	double COSY = VVY / sqrt(VVV);
	double COSZ = VVZ / sqrt(VVV);

	// CALCULATING EKIN AND TEMPERATURE WRT THE CALCULATED Vmean
	EKIN = 0.5 * RM * (VVV * (9.0/4.0));
	TCALC = EKIN / (1.5 * BK);

	// CALCULATING THE SCALING FACTOR 
	SCFAC = sqrt(TE / TCALC);

	// REDISTRIBUTING THE INITIAL VELOCITIES WRT SCALING FACTOR
	VVV = sqrt(VVV);
	double VVXNEW = COSX * VVV * SCFAC;
	double VVYNEW = COSY * VVV * SCFAC;
	double VVZNEW = COSZ * VVV * SCFAC;
	double XSCALE = (VVXNEW-VVX);
	double YSCALE = (VVYNEW-VVY);
	double ZSCALE = (VVZNEW-VVZ);
	for(int i=0; i<NA; i++){
		VV[i][0] += XSCALE;
		VV[i][1] += YSCALE;
		VV[i][2] += ZSCALE;
		VT[i] = pow(VV[i][0],2.0) + pow(VV[i][1],2) + pow(VV[i][2],2);
	}

	// CALCULATING AVERAGES  OF SCALED VELOCITIES
	VVX = 0;
	VVY = 0;
	VVZ = 0;
	for(int i=0; i<NA; i++){
		VVX += VV[i][0];
		VVY += VV[i][1];
		VVZ += VV[i][2];
	}
	VVX /= NA;
	VVY /= NA;
	VVZ /= NA;

	// CALCULATING EKIN AND TEMPERATURE WRT THE SCALED Vmean
	VVV = VVX*VVX + VVY*VVY + VVZ*VVZ;
	EKIN = 0.5 * RM * (VVV * (9/4));
	TCALC = EKIN / (1.5 * BK);

	ETOT = EPOT + EKIN;
}

// REPOSITIONS COORDINATES WHEN ANY MOVING ATOM CROSSES THE BOUNDARY.
void Reposition()
{
	double PAPL, H, B;

	if(PP[0] > 0){
		PAPL = PA[0] + PL[0];
		for(int i=0; i<NA; i++){
			H = (X[i]-PAPL) / PL[0];
			B = H - 2.0*int(H);
			X[i] = B*PL[0] + PAPL;
		}
	}

	if(PP[1] > 0){
		PAPL = PA[1] + PL[1];
		for(int i=0; i<NA; i++){
			H = (Y[i]-PAPL) / PL[1];
			B = H - 2.0*int(H);
			Y[i] = B*PL[1] + PAPL;
		}
	}

	if(PP[2] > 0){
		PAPL = PA[2] + PL[2];
		for(int i=0; i<NA; i++){
			H = (Z[i]-PAPL) / PL[2];
			B = H - 2.0*int(H);
			Z[i] = B*PL[2] + PAPL;
		}
	}
}

// Sorts atoms by the given axis
void SortAtoms(char sortAxis)
{
	double *sortArray;
	if(sortAxis == 'X')
		sortArray = X;
	else if(sortAxis == 'Y')
		sortArray = Y;
	else
		sortArray = Z;

	double tempX, tempY, tempZ;
	for (int i = 0; i < NA; i++)
	{
		for (int j = i+1; j < NA; j++)
		{
			if (sortArray[i] > sortArray[j])
			{
				tempX = X[i];
				tempY = Y[i];
				tempZ = Z[i];

				X[i] = X[j];
				Y[i] = Y[j];
				Z[i] = Z[j];

				X[j] = tempX;
				Y[j] = tempY;
				Z[j] = tempZ;
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

	NN = 0;
	for(int i=0; i<LAYER; i++)
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

					X[NN] = newCoordinateX;
					Y[NN] = newCoordinateY;
					Z[NN] = newCoordinateZ;
					NN++;
					if(NN > MAX_ATOMS)
					{
						cout << "The number of atoms cannot exceed " << MAX_ATOMS << ". Stopping.";
						exit(1);
					}
				}
			}
		}
	}

	if (NN != NA)
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
	} while ((c != ' ') && (c != ',') && (c != '\n') && (c != '\r') && (fileInp.eof() != true));
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

float GetValueFloat()
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

// Calculates interatomic distance between atoms I and J
double Distance(int i, int j)
{
	double XX = X[i] - X[j];
	double YY = Y[i] - Y[j];
	double ZZ = Z[i] - Z[j];
	
	return XX*XX + YY*YY + ZZ*ZZ;
}

void WritePicFile()
{
	double EB = EPOT / NA;
	filePic << "  NN=" << NN << " NA=" << NA << " TOTPE=" << EPOT << " APEPP=" << EB << endl;
	for(int i=0; i<NA; i++){
		filePic << setw(12) << X[i] << "  " << setw(12) << Y[i] << "  " << setw(12) << Z[i] << endl;
	}
}

// Apply periodic boundry condition and find distances between the two particles
// Because of the periodic boundary, the distance may be the one in this working system or the particle in the adjacent system.
void Period(int i, int j, double &XIJ, double &YIJ, double &ZIJ, double &RIJ2, double &RIJ)
{
	XIJ = X[i] - X[j];
	YIJ = Y[i] - Y[j];
	ZIJ = Z[i] - Z[j];

	double DD, ID;
	
	if(IPBC != 0){
		if(PP[0] > 0){
			DD = XIJ / PP[0];
			ID = int(DD);
			XIJ = XIJ - PP[0]*(ID+int(2.0*(DD-ID)));
		}
		if(PP[1] > 0){
			DD = YIJ / PP[1];
			ID = int(DD);
			YIJ = YIJ - PP[1]*(ID+int(2.0*(DD-ID)));
		}
		if(PP[2] > 0){
			DD = ZIJ / PP[2];
			ID = int(DD);
			ZIJ = ZIJ - PP[2]*(ID+int(2.0*(DD-ID)));
		}
	}
	RIJ2 = XIJ*XIJ + YIJ*YIJ + ZIJ*ZIJ;
	RIJ = sqrt(RIJ2);
}

// Check program starting parameters
bool CheckParameters(int argc, char* argv[])
{
	PSilent = false;
	OnlyCpu = false;
	SetPriorityClass(GetCurrentProcess(),BELOW_NORMAL_PRIORITY_CLASS);
	
	for(int i=1; i<argc; i++)
	{
		string parameter = argv[i];
		
		if(parameter == "-help"){
			cout << "Use parameter '-s' for silent mode. No output will be given to the console." << endl;
			cout << "Use parameter '-cpu' for cpu calculations only (otherwise Nvidia graphics card required)." << endl;
			return false; 
		}
		else if(parameter == "-s"){
			PSilent = true;
		}
		else if(parameter == "-cpu"){
			OnlyCpu = true;
			cout << "-cpu option received. Will use only cpu for computations." << endl;
		}
	}

	return true;
}