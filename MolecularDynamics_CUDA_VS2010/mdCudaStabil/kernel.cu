//#include "mdCuda.h"

__global__ void kernelForce(int NA, double* FFX, double* FFY, double* FFZ, double* EE, double* X, double* Y, double* Z, int IPBC, double *Params)
{
	double XIJ, YIJ, ZIJ, RIJ, RIJ2, EPP, FX2, FY2, FZ2;
	double ARG1, ARG2, EXP1, EXP2, UIJ1, UIJ2, UIJ;
	double FAC1, FAC2, FAC12, XRIJ, YRIJ, ZRIJ;

	double PP0 = Params[0];
	double PP1 = Params[1];
	double PP2 = Params[2];
	double AL1 = Params[3];
	double AL2 = Params[4];
	double A1  = Params[5];
	double A2  = Params[6];
	double RL1 = Params[7];
	double RL2 = Params[8];
	double D21 = Params[9];
	double D22 = Params[10];

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	EPP = 0;
	// Forces that effect atoms indexed with i in all three axes
	FX2 = 0;
	FY2 = 0;
	FZ2 = 0;

	for(int j=0; j<NA; j++)
	{
		if(i == j)
			continue;

		// Apply periodic boundaries and find distances between atom I and j. RIJ2 is square of RIJ
		XIJ = X[i] - X[j];
		YIJ = Y[i] - Y[j];
		ZIJ = Z[i] - Z[j];

		double DD, ID;
	
		if(IPBC != 0){
			if(PP0 > 0){
				DD = XIJ / PP0;
				ID = int(DD);
				XIJ = XIJ - PP0*(ID+int(2.0*(DD-ID)));
			}
			if(PP1 > 0){
				DD = YIJ / PP1;
				ID = int(DD);
				YIJ = YIJ - PP1*(ID+int(2.0*(DD-ID)));
			}
			if(PP2 > 0){
				DD = ZIJ / PP2;
				ID = int(DD);
				ZIJ = ZIJ - PP2*(ID+int(2.0*(DD-ID)));
			}
		}
		RIJ2 = XIJ*XIJ + YIJ*YIJ + ZIJ*ZIJ;
		RIJ = sqrt(RIJ2);

		// Calculate potential energy U(r)
		ARG1 = AL1*RIJ2;
		ARG2 = AL2*RIJ2;
		EXP1 = exp(-ARG1);
		EXP2 = exp(-ARG2);
		UIJ1 = A1*EXP1/(pow(RIJ,RL1));
		UIJ2 = A2*EXP2/(pow(RIJ,RL2));
		UIJ = D21*UIJ1 + D22*UIJ2;
		EPP = EPP+UIJ;

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

	FFX[i] = -FX2;
	FFY[i] = -FY2;
	FFZ[i] = -FZ2;
	EE[i] = EPP;
}