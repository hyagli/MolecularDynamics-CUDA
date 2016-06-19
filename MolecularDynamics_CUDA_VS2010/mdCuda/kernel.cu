//#include "mdCuda.h"

__device__ void cudaReposition(float PP0, float PP1, float PP2, float PA0, float PA1, float PA2, float PL0, float PL1, float PL2, float &Xi, float &Yi, float &Zi, float NA)
{
	float PAPL, H, B;

	if(PP0 > 0){
		PAPL = PA0 + PL0;
		H = (Xi-PAPL) / PL0;
		B = H - 2.0*int(H);
		Xi = B*PL0 + PAPL;
	}

	if(PP1 > 0){
		PAPL = PA1 + PL1;
		H = (Yi-PAPL) / PL1;
		B = H - 2.0*int(H);
		Yi = B*PL1 + PAPL;
	}

	if(PP2 > 0){
		PAPL = PA2 + PL2;
		H = (Zi-PAPL) / PL2;
		B = H - 2.0*int(H);
		Zi = B*PL2 + PAPL;
	}
}

__global__ void mdKernel(int NA, float* FFX, float* FFY, float* FFZ, float* EE,	float* X, float* Y, float* Z, int IPBC, 
	float PP0, float PP1, float PP2, float AL1, float AL2, float A1, float A2, float RL1, float RL2, float D21, float D22, 
	float PA0, float PA1, float PA2, float PB0, float PB1, float PB2, float PL0, float PL1, float PL2)
{
	float XIJ, YIJ, ZIJ, RIJ, RIJ2, EPP, FX2, FY2, FZ2;
	float ARG1, ARG2, EXP1, EXP2, UIJ1, UIJ2, UIJ;
	float FAC1, FAC2, FAC12, XRIJ, YRIJ, ZRIJ;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float Xi = X[i];
	float Yi = Y[i];
	float Zi = Z[i];

	cudaReposition(PP0, PP1, PP2, PA0, PA1, PA2, PL0, PL1, PL2, Xi, Yi, Zi, NA);

	///////////////////////////////////////////////////
	// FORCE

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