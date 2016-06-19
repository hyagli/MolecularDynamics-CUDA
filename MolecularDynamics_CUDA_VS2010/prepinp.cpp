#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <iomanip>

using namespace std;

ifstream inpFile, picFile;
ofstream newFile;

int NumberOfAtoms, oldMDSL;
double oldPeriodicBoundary_Z, newPeriodicBoundary_Z;
bool PStretch;
int PReduceX, PReduceY;

void CopyFromInpFile();
void CopyFromPicFile();
void WriteStepFile();
bool CheckParameters(int argc, char* argv[]);

int main (int argc, char *argv[])
{
	if(CheckParameters(argc, argv) == false)
		return 1;
	
    inpFile.open("mdse.inp");
    if(!inpFile)
    {
        cout << "mdse.inp not found, exiting." << endl;
        return 1;
    }
    picFile.open("mdse.pic");
    if(!picFile)
    {
        cout << "mdse.pic not found, exiting." << endl;
        inpFile.close();
        return 1;
    }

    newFile.open("mdse_new.inp");
    newFile << fixed << setprecision(5);

    CopyFromInpFile();

	CopyFromPicFile();

    inpFile.close();
    picFile.close();
    newFile.close();
    cout << "New InpFile ready. Writing step file." << endl;

    WriteStepFile();

    return 0;
}

string GetValue()
{
    string val = "";
    char c;
    do {
        inpFile.get(c);
        val += c;
    } while ((c != ' ') && (c != ','));
    val = val.substr(0, val.size() - 1);
    inpFile.unget();

    return val;
}

string GetSpace()
{
    string val = "";
    char c;
    do {
        inpFile.get(c);
        val += c;
    } while ((c == ' ') || (c == ','));
    val = val.substr(0, val.size() - 1);
    inpFile.unget();

    return val;
}

// Copy the values from input file, increment Lz, set the layer = NA
void CopyFromInpFile()
{
    // Copy the first two lines
    string str, WjkLine;
    getline(inpFile, str);
    newFile << str << endl;
    getline(inpFile, str);
    newFile << str << endl;


    string value, space;

    // MDSL
    inpFile >> oldMDSL;
    space = GetSpace();
    newFile << oldMDSL << space;

    // IAVL, IPPL, ISCAL, IPD, TE
    for(int i=0; i<5; i++)
    {
        value = GetValue();
        space = GetSpace();
        newFile << value << space;
    }

    // Read NA
    inpFile >> NumberOfAtoms;
    space = GetSpace();
    newFile << NumberOfAtoms << space;

    // for LAYER write NA again
    value = GetValue();
    space = GetSpace();
    newFile << NumberOfAtoms << space;

    // IPBC, PP(1), PP(2)
    for(int i=0; i<3; i++)
    {
        value = GetValue();
        space = GetSpace();
        newFile << value << space;
    }

    // Read PP(3) -> Lz and multiply by 1.05 if stretching
    inpFile >> oldPeriodicBoundary_Z;
    if(PStretch == true)
        newPeriodicBoundary_Z = oldPeriodicBoundary_Z * 1.05;
    else
        newPeriodicBoundary_Z = oldPeriodicBoundary_Z;
    char buf[100];
    sprintf(buf, "%.8f", newPeriodicBoundary_Z);
    newFile << buf << endl;
    getline(inpFile, space);
    
    // Copy the fourth line
    getline(inpFile, str);
    newFile << str << endl;

}

void WriteStepFile()
{
	int stepNo = 1;
    ifstream stepFile("steps.txt");
    if(stepFile) {
        // Go to last line
        string line1, line2;
        while(stepFile.eof() == false) {
            getline(stepFile, line1);
            if(line1.size() > 0)
				line2 = line1;
        }

        // read step number from last line
        istringstream ss (line2);
        ss >> stepNo;
        stepFile.close();

        cout << "Step file found. Last done step: " << stepNo << endl;
        stepNo++;

        // add a new line at the end of the file.
        fstream editStepFile;
        editStepFile.open("steps.txt", fstream::in | fstream::out | fstream::app);
        editStepFile << stepNo << " , " << oldMDSL << " , " << fixed << oldPeriodicBoundary_Z << endl;
        editStepFile.close();
    }
    else {
		cout << "Step file not found. Creating new step file." << endl;
        ofstream newStepFile("steps.txt");
        newStepFile << "StepNo , MDSL , Lz" << endl;
        newStepFile << "1 , " << oldMDSL << " , " << fixed << oldPeriodicBoundary_Z << endl;
        newStepFile.close();
    }

    ostringstream ss1;
    ss1 << "mkdir step" << stepNo;
    system(ss1.str().c_str());
    ss1.str("");
    ss1 << "mv mdse.* step" << stepNo;
    system(ss1.str().c_str());
    system("mv mdse_new.inp mdse.inp");
}

bool CheckParameters(int argc, char* argv[])
{
	PStretch = false;
	PReduceX = 0;
	PReduceY = 0;
	
	for(int i=1; i<argc; i++)
	{
		string parameter = argv[i];
		
		if(parameter == "--help"){
			cout << "Use 's' for stretching. low=XXX for amount to reduce from x and y axis coordinates." << endl;
			return false; 
		}
		else if(parameter == "s"){
			PStretch = true;
			cout << "'s' parameter received. Will stretch Z boundary by 5%." << endl;
		}
		else if(parameter.substr(0, 3) == "-x=") {
			istringstream(parameter.substr(3, 10)) >> PReduceX;
			cout << "'-x' parameter received. Reducing X coordinates by: " << PReduceX << endl;
		}
		else if(parameter.substr(0, 3) == "-y=") {
			istringstream(parameter.substr(3, 10)) >> PReduceY;
			cout << "'-y' parameter received. Reducing Y coordinates by: " << PReduceY << endl;
		}
	}
	
	return true;
}

void CopyFromPicFile()
{
    // Copy the coordinates from pic file
    double dx, dy, dz;
    string str;
    getline(picFile, str); // skip the first line
    for(int i=0; i<NumberOfAtoms; i++)
    {
        picFile >> dx >> dy >> dz;
        dx = dx - PReduceX;
        dy = dy - PReduceY;
        newFile << dx << " , " << dy << " , " << dz << " , 1.0 , 1.0 , 1.0 , 1 , 1 , 1" << endl;
    }
}
