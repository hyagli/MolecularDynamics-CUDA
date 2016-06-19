#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <iomanip>

using namespace std;

ifstream mdseFile;

string GSteps = "";
int GNumberOfAtoms = 0;

bool OpenInput();
void EchoSteps();
bool WritePicFile();
void WriteCommand(string command);
int StringToInteger(string str);
void RewindInput(); // Go to the beginning of file

int main (int argc, char *argv[])
{
	if(OpenInput() == false)
		return -1;

	string command = "";
	while(command != "q")
	{
		cout << endl << "Commands: 'q' to quit, 'l' to list steps with printed coordinates, a number to make pic file with that step";
		cout << endl << "Enter command: ";
		cin >> command;

		if(command == "l")
			EchoSteps();
		else if(command != "q")
			WriteCommand(command);
	}

	return 0;
}

bool OpenInput()
{
	mdseFile.open("mdse.out");
	if(mdseFile.is_open() == false)
	{
		cout << "mdse.out not found, exiting." << endl;
		return false;
	}

	string line;

	// Go to line with NA=XX
	for(int i=0; i<8; i++)
		getline(mdseFile, line);

	string strNum = line.substr(5, 3);
	GNumberOfAtoms = StringToInteger(strNum);

	return true;
}

void EchoSteps()
{
	if(GSteps == "")
	{
		string line;

		// Search for periodic printing lines
		RewindInput();
		while(mdseFile.eof() == false)
		{
			getline(mdseFile, line);
			if(line.substr(0, 34) == "  PERIODIC PRINTING OF COORDINATES")
				GSteps += " " + line.substr(64);				
		}
	}

	cout << endl << "Periodic printed steps:" << GSteps << endl;
}

void WriteCommand(string command)
{
	int inputNumber = StringToInteger(command);
	if(inputNumber == -1)
	{
		cout << endl << "Invalid number given" << endl;
		return;
	}

	RewindInput();
	string line;	
	while(mdseFile.eof() == false)
	{
		getline(mdseFile, line);
		if(line.substr(0, 34) == "  PERIODIC PRINTING OF COORDINATES")
		{
			string str = line.substr(64);
			int printedStep = StringToInteger(str);
			if(printedStep == inputNumber)
			{
				WritePicFile();
				return;
			}
		}
	}

	cout << endl << "Step not found." << endl;
}

bool WritePicFile()
{
	ofstream newFile;
	newFile.open("mdse.pic");
	if(newFile.is_open() == false)
	{
		cout << endl << "Couldn't create mdse.pic file" << endl;
		return false;
	}

	newFile << "  NN=" << GNumberOfAtoms << " NA=" << GNumberOfAtoms << " TOTPE=0 APEPP=0" << endl;

	string line;

	// Skip 5 lines
	for(int i=0; i<5; i++)
		getline(mdseFile, line);

	for(int i=0; i<GNumberOfAtoms; i++)
	{
		getline(mdseFile, line);
		string coordinates = line.substr(6, 42);
		newFile << coordinates << endl;
	}

	newFile.close();
	cout << endl << "Pic file written." << endl;
	return true;
}

int StringToInteger(string str)
{
	istringstream convert(str);
	int number;
	if(!(convert >> number))
		number = -1;
	return number;
}

void RewindInput()
{
	mdseFile.clear();
	mdseFile.seekg(0, ios::beg);
}