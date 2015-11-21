#include <iostream>
#include <tclap/CmdLine.h>



int main(int argc, char* argv[]) {
	try {
		TCLAP::CmdLine cmd("Voronoi Demo", ' ', "0.0");
		cmd.parse(argc, argv);
	} catch (TCLAP::ArgException& e) {
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}




	return 0;
}