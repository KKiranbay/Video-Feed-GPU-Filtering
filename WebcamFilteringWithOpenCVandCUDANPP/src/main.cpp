#define SDL_MAIN_HANDLED
#include "Webcam/WebcamView.h"

int main(int argc, char* argv[])
{
	// return memcpyTutorialFunction();

	// return convertImageTutorialFunction();

	/*std::thread worker(getWebcamFeed);

	showWebcamWindow();

	worker.join();*/

	WebcamView gui;
	gui.startMainLoop();

	return 0;
}
