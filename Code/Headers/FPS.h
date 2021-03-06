/****************************************************************************
		Description:	Defines the FPS Class.

		Classes:		FPS

		Project:		MATE 2022

		Copyright 2021 MST Design Team - Underwater Robotics
****************************************************************************/
#ifndef Drive_h
#define Drive_h

#include <ctime>
///////////////////////////////////////////////////////////////////////////////


class FPS
{
public:
    // Declare class methods.
	FPS();
	~FPS();
	void Increment();
	int FramesPerSec();

private:
    // Declare class variables.
	time_t					startTime;
	int						iterations;
	int						tick;
	int						FPSCount;
};
///////////////////////////////////////////////////////////////////////////////
#endif