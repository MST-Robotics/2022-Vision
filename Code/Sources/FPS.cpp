/****************************************************************************
		Description:	Implements the FPS Class.

		Classes:		FPS

		Project:		MATE 2022

		Copyright 2021 MST Design Team - Underwater Robotics
****************************************************************************/
#include "../Headers/FPS.h"
///////////////////////////////////////////////////////////////////////////////


/****************************************************************************
			Description:	FPS constructor.

			Arguments:		None

			Derived From:	Nothing
****************************************************************************/
FPS::FPS()
{
    // Initialize member variables.
    iterations = 0;
    startTime = time(0);
    FPSCount = 0;
}

/****************************************************************************
			Description:	FPS destructor.

			Arguments:		None

			Derived From:	Nothing
****************************************************************************/
FPS::~FPS()
{

}

/****************************************************************************
			Description:	Counts number of interations.

			Arguments: 		None
	
			Returns: 		Nothing
****************************************************************************/
void FPS::Increment()
{
	// Increment Counter.
	iterations++;
	
	// Get seconds since start time.
	time_t timeNow = time(0) - startTime;
	// Calculate FPS if 1 second has passed.
	if (int(timeNow) >= 1)
	{
		// Store current number of iterations.
		FPSCount = iterations;

		// Reset Counter.
		iterations = 0;
		// Reset start time.
		startTime = time(0);
	}
}

/****************************************************************************
			Description:	Calculate average FPS.

			Arguments: 		None
	
			Returns: 		INT
****************************************************************************/
int FPS::FramesPerSec()
{
    // Return FPS value.
    return FPSCount;
}
///////////////////////////////////////////////////////////////////////////////