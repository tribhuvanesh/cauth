// spacemanspiff
// If trying to debug the colour detector code, enable SHOW_DEBUG_IMAGE:
#define SHOW_DEBUG_IMAGE 0

#include <cstdio>	// Used for "printf"
#include <string>	// Used for C++ strings
#include <iostream>	// Used for C++ cout print statements
#include <cmath>	// Used to calculate square-root for statistics

//ImageUtil includes
#include <vector>	// Used for C++ vectors
#include <map>
#include <algorithm>
#include <numeric>
//#include <sstream>	// for printing floats in C++
//#include <fstream>	// for opening files in C++

// Include OpenCV libraries
#include <cv.h>
#include <cvaux.h>
#include <cxcore.h>
#include <highgui.h>

// Custom libraries
#include "detect.h"
#include "utils.h"

#define AVG_TEMP_COUNT 10
#define EPSILON 0.01

extern bool run_shirt;
// Various colour types for detected shirt colours.
enum                             {cBLACK=0,cWHITE, cGREY, cRED, cORANGE, cYELLOW, cGREEN, cAQUA, cBLUE, cPURPLE, cPINK,  NUM_COLOUR_TYPES};
extern char sCTypes[][NUM_COLOUR_TYPES];
extern uchar cCTHue[NUM_COLOUR_TYPES];
extern uchar cCTSat[NUM_COLOUR_TYPES];
extern uchar cCTVal[NUM_COLOUR_TYPES];

extern string colour_types[];

// Initialize vector with colours
// vector<string> colour_vect(sCTypes, sCTypes + NUM_COLOUR_TYPES);

// Range for normalizing RMSD
extern float range_norm;

extern const char* cascadeFileFace;	// Path to the Face Detection HaarCascade XML file
#ifndef GET_PIXEL_COLOR_TYPE
#define GET_PIXEL_COLOR_TYPE
    int getPixelColorType(int, int, int);
#endif

#ifndef GET_TEMPLATE
#define GET_TEMPLATE
    map<string, float> getTemplate(IplImage*, CvHaarClassifierCascade*);
#endif

#ifndef CREATE_AVERAGE
#define CREATE_AVERAGE
    map<string, float> createAverage(vector< map<string, float> >);
#endif

#ifndef SIGMOID
#define SIGMOID
    float sigmoid(float);
#endif

#ifndef NRMSD
#define NRMSD
    float nrmsd(map<string, float>, map<string, float>);
#endif

#ifndef CREATE_TEMPLATE
#define CREATE_TEMPLATE
    map<string, float> createTemplate(CvCapture*, CvHaarClassifierCascade*, int);
#endif

#ifndef SOFT_MAIN
#define SOFT_MAIN
    int soft_main();
#endif
