#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <deque>
#include <stack>
#include <queue>
#include <string>
#include <list>
#include <map>
#include <set>
#include <bitset>
#include <complex>
#include <sstream>
#include <fstream>
#include <functional>
#include <numeric>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <climits>
#include <cctype>
#include <cmath>
#include <cassert>
#include <ctime>

#include "detect.h"
#include "utils.h"

#define vi vector<int>
#define v2di vector< vector<int> >
 
#define fR(i,L,U) for( int i=L; i<U; ++i )
#define f0R(i,U) for( int i=0; i<U; ++i)
 
#define ull unsigned long long int
#define ll long long int

#define DEBUG 0
#define FILEOP 0

using namespace std;

CvRect detectFace(IplImage* image, CvHaarClassifierCascade* cascade)
{
	CvSize minFaceSize = cvSize(20, 20); // Restrict to images greater than 20x20 pixels
	int flags = CV_HAAR_FIND_BIGGEST_OBJECT
		    | CV_HAAR_DO_ROUGH_SEARCH; // Terminate search when first candidate if found
		    // | CV_HAAR_DO_CANNY_PRUNING; // Ignore flat regions
	float searchScaleFactor = 1.1f; // Increase search window size by 10%
	int minNeighbours = 3; // Minimum Neighbours. Prevent false positives by detecting faces with atleast 3 overlapping regions
	IplImage *detectImage;
	IplImage *grayImage = 0;
	CvMemStorage *storage;
	CvRect rect;
	CvSeq* rects;
	CvSize size;
	int nFaces;

	// Create empty storage with block size set to default value
	storage = cvCreateMemStorage(0);
	cvClearMemStorage(storage);

	// If image is RGB, convert to grayscale
	detectImage = convertImageToGrayscale((IplImage*)image);

	// Detect all faces
	rects = cvHaarDetectObjects(detectImage, cascade, storage, searchScaleFactor, minNeighbours,
				    flags, minFaceSize);
	nFaces = rects->total;

#if DEBUG
	printf("Detected %d faces\n", nFaces);
#endif
	// Return first detected object, else return negative
	rect = (nFaces > 0)? *(CvRect*)cvGetSeqElem(rects, 0) : cvRect(-1, -1, -1, -1);

	if(grayImage)
		cvReleaseImage(&grayImage);
	cvReleaseMemStorage(&storage);

	return rect;
}



