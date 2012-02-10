// spacemanspiff
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

#include <cv>
#include <highgui>

#define vi vector<int>
#define v2di vector< vector<int> >
 
#define fR(i,L,U) for( int i=L; i<U; ++i )
#define f0R(i,U) for( int i=0; i<U; ++i)
 
#define ull unsigned long long int
#define ll long long int

#define DEBUG 0
#define FILEOP 0

using namespace std;

//map<string, string> cascadeFileMap;
//map<string, int*> colourMap;

// Return a CvRect bounding detected faces in the gray-scale converted image
CvRect detectFace(IplImage* image, CvHaarClassifierCascade* cascade)
{
	CvSize minFaceSize = cvSize(20, 20); // Restrict to images greater than 20x20 pixels
	int flags = CV_HAAR_FIND_BIGGEST_OBJECT
		    | CV_HAAR_DO_ROUGH_SEARCH; // Terminate search when first candidate if found
		    // | CV_HAAR_DO_CANNY_PRUNING; // Ignore flat regions
	float searchScaleFactor = 1.1f; // Increase search window size by 10%
	int minNeighbours = 3, // Minimum Neighbours. Prevent false positives by detecting faces with atleast 3 overlapping regions
	IplImage *detectImg;
	IplImage *grayImg = 0;
	CvMemStorage *storage;
	CvRect rect;
	CvSeq* rects;
	CvSize size;
	int nFaces;

	// Create empty storage with block size set to default value
	storage = cvCreateMemStorage(0);
	cvClearMemStorage(storage);

	// If image is RGB, convert to grayscale
	detectImg = (IplImage*)image;
	if(image->nChannels > 1)
	{
		size = cvSize(image->width, image->height);
		grayImg = cvCreateImage(size, IPL_DEPTH_8U, 1);
		cvCvtColor(image, grayImg, CV_BGR2GRAY);
		detectImg = grayImg;
	}

	// Detect all faces
	rects = cvHaarDetectObjects(detectImg, cascade, storage, searchScaleFactor, minNeighbours,
				    flags, minFaceSize);
	nFaces = rects->total;

#if DEBUG
	printf("Detected %d faces", nFaces);
#endif
	// Return first detected object, else return negative
	rect = (nFaces > 0)? *(CvRect*)cvGetSeqElem(rects, 0) : cvRect(-1, -1, -1, -1);

	if(grayImg)
		cvReleaseImage(grayImg);
	cvReleaseMemStorage(storage);

	return rc;
}

void drawBox(IplImage* image, CvRect rect)
{
	static CvScalar white[] = {255, 255, 255};
	cvRectangle(image,
		    cvPoint(rect.x,rect.y),
		    cvPoint(rect.x + rect.width, rect.y + rect.height),
		    white);
}

int main(int argc, char** argv)
{
#if FILEOP
	ifstream f (argv[1]);

	if( f.is_open() )
	{
		while( f.good() )
		{
			// File operations
		}
	}
#endif
	
	cvNamedWindow("CA", CV_WINDOW_AUTOSIZE);
	CvCapture* capture;

	switch(argc)
	{
		case 2: capture = cvCreateFileCapture( argv[1] );
			break;

		default: capture = cvCreateCameraCapture( -1 );
			 break;
	}
	assert( capture != NULL );

	IplImage* frame;
	char faceCascadeFileName = "haarcascade_frontalface_default.xml";
	CvHaarClassifierCascade* faceCascade;
	faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFileName, 0, 0, 0);
	assert(faceCascade != NULL);

	while(true)
	{
		frame = cvQueryFrame(capture);
		if( !frame )
			break;
		
		CvRect faceRect = detectFace(frame, faceCascade);
		drawBox(frame, faceRect);
		cvShowImage("CA", frame);

		char c = cvWaitKey(33);
		if( c == 27 )
			break;

	}

	cvReleaseHaarClassifierCascade(&faceCascade);
	cvReleaseCapture(&capture);
	cvDestroyWindow("CA");

	return 0;
}
