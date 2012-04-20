// If trying to debug the colour detector code, enable SHOW_DEBUG_IMAGE:
#define SHOW_DEBUG_IMAGE 0

#include <cstdio>	// Used for "printf"
#include <string>	// Used for C++ strings
#include <iostream>	// Used for C++ cout print statements
//#include <cmath>	// Used to calculate square-root for statistics

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
#include "detect.h"
#include "utils.h"

#define AVG_TEMP_COUNT 10
#define EPSILON 0.01
bool run_shirt=true;

// Various colour types for detected shirt colours.
enum                             {cBLACK=0,cWHITE, cGREY, cRED, cORANGE, cYELLOW, cGREEN, cAQUA, cBLUE, cPURPLE, cPINK,  NUM_COLOUR_TYPES};
char sCTypes[][NUM_COLOUR_TYPES] = {"Black", "White","Grey","Red","Orange","Yellow","Green","Aqua","Blue","Purple","Pink"};
uchar cCTHue[NUM_COLOUR_TYPES] =    {0,       0,      0,     0,     20,      30,      55,    85,   115,    138,     161};
uchar cCTSat[NUM_COLOUR_TYPES] =    {0,       0,      0,    255,   255,     255,     255,   255,   255,    255,     255};
uchar cCTVal[NUM_COLOUR_TYPES] =    {0,      255,    120,   255,   255,     255,     255,   255,   255,    255,     255};

string colour_types[] = {"Black", "White","Grey","Red","Orange","Yellow","Green","Aqua","Blue","Purple","Pink"};

// Function prototypes
int		    getPixelColorType(int H, int S, int V);
map<string, float>  getTemplate( IplImage*, CvHaarClassifierCascade* );
map<string, float>  createAverage( vector< map<string, float> > );
float		    sigmoid( float );
float		    nrmsd( map<string, float>, map<string, float> );

// Initialize vector with colours
vector<string> colour_vect(sCTypes, sCTypes + NUM_COLOUR_TYPES);

// Range for normalizing RMSD
float range_norm = sqrt((2*100*100) / NUM_COLOUR_TYPES);

// Face Detection HaarCascade Classifier file for OpenCV (downloadable from "http://alereimondo.no-ip.org/OpenCV/34").
const char* cascadeFileFace = "haar/haarcascade_frontalface_alt.xml";	// Path to the Face Detection HaarCascade XML file


// Determine what type of colour the HSV pixel is. Returns the colourType between 0 and NUM_COLOUR_TYPES.
int getPixelColorType(int H, int S, int V)
{
	int colour;
	if (V < 75)
		colour = cBLACK;
	else if (V > 190 && S < 27)
		colour = cWHITE;
	else if (S < 53 && V < 185)
		colour = cGREY;
	else {	// Is a colour
		if (H < 14)
			colour = cRED;
		else if (H < 25)
			colour = cORANGE;
		else if (H < 34)
			colour = cYELLOW;
		else if (H < 73)
			colour = cGREEN;
		else if (H < 102)
			colour = cAQUA;
		else if (H < 127)
			colour = cBLUE;
		else if (H < 149)
			colour = cPURPLE;
		else if (H < 175)
			colour = cPINK;
		else	// full circle 
			colour = cRED;	// back to Red
	}
	return colour;
}


map<string, float> getTemplate(IplImage* imageIn, CvHaarClassifierCascade *cascadeFace)
{	
						
		assert( imageIn != NULL );			
		//std::cout<< "captured frame"<<std::endl;		
		//std::cout << "(got a " << imageIn->width << "x" << imageIn->height << " colour image)." << std::endl;
		IplImage* imageDisplay = cvCloneImage(imageIn);
		map<string, float> cmap;

		// If trying to debug the colour detector code, enable this:
		#ifdef SHOW_DEBUG_IMAGE
			// Create a HSV image showing the colour types of the whole image, for debugging.
			IplImage *imageInHSV = cvCreateImage(cvGetSize(imageIn), 8, 3);
			cvCvtColor(imageIn, imageInHSV, CV_BGR2HSV);	// (note that OpenCV stores RGB images in B,G,R order.
				IplImage* imageDisplayHSV = cvCreateImage(cvGetSize(imageIn), 8, 3);	// Create an empty HSV image
				//cvSet(imageDisplayHSV, cvScalar(0,0,0, 0));	// Clear HSV image to blue.
				int hIn = imageDisplayHSV->height;
				int wIn = imageDisplayHSV->width;
				int rowSizeIn = imageDisplayHSV->widthStep;		// Size of row in bytes, including extra padding
				char *imOfsDisp = imageDisplayHSV->imageData;	// Pointer to the start of the image HSV pixels.
				char *imOfsIn = imageInHSV->imageData;	// Pointer to the start of the input image HSV pixels.
				for (int y=0; y<hIn; y++) {
					for (int x=0; x<wIn; x++) {
						// Get the HSV pixel components
						uchar H = *(uchar*)(imOfsIn + y*rowSizeIn + x*3 + 0);	// Hue
						uchar S = *(uchar*)(imOfsIn + y*rowSizeIn + x*3 + 1);	// Saturation
						uchar V = *(uchar*)(imOfsIn + y*rowSizeIn + x*3 + 2);	// Value (Brightness)
						// Determine what type of colour the HSV pixel is.
						int ctype = getPixelColorType(H, S, V);
						//ctype = x / 60;
						// Show the colour type on the displayed image, for debugging.
						*(uchar*)(imOfsDisp + (y)*rowSizeIn + (x)*3 + 0) = cCTHue[ctype];	// Hue
						*(uchar*)(imOfsDisp + (y)*rowSizeIn + (x)*3 + 1) = cCTSat[ctype];	// Full Saturation (except for black & white)
						*(uchar*)(imOfsDisp + (y)*rowSizeIn + (x)*3 + 2) = cCTVal[ctype];	// Full Brightness
					}
				}
				// Display the HSV debugging image
				IplImage *imageDisplayHSV_RGB = cvCreateImage(cvGetSize(imageDisplayHSV), 8, 3);
				cvCvtColor(imageDisplayHSV, imageDisplayHSV_RGB, CV_HSV2BGR);	// (note that OpenCV stores RGB images in B,G,R order.
				cvNamedWindow("Colors", 1);
				cvShowImage("Colors", imageDisplayHSV_RGB);
		#endif	// SHOW_DEBUG_IMAGE

		CvRect rectFace;
		//double timeFaceDetectStart = (double)cvGetTickCount();	// Record the timing.
		rectFace = detectFace(imageIn, cascadeFace);
		//double tallyFaceDetectTime = (double)cvGetTickCount() - timeFaceDetectStart;
		//cout << "Found " << rectFaces.size() << " faces in " << tallyFaceDetectTime/((double)cvGetTickFrequency()*1000.) << "ms\n";	
#if DEBUG		
		cout<< "detected face";
		// Process each detected face
		cout << "Detecting shirt colour below the face." << endl;
#endif

		float initialConfidence = 1.0f;
		int bottom;
		drawBox(imageDisplay, rectFace, CV_RGB(255,0,0));
		// Create the shirt region, to be below the detected face and of similar size.
		float SHIRT_DY = 1.4f;	// Distance from top of face to top of shirt region, based on detected face height.
		float SHIRT_SCALE_X = 0.6f;	// Width of shirt region compared to the detected face
		float SHIRT_SCALE_Y = 0.6f;	// Height of shirt region compared to the detected face
		CvRect rectShirt;
		rectShirt.x = rectFace.x + (int)(0.5f * (1.0f-SHIRT_SCALE_X) * (float)rectFace.width);
		rectShirt.y = rectFace.y + (int)(SHIRT_DY * (float)rectFace.height) + (int)(0.5f * (1.0f-SHIRT_SCALE_Y)* (float)rectFace.height);
		rectShirt.width = (int)(SHIRT_SCALE_X * rectFace.width);
		rectShirt.height = (int)(SHIRT_SCALE_Y * rectFace.height);
#if DEBUG		
		cout << "Shirt region is from " << rectShirt.x << ", " << rectShirt.y << " to " << rectShirt.x + rectShirt.width - 1 << ", " << rectShirt.y + rectShirt.height - 1 << endl;
#endif
		
		// If the shirt region goes partly below the image, try just a little below the face
		bottom = rectShirt.y+rectShirt.height-1;
		if (bottom > imageIn->height-1) 
		{
			SHIRT_DY = 0.95f;	// Distance from top of face to top of shirt region, based on detected face height.
			SHIRT_SCALE_Y = 0.3f;	// Height of shirt region compared to the detected face
			// Use a higher shirt region
			rectShirt.y = rectFace.y + (int)(SHIRT_DY * (float)rectFace.height) + (int)(0.5f * (1.0f-SHIRT_SCALE_Y) * (float)rectFace.height);
			rectShirt.height = (int)(SHIRT_SCALE_Y * rectFace.height);
			initialConfidence = initialConfidence * 0.5f;	// Since we are using a smaller region, we are less confident about the results now.
#if DEBUG
			cout << "Warning: Shirt region goes past the end of the image. Trying to reduce the shirt region position to " ;
			cout<< rectShirt.y << " with a height of " << rectShirt.height << endl;
#endif		
		}

		// Try once again if it is partly below the image.
		bottom = rectShirt.y+rectShirt.height-1;
		if (bottom > imageIn->height-1) {
		bottom = imageIn->height-1;	// Limit the bottom
		rectShirt.height = bottom - (rectShirt.y-1);	// Adjust the height to use the new bottom
		initialConfidence = initialConfidence * 0.7f;	// Since we are using a smaller region, we are less confident about the results now.
#if DEBUG
		cout << "Warning: Shirt region still goes past the end of the image. Trying to reduce the shirt region height to ";
		cout << rectShirt.height << endl;
#endif	
	}
	// Make sure the shirt region is in the image
		if (rectShirt.height <= 1) 
		{
			cout << "Warning: Shirt region is not in the image at all, so skipping this face." << endl;
		}
		else 
		{
	
			// Show the shirt region
			drawBox(imageDisplay, rectShirt, CV_RGB(255,255,255));
			// Convert the shirt region from RGB colours to HSV colours
			//cout << "Converting shirt region to HSV" << endl;
			IplImage *imageShirt = cropImage(imageIn, rectShirt);
			IplImage *imageShirtHSV = cvCreateImage(cvGetSize(imageShirt), 8, 3);
			cvCvtColor(imageShirt, imageShirtHSV, CV_BGR2HSV);	// (note that OpenCV stores RGB images in B,G,R order.	
			if( !imageShirtHSV ) {
				cerr << "ERROR: Couldn't convert Shirt image from BGR2HSV." << endl;
				exit(1);
			}
			//cout << "Determining colour type of the shirt" << endl;
			int h = imageShirtHSV->height;				// Pixel height
			int w = imageShirtHSV->width;				// Pixel width
			int rowSize = imageShirtHSV->widthStep;		// Size of row in bytes, including extra padding
			char *imOfs = imageShirtHSV->imageData;	// Pointer to the start of the image HSV pixels.
				// Create an empty tally of pixel counts for each colour type
				int tallyColors[NUM_COLOUR_TYPES];
				for (int i=0; i<NUM_COLOUR_TYPES; i++)
					tallyColors[i] = 0;
				// Scan the shirt image to find the tally of pixel colours
				for (int y=0; y<h; y++) 
				{
					for (int x=0; x<w; x++) {
						// Get the HSV pixel components
						uchar H = *(uchar*)(imOfs + y*rowSize + x*3 + 0);	// Hue
						uchar S = *(uchar*)(imOfs + y*rowSize + x*3 + 1);	// Saturation
						uchar V = *(uchar*)(imOfs + y*rowSize + x*3 + 2);	// Value (Brightness)
	
						// Determine what type of colour the HSV pixel is.
						int ctype = getPixelColorType(H, S, V);
						// Keep count of these colours.
						tallyColors[ctype]++;
					}
			}
			// Print a report about colour types, and find the max tally
			//cout << "Number of pixels found using each colour type (out of " << (w*h) << ":\n";
			int tallyMaxIndex = 0;
			int tallyMaxCount = -1;
			int pixels = w * h;
			for (int i=0; i<NUM_COLOUR_TYPES; i++) 
			{
				int v = tallyColors[i];
#if DEBUG				
				cout << sCTypes[i] << " " << (v*100/pixels) << "%, ";
#endif				
				string col = sCTypes[i];
				cmap[col] = (v * 100 / pixels);
				if (v > tallyMaxCount) 
				{
					tallyMaxCount = tallyColors[i];
					tallyMaxIndex = i;
				}
			}
			cout << endl;
			int percentage = initialConfidence * (tallyMaxCount * 100 / pixels);
#if DEBUG
			cout << "Color of shirt: " << sCTypes[tallyMaxIndex] << " (" << percentage << "% confidence)." << endl << endl;
#endif
			// Display the colour type over the shirt in the image.
			CvFont font;
			//cvInitFont(&font,CV_FONT_HERSHEY_PLAIN,0.55,0.7, 0,1,CV_AA);	// For OpenCV 1.1
			cvInitFont(&font,CV_FONT_HERSHEY_PLAIN,0.8,1.0, 0,1, CV_AA);	// For OpenCV 2.0
			char text[256];
			snprintf(text, sizeof(text)-1, "%d%%", percentage);		
			cvPutText(imageDisplay, sCTypes[tallyMaxIndex], cvPoint(rectShirt.x, rectShirt.y + rectShirt.height + 12), &font, CV_RGB(255,0,0));
			cvPutText(imageDisplay, text, cvPoint(rectShirt.x, rectShirt.y + rectShirt.height + 24), &font, CV_RGB(255,0,0));
			
			// Free resources.
			cvReleaseImage( &imageShirtHSV );
			cvReleaseImage( &imageShirt );
		}//end if valid height
			
		// Display the RGB debugging image
		// cvNamedWindow("Shirt", 1);
		cvShowImage("Shirt", imageDisplay);
			
	// Pause
	//cvWaitKey();
	
	// Free resources.
	cvReleaseImage(&imageDisplay);

	return cmap;
}

map<string, float> createAverage( vector< map<string, float> > cmapVect )
{
	int i, j;
	map<string, float> avgTemplate;

	for (i = 0; i < colour_vect.size(); i++)
	{
		for (j = 0; j < cmapVect.size(); j++)
		{
			avgTemplate[colour_vect[i]] += cmapVect[j][colour_vect[i]];
		}
		avgTemplate[colour_vect[i]] /= cmapVect.size();
	}

	return avgTemplate;
}

float sigmoid(float x)
{
	float exp_val = exp(-x);
	return 1 / (1 + exp_val);
}

float nrmsd(map<string, float> avgTemplate, map<string, float> currentTemplate)
{
	int i;
	float rmsd;
	vector<float> delta (NUM_COLOUR_TYPES, 0);
	//for_each(colour_vect.begin(), colour_vect.end(), func)

	for (i = 0; i < colour_vect.size(); i++) 
	{
		delta[i] = avgTemplate[colour_vect[i]] - currentTemplate[colour_vect[i]];
		delta[i] *= delta[i];
	}

	rmsd = sqrt( accumulate(delta.begin(), delta.end(), 0) / delta.size() );

	return 1 - rmsd/range_norm;
}

map<string, float> createTemplate(CvCapture* capture, CvHaarClassifierCascade* cascadeFace, int avgIterations)
{
	vector< map<string, float> > cmapVect;
	IplImage* imageIn;

	while(true)
	{
		imageIn = cvQueryFrame(capture);
		map<string, float> cmap = getTemplate(imageIn, cascadeFace);
		
		if(cmapVect.size() == avgIterations)
			break;
		else
			if(cmap.size() != 0)
				cmapVect.push_back(cmap);

		char c = cvWaitKey(30);
		if( c == 27 )
			break;
		
	}

	return createAverage(cmapVect);
}


int main(int argc, char** argv)
{
	int i;
	IplImage * imageIn;
	CvCapture* capture;
	
	map<string, float>::iterator it;
	map<string, float> avgTemplate;
	vector< map<string, float> > cmapVect;

	capture = cvCreateCameraCapture(getenv("CAM")==NULL? -1:1);
	assert( capture != NULL );

	cvNamedWindow("Shirt", CV_WINDOW_AUTOSIZE);
	printf("Initialized template");

	//Load the HaarCascade classifier for face detection. Added by Shervin on 22/9/09
	cout << "Loading Face HaarCascade in '" << cascadeFileFace << "'" << endl;
	CvHaarClassifierCascade* cascadeFace = (CvHaarClassifierCascade*)cvLoad(cascadeFileFace, 0, 0, 0 );
	if( !cascadeFace ) 
	{
		cerr << "ERROR: Couldn't load face detector classifier in '" << cascadeFileFace << "'\n";
		exit(1);
	}	

	// Creating average template by using the next 5 valid frames
	avgTemplate = createTemplate(capture, cascadeFace, AVG_TEMP_COUNT);

#if DEBUG
	cout<<"Printing the 5 colours templates"<<endl;
	for (i = 0; i < cmapVect.size(); i++)
	{
		for (it = cmapVect[i].begin(); it != cmapVect[i].end(); it++)
		{
			cout<<(*it).first<<" => "<<(*it).second<<"  ";
		}
		cout<<endl;	
	}

	cout<<"Average template: "<<endl;
	for (it = avgTemplate.begin(); it != avgTemplate.end(); it++)
	{
		cout<<(*it).first<<" => "<<(*it).second<<"  ";
	}
#endif

	string states[] = {"Lock-down", "Authorized"};
	bool current_state = 1;
	while(true)
	{
		imageIn = cvQueryFrame(capture);
		map<string, float> currentmap = getTemplate(imageIn, cascadeFace);

		if(currentmap.size() != 0)
		{	
			float confidence = nrmsd(avgTemplate, currentmap);
			bool new_state = ( (confidence >= 0.5)? 1 : 0 );
#if DEBUG		
			cout<<"confidence: "<<confidence;
#endif			
			//cout<<(( sigmoid(confidence) <= EPSILON )?"Authorized":"Lock-down")<<endl;
			//cout<<( (confidence >= 0.5)?"Authorized":"Lock-down" )<<endl;
			
			if(current_state != new_state)
				current_state = new_state;
			cout<<states[(int)current_state];
		}
		else
			continue;

		char c = cvWaitKey(30);
		if( c == 27 )
			break;
	}
	cvDestroyWindow("Shirt");

	cvReleaseImage(&imageIn);
	
	// Close everything	
	cvReleaseHaarClassifierCascade( &cascadeFace );
	cvReleaseCapture(&capture);

	return 0;
}
