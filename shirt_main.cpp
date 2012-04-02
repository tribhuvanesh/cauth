// If trying to debug the color detector code, enable SHOW_DEBUG_IMAGE:
#define SHOW_DEBUG_IMAGE 0

#include <cstdio>	// Used for "printf"
#include <string>	// Used for C++ strings
#include <iostream>	// Used for C++ cout print statements
#include <cmath>	// Used to calculate square-root for statistics

//ImageUtil includes
//#include <tchar.h>
//#include <vector>	// Used for C++ vectors
#include <sstream>	// for printing floats in C++
#include <fstream>	// for opening files in C++

// Include OpenCV libraries
#include <cv.h>
#include <cvaux.h>
#include <cxcore.h>
#include <highgui.h>
#include "detect.h"
#include "utils.h"


bool run_shirt=true;

// Face Detection HaarCascade Classifier file for OpenCV (downloadable from "http://alereimondo.no-ip.org/OpenCV/34").
const char* cascadeFileFace = "haar/haarcascade_frontalface_alt.xml";	// Path to the Face Detection HaarCascade XML file

// Various color types for detected shirt colors.
enum                             {cBLACK=0,cWHITE, cGREY, cRED, cORANGE, cYELLOW, cGREEN, cAQUA, cBLUE, cPURPLE, cPINK,  NUM_COLOR_TYPES};
char sCTypes[][NUM_COLOR_TYPES] = {"Black", "White","Grey","Red","Orange","Yellow","Green","Aqua","Blue","Purple","Pink"};
uchar cCTHue[NUM_COLOR_TYPES] =    {0,       0,      0,     0,     20,      30,      55,    85,   115,    138,     161};
uchar cCTSat[NUM_COLOR_TYPES] =    {0,       0,      0,    255,   255,     255,     255,   255,   255,    255,     255};
uchar cCTVal[NUM_COLOR_TYPES] =    {0,      255,    120,   255,   255,     255,     255,   255,   255,    255,     255};

struct shirt_template
{
	IplImage * shirt_hsv;
	int color_perc[NUM_COLOR_TYPES];
};
typedef struct shirt_template * S_TEMPL;

// Determine what type of color the HSV pixel is. Returns the colorType between 0 and NUM_COLOR_TYPES.
int getPixelColorType(int H, int S, int V)
{
	int color;
	if (V < 75)
		color = cBLACK;
	else if (V > 190 && S < 27)
		color = cWHITE;
	else if (S < 53 && V < 185)
		color = cGREY;
	else {	// Is a color
		if (H < 14)
			color = cRED;
		else if (H < 25)
			color = cORANGE;
		else if (H < 34)
			color = cYELLOW;
		else if (H < 73)
			color = cGREEN;
		else if (H < 102)
			color = cAQUA;
		else if (H < 127)
			color = cBLUE;
		else if (H < 149)
			color = cPURPLE;
		else if (H < 175)
			color = cPINK;
		else	// full circle 
			color = cRED;	// back to Red
	}
	return color;
}

void create_template(S_TEMPL init_templ,CvCapture* capture,CvHaarClassifierCascade* cascadeFace)
{
	IplImage * imageIn;
	//CvCapture* capture;

	//capture = cvCreateCameraCapture(getenv("CAM")==NULL? -1:1);
	assert( capture != NULL );
	
	// Load the HaarCascade classifier for face detection. Added by Shervin on 22/9/09
	//cout << "Loading Face HaarCascade in '" << cascadeFileFace << "'" << endl;
	//CvHaarClassifierCascade* cascadeFace = (CvHaarClassifierCascade*)cvLoad(cascadeFileFace, 0, 0, 0 );
	if( !cascadeFace ) 
	{
		cerr << "ERROR: Couldn't load face detector classifier in '" << cascadeFileFace << "'\n";
		exit(1);
	}	
	cout<<"initialising initial template"<<endl;
	for(int k=0; k<NUM_COLOR_TYPES; k++)
		init_templ->color_perc[k]=0;		

	cvNamedWindow("Shirt", 1);	
	cout<<"beginning capture for initial template"<<endl;	
	int j=0;	
	while(j< 5)
	{
						
		imageIn = cvQueryFrame(capture);
		assert( imageIn != NULL );			
		
		IplImage* imageDisplay = cvCloneImage(imageIn);	//display image with rectangles
		CvRect rectFace;
		cvShowImage("Shirt", imageDisplay);		
		rectFace = detectFace(imageIn, cascadeFace);
		drawBox(imageDisplay, rectFace, CV_RGB(255,0,0));
		//cout << "Detecting shirt color below the face." << endl;
		
		int bottom;
		// Create the shirt region, to be below the detected face and of similar size.
		float SHIRT_DY = 1.4f;	// Distance from top of face to top of shirt region, based on detected face height.
		float SHIRT_SCALE_X = 0.6f;	// Width of shirt region compared to the detected face
		float SHIRT_SCALE_Y = 0.6f;	// Height of shirt region compared to the detected face
		
		CvRect rectShirt;
		rectShirt.x = rectFace.x + (int)(0.5f * (1.0f-SHIRT_SCALE_X) * (float)rectFace.width);
		rectShirt.y = rectFace.y + (int)(SHIRT_DY * (float)rectFace.height) + (int)(0.5f * (1.0f-SHIRT_SCALE_Y)* (float)rectFace.height);
		rectShirt.width = (int)(SHIRT_SCALE_X * rectFace.width);
		rectShirt.height = (int)(SHIRT_SCALE_Y * rectFace.height);
		//cout << "Shirt region is from " << rectShirt.x << ", " << rectShirt.y << " to " << rectShirt.x + rectShirt.width - 1 << ", " << rectShirt.y + rectShirt.height - 1 << endl;
		
		// If the shirt region goes partly below the image, try just a little below the face
		bottom = rectShirt.y+rectShirt.height-1;
		if (bottom > imageIn->height-1) 
		{
			SHIRT_DY = 0.95f;	// Distance from top of face to top of shirt region, based on detected face height.
			SHIRT_SCALE_Y = 0.3f;	// Height of shirt region compared to the detected face
			// Use a higher shirt region
			rectShirt.y = rectFace.y + (int)(SHIRT_DY * (float)rectFace.height) + (int)(0.5f * (1.0f-SHIRT_SCALE_Y) * (float)rectFace.height);
			rectShirt.height = (int)(SHIRT_SCALE_Y * rectFace.height);
			cout << "Warning: Shirt region goes past the end of the image. Trying to reduce the shirt region position to " ;
			cout<< rectShirt.y << " with a height of " << rectShirt.height << endl;
		}
		// Try once again if it is partly below the image.
		bottom = rectShirt.y+rectShirt.height-1;
		if (bottom > imageIn->height-1) 
		{
			bottom = imageIn->height-1;	// Limit the bottom
			rectShirt.height = bottom - (rectShirt.y-1);	// Adjust the height to use the new bottom
			cout << "Warning: Shirt region still goes past the end of the image. Trying to reduce the shirt region height to ";
			cout << rectShirt.height << endl;
		}
		// Make sure the shirt region is in the image
		if (rectShirt.height <= 1) 
		{
			cout << "Warning: Shirt region is not in the image at all, so skipping this face." << endl;
			continue;
		}
		else 
		{
				drawBox(imageDisplay, rectFace, CV_RGB(255,0,0));
					
			j++;
			// Convert the shirt region from RGB colors to HSV colors
			cout << "Converting shirt region to HSV" << endl;
			IplImage *imageShirt = cropImage(imageIn, rectShirt);
			IplImage *imageShirtHSV = cvCreateImage(cvGetSize(imageShirt), 8, 3);	
			cvCvtColor(imageShirt, imageShirtHSV, CV_BGR2HSV);	// (note that OpenCV stores RGB images in B,G,R order.)	
			if( !(imageShirtHSV) ) 
			{
				cerr << "ERROR: Couldn't convert Shirt image from BGR2HSV." << endl;
				exit(1);
			}
			cout << "Determining color type of the shirt" << endl;
			int h = imageShirtHSV->height;				// Pixel height
			int w = imageShirtHSV->width;				// Pixel width
			int rowSize = imageShirtHSV->widthStep;		// Size of row in bytes, including extra padding
			char *pixOfsh = imageShirtHSV->imageData;	// Pointer to the start of the image HSV pixels.
			cout<<"reached here";
			// Create an empty tally of pixel counts for each color type
			int tallyColors[NUM_COLOR_TYPES];
			for (int i=0; i<NUM_COLOR_TYPES; i++)
			{	tallyColors[i] = 0;}
				
			// Scan the shirt image to find the tally of pixel colors
			for (int y=0; y<h; y++)	//row no. 
			{
				for (int x=0; x<w; x++) //col no.
				{
					// Get the HSV pixel components
					uchar H = *(uchar*)(pixOfsh + y*rowSize + x*3 + 0);	// Hue
					uchar S = *(uchar*)(pixOfsh + y*rowSize + x*3 + 1);	// Saturation
					uchar V = *(uchar*)(pixOfsh + y*rowSize + x*3 + 2);	// Value (Brightness)

					// Determine what type of color the HSV pixel is.
					int ctype = getPixelColorType(H, S, V);
					// Keep count of these colors.
					tallyColors[ctype]++;
				}
			}
			
			
	    		cvShowImage("Shirt", imageDisplay);
		
			cout<<" Store into template the sum of 5 shirt images based on color types";
			int pixels = w * h;
			for (int i=0; i<NUM_COLOR_TYPES; i++) 
				init_templ->color_perc[i] += (tallyColors[i]*100/pixels);					
			
			// Free resources.
			cvReleaseImage( &imageShirt );
			cvReleaseImage( &imageShirtHSV );		
		}//end if valid height
		cvReleaseImage(&imageDisplay);
		cvReleaseImage(&imageIn);
		
	
	}//end while loop for 5 shirt images
	for (int i=0; i<NUM_COLOR_TYPES; i++)	//calculate avg of image HSV colours
		init_templ->color_perc[i] /= 5;					

	// Free resources.
	//cvReleaseHaarClassifierCascade( &cascadeFace );
	//cvReleaseCapture(&capture);
	cvDestroyWindow("Shirt");
}

float compare_shirt(S_TEMPL init_templ,S_TEMPL sCompare)
{
	S_TEMPL delta_sq, rating;
	int sum=0;
	for (int i=0; i<NUM_COLOR_TYPES; i++)
	{	
		delta_sq->color_perc[i] = (init_templ->color_perc[i]-sCompare->color_perc[i])^2;
		rating->color_perc[i] = 1 - sqrt(delta_sq->color_perc[i]/(init_templ->color_perc[i]^2));
	}
	for (int i=0; i<NUM_COLOR_TYPES; i++)
		sum += rating->color_perc[i];
	return sum/5;				
}

// C/C++ entry point
int main(int argc, char **argv)
{	
	IplImage * imageIn;
	CvCapture* capture;
	S_TEMPL init_templ= (struct shirt_template*) malloc(sizeof(struct shirt_template));
	//int flag =0;
	
	capture = cvCreateCameraCapture(getenv("CAM")==NULL? -1:1);
	assert( capture != NULL );

	//Load the HaarCascade classifier for face detection. Added by Shervin on 22/9/09
	cout << "Loading Face HaarCascade in '" << cascadeFileFace << "'" << endl;
	CvHaarClassifierCascade* cascadeFace = (CvHaarClassifierCascade*)cvLoad(cascadeFileFace, 0, 0, 0 );
	if( !cascadeFace ) 
	{
		cerr << "ERROR: Couldn't load face detector classifier in '" << cascadeFileFace << "'\n";
		exit(1);
	}	

	create_template(init_templ,capture,cascadeFace);	//CREATES INITIAL TEMPLATE
	cout<<"template creation completed";	
	
	while(run_shirt)
	{						
		S_TEMPL sCompare= (struct shirt_template*) malloc(sizeof(struct shirt_template));
	
		for (int i=0; i<NUM_COLOR_TYPES; i++)
			sCompare->color_perc[i]=0;			
	
		imageIn = cvQueryFrame(capture);
		assert( imageIn != NULL );			

		IplImage* imageDisplay = cvCloneImage(imageIn);
	
		// If trying to debug the color detector code, enable this:
		/*#ifdef SHOW_DEBUG_IMAGE
			// Create a HSV image showing the color types of the whole image, for debugging.
			IplImage *imageInHSV = cvCreateImage(cvGetSize(imageIn), 8, 3);
			cvCvtColor(imageIn, imageInHSV, CV_BGR2HSV);	// (note that OpenCV stores RGB images in B,G,R order.
			IplImage* imageDisplayHSV = cvCreateImage(cvGetSize(imageIn), 8, 3);	// Create an empty HSV image
			//cvSet(imageDisplayHSV, cvScalar(0,0,0, 0));	// Clear HSV image to blue.
			int hIn = imageDisplayHSV->height;
			int wIn = imageDisplayHSV->width;
			int rowSizeIn = imageDisplayHSV->widthStep;		// Size of row in bytes, including extra padding
			char *pixOfshDisp = imageDisplayHSV->imageData;	// Pointer to the start of the image HSV pixels.
			char *pixOfshIn = imageInHSV->imageData;	// Pointer to the start of the input image HSV pixels.
			for (int y=0; y<hIn; y++) {
				for (int x=0; x<wIn; x++) {
					// Get the HSV pixel components
					uchar H = *(uchar*)(pixOfshIn + y*rowSizeIn + x*3 + 0);	// Hue
					uchar S = *(uchar*)(pixOfshIn + y*rowSizeIn + x*3 + 1);	// Saturation
					uchar V = *(uchar*)(pixOfshIn + y*rowSizeIn + x*3 + 2);	// Value (Brightness)
					// Determine what type of color the HSV pixel is.
					int ctype = getPixelColorType(H, S, V);
					//ctype = x / 60;
					// Show the color type on the displayed image, for debugging.
					*(uchar*)(pixOfshDisp + (y)*rowSizeIn + (x)*3 + 0) = cCTHue[ctype];	// Hue
					*(uchar*)(pixOfshDisp + (y)*rowSizeIn + (x)*3 + 1) = cCTSat[ctype];	// Full Saturation (except for black & white)
					*(uchar*)(pixOfshDisp + (y)*rowSizeIn + (x)*3 + 2) = cCTVal[ctype];	// Full Brightness
				}
			}
			// Display the HSV debugging image
			IplImage *imageDisplayHSV_RGB = cvCreateImage(cvGetSize(imageDisplayHSV), 8, 3);
			cvCvtColor(imageDisplayHSV, imageDisplayHSV_RGB, CV_HSV2BGR);	// (note that OpenCV stores RGB images in B,G,R order.
			cvNamedWindow("Colors", 1);
			cvShowImage("Colors", imageDisplayHSV_RGB);
		#endif	// SHOW_DEBUG_IMAGE
		*/
		CvRect rectFace;
		rectFace = detectFace(imageIn, cascadeFace);
		cout<< "detected face";
		// Process each detected face
		cout << "Detecting shirt color below the face." << endl;
						
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
		cout << "Shirt region is from " << rectShirt.x << ", " << rectShirt.y << " to " << rectShirt.x + rectShirt.width - 1 << ", " << rectShirt.y + rectShirt.height - 1 << endl;
		
		// If the shirt region goes partly below the image, try just a little below the face
		bottom = rectShirt.y+rectShirt.height-1;
		if (bottom > imageIn->height-1) 
		{
			SHIRT_DY = 0.95f;	// Distance from top of face to top of shirt region, based on detected face height.
			SHIRT_SCALE_Y = 0.3f;	// Height of shirt region compared to the detected face
			// Use a higher shirt region
			rectShirt.y = rectFace.y + (int)(SHIRT_DY * (float)rectFace.height) + (int)(0.5f * (1.0f-SHIRT_SCALE_Y) * (float)rectFace.height);
			rectShirt.height = (int)(SHIRT_SCALE_Y * rectFace.height);
			cout << "Warning: Shirt region goes past the end of the image. Trying to reduce the shirt region position to " ;
			cout<< rectShirt.y << " with a height of " << rectShirt.height << endl;
		}
		// Try once again if it is partly below the image.
		bottom = rectShirt.y+rectShirt.height-1;
		if (bottom > imageIn->height-1) 
		{
			bottom = imageIn->height-1;	// Limit the bottom
			rectShirt.height = bottom - (rectShirt.y-1);	// Adjust the height to use the new bottom
			cout << "Warning: Shirt region still goes past the end of the image. Trying to reduce the shirt region height to ";
			cout << rectShirt.height << endl;
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
			// Convert the shirt region from RGB colors to HSV colors
			cout << "Converting shirt region to HSV" << endl;
			IplImage *imageShirt = cropImage(imageIn, rectShirt);
			IplImage *imageShirtHSV = cvCreateImage(cvGetSize(imageShirt), 8, 3);
			cvCvtColor(imageShirt, imageShirtHSV, CV_BGR2HSV);	// (note that OpenCV stores RGB images in B,G,R order.)	
			if( !imageShirtHSV ) 
			{
				cerr << "ERROR: Couldn't convert Shirt image from BGR2HSV." << endl;
				exit(1);
			}
			//cout << "Determining color type of the shirt" << endl;
			int h = imageShirtHSV->height;				// Pixel height
			int w = imageShirtHSV->width;				// Pixel width
			int rowSize = imageShirtHSV->widthStep;		// Size of row in bytes, including extra padding
			char *pixOfsh = imageShirtHSV->imageData;	// Pointer to the start of the image HSV pixels.
			// Create an empty tally of pixel counts for each color type
			int tallyColors[NUM_COLOR_TYPES];
			for (int i=0; i<NUM_COLOR_TYPES; i++)
				tallyColors[i] = 0;
			// Scan the shirt image to find the tally of pixel colors
			for (int y=0; y<h; y++)	//row no. 
			{
				for (int x=0; x<w; x++) //col no.
				{
					// Get the HSV pixel components
					uchar H = *(uchar*)(pixOfsh + y*rowSize + x*3 + 0);	// Hue
					uchar S = *(uchar*)(pixOfsh + y*rowSize + x*3 + 1);	// Saturation
					uchar V = *(uchar*)(pixOfsh + y*rowSize + x*3 + 2);	// Value (Brightness)

					// Determine what type of color the HSV pixel is.
					int ctype = getPixelColorType(H, S, V);
					// Keep count of these colors.
					tallyColors[ctype]++;
				}
			}
			// Compare the two templates
			//cout << "Number of pixels found using each color type (out of " << (w*h) << ":\n";
			int tallyMaxIndex = 0;
			int tallyMaxCount = -1;
			int pixels = w * h;
			for (int i=0; i<NUM_COLOR_TYPES; i++) 
			{
				int v = tallyColors[i];
				
				sCompare->color_perc[i] = (v*100/pixels);
				
				if (v > tallyMaxCount) 
				{
					tallyMaxCount = tallyColors[i];
					tallyMaxIndex = i;
				}
			}
			float confidence= compare_shirt(init_templ,sCompare);
			cout << endl;
			cout << "Color of shirt: " << sCTypes[tallyMaxIndex] << " (" << sCompare->color_perc[tallyMaxIndex] << "% .)" << endl << endl;

			// Display the color type over the shirt in the image.
			CvFont font;
			//cvInitFont(&font,CV_FONT_HERSHEY_PLAIN,0.55,0.7, 0,1,CV_AA);	// For OpenCV 1.1
			cvInitFont(&font,CV_FONT_HERSHEY_PLAIN,0.8,1.0, 0,1, CV_AA);	// For OpenCV 2.0
			char text[256];
			snprintf(text, sizeof(text)-1, "%.2f%%", confidence);		
			cvPutText(imageDisplay, sCTypes[tallyMaxIndex], cvPoint(rectShirt.x, rectShirt.y - 12), &font, CV_RGB(255,0,0));
			cvPutText(imageDisplay, text, cvPoint(rectShirt.x, rectShirt.y - 24), &font, CV_RGB(255,0,0));
			
			// Free resources.
			cvReleaseImage( &imageShirtHSV );
			cvReleaseImage( &imageShirt );
		}//end if valid height
		
		// Display the RGB debugging image
		cvNamedWindow("Shirt", 1);
    		cvShowImage("Shirt", imageDisplay);
		
		char c = cvWaitKey(30);
		if( c == 27 )
			break;

		// Free resources.
		cvReleaseImage(&imageDisplay);
	   	
	}
	
	// Close everything	
	cvReleaseImage(&imageIn);
	cvReleaseHaarClassifierCascade( &cascadeFace );
	cvReleaseCapture(&capture);
	cvDestroyWindow("Shirt");
	return 0;
}
