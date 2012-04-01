#include <iostream>

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>

#include "utils.h"

int                   _faceHeight	    = 180;
int                   _faceWidth	    = 180;


using namespace std;

IplImage* convertImageToGrayscale(IplImage* srcImage)
{
	IplImage* grayImage;
	if(srcImage->nChannels > 1)
	{
		grayImage = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_8U, 1);
		cvCvtColor(srcImage, grayImage, CV_BGR2GRAY);
	}
	else
	{
		grayImage = cvCloneImage(srcImage);
	}
	return grayImage;
}

// Return image of the face in the frame defined by faceRect
IplImage* cropImage(IplImage* srcImage, CvRect faceRect)
{
	IplImage* tempImage;
	IplImage* rgbImage;
	CvSize size = cvSize(srcImage->width, srcImage->height);

	// Assert if image is not a BGR, 8-bit per pixel image
	assert(srcImage->depth == IPL_DEPTH_8U);

	// Create tempImage and initially store the original contents
	tempImage = cvCreateImage(size, IPL_DEPTH_8U, srcImage->nChannels);
	cvCopy(srcImage, tempImage, NULL);

	// Set region of interest as faceRect
	cvSetImageROI(tempImage, faceRect);

	// Copy the region of interest into rgbImage
	size = cvSize(faceRect.width, faceRect.height);
	rgbImage = cvCreateImage(size, IPL_DEPTH_8U, srcImage->nChannels);
	cvCopy(tempImage, rgbImage, NULL);

	cvReleaseImage(&tempImage);
	return rgbImage;
}


IplImage* resizeImage(IplImage* srcImage, bool preserveAspectRatio = true,
		      int newHeight = _faceHeight, int newWidth = _faceWidth)
{
	IplImage* outImage;
        int origWidth;
        int origHeight;
        CvRect rect;
        if(srcImage)
        {
            origWidth = srcImage->width;
            origHeight = srcImage->height;
        }
        if(preserveAspectRatio)
        {
            float origAspectRatio = (float) origWidth / origHeight;
            float newAspectRatio = (float) newWidth / newHeight;

            if(origAspectRatio > newAspectRatio)
            {
                int wTemp = ( origHeight * newWidth ) / newHeight;
                rect = cvRect((origWidth - wTemp)/2, 0, wTemp, origHeight); 
            }
            else
            {
                int hTemp = ( origWidth * newHeight ) / newWidth;
                rect = cvRect(0, (origHeight - hTemp)/2, origWidth, hTemp); 
            }
            
            IplImage* croppedImage = cropImage(srcImage, rect);
            outImage = resizeImage(croppedImage, false);
        }
        else
        {
            outImage = cvCreateImage(cvSize(newWidth, newHeight), srcImage->depth,
                                     srcImage->nChannels);
            if((newWidth > srcImage->width) && (newHeight > srcImage->height))
            {
                cvResetImageROI((IplImage*)srcImage);
                // To enlarge
                cvResize(srcImage, outImage, CV_INTER_LINEAR);
            }
            else
            {
                cvResetImageROI((IplImage*)srcImage);
                // To shrink
                cvResize(srcImage, outImage, CV_INTER_AREA);
            }
        }
        return outImage;
}

void drawBox(IplImage* image, CvRect rect, CvScalar colour)
{
	cvRectangle(image,
		    cvPoint(rect.x,rect.y),
		    cvPoint(rect.x + rect.width, rect.y + rect.height),
		    colour);
}

