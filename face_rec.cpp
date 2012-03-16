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

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
//#include "face_rec.h"

#define vi vector<int>
#define v2di vector< vector<int> >
 
#define fR(i,L,U) for( int i=L; i<U; ++i )
#define f0R(i,U) for( int i=0; i<U; ++i)
 
#define ull unsigned long long int
#define ll long long int

#define DEBUG 1
#define FILEOP 0

using namespace std;


// Global variables
map<string, CvScalar> colourMap;
map<string, string>   cascadeFileMap;
int                   faceHeight	    = 180;
int                   faceWidth		    = 180;
int                   nTrainFaces	    = 0; // No. of training images
int                   nEigens		    = 0; // No. of eigenvalues
IplImage**	      faceImageArr            = 0; // Array of face images
CvMat*		      personNumTruthMat     = 0; // Array of person numbers
IplImage*	      pAvgTrainImage        = 0; // Average image
IplImage**	      eigenVectArr          = 0; // eigenvectors
CvMat*		      eigenValMat	    = 0; // eigenvalues
CvMat*		      projectedTrainFaceMat = 0; // Projected training faces


// Function prototypes
void      init(void);
IplImage* cropImage(IplImage* srcImage, CvRect faceRect);
IplImage* resizeImage(IplImage* srcImage, bool preserveAspectRatio, int newHeight, int newWidth);
IplImage* convertImageToGrayscale(IplImage* srcImage);
CvRect    detectFace(IplImage* image, CvHaarClassifierCascade* cascade);
void      drawBox(IplImage* image, CvRect rect);

void learn();
void doPCA();
void storeTrainingData();
int loadTrainingData(CvMat** pTrainPersonNumMat);
int findNearestNeighbour(float* projectedTestFace, float* confidence);
int loadFaceImageArr(char* filename);


// Initializes constants and static data
void init()
{
	cascadeFileMap["default"] = "haarcascade_frontalface_default.xml";
	cascadeFileMap["alt"] = "haarcascade_frontalface_alt.xml";
	cascadeFileMap["alt2"] = "haarcascade_frontalface_alt2.xml";
	cascadeFileMap["alt_tree"] = "haarcascade_frontalface_alt_tree.xml";

	colourMap["white"] = cvScalar(255, 255, 255);
	colourMap["red"]   = cvScalar(255, 0, 0);
	colourMap["green"] = cvScalar(0, 255, 0);
	colourMap["blue"]  = cvScalar(0, 0, 255);
}


void learn()
{
	int i;

	// Load training data
	char trainingFile[] = "train.txt";
	nTrainFaces = loadFaceImageArr(trainingFile);
        printf("Read %d faces..\n", nTrainFaces);
	assert(nTrainFaces > 2);

	// Do PCA on training images to find a subspace
	doPCA();
        printf("Completed PCA\n");

	// Project the training images on to the PCA subspace
        projectedTrainFaceMat = cvCreateMat(nTrainFaces, nEigens, CV_32FC1);
	for (i = 0; i < nTrainFaces; i++) 
	{
                printf("Projecting face %d\n", i);
		cvEigenDecomposite( faceImageArr[i], // Input object
				    nEigens,         // no. of eigenvalues
				    eigenVectArr,    // Pointer to array of IplImage input objects
				    0, 0,            // ioFlags and userData
				    pAvgTrainImage,  // Averaged object
				    projectedTrainFaceMat->data.fl + i * nEigens // Output - calculated coefficients
				  );
	}

        printf("Completed PCA projection\n");
	// Store data as an xml file
	storeTrainingData();
}


void doPCA()
{
	int i;
	CvTermCriteria calcLimit;
	CvSize faceImageSize;

	// Set number of eigenvalues
	nEigens = nTrainFaces - 1;

	// Allocate the eigenvector images
	faceImageSize.width = faceImageArr[0]->width;
	faceImageSize.height = faceImageArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
	for (i = 0; i < nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImageSize, IPL_DEPTH_32F, 1);
	
	// Allocate eigenvalue array
	eigenValMat = cvCreateMat(1, nEigens, CV_32FC1);

	// Allocate the averaged image
	pAvgTrainImage = cvCreateImage(faceImageSize, IPL_DEPTH_32F, 1);

	// Set PCA termination criterion
	calcLimit = cvTermCriteria(CV_TERMCRIT_ITER, nEigens, 1);

	// Compute average image, eigenvalue and eigenvectors
	cvCalcEigenObjects( nTrainFaces,            // No. of source objects
			    (void*) faceImageArr,   // input array
			    (void*) eigenVectArr,   // output array
			    CV_EIGOBJ_NO_CALLBACK,  // flags
			    0,                      // IO buffer size 
			    0,			    // Pointer to DS containing data for callbacks
			    &calcLimit,             // PCA termination criterion
			    pAvgTrainImage,         // Averaged object
			    eigenValMat->data.fl    // Pointer to the data values in eigenValMat
			);
	
	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);

        cvSaveImage("avg_image.jpeg", pAvgTrainImage);
}


int findNearestNeighbour(float* projectedTestFace, float* confidence)
{
	double leastDistSq = DBL_MAX;
	int i, iTrain, iNearest = 0;

	for (iTrain = 0; iTrain < nTrainFaces; iTrain++) {
		double distSq = 0;

		for (i = 0; i < nEigens; i++) {
			float d_i = projectedTestFace[i] - projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
#ifdef USE_MAHALANOBIS_DISTANCE
			distSq += d_i * d_i / eigenValMat->data.fl[i];
#else
			distSq += d_i * d_i;
#endif
		}

		if(distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}
	
	// Return confidence based on Euclidean distance
	*confidence = 1.0f - sqrt( leastDistSq / (float)(nTrainFaces * nEigens) ) / 255.0f;

	return iNearest;
}


int loadTrainingData(CvMat** pTrainPersonNumMat)
{
	CvFileStorage* fileStorage;
	int i;

	fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_READ);
	if(!fileStorage)
	{
		printf("Can't open training database file\n");
		return 0;
	}

	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat*)cvReadByName(fileStorage, 0, "trainPersonNumMat");
	eigenValMat = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat");
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat");
	pAvgTrainImage = (IplImage *)cvReadByName(fileStorage, 0, "pAvgTrainImage");
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces * sizeof(IplImage *));
	for (i = 0; i < nEigens; i++)
	{
		char varname[200];
		sprintf(varname, "eigenVect_%d", i);
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}

	cvReleaseFileStorage(&fileStorage);
	return 1;
}


int loadFaceImageArr(char* filename)
{
#if DEBUG
        printf("Loading face image array...\n");
#endif
	FILE* imgListFile = 0;
	char imgFileName[512];
	int iFace, nFaces = 0;

	// Open input file
	imgListFile = fopen(filename, "r");
        assert(imgListFile != NULL);
#if DEBUG
        printf("Read image file list\n");
#endif

	// Count number of faces
	while( fgets(imgFileName, 512, imgListFile) )
		++nFaces;
	rewind(imgListFile);

#if DEBUG
        printf("Read %d faces\n", nFaces);
#endif

	// Allocate 
	// 1. faceImageArr : face image array, 
	// 2. personNumTruthMat : The "ground truth", i.e. values in this variable are the true values
	//			  for each face image. CV_32SC1 => 32-bit OS, Signed, 1 Channel
	faceImageArr = (IplImage**) cvAlloc(nFaces * sizeof(IplImage*));
	personNumTruthMat = cvCreateMat(1, nFaces, CV_32SC1);

	// Store the face images from disk into faceImageArr
	for (iFace = 0; iFace < nFaces; iFace++) 
	{
		// Read person number and path to image from file
		// data is a union in CvMat, and is accessed as data.i. Add iFace to account for offset.
		fscanf(imgListFile, "%d %s", personNumTruthMat->data.i + iFace, imgFileName);
#if DEBUG
                printf("Read %s\n", imgFileName);
#endif

		// Load face image
		faceImageArr[iFace] = cvLoadImage(imgFileName, CV_LOAD_IMAGE_GRAYSCALE);
	}

	fclose(imgListFile);

	return nFaces;
}

void storeTrainingData()
{
	CvFileStorage* fileStorage;
	int i;

	// Create a file-storage interface
	fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_WRITE);
	
	// Store all data
	cvWriteInt(fileStorage, "nEigens", nEigens);
	cvWriteInt(fileStorage, "nTrainFaces", nTrainFaces);
	cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
	cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
	cvWrite(fileStorage, "avgTrainImage", pAvgTrainImage, cvAttrList(0,0));

	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		snprintf( varname, sizeof(varname)-1, "eigenVect_%d", i );
		cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
	}

	// Release the file-storage interface
	cvReleaseFileStorage( &fileStorage );
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
		      int newHeight = faceHeight, int newWidth = faceWidth)
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


// Return a CvRect bounding detected faces in the gray-scale converted image
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
	// if(image->nChannels > 1)
	// {
	// 	size = cvSize(image->width, image->height);
	// 	grayImage = cvCreateImage(size, IPL_DEPTH_8U, 1);
	// 	cvCvtColor(image, grayImage, CV_BGR2GRAY);
	// 	detectImage = grayImage;
	// }

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


void drawBox(IplImage* image, CvRect rect)
{
	static CvScalar white = {255, 255, 255};
	cvRectangle(image,
		    cvPoint(rect.x,rect.y),
		    cvPoint(rect.x + rect.width, rect.y + rect.height),
		    colourMap["white"]);
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
	IplImage* frame;
	IplImage *faceImage = 0;
	IplImage *resizedImage;
	IplImage *equalizedImage;
	CvMat* trainPersonNumMat;
	// char faceCascadeFileName[] = "haarcascade_frontalface_default.xml";
	CvHaarClassifierCascade* faceCascade;
	CvCapture* capture;

	int delay = 33;
	bool runFlag = true;

	init();
	cvNamedWindow("CA", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("test", CV_WINDOW_AUTOSIZE);

	bool collectFlag = false;
	int collectCount = 50;
	int count = 0;
        unsigned int i;
	string prefix;
	string extension = ".jpeg";
	string filename;
	stringstream sstm;

	// switch(argc)
	// {
	// 	case 2: cout<<argv[1];
	// 		if(argv[1] == "train")
	// 		{
	// 			capture = cvCreateFileCapture( argv[1] );
	// 			cout<<"----- Data collection mode -----"<<endl;
	// 			collectFlag = true;
	// 		}
	// 		break;

	// 	default: capture = cvCreateCameraCapture( -1 );
	// 		 break;
	
        for(i = 1; i < argc; i++)
	{
		string cmd = argv[i];	
		if( (cmd == "--collect") || (cmd == "-c") )
		{
                    cout<<"----- Data collection mode -----"<<endl;
                    collectFlag = true;
                    delay = 500;
                    cout<<"Enter prefix: ";
                    cin>>prefix;
		}
                else if(( cmd == "--learn") || (cmd == "-l") )
                {
                    printf("Now training...\n");
                    learn();
                    printf("Training completed.\n");
                    return 0;
                }
		else if( (cmd == "--help") || (cmd == "-h") || (cmd == "-?") )
		{
                    cout<<"Usage:"<<endl;
                    cout<<"face_rec [--collect | -c] || [--help | -h | -?]";
		}
	}

	capture = cvCreateCameraCapture(-1);
	assert( capture != NULL );

#if DEBUG
	cout<<"Cam dimensions: "<<cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH)<<" "<<cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT)<<endl;
	//cout<<"FPS: "<<cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);

	// cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 500);
	// cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 500);

	cout<<"Cam dimensions now set to: "<<cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH)<<" "<<cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT)<<endl;
	//cout<<"FPS: "<<cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
#endif

	// Choose from default, alt, alt2 or alt_tree
	faceCascade = (CvHaarClassifierCascade*)cvLoad(cascadeFileMap["default"].c_str(), 0, 0, 0);
	assert(faceCascade != NULL);

	while(runFlag)
	{
		frame = cvQueryFrame(capture);
		//frame = convertImageToGrayscale(frame);
		if( !frame )
			break;

		CvRect faceRect = detectFace(frame, faceCascade);
		drawBox(frame, faceRect);

#if DEBUG
		cout<<"Size of face: "<<faceRect.width<<" x "<<faceRect.height<<endl;
#endif

		cvShowImage("CA", frame);

		if(faceRect.width > 0)
		{
			// 1. Get image content from faceRect
			faceImage = cropImage(frame, faceRect);
			// 2. Resize image to 180x180 pixels
			resizedImage = resizeImage(faceImage, faceWidth, faceHeight);
                        // 3. Convert to grayscale and equalize image
                        equalizedImage = cvCreateImage(cvGetSize(resizedImage), 8, 1);
			cvEqualizeHist(convertImageToGrayscale(resizedImage), equalizedImage);
			
			cvShowImage("test", equalizedImage);

			if(collectFlag)
			{
				//sstm<<"./data/"<<prefix<<"-"<<++count<<extension;
				//filename = sstm.str();

				char numstr[20];
				sprintf(numstr, "%d", ++count);
				filename = prefix + "-" + numstr + extension;
				cout<<"Saving "<<filename<<endl;
				cvSaveImage(filename.c_str(), equalizedImage);
				if(count > collectCount)
                                {
                                    // Stop collecting training data. Use data to train/retrain images.
                                    runFlag = false;
                                    printf("Training...\n");
                                    learn();
                                }
			}
			else
			{
				if (nEigens > 0)
				{
					int iNearest, nearest, truth;
					float *projectedTestFace = 0, confidence;

					cvEigenDecomposite( equalizedImage, // Input object
							    nEigens,         // no. of eigenvalues
							    eigenVectArr,    // Pointer to array of IplImage input objects
							    0, 0,            // ioFlags and userData
							    pAvgTrainImage,  // Averaged object
							    projectedTestFace// Output - calculated coefficients
							  );
					iNearest = findNearestNeighbour(projectedTestFace, &confidence);
					nearest = trainPersonNumMat->data.i[iNearest];

					printf("Nearest = %d, Confidence = %f\n", nearest, confidence);
				}
			}
		}
		
		char c = cvWaitKey(delay);
		if( c == 27 )
			break;

		cvReleaseImage(&faceImage);
		cvReleaseImage(&resizedImage);
		cvReleaseImage(&equalizedImage);
	}


	cvReleaseHaarClassifierCascade(&faceCascade);
	cvReleaseCapture(&capture);
	cvDestroyWindow("CA");
	cvDestroyWindow("test");

	return 0;
}
