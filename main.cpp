/*

Basic functions used by recognition and detection 
utils.h:

Calls to detect face in a given frame
detect.h:

*/
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <map>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <climits>
#include <cctype>
#include <cmath>
#include <cassert>
#include <ctime>

// To obtain password from the user
#include <unistd.h>

// For SHA1 encoding of password before storing it in the xml file
#include<cryptopp/sha.h>
#include<cryptopp/filters.h>
#include<cryptopp/hex.h>

// OpenCV libraries
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>

// Other independent modules
#include "detect.h"
#include "utils.h"
#include "soft.h"

#define vi vector<int>
#define v2di vector< vector<int> >
 
#define fR(i,L,U) for( int i=L; i<U; ++i )
#define f0R(i,U) for( int i=0; i<U; ++i)
 
#define ull unsigned long long int
#define ll long long int

#define DEBUG 1
#define FILEOP 0
#define STORE_EIGEN 1
#define COLLECT_COUNT 50
#define COUNT_FREQ 5
#define EXTENSION ".jpeg"

using namespace std;


// Global variables
map<string, CvScalar> colourMap;
map<string, string>   cascadeFileMap;
int                   faceHeight	    = 180;
int                   faceWidth		    = 180;
int                   nTrainFaces	    = 0; // No. of training images
int                   nEigens		    = 0; // No. of eigenvalues
int		      nPersons		    = 0; // No. of people in the training set
IplImage**	      faceImageArr          = 0; // Array of face images
CvMat*		      personNumTruthMat     = 0; // Array of person numbers
IplImage*	      pAvgTrainImage        = 0; // Average image
IplImage**	      eigenVectArr          = 0; // eigenvectors
CvMat*		      eigenValMat	    = 0; // eigenvalues
CvMat*		      projectedTrainFaceMat = 0; // Projected training faces
vector<string>        personNames;

// If a new account has been created, store the following
// bool		      createAccount = false;
// char		      uname[200];
// char		      upwd[200];

char prompt[]                               = "Enter password: ";
char promptAgain[]                          = "Re-enter password: ";

// Initialize variables for soft biometric tests
bool run_shirt=true;
// Various colour types for detected shirt colours.
//enum                             {cBLACK=0,cWHITE, cGREY, cRED, cORANGE, cYELLOW, cGREEN, cAQUA, cBLUE, cPURPLE, cPINK,  NUM_COLOUR_TYPES};
char sCTypes[][NUM_COLOUR_TYPES] = {"Black", "White","Grey","Red","Orange","Yellow","Green","Aqua","Blue","Purple","Pink"};
uchar cCTHue[NUM_COLOUR_TYPES] =    {0,       0,      0,     0,     20,      30,      55,    85,   115,    138,     161};
uchar cCTSat[NUM_COLOUR_TYPES] =    {0,       0,      0,    255,   255,     255,     255,   255,   255,    255,     255};
uchar cCTVal[NUM_COLOUR_TYPES] =    {0,      255,    120,   255,   255,     255,     255,   255,   255,    255,     255};
string colour_types[] = {"Black", "White","Grey","Red","Orange","Yellow","Green","Aqua","Blue","Purple","Pink"};
float range_norm = sqrt((2*100*100) / NUM_COLOUR_TYPES);
//const char* cascadeFileFace = "haar/haarcascade_frontalface_alt.xml";	// Path to the Face Detection HaarCascade XML file

// Function prototypes
void      init(void);
void      storeEigenfaceImages();

void learn();
void doPCA();
void storeTrainingData();
int loadTrainingData(CvMat** pTrainPersonNumMat);
int findNearestNeighbour(float* projectedTestFace, float* confidence);
int loadFaceImageArr(char* filename);

void recognizeFromCam(string user);
void collect();

// Initializes constants and static data
void init()
{
	cascadeFileMap["default"] = "haar/haarcascade_frontalface_default.xml";
	cascadeFileMap["alt"] = "haar/haarcascade_frontalface_alt.xml";
	cascadeFileMap["alt2"] = "haar/haarcascade_frontalface_alt2.xml";
	cascadeFileMap["alt_tree"] = "haar/haarcascade_frontalface_alt_tree.xml";

	colourMap["white"] = cvScalar(255, 255, 255);
	colourMap["red"]   = cvScalar(255, 0, 0);
	colourMap["green"] = cvScalar(0, 255, 0);
	colourMap["blue"]  = cvScalar(0, 0, 255);
}

// Get an 8-bit equivalent of the 32-bit Float image.
// Returns a new image, so remember to call 'cvReleaseImage()' on the result.
IplImage* convertFloatImageToUcharImage(const IplImage *srcImg)
{
	IplImage *dstImg = 0;
	if ((srcImg) && (srcImg->width > 0 && srcImg->height > 0)) {

		// Spread the 32bit floating point pixels to fit within 8bit pixel range.
		double minVal, maxVal;
		cvMinMaxLoc(srcImg, &minVal, &maxVal);

		//cout << "FloatImage:(minV=" << minVal << ", maxV=" << maxVal << ")." << endl;

		// Deal with NaN and extreme values, since the DFT seems to give some NaN results.
		if (cvIsNaN(minVal) || minVal < -1e30)
			minVal = -1e30;
		if (cvIsNaN(maxVal) || maxVal > 1e30)
			maxVal = 1e30;
		if (maxVal-minVal == 0.0f)
			maxVal = minVal + 0.001;	// remove potential divide by zero errors.

		// Convert the format
		dstImg = cvCreateImage(cvSize(srcImg->width, srcImg->height), 8, 1);
		cvConvertScale(srcImg, dstImg, 255.0 / (maxVal - minVal), - minVal * 255.0 / (maxVal-minVal));
	}
	return dstImg;
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
	printf("Starting PCAin learn()\n");
        printf("Completed PCA\n");

	// Project the training images on to the PCA subspace
        projectedTrainFaceMat = cvCreateMat(nTrainFaces, nEigens, CV_32FC1);
	for (i = 0; i < nTrainFaces; i++) 
	{
                // printf("Projecting face %d\n", i);
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

	// printf("Started PCA... %d\n", nTrainFaces);
	printf("Started PCA...\n");

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

int loadTrainingData(CvMat **pTrainPersonNumMat)
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interface
	printf("Loading facedata.xml!\n");
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
	if( !fileStorage ) {
		printf("Can't open training database file 'facedata.xml'.\n");
		return 0;
	}

	personNames.clear();
	nPersons = cvReadIntByName(fileStorage, 0, "nPersons", 0);
	if(nPersons == 0)
	{
		printf("Database is empty. \n");
		return 0;
	}

	cout<<"nPersons: "<<nPersons<<endl;
	for (i = 0; i < nPersons; i++)
	{
		string sPersonName;
		char varname[200];
		snprintf(varname, sizeof(varname)-1, "personName_%d", (i+1));
		cout<<"Reading string "<<varname<<endl;
		sPersonName = cvReadStringByName(fileStorage, 0, varname);
		cout<<"Loaded person: "<<sPersonName<<endl;
		personNames.push_back(sPersonName);
	}
	cout<<"Loaded nPersons"<<endl;

	// Load the data
	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImage = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImage", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		snprintf( varname, sizeof(varname)-1, "eigenVect_%d", i );
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}
	
	// release the file-storage interface
	cvReleaseFileStorage( &fileStorage );

	printf("Training data loaded of %d people.\n", nPersons);

}


int loadFaceImageArr(char* filename)
{
#if DEBUG
        printf("Loading face image array...\n");
#endif
	FILE* imgListFile = 0;
	char imgFileName[512];
	int iFace, nFaces = 0;
	int i;

	// Open input file
	imgListFile = fopen(filename, "r");
        assert(imgListFile != NULL);
#if DEBUG
        printf("Read image file list\n");
#endif

	// Count number of faces
	while( fgets(imgFileName, sizeof(imgFileName)-1, imgListFile) )
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
		char personName[256];
		string sPersonName;
		int personNumber;

		// Read person number and path to image from file
		// data is a union in CvMat, and is accessed as data.i. Add iFace to account for offset.
		fscanf(imgListFile, "%d %s %s", &personNumber, personName, imgFileName);
		sPersonName = personName;
#if DEBUG
                //printf("Read %s\n", imgFileName);
#endif
		if(personNumber > nPersons)
		{
			for (i = nPersons; i < personNumber; i++)
			{
				personNames.push_back(sPersonName);
			}
			nPersons = personNumber;
		}

		personNumTruthMat->data.i[iFace] = personNumber;

		// Load face image
		faceImageArr[iFace] = cvLoadImage(imgFileName, CV_LOAD_IMAGE_GRAYSCALE);
	}

	fclose(imgListFile);

	printf("Data loaded from '%s': (%d images of %d people).\n", filename, nFaces, nPersons);
	printf("People: ");
	if (nPersons > 0)
		printf("<%s>", personNames[0].c_str());
	for (i=1; i<nPersons; i++)
	{
		printf(", <%s>", personNames[i].c_str());
	}
	printf(".\n");

	return nFaces;
}

void storeTrainingData()
{
	CvFileStorage* fileStorage;
	int i;

	// Create a file-storage interface
	fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_WRITE);

	cvWriteInt(fileStorage, "nPersons", nPersons);
	printf("Stored data for %d persons.\n", nPersons);
	for (i = 0; i < nPersons; i++)
	{
		char varname[200];
		snprintf( varname, sizeof(varname)-1, "personName_%d", (i+1) );
		cvWriteString(fileStorage, varname, personNames[i].c_str(), 0);
	}
	
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
                char fname[200];
                strcpy(fname, varname);
                strcat(fname, ".jpeg");
                // cout<<"Storing "<<fname<<endl;
	        cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
                // cvSave(fname, convertFloatImageToUcharImage(eigenVectArr[i]));
	}

	if(STORE_EIGEN)
	{
            cvSaveImage("avg_image.jpeg", pAvgTrainImage);
	    storeEigenfaceImages();
	}

	// Release the file-storage interface
	cvReleaseFileStorage( &fileStorage );
}

// Save all the eigenvectors as images, so that they can be checked.
void storeEigenfaceImages()
{
	// Store the average image to a file
	printf("Saving the image of the average face as 'out_averageImage.bmp'.\n");
	cvSaveImage("out_averageImage.bmp", pAvgTrainImage);
	// Create a large image made of many eigenface images.
	// Must also convert each eigenface image to a normal 8-bit UCHAR image instead of a 32-bit float image.
	printf("Saving the %d eigenvector images as 'out_eigenfaces.bmp'\n", nEigens);
	if (nEigens > 0) {
		// Put all the eigenfaces next to each other.
		int COLUMNS = 8;	// Put upto 8 images on a row.
		int nCols = min(nEigens, COLUMNS);
		int nRows = 1 + (nEigens / COLUMNS);	// Put the rest on new rows.
		int w = eigenVectArr[0]->width;
		int h = eigenVectArr[0]->height;
		CvSize size;
		size = cvSize(nCols * w, nRows * h);
		IplImage *bigImg = cvCreateImage(size, IPL_DEPTH_8U, 1);	// 8-bit Greyscale UCHAR image
		for (int i=0; i<nEigens; i++) {
			// Get the eigenface image.
			IplImage *byteImg = convertFloatImageToUcharImage(eigenVectArr[i]);
			// Paste it into the correct position.
			int x = w * (i % COLUMNS);
			int y = h * (i / COLUMNS);
			CvRect ROI = cvRect(x, y, w, h);
			cvSetImageROI(bigImg, ROI);
			cvCopyImage(byteImg, bigImg);
			cvResetImageROI(bigImg);
			cvReleaseImage(&byteImg);
		}
		cvSaveImage("out_eigenfaces.bmp", bigImg);
		cvReleaseImage(&bigImg);
	}
}

int getID(string user)
{
	int i;
	for (i = 0; i < personNames.size(); i++)
	{
		if(personNames[i] == user)
			return i;
	}
}


void recognizeFromCam(string user)
{
	IplImage* frame;
	IplImage *faceImage = 0;
	IplImage *resizedImage;
	IplImage *equalizedImage;
	IplImage *processedFaceImage;
	CvMat* trainPersonNumMat;
	CvHaarClassifierCascade* faceCascade;
	CvCapture* capture;

	int delay = 33;
	bool runFlag = true;

	cvNamedWindow("CA", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("test", CV_WINDOW_AUTOSIZE);

	bool collectFlag = false;
	int count = 0;
        unsigned int i;
	string prefix;
	string extension = EXTENSION;
	string filename;
	stringstream sstm;

	// Add 1, since indexing in vector starts from 0, and UIDs start from 1
	int uid = getID(user) + 1;

	// Estimates
	float mu = 0;
	float sig = 10000;
	// Error in estimation
	float r_mu;
	float r_sig = 2;

    // No extra arguments. Load training data and start recognition phase.
    if( loadTrainingData(&trainPersonNumMat) )
    {
	faceWidth = pAvgTrainImage->width;
	faceHeight = pAvgTrainImage->height;
	printf("Loaded training data successfully\n");
    }
    else
    {
	printf("Unable to load training data. Aborting\n");
	exit(0);
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
		drawBox(frame, faceRect, colourMap["white"]);

#if DEBUG
		cout<<"Size of face: "<<faceRect.width<<" x "<<faceRect.height<<endl;
#endif

		cvShowImage("CA", frame);
                
                printf("Dimentsions of faceRect = %d x %d\n", faceRect.width, faceRect.height);

		if(faceRect.width > 0)
		{
			// 1. Get image content from faceRect
			faceImage = cropImage(frame, faceRect);
			// 2. Resize image to 180x180 pixels
			resizedImage = resizeImage(faceImage, true, faceWidth, faceHeight);
                        // 3. Convert to grayscale and equalize image
                        equalizedImage = cvCreateImage(cvGetSize(resizedImage), 8, 1);
			cvEqualizeHist(convertImageToGrayscale(resizedImage), equalizedImage);

			printf("Recognizing! %d\n", nEigens);
			cvShowImage("test", equalizedImage);
			processedFaceImage = equalizedImage;

			if (nEigens > 0)
			{
				int iNearest, nearest, truth;
				float *projectedTestFace = 0, confidence;

				cvFree(&projectedTestFace);
				projectedTestFace = (float *)cvAlloc(nEigens*sizeof(float));

				printf("Projecting! \n");
				cvEigenDecomposite( processedFaceImage,   // Input object
						    nEigens,          // no. of eigenvalues
						    eigenVectArr,     // Pointer to array of IplImage input objects
						    0, 0,             // ioFlags and userData
						    pAvgTrainImage,   // Averaged object
						    projectedTestFace // Output - calculated coefficients
						  );
				printf("Done projecting! \n");
				iNearest = findNearestNeighbour(projectedTestFace, &confidence);
				nearest = trainPersonNumMat->data.i[iNearest];

				r_mu = nearest;

				mu = ((r_mu * sig) + (mu * r_sig)) / (sig + r_sig);
				sig = (sig * r_sig) / (sig + r_sig);

				cout<<"UID: "<<uid<<endl;
				printf("Nearest = %d, Person = %s, Confidence = %f\n", nearest, personNames[nearest-1].c_str(), confidence);
				printf("Mu = %f\tSigma = %f\n", mu, sig);

				double areaUnderCurve = 0.00;


			}
		}
		else
			continue;
		
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

}

void collect(string prefix, int collectCount)
{
	int delay = 33;
	bool runFlag = true;

	IplImage* frame;
	IplImage *faceImage = 0;
	IplImage *resizedImage;
	IplImage *equalizedImage;
	IplImage *processedFaceImage;
	CvCapture* capture;
	CvHaarClassifierCascade* faceCascade;

	cvNamedWindow("CA", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("test", CV_WINDOW_AUTOSIZE);

        unsigned int i, run = 0, collected = 0;
	string extension = ".jpeg";
	string filename;
	stringstream sstm;

	// Choose from default, alt, alt2 or alt_tree
	faceCascade = (CvHaarClassifierCascade*)cvLoad(cascadeFileMap["default"].c_str(), 0, 0, 0);
	assert(faceCascade != NULL);

	cvNamedWindow("CA", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("test", CV_WINDOW_AUTOSIZE);

	capture = cvCreateCameraCapture(-1);
	assert( capture != NULL );

	while(collected < collectCount)
	{
		frame = cvQueryFrame(capture);
		if( !frame )
			break;

		CvRect faceRect = detectFace(frame, faceCascade);
		drawBox(frame, faceRect, colourMap["white"]);

		// Collect 1 in 5 frames. 
		run += 1;
		if((run >= COUNT_FREQ) && (faceRect.width > 0))
		{
			// 1. Crop image
			faceImage = cropImage(frame, faceRect);
			// 2. Resize image to 180x180 pixels
			resizedImage = resizeImage(faceImage, true, faceWidth, faceHeight);
			// 3. Convert to grayscale and equalize image
			equalizedImage = cvCreateImage(cvGetSize(resizedImage), 8, 1);
			cvEqualizeHist(convertImageToGrayscale(resizedImage), equalizedImage);

			char numstr[20];
			sprintf(numstr, "%d", ++collected);
			filename = prefix + "-" + numstr + extension;
			cout<<"Saving "<<filename<<endl;
			cvSaveImage(filename.c_str(), equalizedImage);

			cvShowImage("test", equalizedImage);

			// Revert run back to 0. Increment count.
			run = 0;
		}

		cvShowImage("CA", frame);

		char c = cvWaitKey(delay);
		if( c == 27 )
			break;
	}

	// Data collection completed. Reorganize files and learn.
	if( system("python2 reorganize.py") == -1 )
	{
		printf("Failed to reorganize files. Now exiting...\n");
		exit(1);
	}
	printf("Reorganization of files completed.\n");

	cvReleaseImage(&faceImage);
	cvReleaseImage(&resizedImage);
	cvReleaseImage(&equalizedImage);

	cvDestroyWindow("CA");
	cvDestroyWindow("test");
}

string create_hash(string source)
{
	CryptoPP::SHA1 sha1;
	string hash = "";
	CryptoPP::StringSource(source, true, 
				new CryptoPP::HashFilter(sha1, 
				new CryptoPP::HexEncoder(new CryptoPP::StringSink(hash))));
	return hash;
}

string verify_pwd()
{
	int timeout = 3;
	string ipwd, ihash = "", storedHash, prefix;
	CvFileStorage* fileStorage;
	fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_READ);

	// Obtain username and password of the user, and hash it
	cout<<"Enter username: ";
	cin>>prefix;

	cout<<"Starting verification"<<endl;

	// Obtain hash of this user stored during account creation
	char user_[] = "user_";
	char username[200];
	//snprintf(username, sizeof(username)-1, "user_%s", prefix.c_str());
	strcpy(username, user_);
	strcat(username, prefix.c_str());
	cout<<username;
	cout<<"Fetching from db"<<endl;
	storedHash = cvReadStringByName(fileStorage, 0, username);
	cout<<"Read Hash: "<<storedHash<<endl;
	//storedHash = "ABC";
#if DEBUG
	cout<<storedHash<<endl;
#endif
	while((storedHash != ihash) && (timeout--))
	{
		ipwd = getpass(prompt);
		printf("\n");
		ihash = create_hash(ipwd);
	}

	cvReleaseFileStorage(&fileStorage);

	// Compare hashes and return
	 if (storedHash == ihash)
		 return prefix;
	else
		return string("NULL");
}

int main(int argc, char** argv)
{
	init();
	string hash = "";
	string user;

	switch(argc)
	{
		case 1: user = verify_pwd();
			if(user != "NULL")
			{
				recognizeFromCam(user);
				//soft_main();
			}
			else
			{
				cout<<"Password time-out. Now exiting."<<endl;
				return 0;
			}
			break;

		case 2: string cmd = argv[1];
			if (cmd == "--learn")
			{
				printf("Learning phase initiated.\n");
				learn();
				printf("Learning phase completed.\n");
			}
			else if (cmd == "--collect")
			{
				string prefix, pwd1, pwd2;
				string pwd_sha1_str;
				int time_out = 3;

				printf("Enter username: ");
				cin>>prefix;

				pwd1 = getpass(prompt);

				while((pwd1 != pwd2) && (time_out--))
				{
					pwd2 = getpass(promptAgain);
				}

				if(pwd1 == pwd2)
				{
					// unsigned char pwd_sha1[100];
					// unsigned char* pwd_org = new unsigned char[pwd1.size() + 1];
					// copy(pwd1.begin(), pwd1.end(), pwd_org);
					// /* SHA1(const_cast<unsigned char*>(pwd1.c_str()), pwd1.length(), pwd_sha1); */
					// SHA1(pwd_org, pwd1.length(), pwd_sha1);
					// //printf("%u\n%c\n%X\n", pwd_sha1, pwd_sha1, pwd_sha1);
					// // vector<char> pwd_enc;
					// // fR(i, 0, strlen((char*)pwd_sha1)) pwd_enc.push_back(pwd_sha1[i]);
					// fR(i, 0, strlen((char*)pwd_sha1)) printf("%02x", pwd_sha1[i]);
					hash = create_hash(pwd1);
				}
				printf("\n");

				// printf("**Printing names!! 1: \n");
				// fR(i, 0, personNames.size()) cout<<personNames[i]<<"\t";
				//
				collect(prefix, COLLECT_COUNT);
				printf("Learning phase initiated.\n");
				learn();
				printf("Learning phase completed.\n");
			
				// Write username and password to file	
				CvFileStorage* fileStorage;
				fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_APPEND);

				char user_[] = "user_";
				char username[200];
				//snprintf(username, sizeof(username)-1, "user_%s", prefix.c_str());
				strcpy(username, user_);
				strcat(username, prefix.c_str());
				cout<<username<<"\t"<<hash<<endl;
				cvWriteString(fileStorage, username, hash.c_str(), 0);
				cvReleaseFileStorage(&fileStorage);

				// fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_READ);
				// string s = cvReadString(fileStorage, 0, username);
				// cout<<s<<endl;
				// cvReleaseFileStorage(&fileStorage);

				// printf("**Printing names!! 2: \n");
				// fR(i, 0, personNames.size()) cout<<personNames[i]<<"\t";
				

			}
	}
	return 0;
}

