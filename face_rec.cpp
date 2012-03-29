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
#include "detect.h"
#include "utils.h"

#define vi vector<int>
#define v2di vector< vector<int> >
 
#define fR(i,L,U) for( int i=L; i<U; ++i )
#define f0R(i,U) for( int i=0; i<U; ++i)
 
#define ull unsigned long long int
#define ll long long int

#define DEBUG 1
#define FILEOP 0
#define STORE_EIGEN 1

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


// Function prototypes
void      init(void);
void      drawBox(IplImage* image, CvRect rect);
void      storeEigenfaceImages();

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
	doPCA();
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
                printf("Read %s\n", imgFileName);
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
	for (i=1; i<nPersons; i++) {
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
	IplImage *processedFaceImage;
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
	int collectCount = 10;
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

        if(argc == 1)
        {
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
        }
	else
        {
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
                                    // TODO Insert reorganize.py here
				    if( system("python2 reorganize.py") == -1 )
				    {
					    printf("Failed to reorganize files. Now exiting...\n");
					    exit(1);
				    }
				    printf("Reorganized files!\n");
                                    learn();
                                }
			}
			else
			{
                                printf("Recognizing! %d\n", nEigens);
			        cvShowImage("test", equalizedImage);
                                processedFaceImage = equalizedImage;
				 // /*
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

					printf("Nearest = %d, Person = %s, Confidence = %f\n", nearest, personNames[nearest-1].c_str(), confidence);
				}
				// */
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

	return 0;
}
