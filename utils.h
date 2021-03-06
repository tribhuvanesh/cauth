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

#define vi vector<int>
#define v2di vector< vector<int> >
 
#define fR(i,L,U) for( int i=L; i<U; ++i )
#define f0R(i,U) for( int i=0; i<U; ++i)
 
#define ull unsigned long long int
#define ll long long int

#define FILEOP 0

using namespace std;

#ifndef CONVERT_TO_GRAYSCALE
#define CONVERT_TO_GRAYSCALE
    IplImage* convertImageToGrayscale(IplImage*);
#endif

#ifndef CROP_IMAGE
#define CROP_IMAGE
    IplImage* cropImage(IplImage*, CvRect);
#endif

#ifndef RESIZE_IMAGE
#define RESIZE_IMAGE
    IplImage* resizeImage(IplImage*, bool, int, int);
#endif

#ifndef DRAW_BOX
#define DRAW_BOX 
    void drawBox(IplImage*, CvRect, CvScalar);
#endif
