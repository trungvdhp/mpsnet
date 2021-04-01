//
//  pvlib.h
//  BioID
//
//  Created by trung on 6/5/20.
//  Copyright © 2020 DeepSky. All rights reserved.
//

#ifndef palmlib_h
#define palmlib_h
#endif /* palmlib_h */

#import <stdlib.h>
#import <algorithm>
#import <string>
#import <time.h>
#import <opencv2/core.hpp>
#import <opencv2/opencv.hpp>

#import <android/log.h>
#define  LOG_TAG    "LOG_JNI"
#define  ALOG(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

using namespace cv;
using namespace std;

const double M2PI = 2*M_PI;
const int binaryMaskSize = 160;
const int blockSize=27;
const int constant=11;
const Scalar greenColor=Scalar(0,255,0);
const Scalar whiteColor=Scalar(255,255,255);
const Scalar blackColor=Scalar(0,0,0);

// Structs
typedef struct Landmarks{
    Point2i topLeft=Point2i(INT_MAX, INT_MAX);
    Point2i topRight=Point2i(INT_MAX, INT_MAX);
    Point2i bottomLeft=Point2i(INT_MAX, INT_MAX);
    Point2i bottomRight=Point2i(INT_MAX, INT_MAX);
    int edgeSize=INT_MAX;

    void init(Point2i top_left, Point2i top_right, Point2i bottom_left, Point2i bottom_right){
        topLeft=top_left;
        topRight=top_right;
        bottomLeft=bottom_left;
        bottomRight=bottom_right;
        int dx=(topRight.x-topLeft.x);
        int dy=(topRight.y-topLeft.y);
        edgeSize = int(sqrt(dx*dx+dy*dy));

        //("TopLeft(%d, %d), TopRight(%d, %d), edgeSize: %d\n", topLeft.x, topLeft.y,
               //topRight.x, topRight.y, edgeSize);
        //printf("------------------------------------------------------------\n");
    };

    void translate(int biasX, int biasY) {
        topLeft = Point2i(topLeft.x+biasX, topLeft.y+biasY);
        topRight = Point2i(topRight.x+biasX, topRight.y+biasY);
        bottomLeft = Point2i(bottomLeft.x+biasX, bottomLeft.y+biasY);
        bottomRight = Point2i(bottomRight.x+biasX, bottomRight.y+biasY);
    }

    Point2f * inputQuad(){
        Point2f *input4 = new Point2f[4];

        input4[0] = Point2f(topLeft);
        input4[1] = Point2f(topRight);
        input4[2] = Point2f(bottomLeft);
        input4[3] = Point2f(bottomRight);

        return input4;
    };

    static Point2f * outputQuad(int roiSize=128){
        Point2f *output4 = new Point2f[4];
        output4[0] = Point2f(0.0f,0.0f);
        output4[1] = Point2f(roiSize,0);
        output4[2] = Point2f(0,roiSize);
        output4[3] = Point2f(roiSize,roiSize);
        return output4;
    };

} Landmarks;

typedef struct ConvexHull{
    double area;
    int begin;
    int mid;
    int end;
    int dist;
    double angle;
} ConvexHull;

typedef struct Valley{
    int cid;
    Point2i center;
    Point2i ref;
    ConvexHull hull;
} Valley;

typedef struct ValleyEdge{
    pair<int, int> edge1;
    pair<int, int> edge2;
} ValleyEdge;

typedef struct LineCoefficient{
    int a;
    int b;
    int c;
    int d;
    int d2;
} LineCoefficient;

typedef struct Intersection{
    Point2i p = Point2i(INT_MAX, INT_MAX);
    int d1=INT_MAX;
    int d2=INT_MAX;
    int d11=INT_MAX;
    int d12=INT_MAX;
    int d21=INT_MAX;
    int d22=INT_MAX;
} Intersection;

// Inline Template Functions
template<typename T>
inline vector<T> subVector(const vector<T> &v, int size, int first, int last) {
    vector<T> res;

    try {

        if(first>last) swap(first, last);

        if(last>=size) last=(last+1)%size;

        if (first>=0 && first<last)
            return vector<T>(v.begin() + first, v.begin() + last);

        if (first<0) first+=size;
        if (last<0) last+=size;

        //vector<T> sv1(v.begin() + first, v.begin() + size - 1);
        //vector<T> sv2(v.begin(), v.begin() + last);
        //move(sv2.begin(), sv2.end(), back_inserter(sv1));
        copy(v.begin()+first, v.begin()+size, back_inserter(res));
        copy(v.begin(), v.begin()+last, back_inserter(res));
    } catch(Exception e){
        //printf("ERR: ", e.msg.c_str());
    }

    return res;
}

template<typename T>
inline T clip( const T &a, const T &b, const T &c){
    return min(max(a, b), c);
}

template<typename T>
inline int sign(const T &a){
    return a<0?-1:a>0;
}

template<typename T>
inline T valueAt(const vector<T> &v, int size, int index) {
    if(index<0) index+=size;

    return v[index%size];
}

// Inline Functions
inline double toDegree(double rad) {
    return rad*180.0/M_PI;
}

inline int cross(const Point2i &p1, const Point2i &p2) {
    return p1.x*p2.y - p1.y*p2.x;
}

inline int cross4(int a1, int b1, int a2, int b2){
    return a1*b2 - a2*b1;
}

inline int magnitude(const Point2i &p){
    return int(sqrt(p.x*p.x + p.y*p.y));
}

inline int magnitude2(int x, int y) {
    return x*x + y*y;
}

inline int midValue(int x, int y){
    return (x + y)/2;
}

inline Point2i midpoint(const Point2i &p1, const Point2i &p2){
    return Point2i((p1.x+p2.x)/2, (p1.y+p2.y)/2);
}

inline Point2i sub(const Point2i &p1, const Point2i &p2){
    return Point2i(p2.x-p1.x,p2.y-p1.y);
}

inline bool outBound(const Point2i &p, int xmin, int xmax, int ymin, int ymax){
    //printf("(%d,%d): X(%d->%d) Y(%d->%d)",p.x,p.y,xmin,xmax,ymin,ymax);
    return p.x<xmin || p.x>xmax || p.y<ymin || p.y>ymax;
}

inline Point2i scalePoint(const Point2i &p, double scaleFactor){
    //printf("Point(%d, %d): scaleFactor=%.3f, biasX=%d, //biasY=%d\n",p.x,p.y,scaleFactor,biasX,biasY);
    return Point2i(int(p.x*scaleFactor),int(p.y*scaleFactor));
}

inline int distance2Points(const Point2i &p1, const Point2i &p2){
    Point2i p = sub(p1, p2);
    return magnitude(p);
}

inline int midByRate(
        const int size,
        const int a1,
        const int a2
){

    if(a1+a2==0) return -1;

    return min(size*a1/(a1+a2),size-1);
}

inline double angle3pt(
        const Point2i &p1,
        const Point2i &p2,
        const Point2i &p3){
    double a = atan2(p3.y-p2.y, p3.x-p2.x) - atan2(p1.y-p2.y, p1.x-p2.x);

    if(a<0) a += M2PI;

    if(a>M_PI) a = M2PI - a;

    return toDegree(a);
}

inline double angle2VectWithDirection(
        const Point2i &p1,
        const Point2i &p2
){

    double a = atan2(-p2.y, p2.x) - atan2(-p1.y, p1.x);

    if(a>M_PI) a -= M2PI;
    if(a<-M_PI) a += M2PI;

    return toDegree(a);
}

inline LineCoefficient getLineCoefficient(
        const Point2i &p1,
        const Point2i &p2
){
    auto p = sub(p1, p2);
    LineCoefficient coef;
    coef.a = p.y;
    coef.b = -p.x;
    coef.c = cross(p2, p1);
    coef.d2 = magnitude2(p.x, p.y);
    coef.d = int(sqrt(coef.d2));

    return coef;
}

inline int pointLineScore(
        const Point2i &p,
        const LineCoefficient &coef
){
    return coef.a*p.x + coef.b*p.y + coef.c;
}

inline int pointLineDistance(
        const Point2i &p,
        const LineCoefficient &coef){
    return abs(pointLineScore(p, coef))/coef.d;
}

// Other functions
Intersection intersection2Vectors(
        const Point2i &t1,
        const Point2i &h1,
        const Point2i &t2,
        const Point2i &h2 );

double calculateContourAngle(
        const vector<Point2i> & contour,
        int contourSize,
        int index,
        int step= 5,
        double maxAngle= 170.0 );

vector<Point2i> samplingLine(
        const Point2i &p1,
        const Point2i &p2,
        int xMax=INT_MAX,
        int yMax=INT_MAX);

pair<int, int> countBlackWhite(
        const Mat &img,
        const Point2i &p1,
        const Point2i &p2 );

vector<Point2i> bisector(
        const Point2i& t1,
        const Point2i& h1,
        const Point2i& t2,
        const Point2i& h2,
        bool &direction);

vector<ConvexHull> correctHull(
        const Mat &img,
        const vector<int> &cvHull,
        const vector<Point2i> &contour,
        vector<double> &angles,
        size_t t1=2,
        size_t t2=60,
        size_t t3=6,
        double t4=100.0,
        double bias=40.0,
        double maxAngle=170.0 );

bool isValidValleyEdge(
        const Point2i &edgeBegin,
        const Point2i &edgeEnd,
        const Point2i &hullBegin,
        const Point2i &hullEnd,
        const double distanceThreshold);

ValleyEdge findValleyEdges(
        const vector<Point2i> &contour,
        const vector<double> &angles,
        const ConvexHull &hull,
        int distanceThreshold,
        double angleThreshold );

int lineContourIntersectionId(
        const vector<Point2i>&contour,
        const LineCoefficient &coef,
        const Point2i &ref,
        int contourSize,
        int beginId,
        int endId,
        int step,
        bool flag = true );

pair<Point2i,Point2i> findReferenceLine(
        const vector<Point2i> &contour,
        const LineCoefficient &coef,
        const Point2i &ref,
        const Valley &v1,
        const Valley &v2 );

Landmarks getROICoordinates(
        const Mat &binaryImage,
        const vector<Point2i> &contour,
        const Valley &v1,
        const Valley &v2,
        const Point2i &ref,
        int minDistance=24,
        int maxDistance=72);

vector< vector<int> > getValleyTriplets(
        const vector<Valley> &valleys );

vector< pair<int,int> > selectTwoOptimalKeyVectors(
        const vector<Valley> &valleys,
        int angle,
        int angleThreshold=130 );

Landmarks selectROICandidate(
        const Mat &binaryImage,
        const vector<Point2i> &contour,
        const vector<Valley> &valleys,
        int minDistance,
        int maxDistance,
        int angleThreshold=130 );

void printValleys(const vector<Valley> &valleys, const string &title);

void correctGamma(Mat &src, double gamma, bool isAutoMode=true);

void enhanceVein(Mat &src);

void enhanceVein(Mat &src, int featureDensity);

double getFocusScore(const Mat &img);

void normalizeGrayImage(Mat &grayImage);

void prepareInputForPLDNet(Mat &frame, Mat &dst, int input_size, int code2Gray=COLOR_RGBA2GRAY);

void extractPalmVeinROI(Mat &frame, Mat &palmVeinROI, const float *landmarks, int roi_size, int preference_code, int code2RGB=COLOR_RGBA2RGB);

bool extractPalmVeinROI(Mat &frame, Mat &palmVeinImage, int roi_size, int preference_code, int code2RGB=COLOR_RGBA2RGB);