//
//  palmlib.cpp
//
//  Copyright © 2021 RB304-3, NTUST. All rights reserved.
//

#import "palmlib.h"

// Variables
const int binaryMaskSize = 160;
const double rmin = 0.3;
const double rmax = 0.8;
const double marginTop = 0.13;
const double marginLeft = 0.13;
const double whiteArea_threshold = 0.8;
string errorString;

double calculateContourAngle(const vector<Point2i> & contour, int contourSize, int index, int step, double maxAngle) {

    int backward=index-step;
    int forward=(index+step)%contourSize;

    return min(angle3pt(
            valueAt(contour,contourSize,backward),
            valueAt(contour,contourSize,index),
            valueAt(contour,contourSize,forward)), maxAngle);
}

vector<Point2i> samplingLine(const Point2i &p1, const Point2i &p2, int maxX, int maxY) {
    Point2i p = sub(p1, p2);
    vector<Point2i> points;
    int b = cross(p2, p1);
    int t;

    if(p.x == 0 && p.y==0) return points;

    if(abs(p.x) > abs(p.y)){
        int sign_x=sign(p.x);

        for(int x=p1.x+sign_x; x*sign_x<p2.x*sign_x; x+=sign_x){

            t = (p.y*x + b)/p.x;

            if(x>=0 && x<maxX && t>=0 && t<maxY){
                points.emplace_back(Point2i(x,t));
            }
        }
    }else{
        int sign_y=sign(p.y);

        for(int y=p1.y+sign_y; y*sign_y<p2.y*sign_y; y+=sign_y){
            t = (p.x*y - b)/p.y;

            if(t>=0 && t<maxX && y>=0 && y<maxY){
                points.emplace_back(Point2i(t,y));
            }
        }
    }

    if(points.size()>3){
        points.erase(points.begin(), points.begin()+1);
    }

    return points;
}

pair<int, int> countBlackWhite(const Mat &img, const Point2i &p1, const Point2i &p2) {

    auto points = samplingLine(p1, p2);
    int black=0;
    int white=0;

    for(auto p: points){
        img.at<uchar>(p.y, p.x) == 0 ? black+=1: white+=1;
    }

    return make_pair(black, white);
}

Intersection intersection2Vectors(const Point2i &t1, const Point2i &h1, const Point2i &t2, const Point2i &h2
){
    auto f1 = getLineCoefficient(t1, h1);
    auto f2 = getLineCoefficient(t2, h2);
    auto d11 = pointLineDistance(t1,f2);
    auto d12 = pointLineDistance(h1,f2);
    auto d21 = pointLineDistance(t2,f1);
    auto d22 = pointLineDistance(h2,f1);
    Intersection inter = Intersection();

    if( (d11<2 && d12<2) || (d21<2 && d22<2) )
        return inter;

    f1.c = -f1.c;
    f2.c = -f2.c;
    auto d  = cross4(f1.a, f1.b, f2.a, f2.b);
    auto dx = cross4(f1.c, f1.b, f2.c, f2.b);
    auto dy = cross4(f1.a, f1.c, f2.a, f2.c);
    inter.d1 = f1.d;
    inter.d2 = f2.d;

    if(d != 0){
        inter.p = Point2i(dx/d, dy/d);
        inter.d11 = distance2Points(inter.p, t1);
        inter.d12 = distance2Points(inter.p, h1);
        inter.d21 = distance2Points(inter.p, t2);
        inter.d22 = distance2Points(inter.p, h2);
    }

    return inter;
}

vector<Point2i> bisector(const Point2i& t1, const Point2i& h1, const Point2i& t2, const Point2i& h2, bool &direction) {

    auto inter = intersection2Vectors(t1, h1, t2, h2);
    direction = inter.d11 >= inter.d12;
    vector<Point2i> res(3);
    res[0] = midpoint(t1, t2);
    res[1] = midpoint(h1, h2);
    res[2] = res[1];

    if (inter.p.x == INT_MAX) {
        res[0] = midpoint(t1, t2);
        res[1] = midpoint(h1, h2);
        res[2] = res[1];
    }
    else {
        res[0] = inter.p;
        auto ps1 = samplingLine(h1, h2);
        auto id1 = (inter.d12 < 0 || inter.d22 < 0) ? -1 : midByRate(int(ps1.size()), inter.d12, inter.d22);
        res[1] = (id1 < 0 || id1 >= ps1.size()) ? midpoint(h1, h2) : ps1[id1];
        auto ps2 = samplingLine(t1, t2);
        auto id2 = (inter.d11 < 0 || inter.d21 < 0) ? -1 : midByRate(int(ps2.size()), inter.d11, inter.d21);
        res[2] = (id2 < 0 || id2 >= ps2.size()) ? midpoint(t1, t2) : ps2[id2];
    }

    return res;
}

vector<ConvexHull> correctHull(const Mat &img, const vector<int> &cvHull, const vector<Point2i> &contour, vector<double> &angles, size_t minHullEdgeSize, size_t maxHullEdgeSize, double maxHullAngle, double maxContourAngle){

    int contourSize=int(contour.size());
    int cvHullSize=int(cvHull.size());
    vector<ConvexHull> newHull;
    auto prev = cvHull[cvHullSize - 1];
    angles = vector<double>((unsigned long)(contourSize));
    size_t diff=0;
    int max_d=0;
    int max_id=prev;
    int max_first=prev;
    int last=0;
    int begin, end;
    int j,s,e;
    int nde, nid;
    int last_zero;
    int global_max_dist;
    int global_max_id;
    int global_max_first;
    bool update_last;
    pair<int,int> bw;
    double ang_t, angle;
    bool new_seg;
    int dist=0, d1=0;
    size_t temp_size;
    int mid;

    // Track the distance changing from contour points to the corresponding convex hull edges
    for(int i = cvHullSize - 1; i >= 0; --i){
        s=i;
        e=i>0?i-1: cvHullSize - 1;
        begin=cvHull[s];
        end=cvHull[e];
        prev=begin;
        max_d=0;
        diff=0;
        last=0;

        if(end==0){
            end = contourSize-1;
        }

        if(begin>end){
            if(2*(begin-end)>contourSize){
                begin -= contourSize;
            }
            else{
                swap(begin,end);
            }
        }else{
            if(1.1*(end-begin) > contourSize){
                end -= contourSize;
                swap(begin,end);
            }
        }

        Point2i v1=valueAt(contour, contourSize, begin);
        Point2i v2=valueAt(contour,contourSize, end);

        auto coef = getLineCoefficient(v1,v2);
        vector<ConvexHull> temp;
        nde=INT_MAX;
        nid=INT_MAX;
        global_max_dist=0;
        global_max_id=begin;
        global_max_first=begin;
        update_last=true;
        new_seg=true;
        j=begin;

        while(j<end){

            if(new_seg){

                int curr_dist=max_d;
                last_zero=j;

                while(j<end){
                    angle = calculateContourAngle(contour,contourSize,j);
                    j<0?angles[j+contourSize]=angle:angles[j]=angle;

                    if(angle<maxContourAngle){
                        dist = pointLineDistance(valueAt(contour,contourSize,j), coef);

                        if(dist>global_max_dist){
                            global_max_dist=dist;
                            global_max_first=j;
                        }else if(dist==global_max_dist){
                            global_max_id=j;
                        }

                        if(dist-curr_dist>0){
                            new_seg=false;
                            max_d=dist;
                            max_first=last_zero;
                            prev=last_zero;
                            j++;
                            break;
                        }else{
                            last_zero=j;
                        }
                    }
                    j++;
                }
            }
            angle = calculateContourAngle(contour,contourSize,j);
            j<0?angles[j+contourSize]=angle:angles[j]=angle;

            if(angle < maxContourAngle){
                dist=pointLineDistance(valueAt(contour,contourSize,j), coef);

                if(dist>global_max_dist){
                    global_max_dist=dist;
                    global_max_first=j;
                }else if(dist==global_max_dist){
                    global_max_id=j;
                }

                if(dist>0){
                    diff++;
                    last=j;
                }

                if(dist<max_d){
                    if(dist<nde){
                        nid=j;
                        nde=dist;
                    }else{
                        update_last=dist!=0;

                        if(dist==0.0 || (dist-nde>minHullEdgeSize && j-nid>minHullEdgeSize && max_d-nde>minHullEdgeSize && max_d-dist>minHullEdgeSize)) {
                            if(diff>minHullEdgeSize) {
                                if(max_id>j-minHullEdgeSize)
                                    max_id = int(j-minHullEdgeSize);

                                Point2i pprev = valueAt(contour,contourSize,prev);
                                Point2i pnid = valueAt(contour,contourSize,nid);
                                mid=midValue(max_first, max_id);
                                Point2i pmax = valueAt(contour,contourSize,mid);
                                angle = angle3pt(pprev, pmax, pnid);
                                d1 = distance2Points(pnid, pprev);

                                if(angle<maxHullAngle && d1>minHullEdgeSize && d1<maxHullEdgeSize) {
                                    if(prev>j) {
                                        prev -= contourSize;
                                    }

                                    if(dist==0){
                                        bw=make_pair(1,0);
                                    }else{
                                        bw=countBlackWhite(img,pprev,pnid);
                                    }

                                    if(bw.second<0.5*bw.first){
                                        double area=contourArea(
                                                subVector(contour, contourSize, prev, nid));
                                        ConvexHull hull;
                                        hull.dist=distance2Points(midpoint(pprev,pnid), pmax);
                                        hull.area=area;
                                        hull.begin=prev;
                                        hull.farthest=mid;
                                        hull.end=nid;
                                        hull.angle=0.0;
                                        temp_size=temp.size();

                                        if(temp_size>0){
                                            auto flag=true;
                                            Point2i pbegin=valueAt(contour,contourSize,temp[temp_size-1].begin);
                                            Point2i pend=valueAt(contour,contourSize,temp[temp_size-1].end);
                                            auto a1=angle3pt(pbegin,pprev,pend);

                                            if(a1>=maxContourAngle) flag=false;
                                            else{
                                                auto a2=angle3pt(pbegin,pnid,pend);

                                                if(a2>=maxContourAngle) flag=false;
                                                else{
                                                    auto a3=angle3pt(pprev,pbegin,pnid);

                                                    if(a3>=maxContourAngle) flag=false;
                                                    else{
                                                        auto a4=angle3pt(pprev,pend,pnid);

                                                        if(a4>=maxContourAngle) flag=false;
                                                    }
                                                }
                                            }
                                            if(flag){
                                                temp.emplace_back(hull);
                                            }else{
                                                if(j-prev>temp[temp_size-1].end-temp[temp_size-1].begin){
                                                    temp.back()=hull;
                                                }
                                            }
                                        }else{
                                            temp.emplace_back(hull);
                                        }

                                        diff=minHullEdgeSize;
                                        max_d=dist;
                                        max_id=j;
                                        max_first=j;
                                        prev=nid;
                                        new_seg=true;
                                    }else{
                                        if(dist > minHullEdgeSize){
                                            nid=INT_MAX;
                                            nde=INT_MAX;
                                        }else{
                                            diff=0;
                                            max_d=dist;
                                            max_id=j;
                                            max_first=j;
                                            prev=nid;
                                            new_seg=true;
                                        }
                                    }
                                }else{
                                    if(dist > minHullEdgeSize){
                                        nid=INT_MAX;
                                        nde=INT_MAX;
                                    }else{
                                        diff=0;
                                        max_d=dist;
                                        max_id=j;
                                        max_first=j;
                                        prev=nid;
                                        new_seg=true;
                                    }
                                }
                            }else{
                                diff=minHullEdgeSize;
                                max_d=dist;
                                max_id=j;
                                max_first=j;
                                prev=nid;
                                new_seg=true;
                            }
                        }
                    }
                }else if(dist>max_d){
                    max_d=dist;
                    max_id=j;
                    max_first=j;
                    nid=INT_MAX;
                    nde=INT_MAX;
                }else{
                    max_id=j;
                    nid=INT_MAX;
                    nde=INT_MAX;
                }
            }
            j++;
        }
        // Last one
        if(diff > minHullEdgeSize){

            if(max_id> last - minHullEdgeSize){
                max_id= int(last-minHullEdgeSize);
            }
       
            Point2i pprev = valueAt(contour,contourSize,prev);
            Point2i plast = valueAt(contour,contourSize,last);
            mid=midValue(max_first,max_id);
            Point2i pmax = valueAt(contour,contourSize,mid);
            angle = angle3pt(pprev, pmax, plast);
            d1 = distance2Points(plast, pprev);

            if(angle<maxHullAngle && d1>minHullEdgeSize && d1<maxHullEdgeSize) {
                if (prev > last) {
                    prev -= contourSize;
                }

                if (dist == 0) {
                    bw=make_pair(10,0);
                } else {
                    bw = countBlackWhite(img, pprev, plast);
                }

                if (bw.second<0.5*bw.first) {
                    double area = contourArea(
                            subVector(contour, contourSize, prev, last));
                    ConvexHull hull;
                    hull.dist=distance2Points(midpoint(pprev,plast), pmax);
                    hull.area = area;
                    hull.begin = prev;
                    hull.farthest = mid;
                    hull.end = last;
                    hull.angle = 0.0;
                    temp.emplace_back(hull);
                    update_last=false;
                }else update_last=true;
            }
        }

        temp_size = temp.size();

        if(temp_size==0){
            mid=midValue(global_max_first,global_max_id);
            Point2i pmax = valueAt(contour,contourSize,mid);
            angle = angle3pt(v1, pmax, v2);
            d1 = distance2Points(v1, v2);

            if(angle<maxHullAngle && d1>minHullEdgeSize && d1<maxHullEdgeSize) {
                bw = countBlackWhite(img, v1, v2);

                if (bw.second<0.5*bw.first) {
                    double area = contourArea(
                            subVector(contour, contourSize, begin, end));
                    ConvexHull hull;
                    hull.dist = distance2Points(midpoint(v1, v2), pmax);
                    hull.area = area;
                    hull.begin = begin;
                    hull.farthest = mid;
                    hull.end = end;
                    hull.angle = 0.0;
                    temp_size = 1;
                    temp.emplace_back(hull);
                }
            }
        } else if(update_last){
            temp.back().end = end;

            if(max_d>temp[temp_size-1].dist){
                temp.back().dist=max_d;
                temp.back().mid=midValue(max_id,max_first);
            }
            v1=valueAt(contour,contourSize,temp.back().begin);
            v2=valueAt(contour,contourSize,temp.back().end);
            Point2i pmax = valueAt(contour,contourSize,temp.back().mid);
            angle = angle3pt(v1, pmax, v2);
            d1 = distance2Points(v1, v2);

            if(angle<maxHullAngle && d1>minHullEdgeSize && d1<maxHullEdgeSize) {
                bw = countBlackWhite(img, v1, v2);

                if (bw.second<0.5*bw.first) {
                    temp.back().area = contourArea(subVector(contour, contourSize, temp.back().begin, temp.back().end));
                }else{
                    temp.pop_back();
                    temp_size--;
                }
            }
        }

        if(temp_size>1){
            double areas=0.0;
            double max_area=0.0;
            int maxn=1;

            for(auto hull: temp){
                areas += hull.area;
                if(hull.area>max_area){
                    max_area=hull.area;
                    maxn=1;
                }else if(hull.area==max_area){
                    maxn++;
                }
            }

            if(temp_size-maxn==0){
                areas=max_area;
            }else{
                areas-=max_area*maxn;
                areas/=temp_size-maxn;
            }

            sort(temp.begin(),
                 temp.end(),
                 [](const ConvexHull &x, const ConvexHull &y) {
                     return x.area < y.area;
                 });

            while(temp[temp_size-1].area<0.1*areas){
                temp_size--;
            }
            if(temp_size>4){
                temp_size=4;
            }
        }

        if(temp_size>0){
            max_d=0;

            for(size_t k=0; k<temp_size; ++k){
                if(temp[k].dist>max_d){
                    max_d=temp[k].dist;
                }
            }
            if(temp_size==1 || max_d > t3){
                move(temp.begin(),temp.begin()+temp_size,back_inserter(newHull));
            }
        }
    }

    sort(newHull.begin(),newHull.end(),
         [](const ConvexHull &a, const ConvexHull &b) {
             return int(a.area > b.area);
         });

    // Get top 4
    size_t top = 4;

    if(newHull.size()>top)
        newHull=vector<ConvexHull>(
                newHull.begin(),
                newHull.begin()+top);

    return newHull;
}

bool isValidValleyEdge(const Point2i &edgeBegin, const Point2i &edgeEnd, const Point2i &hullBegin, const Point2i &hullEnd, const double distanceThreshold){
    auto intersection = intersection2Vectors(
            edgeBegin,
            edgeEnd,
            hullBegin,
            hullEnd
    );

    return intersection.d21 < distanceThreshold;
}

ValleyEdge findValleyEdges(const vector<Point2i> &contour, const vector<double> &angles, const ConvexHull &hull, double angleThreshold) {

    ValleyEdge valley;
    int contourSize = (int)(contour.size());
    auto hull_begin = valueAt(contour, contourSize, hull.begin);
    auto hull_end = valueAt(contour, contourSize, hull.end);
    auto hull_farthest = valueAt(contour, contourSize, hull.farthest);

    // Find the first valid valley edge
    // Search from the farthest point
    int edge_length = (hull.mid - hull.begin + 1) / 3;
    int edge_end_id = hull.farthest;
    int edge_begin_id = edge_end_id - edge_length;
    int limit_id = hull.begin + edge_length;
    bool valid = false;

    while (edge_end_id > limit_id) {
        edge_end_id--;

        if (valueAt(angles, contourSize, edge_end_id) > angleThreshold) {
            edge_end = valueAt(contour, contourSize, edge_end_id);
            edge_begin = valueAt(contour, contourSize, edge_end_id - edge_length);

            if (isValidValleyEdge(edge_begin, edge_end, hull_begin, hull_end, 5)) {
                valid = true;
                break;
            }
        }
    }
    if (!valid) {
        edge_begin_id = hull.begin;
        edge_end_id = edge_begin_id + edge_length;
    }
    valley.edge1 = make_pair(edge_begin_id, edge_end_id);

    // Find second valid valley edge
    // Search from the farthest point
    edge_length = (hull.end - hull.mid + 1) / 3;
    edge_end_id = hull.farthest;
    edge_begin_id = edge_end_id + edge_length;
    limit_id = hull.end - edge_length;
    valid = false;

    while (edge_end_id < limit_id) {
        edge_end_id++;

        if (valueAt(angles, contourSize, edge_end_id) > angleThreshold) {
            edge_end = valueAt(contour, contourSize, edge_end_id);
            edge_begin = valueAt(contour, contourSize, edge_end_id + edge_length);

            if (isValidValleyEdge(edge_begin, edge_end, hull_begin, hull_end, 5)) {
                valid = true;
                break;
            }
        }
    }
    if (!valid) {
        edge_begin_id = hull.end;
        edge_end_id = edge_begin_id - edge_length;
    }
    valley.edge2 = make_pair(edge_begin_id, edge_end_id);

    return valley;
}

int lineContourIntersectionId(const vector<Point2i>& contour, const LineCoefficient& coef, const Point2i& ref, int contourSize, int beginId, int endId, int step, bool flag = true) {

    int id = midValue(beginId,endId);
    int minDistance=INT_MAX,maxDistance=0,distance;
    Point2i p;

    for(int i=beginId; step*i<step*endId; i+=step){
        p = valueAt(contour,contourSize,i);
        distance = pointLineDistance(p,coef);

        if(distance<1) {
            distance=distance2Points(ref,p);

            if(flag){
                if(distance>maxDistance){
                    id = i;
                    maxDistance=distance;
                }
            } else {
                if(distance<minDistance) {
                    id = i;
                    minDistance=distance;
                }
            }
        }
    }

    return id;
}

pair<Point2i,Point2i> findReferenceLine(const vector<Point2i> &contour, const LineCoefficient &coef, const Point2i &ref, const Valley &v1, const Valley &v2){
    pair<Point2i,Point2i> result;
    int contourSize = int(contour.size());
    int expand = contourSize/5;

    // Find left intersection id
    Point2i begin = valueAt(contour,contourSize, v1.hull.begin);
    Point2i end = valueAt(contour,contourSize, v1.hull.end);
    auto angle1 = angle3pt(begin,v1.center,v2.center);
    auto angle2 = angle3pt(end,v1.center,v2.center);
    int begin_id, end_id, step1, step2;

    if(angle1>angle2){
        begin_id = v1.hull.begin;
        end_id = begin_id-expand;
        step1=-1;
    } else{
        begin_id = v1.hull.end;
        end_id = begin_id+expand;
        step1=1;
    }

    auto left_id = lineContourIntersectionId(contour,coef,ref,contourSize,begin_id,end_id,step1,false);

    // Find right intersection id
    begin = valueAt(contour,contourSize, v2.hull.begin);
    end = valueAt(contour,contourSize, v2.hull.end);
    angle1 = angle3pt(begin,v2.center,v1.center);
    angle2 = angle3pt(end,v2.center,v1.center);

    if(angle1>angle2){
        begin_id = v2.hull.begin;
        end_id = begin_id-expand;
        step2=-1;
    } else{
        begin_id = v2.hull.end;
        end_id = begin_id+expand;
        step2=1;
    }

    auto right_id = lineContourIntersectionId(contour,coef,ref,contourSize,begin_id,end_id,step2,false);

    // Find reference size which is length of line that is parallel to line (contour[left_id],contour[right_id])
    // This line is shortest compared to other parallel lines
    int k=distance2Points(v1.center,v2.center)/2;
    int curr_dist=0;
    int best_size=INT_MAX;
    int loop1=100;
    int loop2;
    int d1=0,d2=0,delta=0;
    Point2i best_left, best_right;

    while(curr_dist<k and loop1--){

        left_id+=step1;
        right_id+=step2;
        loop2=5;

        while(loop2--){
            begin=valueAt(contour,contourSize,left_id);
            end=valueAt(contour,contourSize,right_id);
            d1=pointLineDistance(begin,coef);
            d2=pointLineDistance(end,coef);
            delta = d1-d2;

            if(abs(delta)>1){
                if(delta<0){
                    left_id+=step1;

                } else{
                    right_id+=step2;
                }
            } else break;
        }

        curr_dist = max(d1,d2);
        d1=distance2Points(begin,end);

        if(d1<best_size){
            best_size=d1;
            best_left=begin;
            best_right=end;

        }
    }

    result = make_pair(best_left,best_right);

    return result;
}

BoundingBox getROICoordinates(const Mat &binary_image, const vector<Point2i> &contour, const Valley &v1, const Valley &v2, const Point2i &ref, int minROISize, int maxROISize){

    auto mid1 = midpoint(v1.center,v2.center);
    auto n = distance2Points(v1.center,v2.center);
    BoundingBox bbox;

    // Find edge size
    auto coef=getLineCoefficient(ref,mid1);
    double m=1.0;
    double k1=n/2,k2;

    if(coef.b!=0){
        m=abs(coef.a*1.0/coef.b);
        k1*=sqrt(1.0/(1.0+m*m));
    }
    auto p=sub(ref,mid1);
    p.x=int(mid1.x+sign(p.x)*k1);
    p.y=int(mid1.y+sign(p.y)*m*k1);

    // Find coef of line(v1,v2)
    coef = getLineCoefficient(v1.center,v2.center);
    // Find coef of line L1 that is parallel to line(v1,v2) and p is on this line
    coef.c=-(coef.a*p.x+coef.b*p.y);

    // Find two contour points that are on line L1
    // Left contour point is same side with left point v1
    // Right contour point is same side with right point v2
    auto refLine = findReferenceLine(contour,coef,p,v1,v2);
    auto refDistance = distance2Points(refLine.first,refLine.second)*(1.0-marginLeft*2);

    if(refDistance<minROISize || refDistance>maxROISize){
        errorString = "Out_of_distance";
        return bbox;
    }
    auto mid2 = midpoint(refLine.first,refLine.second);

    // Fetermine point mid_top_edge, mid_bottom_edge on line(mid(v1,v2), ref)
    // and distance(mid_top_edge,mid(v1,v2))=padding
    // Distance(mid_bottom_edge,mid(v1,v2))=edge_size/2
    coef=getLineCoefficient(mid1,mid2);
    m=1.0;
    k1=refDistance*marginTop;
    k2=refDistance;

    if(coef.b!=0){
        m=abs(coef.a*1.0/coef.b);
        k1*=sqrt(1.0/(1.0+m*m));
        k2*=sqrt(1.0/(1.0+m*m));
    }
    p=sub(mid1,mid2);
    Point2i mid_top_edge,mid_bottom_edge;

    mid_top_edge.x=int(mid1.x+sign(p.x)*k1);
    mid_top_edge.y=int(mid1.y+sign(p.y)*m*k1);

    mid_bottom_edge.x=int(mid_top_edge.x+sign(p.x)*k2);
    mid_bottom_edge.y=int(mid_top_edge.y+sign(p.y)*m*k2);

    // Determine top_left and top_right on line is parallel line(v1,v2)
    // and distance(top_left,mid_top_edge)=edge_size/2
    // Distance(top_right,mid_top_edge)=edge_size/2
    m=1.0;
    k1=refDistance/2;
    coef = getLineCoefficient(v1.center,v2.center);

    if(coef.b!=0){
        m=abs(coef.a*1.0/coef.b);
        k1=k1*sqrt(1.0/(1.0+m*m));
    }
    auto top_left=sub(mid1,v1.center);
    top_left.x=int(mid_top_edge.x+sign(top_left.x)*k1);
    top_left.y=int(mid_top_edge.y+sign(top_left.y)*m*k1);

    auto top_right=sub(mid1,v2.center);
    top_right.x=int(mid_top_edge.x+sign(top_right.x)*k1);
    top_right.y=int(mid_top_edge.y+sign(top_right.y)*m*k1);

    // Determine bottom_left, bottom_right on line is parallel line(v1,v2) and
    // distance(bottom_left,mid_bottom_edge)=edge_size
    // distance(bottom_right,mid_bottom_edge)=edge_size
    auto bottom_left=sub(mid1,v1.center);
    bottom_left.x=int(mid_bottom_edge.x+sign(bottom_left.x)*k1);
    bottom_left.y=int(mid_bottom_edge.y+sign(bottom_left.y)*m*k1);

    auto bottom_right=sub(mid1,v2.center);
    bottom_right.x=int(mid_bottom_edge.x+sign(bottom_right.x)*k1);
    bottom_right.y=int(mid_bottom_edge.y+sign(bottom_right.y)*m*k1);

    // Validate roi coordinates
    int x_min=0;
    int y_min=0;
    int x_max=binary_image.cols-1;
    int y_max=binary_image.rows-1;

    if ( outBound(bottom_left, x_min, x_max, y_min, y_max) || outBound(bottom_right, x_min, x_max, y_min, y_max) ){
        errorString = "Outbound";
        return bbox;
    }
    bbox.init(top_left, top_right, bottom_left, bottom_right);

    return bbox;
}

vector< vector<int> > getValleyTriplets( const vector<Valley> &valleys){

    auto valleysSize = valleys.size();
    int max_angle=0,angle;
    int left=0,middle=1,right=2;
    vector< vector<int> > result;

    for(int i=0; i<valleysSize; ++i){

        max_angle=0;

        for(int j=0; j<valleysSize; ++j){

            if(j!=i){

                for(int k=0; k<valleysSize; ++k){

                    if(k!=i && k!=j){

                        angle = int(angle3pt(
                                valleys[j].center,
                                valleys[i].center,
                                valleys[k].center
                        ));
                        if(angle>max_angle){
                            max_angle=angle;
                            left=j;
                            middle=i;
                            right=k;
                        }
                    }
                }
            }
        }

        vector<int> item{max_angle,left,middle,right};
        result.emplace_back(item);
    }

    sort(result.rbegin(),result.rend());


    return result;
}

vector< pair<int,int> > selectTwoOptimalKeyVectors(const vector<Valley> &valleys, int angle, int angleThreshold){

    vector< pair<int, int> > res;

    auto d1=distance2Points(valleys[1].center,valleys[0].center);
    auto d2=distance2Points(valleys[1].center,valleys[2].center);
    auto delta=max(d1,d2)/2;

    if(min(d1,d2)>delta){
        res.emplace_back(make_pair(0,2));
    } else {
        errorString = "Triangle_failure";
    }

    if(angle<=angleThreshold){

        auto a1=max(
                angle3pt(
                        valleys[0].ref,
                        valleys[0].center,
                        valleys[1].center),
                angle3pt(
                        valleys[0].ref,
                        valleys[0].center,
                        valleys[2].center)
        );

        auto a2=max(
                angle3pt(
                        valleys[1].ref,
                        valleys[1].center,
                        valleys[0].center),
                angle3pt(
                        valleys[1].ref,
                        valleys[1].center,
                        valleys[2].center)
        );

        auto a3=max(
                angle3pt(
                        valleys[2].ref,
                        valleys[2].center,
                        valleys[0].center),
                angle3pt(
                        valleys[2].ref,
                        valleys[2].center,
                        valleys[1].center)
        );

        auto a=min(min(a1,a2), a3);

        if(a==a1) {
            res.emplace_back(make_pair(1,2));
        } else if(a==a3) {
            res.emplace_back(make_pair(0,1));
        }
    }

    return  res;
}

BoundingBox selectROICandidate(const Mat &binaryImage, const vector<Point2i> &contour, const vector<Valley> &valleys, int minROISize, int maxROISize, int angleThreshold) {

    auto valleysSize = valleys.size();
    int contourSize = int(contour.size());
    BoundingBox roi;

    if(valleysSize<2) {
        errorString = "Num_vector_smaller_2";
        return roi;
    }

    auto angle_t = angleThreshold + 30;
    double angle1, angle2;
    Valley valley1, valley2;
    Point2i begin1, end1, begin2, end2, ref;
    int d1,d2;

    if(valleysSize==2){

        valley1 = valleys[0];
        valley2 = valleys[1];

        angle1 = angle2VectWithDirection(
                sub(valley1.center, valley1.ref),
                sub(valley1.center, valley2.center));

        angle2 = angle2VectWithDirection(
                sub(valley2.center, valley2.ref),
                sub(valley2.center, valley1.center));

        if (abs(angle1)>angle_t || abs(angle2)>angle_t || (angle1*angle2>0))
            return roi;

        if (angle1 > 0) {
            valley1=valleys[1];
            valley2=valleys[0];
        }

        begin1 = valueAt(contour,contourSize,valley1.hull.begin);
        end1 = valueAt(contour,contourSize,valley2.hull.end);
        d1 = distance2Points(begin1,end1);
        begin2 = valueAt(contour,contourSize,valley2.hull.begin);
        end2 = valueAt(contour,contourSize,valley1.hull.end);
        d2=distance2Points(begin2,end2);

        if(d1<d2){
            ref = midpoint(begin1, end1);
        } else {
            ref = midpoint(begin2, end2);
        }
        roi = getROICoordinates(binaryImage,contour,valley1,valley2,ref, minROISize,maxROISize);
    } else {

        auto triplets = getValleyTriplets(valleys);

        for(auto item: triplets){

            vector<Valley> triplet{valleys[item[1]],valleys[item[2]],valleys[item[3]]};
            auto candidates = selectTwoOptimalKeyVectors(triplet,item[0],angleThreshold);

            if(candidates.empty()) continue;

            for(auto c: candidates){

                valley1 = triplet[c.first];
                valley2 = triplet[c.second];
                angle1 = angle2VectWithDirection(
                        sub(valley1.center, valley1.ref),
                        sub(valley1.center, valley2.center));
                angle2 = angle2VectWithDirection(
                        sub(valley2.center, valley2.ref),
                        sub(valley2.center, valley1.center));

                if (abs(angle1)>angle_t || abs(angle2)>angle_t || angle1*angle2>0) {
                    errorString = "Vector_angle_error";
                    continue;
                }

                if (angle1>0) {
                    valley1=triplet[c.second];
                    valley2=triplet[c.first];
                }
                begin1 = valueAt(contour,contourSize,valley1.hull.begin);
                end1 = valueAt(contour,contourSize,valley2.hull.end);
                d1=distance2Points(begin1,end1);
                begin2 = valueAt(contour,contourSize,valley2.hull.begin);
                end2 = valueAt(contour,contourSize,valley1.hull.end);
                d2 = distance2Points(begin2,end2);

                if(d1<d2){
                    ref = midpoint(begin1, end1);
                } else {
                    ref = midpoint(begin2, end2);
                }
                roi = getROICoordinates(binaryImage, contour, valley1, valley2, ref, minROISize, maxROISize);

                if(roi.edgeSize != INT_MAX) {
                    break;
                }
            }

            if(roi.edgeSize != INT_MAX){
                break;
            }
        }
    }

    return roi;
}

void segmentHand(const Mat &rgbImage, Mat &binaryImage, vector<Point2i> &contour){
    // Extract the binary mask and the corressponding largest contour
    Mat binary, tmp;
    vector<Mat> lab,yrb;
    vector<vector<Point2i> > contours;
    double new_area;
    double largest_area;
    size_t largest_contour_index;

    // Find most compact channel between channels A* and B* of LAB color space, Cr and Cb of YCrCb color space
    auto thresholding_method = THRESH_OTSU;

    cvtColor(rgbImage, tmp, COLOR_RGB2Lab);
    split(tmp, lab);

    cvtColor(rgbImage, tmp, COLOR_RGB2YCrCb);
    split(tmp, yrb);

    auto bin_size = binaryMaskSize * binaryMaskSize;
    const auto gaussKernel = cv::Size(5, 5);
    Mat bin_a, bin_b, bin_cr, bin_cb;

    GaussianBlur(lab.at(1), lab.at(1), gaussKernel, 0);
    threshold(lab.at(1), bin_a, 0, 255, THRESH_BINARY + thresholding_method);
    auto score_a = countNonZero(bin_a) * 1.0/bin_size;

    GaussianBlur(lab.at(2), lab.at(2), gaussKernel, 0);
    threshold(lab.at(2), bin_b, 0, 255, THRESH_BINARY + thresholding_method);
    auto score_b = countNonZero(bin_b) * 1.0 / bin_size;

    GaussianBlur(yrb.at(1), yrb.at(1), gaussKernel, 0);
    threshold(yrb.at(1), bin_cr, 0, 255, THRESH_BINARY + thresholding_method);
    auto score_cr = countNonZero(bin_cr) * 1.0 / bin_size;

    GaussianBlur(yrb.at(2), yrb.at(2), gaussKernel, 0);
    threshold(yrb.at(2), bin_cb, 0, 255, THRESH_BINARY_INV + thresholding_method);
    auto score_cb = countNonZero(bin_cb) * 1.0 / bin_size;

    double best_score = 0;

    if (score_a<whiteArea_threshold && score_a>best_score) {
        best_score = score_a;
    }

    if (score_b<whiteArea_threshold && score_b>best_score) {
        best_score = score_b;
    }

    if (score_cr<whiteArea_threshold && score_cr>best_score) {
        best_score = score_cr;
    }

    if (score_cb<whiteArea_threshold && score_cb>best_score) {
        best_score = score_cb;
    }

    if (best_score == score_a) {
        binary = bin_a;
    }
    else if (best_score == score_b) {
        binary = bin_b;
    }
    else if (best_score == score_cr) {
        binary = bin_cr;
    }
    else {
        binary = bin_cb;
    }

    largest_area=0.0;
    largest_contour_index=0;
    findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    auto contours_size = contours.size();

    for (size_t i = 0; i < contours_size; i++) {
        new_area = contourArea(contours[i]);

        if (new_area > largest_area) {
            largest_area = new_area;
            largest_contour_index = i;
        }
    }
    binary.setTo(0);
    drawContours(binary, contours, int(largest_contour_index), Scalar(1.0), FILLED);
    contours.clear();

    // Apply adaptive threshold on region limited by binary mask
    cvtColor(rgbImage, tmp, COLOR_RGB2GRAY);
    GaussianBlur(tmp, tmp, gaussKernel, 0);
    multiply(binary, tmp, tmp);
    adaptiveThreshold(tmp, binary, 1.0, ADAPTIVE_THRESH_GAUSSIAN_C, 0, blockSize, constant);
    
    // Remove noises
    medianBlur(binary, binary, 5);
    morphologyEx(binary, binary, MORPH_ERODE, cv::getStructuringElement(MORPH_RECT, cv::Size(3,3)));

    // Find the largest contour and corresponding binary mask after applying Adaptive thresholding method
    largest_area=0.0;
    largest_contour_index=0;
    findContours(binary, contours, RETR_LIST, CHAIN_APPROX_NONE);

    contours_size = contours.size();

    for (size_t i = 0; i < contours_size; i++) {

        new_area = contourArea(contours[i]);

        if (new_area > largest_area) {
            largest_area = new_area;
            largest_contour_index = i;
        }
    }
    contour=contours[largest_contour_index];
    binaryImage = Mat(binary.rows, binary.cols, CV_8UC1, Scalar(0));
    drawContours(binaryImage, contours, int(largest_contour_index), Scalar(255.0), FILLED);
}

Mat extractSaturationFromRGBImage(const Mat& src) {

    Mat dst = Mat(src.rows, src.cols, CV_8UC1);
    double b = 0, g = 0, r = 0, vmin, vmax, L;

    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* pixel1 = src.ptr<uchar>(i);
        uchar* pixel2 = dst.ptr<uchar>(i);

        for (int j = 0; j < src.cols; ++j)
        {
            r = (*pixel1++);
            g = (*pixel1++);
            b = (*pixel1++);

            vmin = min(b, min(g, r));
            vmax = max(b, max(g, r));
            L = vmin + vmax;

            *pixel2++ = 255 * (L < 256 ? (vmax - vmin) / L : (vmax - vmin) / (512 - L));
        }
    }

    return dst;
}

Mat correctGamma(const Mat &src, double gamma, bool isAutoMode){
    double c=1.0;

    if(isAutoMode){
        double val = mean(src).val[0]/255;
        c=log10(0.5)/log10(val);
    }

    gamma *= c;

    Mat lut(1, 256, CV_8U);

    for (int i=0; i<256; ++i) {
        lut.at<uchar>(i) = (uchar) (clip(pow(i * 1.0 / 255.0, gamma) * 255.0, 0.0, 255.0));
    }
    Mat dst;
    LUT(src, lut, dst);

    return dst;
}

Mat enhanceFeatures(const Mat &src){

    Mat dst = Mat();

    if(src.data == nullptr) return dst;

    //Auto Correct Gamma
    dst = correctGamma(src, 1.0, true);

    //Sharpen Image using CLAHE
    auto clahe = createCLAHE();
    clahe->setClipLimit(2);
    clahe->setTilesGridSize(cv::Size(4,4));
    clahe->apply(dst, dst);

    //Low-pass filter
    dst.convertTo(dst, CV_32F, 1.0/255.0);
    GaussianBlur(dst, dst, cv::Size(0,0), 4);

    //High-pass filter
    Laplacian(dst, dst, CV_32F);
    dst = max(dst, 0.0);
    double lapmin, lapmax;
    minMaxLoc(dst, &lapmin, &lapmax);
    double scale = 255.0/lapmax;
    dst.convertTo(dst, CV_8U, scale);

    return dst;
}

void extractPalmROI(const Mat &rgbImage, Mat &roiImage, int roiSize) {
    
    // Step1. Crop to center square image and resize to 160x160 pixels
    Mat croppedImage, resizedImage;  
    bool portrait = rgbImage.cols < rgbImage.rows;
    cv::Rect rect;

    if (portrait) {
        int biasY = (frame.rows - frame.cols) / 2;
        rect = cv::Rect(0, biasY, frame.cols, frame.cols);
    }
    else {
        int biasX = (frame.cols - frame.rows) / 2;
        rect = cv::Rect(biasX, 0, frame.rows, frame.rows);
    }
    rgbImage(rect).copyTo(croppedImage);

    if (croppedImage.rows > binaryMaskSize) {
        resize(croppedImage, resizedImage, cv::Size(binaryMaskSize, binaryMaskSize), 0, 0, INTER_AREA);
    }
    else if (croppedImage.rows < binaryMaskSize) {
        resize(croppedImage, resizedImage, cv::Size(binaryMaskSize, binaryMaskSize), 0, 0, INTER_LINEAR);
    }
    else {
        resizedImage = croppedImage;
    }

    // Step 2. Segment hand
    vector<Point2i> contour;
    roiImage = Mat();
    auto bbox = BoundingBox();
    segmentHand(resizedImage, binaryImage, contour);

    // Step 3. Find potential valleys between the adjacent fingers
    int contourSize = (int)contour.size();
    vector<int> cvHull;
    vector<double> angles;
    double maxContourAngle = 170.0;
    double maxHullAngle = 100;
    size_t minHullEdgeSize=2;
    size_t maxHullEdgeSize=80;
    int minROISize=int(binaryMaskSize*rmin);
    int maxROISize=int(binaryMaskSize*rmax);
    int begin_id, end_id, inter_id;
    LineCoefficient coef;
    convexHull(contour, cvHull, false, false);
    auto hulls=correctHull(binaryImage, cvHull, contour, angles, minHullEdgeSize, maxHullEdgeSize, maxHullAngle, maxContourAngle);

    // Step 4. Find key vector candidates
    vector<Valley> valleys;

    for(auto hull:hulls){

        ValleyEdge ve;

        try {
            ve = findValleyEdges(contour, angles, hull, maxContourAngle);
        } catch (Exception &e){
            ve.edge1 = make_pair(hull.begin, midValue(hull.begin, hull.farthest));
            ve.edge2 = make_pair(hull.end, midValue(hull.end, hull.farthest));
        }
        vector<Point2i> bs;
        Point2i tail1,head1,tail2,head2;
        Valley v;

        tail1=valueAt(contour,contourSize,ve.edge1.first);
        head1=valueAt(contour,contourSize,ve.edge1.second);
        tail2=valueAt(contour,contourSize,ve.edge2.first);
        head2=valueAt(contour,contourSize,ve.edge2.second);
        bool direction;
        bs = bisector(tail1, head1, tail2, head2, direction);

        if(ve.edge1.first<ve.edge2.first){
            begin_id=ve.edge1.first;
            end_id=ve.edge2.first;
        } else{
            begin_id=ve.edge2.first;
            end_id=ve.edge1.first;
        }
        coef = getLineCoefficient(bs[0], bs[1]);
        inter_id = lineContourIntersectionId(contour, coef, bs[1], contourSize, begin_id,end_id, sign(end_id-begin_id), direction);

        v.cid = inter_id;
        v.ref=(distance2Points(head1,bs[1])>t3)?bs[1]:bs[2];
        v.center=valueAt(contour,contourSize,v.cid);

        head1=valueAt(contour,contourSize,hull.begin);
        tail1=valueAt(contour,contourSize,hull.end);
        coef = getLineCoefficient(head1, tail1);
        hull.dist=pointLineDistance(v.center,coef);
        hull.angle=angle3pt(head1,v.center,tail1);

        if(hull.angle > maxHullAngle){
            continue;
        }
        v.hull=hull;
        valleys.emplace_back(v);
    }
    // Step 5+6. Find two optimal key vectors and locate palm ROI
    int angleThreshold=120;
    bbox = selectROICandidate(binaryImage, contour, valleys, minROISize, maxROISize, angleThreshold);

    if(bbox.edgeSize == INT_MAX) return;
    // Scale up to croppedImage size
    double scaleFactor = croppedImage.rows * 1.0 / binaryMaskSize;
    bbox.scale(scaleFactor);

    // Step 7. Extract palm ROI
    auto dsize = cv::Size(roiSize, roiSize);
    auto input4=bbox.inputQuad();
    auto output4=BoundingBox::outputQuad(roiSize);
    auto M = getPerspectiveTransform(input4, output4);
    roiImage = Mat(roiSize, roiSize, CV_8UC1);
    warpPerspective(rgbImage, roiImage, M, dsize);

    // Step 8. Extract saturation image
    roiImage = extractSaturationFromRGBImage(roiImage);
}