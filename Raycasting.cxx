#define _USE_MATH_DEFINES
#include <iostream>
#include <chrono>
#include <cmath>
#include "vtkSmartPointer.h"
#include "vtkSphereSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkInteractorStyle.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkProperty.h"
#include "vtkCamera.h"
#include "vtkLight.h"
#include "vtkOpenGLPolyDataMapper.h"
#include "vtkJPEGReader.h"
#include "vtkImageData.h"
#include <vtkImageMapper.h>
#include <vtkNamedColors.h>
#include <vtkPNGWriter.h>
#include <vtkLookupTable.h>
#include <vtkActor2D.h>

#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkPolyDataReader.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkCellArray.h>
#include <vtkDataSetReader.h>
#include <vtkContourFilter.h>
#include <vtkRectilinearGrid.h>

#include <vtkCamera.h>
#include <vtkDataSetMapper.h>
#include <vtkRenderer.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSetTriangleFilter.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <vtkVolumeProperty.h>
#include <vtkOpenGLProjectedTetrahedraMapper.h>
#include <vtkRectilinearGridToTetrahedra.h>
#include <vtkType.h>
#include <vtkTetra.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridReader.h>

struct Ray {
    double orig[3];
    double dir[3];
};

struct Camera
{
    double          near, far;
    double          angle;
    double          position[3];
    double          focus[3];
    double          up[3];
};


struct TransferFunction
{
    double          min;
    double          max;
    int             numBins;
    unsigned char* colors;  // size is 3*numBins
    double* opacities; // size is numBins

    // Take in a value and applies the transfer function.
    // Step #1: figure out which bin "value" lies in.
    // If "min" is 2 and "max" is 4, and there are 10 bins, then
    //   bin 0 = 2->2.2
    //   bin 1 = 2.2->2.4
    //   bin 2 = 2.4->2.6
    //   bin 3 = 2.6->2.8
    //   bin 4 = 2.8->3.0
    //   bin 5 = 3.0->3.2
    //   bin 6 = 3.2->3.4
    //   bin 7 = 3.4->3.6
    //   bin 8 = 3.6->3.8
    //   bin 9 = 3.8->4.0
    // and, for example, a "value" of 3.15 would return the color in bin 5
    // and the opacity at "opacities[5]".
    void ApplyTransferFunction(double value, double grad, unsigned char* RGB, double& opacity, bool twodTransfer)
    {


        grad = grad * 1e9;

        int bin = 0;
        int grad_bin = 0;
        if (value > max) value = max;
        if (value < min) value = min;
        bin = (int)floor(numBins * (value - min) / (max - min));
        if (grad > max) grad = max;
        if (grad < min) grad = min;
        grad_bin = (int)floor(numBins * (grad - min) / (max - min));

        RGB[0] = colors[3 * bin + 0];
        RGB[1] = colors[3 * bin + 1];
        RGB[2] = colors[3 * bin + 2];
        if (twodTransfer) bin = grad_bin;
        opacity = opacities[bin];

        //cerr<<"mapped to bin " <<bin<<endl;


    }

    void ApplyTransferFunctionTetra(double value, unsigned char* RGB, double& opacity)
    {
        value += 5;
        int bin = 0;

        if (value > max) value = max;
        if (value < min) value = min;
        bin = (int)floor(numBins * (value - min) / (max - min));


        RGB[0] = colors[3 * bin + 0];
        RGB[1] = colors[3 * bin + 1];
        RGB[2] = colors[3 * bin + 2];
        opacity = opacities[bin];

        //cerr<<"mapped to bin " <<bin<<endl;


    }


};

TransferFunction
SetupTransferFunctionTetra()
{
    int  i;

    TransferFunction rv;
    rv.min = 0;
    rv.max = 15;
    rv.numBins = 256;
    rv.colors = new unsigned char[3 * 256];
    rv.opacities = new double[256];
    unsigned char charOpacity[256] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 14, 14, 14, 14, 14, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 5, 4, 3, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 16, 16, 15, 14, 13, 12, 11, 9, 8, 7, 6, 5, 5, 4, 3, 3, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 18, 20, 22, 24, 27, 29, 32, 35, 38, 41, 44, 47, 50, 52, 55, 58, 60, 62, 64, 66, 67, 68, 69, 70, 70, 70, 69, 68, 67, 66, 64, 62, 60, 58, 55, 52, 50, 47, 44, 41, 38, 35, 32, 29, 27, 24, 22, 20, 20, 23, 28, 33, 38, 45, 51, 59, 67, 76, 85, 95, 105, 116, 127, 138, 149, 160, 170, 180, 189, 198, 205, 212, 217, 221, 223, 224, 224, 222, 219, 214, 208, 201, 193, 184, 174, 164, 153, 142, 131, 120, 109, 99, 89, 79, 70, 62, 54, 47, 40, 35, 30, 25, 21, 17, 14, 12, 10, 8, 6, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };


    for (int i = 0; i < 256; i++) {
        charOpacity[i] = i;
    }

    for (i = 0; i < 256; i++)
        rv.opacities[i] = charOpacity[i] / 255.0;
    const int numControlPoints = 8;
    unsigned char controlPointColors[numControlPoints * 3] = {
           71, 71, 219, 0, 0, 91, 0, 255, 255, 0, 127, 0,
           255, 255, 0, 255, 96, 0, 107, 0, 0, 224, 76, 76
    };
    double controlPointPositions[numControlPoints] = { 0, 0.143, 0.285, 0.429, 0.571, 0.714, 0.857, 1.0 };
    for (i = 0; i < numControlPoints - 1; i++)
    {
        int start = controlPointPositions[i] * rv.numBins;
        int end = controlPointPositions[i + 1] * rv.numBins + 1;
        //cerr << "Working on " << i << "/" << i+1 << ", with range " << start << "/" << end << endl;
        if (end >= rv.numBins)
            end = rv.numBins - 1;
        for (int j = start; j <= end; j++)
        {
            double proportion = (j / (rv.numBins - 1.0) - controlPointPositions[i]) / (controlPointPositions[i + 1] - controlPointPositions[i]);
            if (proportion < 0 || proportion > 1.)
                continue;
            for (int k = 0; k < 3; k++)
                rv.colors[3 * j + k] = proportion * (controlPointColors[3 * (i + 1) + k] - controlPointColors[3 * i + k])
                + controlPointColors[3 * i + k];

        }
    }

    return rv;
}


TransferFunction
SetupTransferFunction(bool grad)
{
    int  i;

    TransferFunction rv;
    rv.min = 10;
    rv.max = 15;
    rv.numBins = 256;
    rv.colors = new unsigned char[3 * 256];
    rv.opacities = new double[256];
    unsigned char charOpacity[256] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 14, 14, 14, 14, 14, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 5, 4, 3, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 16, 16, 15, 14, 13, 12, 11, 9, 8, 7, 6, 5, 5, 4, 3, 3, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 18, 20, 22, 24, 27, 29, 32, 35, 38, 41, 44, 47, 50, 52, 55, 58, 60, 62, 64, 66, 67, 68, 69, 70, 70, 70, 69, 68, 67, 66, 64, 62, 60, 58, 55, 52, 50, 47, 44, 41, 38, 35, 32, 29, 27, 24, 22, 20, 20, 23, 28, 33, 38, 45, 51, 59, 67, 76, 85, 95, 105, 116, 127, 138, 149, 160, 170, 180, 189, 198, 205, 212, 217, 221, 223, 224, 224, 222, 219, 214, 208, 201, 193, 184, 174, 164, 153, 142, 131, 120, 109, 99, 89, 79, 70, 62, 54, 47, 40, 35, 30, 25, 21, 17, 14, 12, 10, 8, 6, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    if (grad) {
        for (int i = 0; i < 256; i++) {
            charOpacity[i] = i;
        }
    }


    for (i = 0; i < 256; i++)
        rv.opacities[i] = charOpacity[i] / 255.0;
    const int numControlPoints = 8;
    unsigned char controlPointColors[numControlPoints * 3] = {
           71, 71, 219, 0, 0, 91, 0, 255, 255, 0, 127, 0,
           255, 255, 0, 255, 96, 0, 107, 0, 0, 224, 76, 76
    };
    double controlPointPositions[numControlPoints] = { 0, 0.143, 0.285, 0.429, 0.571, 0.714, 0.857, 1.0 };
    for (i = 0; i < numControlPoints - 1; i++)
    {
        int start = controlPointPositions[i] * rv.numBins;
        int end = controlPointPositions[i + 1] * rv.numBins + 1;
        //cerr << "Working on " << i << "/" << i+1 << ", with range " << start << "/" << end << endl;
        if (end >= rv.numBins)
            end = rv.numBins - 1;
        for (int j = start; j <= end; j++)
        {
            double proportion = (j / (rv.numBins - 1.0) - controlPointPositions[i]) / (controlPointPositions[i + 1] - controlPointPositions[i]);
            if (proportion < 0 || proportion > 1.)
                continue;
            for (int k = 0; k < 3; k++)
                rv.colors[3 * j + k] = proportion * (controlPointColors[3 * (i + 1) + k] - controlPointColors[3 * i + k])
                + controlPointColors[3 * i + k];

        }
    }

    return rv;
}

Camera
SetupCamera(void)
{
    Camera rv;
    rv.focus[0] = 0;
    rv.focus[1] = 0;
    rv.focus[2] = 0;
    rv.up[0] = 0;
    rv.up[1] = -1;
    rv.up[2] = 0;
    rv.angle = 30;
    rv.near = 7.5e+7;
    rv.far = 1.4e+8;
    rv.position[0] = -8.25e+7;
    rv.position[1] = -3.45e+7;
    rv.position[2] = 3.35e+7;

    return rv;
}

Camera
SetupCameraTetra(void)
{
    Camera rv;
    rv.focus[0] = 0.5;
    rv.focus[1] = 0.5;
    rv.focus[2] = 0;
    rv.up[0] = 0;
    rv.up[1] = 1;
    rv.up[2] = 0;
    rv.angle = 30;
    rv.near = 0;
    rv.far = 10;
    rv.position[0] = 2;
    rv.position[1] = 2;
    rv.position[2] = -4;

    return rv;
}


void
CrossProduct(double v_A[], double v_B[], double c_P[])
{
    c_P[0] = v_A[1] * v_B[2] - v_A[2] * v_B[1];
    c_P[1] = -(v_A[0] * v_B[2] - v_A[2] * v_B[0]);
    c_P[2] = v_A[0] * v_B[1] - v_A[1] * v_B[0];
}

void
Normalize(double v_A[])
{
    double div = sqrt(pow(v_A[0], 2) + pow(v_A[1], 2) + pow(v_A[2], 2));
    v_A[0] /= div;
    v_A[1] /= div;
    v_A[2] /= div;
}

double
Interp(double samplePos[], int i, int j, int k, const int* dims,
    const float* X, const float* Y, const float* Z, const float* F) {
    double xd = (samplePos[0] - X[i]) / (X[i + 1] - X[i]);
    double yd = (samplePos[1] - Y[j]) / (Y[j + 1] - Y[j]);
    double zd = (samplePos[2] - Z[k]) / (Z[k + 1] - Z[k]);

    double c000 = F[i + j * dims[0] + k * (dims[0] * dims[1])];
    double c100 = F[(i + 1) + j * dims[0] + (k) * (dims[0] * dims[1])];
    double c110 = F[i + 1 + (j + 1) * dims[0] + (k) * (dims[0] * dims[1])];
    double c010 = F[i + (j + 1) * dims[0] + (k) * (dims[0] * dims[1])];
    double c001 = F[i + (j)*dims[0] + (k + 1) * (dims[0] * dims[1])];
    double c101 = F[i + 1 + (j)*dims[0] + (k + 1) * (dims[0] * dims[1])];
    double c111 = F[i + 1 + (j + 1) * dims[0] + (k + 1) * (dims[0] * dims[1])];
    double c011 = F[i + (j + 1) * dims[0] + (k + 1) * (dims[0] * dims[1])];

    double c00 = c000 * (1 - xd) + c100 * xd;
    double c01 = c001 * (1 - xd) + c101 * xd;
    double c10 = c010 * (1 - xd) + c110 * xd;
    double c11 = c011 * (1 - xd) + c111 * xd;

    double c0 = c00 * (1 - yd) + c10 * yd;
    double c1 = c01 * (1 - yd) + c11 * yd;

    double c = c0 * (1 - zd) + c1 * zd;

    return c;
}

void
Intersect(Ray ray, int samples, const int stepSize,
    TransferFunction transfer, const double near, const double far,
    const int* dims, const int ncells,
    const float* X, const float* Y, const float* Z, const float* F,
    double rayTermination, unsigned char* pixelRGB, bool twodTransfer)
{


    int numx = dims[0] - 1;
    int numy = dims[1] - 1;
    int numz = dims[2] - 1;

    double samplePos[3];
    int sampleCell[3] = { -1,-1,-1 };
    int sign[3];
    double last = 0;

    bool hadCollision = false;

    double RGB_F[] = { 0, 0, 0, 0 };

    sign[0] = ray.dir[0] > 0 ? 1 : -1;
    sign[1] = ray.dir[1] > 0 ? 1 : -1;
    sign[2] = ray.dir[2] > 0 ? 1 : -1;


    for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++) {
        samplePos[0] = ray.dir[0] * stepSize * sampleIndex + ray.orig[0] + ray.dir[0] * near;
        samplePos[1] = ray.dir[1] * stepSize * sampleIndex + ray.orig[1] + ray.dir[1] * near;
        samplePos[2] = ray.dir[2] * stepSize * sampleIndex + ray.orig[2] + ray.dir[2] * near;

        if (!(samplePos[0] > X[0] && samplePos[0]<X[numx] &&
            samplePos[1]>Y[0] && samplePos[1]<Y[numy] &&
            samplePos[2]>Z[0] && samplePos[2] < Z[numz])) {
            if (hadCollision) break; else continue;
        }

        if (hadCollision) {

            int i = sampleCell[0];
            int j = sampleCell[1];
            int k = sampleCell[2];
            int oldi = i;
            int oldj = j;
            int oldk = k;

            if (sign[0] > 0) {
                while (samplePos[0] > X[i + 1])
                    i++;
            }
            else {
                while (samplePos[0] < X[i])
                    i--;
            }

            if (sign[1] > 0) {
                while (samplePos[1] > Y[j + 1])
                    j++;
            }
            else {
                while (samplePos[1] < Y[j])
                    j--;
            }

            if (sign[2] > 0) {
                while (samplePos[2] > Z[k + 1])
                    k++;
            }
            else {
                while (samplePos[2] < Z[k])
                    k--;
            }


            sampleCell[0] = i;
            sampleCell[1] = j;
            sampleCell[2] = k;
            unsigned char* RGB = new unsigned char[3];
            double opacity = 0;

            double c = Interp(samplePos, i, j, k, dims, X, Y, Z, F);
            double grad = c - last;
            grad = pow(grad / (samplePos[0] - X[oldi]), 2) + pow(grad / (samplePos[1] - Y[oldj]), 2) + pow(grad / (samplePos[2] - Z[oldk]), 2);



            transfer.ApplyTransferFunction(c, grad, RGB, opacity, twodTransfer);

            double r = RGB[0];
            double g = RGB[1];
            double b = RGB[2];

            //alpha correction 
            double corrected = 1 - pow((1 - opacity), 500.0 / samples);

            RGB_F[0] = r / 255.0 * (1 - RGB_F[3]) * corrected + RGB_F[0];
            RGB_F[1] = g / 255.0 * (1 - RGB_F[3]) * corrected + RGB_F[1];
            RGB_F[2] = b / 255.0 * (1 - RGB_F[3]) * corrected + RGB_F[2];
            RGB_F[3] = (1 - RGB_F[3]) * corrected + RGB_F[3];

            if (RGB_F[3] > rayTermination) {
                break;
            }


        }
        else {
            for (int i = 0; i < numx; i++) {
                for (int j = 0; j < numy; j++) {
                    for (int k = 0; k < numz; k++) {
                        if (samplePos[0] > X[i] && samplePos[0]<X[i + 1] &&
                            samplePos[1]>Y[j] && samplePos[1]<Y[j + 1] &&
                            samplePos[2]>Z[k] && samplePos[2] < Z[k + 1]) {
                            hadCollision = true;
                            sampleCell[0] = i;
                            sampleCell[1] = j;
                            sampleCell[2] = k;
                            unsigned char* RGB = new unsigned char[3];
                            double opacity = 0;

                            double c = Interp(samplePos, i, j, k, dims, X, Y, Z, F);
                            double grad = c - F[i + j * dims[0] + k * (dims[0] * dims[1])];
                            grad = pow(grad / (samplePos[0] - X[i]), 2) + pow(grad / (samplePos[1] - Y[j]), 2) + pow(grad / (samplePos[2] - Z[k]), 2);

                            last = c;

                            transfer.ApplyTransferFunction(c, grad, RGB, opacity, twodTransfer);

                            double r = RGB[0];
                            double g = RGB[1];
                            double b = RGB[2];

                            //alpha correction 
                            double corrected = 1 - pow((1 - opacity), 500.0 / samples);

                            RGB_F[0] = r / 255.0 * (1 - RGB_F[3]) * corrected + RGB_F[0];
                            RGB_F[1] = g / 255.0 * (1 - RGB_F[3]) * corrected + RGB_F[1];
                            RGB_F[2] = b / 255.0 * (1 - RGB_F[3]) * corrected + RGB_F[2];
                            RGB_F[3] = (1 - RGB_F[3]) * corrected + RGB_F[3];

                            if (RGB_F[3] > rayTermination) {
                                break;
                            }

                        }
                    }

                }
            }

        }

    }

    pixelRGB[0] = round(RGB_F[0] * 255);
    pixelRGB[1] = round(RGB_F[1] * 255);
    pixelRGB[2] = round(RGB_F[2] * 255);

}

void
RayCasting(int windowSize, int samples, const int* dims, const int ncells,
    const float* X, const float* Y, const float* Z, const float* F,
    double rayTermination, bool twodTransfer, unsigned char* data)
{

    Camera cam = SetupCamera();
    TransferFunction transfer = SetupTransferFunction(false);

    double stepSize = (cam.far - cam.near) / (samples - 1);
    double lookdir[] = { cam.focus[0] - cam.position[0],
                        cam.focus[1] - cam.position[1],
                        cam.focus[2] - cam.position[2] };
    Normalize(lookdir);

    double u[3];
    double v[3];

    CrossProduct(lookdir, cam.up, u);
    Normalize(u);

    CrossProduct(lookdir, u, v);
    Normalize(v);

    double delta = 2 * tan(cam.angle * M_PI / 180 / 2) / windowSize;
    double deltaX[] = { delta * u[0], delta * u[1], delta * u[2] };
    double deltaY[] = { delta * v[0], delta * v[1], delta * v[2] };


    unsigned char* pixelRGB = new unsigned char[3];
    for (int x = 0; x < windowSize; x++) {
        for (int y = 0; y < windowSize; y++) {

            Ray ray;
            ray.dir[0] = lookdir[0] + (2.0 * x + 1.0 - (double)windowSize) / 2.0 * deltaX[0] + (2.0 * y + 1.0 - (double)windowSize) / 2.0 * deltaY[0];
            ray.dir[1] = lookdir[1] + (2.0 * x + 1.0 - (double)windowSize) / 2.0 * deltaX[1] + (2.0 * y + 1.0 - (double)windowSize) / 2.0 * deltaY[1];
            ray.dir[2] = lookdir[2] + (2.0 * x + 1.0 - (double)windowSize) / 2.0 * deltaX[2] + (2.0 * y + 1.0 - (double)windowSize) / 2.0 * deltaY[2];
            ray.orig[0] = cam.position[0];
            ray.orig[1] = cam.position[1];
            ray.orig[2] = cam.position[2];

            Intersect(ray, samples, stepSize, transfer, cam.near, cam.far, dims, ncells, X, Y, Z, F, rayTermination, pixelRGB, twodTransfer);

            data[3 * x * windowSize + 3 * y] = pixelRGB[0];
            data[3 * x * windowSize + 3 * y + 1] = pixelRGB[1];
            data[3 * x * windowSize + 3 * y + 2] = pixelRGB[2];

            //cerr<<x<<" "<<y<<" completed "<<endl;

        }
    }

}




void
CreateImage(vtkImageData* image, const unsigned char data[], int size) {
    vtkNamedColors* colors = vtkNamedColors::New();

    unsigned int dim = size;

    image->SetDimensions(dim, dim, 1);
    image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);

    for (unsigned int x = 0; x < dim; x++)
    {
        for (unsigned int y = 0; y < dim; y++)
        {
            auto pixel =
                static_cast<unsigned char*>(image->GetScalarPointer(x, y, 0));

            for (auto i = 0; i < 3; ++i)
            {
                pixel[i] = data[3 * x * size + 3 * y + i];

            }

        }
    }

    image->Modified();

}




int
VolumeRender(int windowSize, int samples, double rayTermination, bool twodTransfer, char* filename)
{
    vtkDataSetReader* reader = vtkDataSetReader::New();
    cerr << "Reading the file" << endl;
    reader->SetFileName(filename);
    reader->Update();

    if (reader->GetOutput() == NULL || reader->GetOutput()->GetNumberOfCells() == 0)
    {
        cerr << "Could not find input file." << endl;
        exit(EXIT_FAILURE);
    }
    cerr << "Reading complete" << endl;

    int dims[3];
    vtkRectilinearGrid* rgrid = (vtkRectilinearGrid*)reader->GetOutput();
    rgrid->GetDimensions(dims);
    float* X = (float*)rgrid->GetXCoordinates()->GetVoidPointer(0);
    float* Y = (float*)rgrid->GetYCoordinates()->GetVoidPointer(0);
    float* Z = (float*)rgrid->GetZCoordinates()->GetVoidPointer(0);
    const char* arrayName = rgrid->GetPointData()->GetArrayName(0);
    vtkDataArray* weights = rgrid->GetPointData()->GetArray(arrayName);

    float* F = (float*)weights->GetVoidPointer(0);
    int ncells = rgrid->GetNumberOfCells();

    unsigned char* data = new unsigned char[3 * windowSize * windowSize];
    for (int i = 0; i < 3 * windowSize * windowSize; i += 3) {
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
    }

    double avgTime = 0;
    cerr << "Raycasting begins..." << endl;
    int i = 0;
    for (; i < 1; i++) {

        auto timerStart = std::chrono::high_resolution_clock::now();
        RayCasting(windowSize, samples, dims, ncells, X, Y, Z, F, rayTermination, twodTransfer, data);
        auto timerEnd = std::chrono::high_resolution_clock::now();
        avgTime += std::chrono::duration_cast<std::chrono::milliseconds>(timerEnd - timerStart).count();

    }
    cerr << " Running raycasting " << i << " times takes an average " << avgTime / i << " milliseconds" << endl;

    vtkNamedColors* colors = vtkNamedColors::New();
    vtkImageData* colorImage = vtkImageData::New();
    CreateImage(colorImage, data, windowSize);

    vtkImageMapper* imageMapper = vtkImageMapper::New();
    imageMapper->SetInputData(colorImage);
    imageMapper->SetColorWindow(windowSize);
    imageMapper->SetColorLevel(120);

    vtkActor2D* imageActor = vtkActor2D::New();
    imageActor->SetMapper(imageMapper);
    imageActor->SetPosition(windowSize / 2, windowSize / 2);

    vtkRenderer* ren = vtkRenderer::New();
    ren->AddActor2D(imageActor);

    vtkRenderWindow* renwin = vtkRenderWindow::New();
    renwin->SetSize(windowSize * 2, windowSize * 2);
    renwin->AddRenderer(ren);

    vtkRenderWindowInteractor* iren = vtkRenderWindowInteractor::New();
    iren->SetRenderWindow(renwin);

    renwin->Render();
    iren->Start();



    while (1);
}




int UnstructuredRendering() {

    vtkDataSetReader* reader = vtkDataSetReader::New();
    reader->SetFileName("astro64.vtk");
    reader->Update();

    if (reader->GetOutput() == NULL || reader->GetOutput()->GetNumberOfCells() == 0)
    {
        cerr << "Could not find input file." << endl;
        exit(EXIT_FAILURE);
    }

    vtkRectilinearGrid* rgrid = (vtkRectilinearGrid*)reader->GetOutput();
    vtkDataArray* scalars = rgrid->GetPointData()->GetScalars();

    //to unstructured grid
    vtkUnstructuredGrid* ugrid = vtkUnstructuredGrid::New();
    vtkPoints* points = vtkPoints::New();
    rgrid->GetPoints(points);
    ugrid->SetPoints(points);
    ugrid->GetPointData()->SetScalars(scalars);

    for (int i = 0; i < rgrid->GetNumberOfCells(); i++) {
        vtkCell* cell = rgrid->GetCell(i);
        ugrid->InsertNextCell(cell->GetCellType(), cell->GetPointIds());
    }



    vtkDataSetTriangleFilter* filter = vtkDataSetTriangleFilter::New();
    filter->SetInputData(ugrid);
    filter->Update();


    vtkOpenGLProjectedTetrahedraMapper* volumeMapper = vtkOpenGLProjectedTetrahedraMapper::New();
    volumeMapper->SetInputConnection(filter->GetOutputPort());
    vtkUnstructuredGrid* triugrid = filter->GetOutput();



    vtkPiecewiseFunction* opacityTransferFunction = vtkPiecewiseFunction::New();


    double opcityv[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.09, 0.11, 0.11, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.2, 0.2, 0.22, 0.23, 0.24, 0.24, 0.25, 0.26, 0.26, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.26, 0.26, 0.25, 0.24, 0.24, 0.23, 0.22, 0.2, 0.2, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.11, 0.11, 0.09, 0.09, 0.08, 0.08, 0.09, 0.11, 0.13, 0.15, 0.18, 0.2, 0.23, 0.26, 0.3, 0.33, 0.37, 0.41, 0.45, 0.5, 0.54, 0.58, 0.63, 0.67, 0.71, 0.74, 0.78, 0.8, 0.83, 0.85, 0.87, 0.87, 0.88, 0.88, 0.87, 0.86, 0.84, 0.82, 0.79, 0.76, 0.72, 0.68, 0.64, 0.6, 0.56, 0.51, 0.47, 0.43, 0.39, 0.35, 0.31, 0.27, 0.24, 0.21, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.07, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    double opacityi[] = { 10.0, 10.02, 10.04, 10.06, 10.08, 10.1, 10.12, 10.14, 10.16, 10.18, 10.2, 10.22, 10.24, 10.25, 10.27, 10.29, 10.31, 10.33, 10.35, 10.37, 10.39, 10.41, 10.43, 10.45, 10.47, 10.49, 10.51, 10.53, 10.55, 10.57, 10.59, 10.61, 10.63, 10.65, 10.67, 10.69, 10.71, 10.73, 10.75, 10.76, 10.78, 10.8, 10.82, 10.84, 10.86, 10.88, 10.9, 10.92, 10.94, 10.96, 10.98, 11.0, 11.02, 11.04, 11.06, 11.08, 11.1, 11.12, 11.14, 11.16, 11.18, 11.2, 11.22, 11.24, 11.25, 11.27, 11.29, 11.31, 11.33, 11.35, 11.37, 11.39, 11.41, 11.43, 11.45, 11.47, 11.49, 11.51, 11.53, 11.55, 11.57, 11.59, 11.61, 11.63, 11.65, 11.67, 11.69, 11.71, 11.73, 11.75, 11.76, 11.78, 11.8, 11.82, 11.84, 11.86, 11.88, 11.9, 11.92, 11.94, 11.96, 11.98, 12.0, 12.02, 12.04, 12.06, 12.08, 12.1, 12.12, 12.14, 12.16, 12.18, 12.2, 12.22, 12.24, 12.25, 12.27, 12.29, 12.31, 12.33, 12.35, 12.37, 12.39, 12.41, 12.43, 12.45, 12.47, 12.49, 12.51, 12.53, 12.55, 12.57, 12.59, 12.61, 12.63, 12.65, 12.67, 12.69, 12.71, 12.73, 12.75, 12.76, 12.78, 12.8, 12.82, 12.84, 12.86, 12.88, 12.9, 12.92, 12.94, 12.96, 12.98, 13.0, 13.02, 13.04, 13.06, 13.08, 13.1, 13.12, 13.14, 13.16, 13.18, 13.2, 13.22, 13.24, 13.25, 13.27, 13.29, 13.31, 13.33, 13.35, 13.37, 13.39, 13.41, 13.43, 13.45, 13.47, 13.49, 13.51, 13.53, 13.55, 13.57, 13.59, 13.61, 13.63, 13.65, 13.67, 13.69, 13.71, 13.73, 13.75, 13.76, 13.78, 13.8, 13.82, 13.84, 13.86, 13.88, 13.9, 13.92, 13.94, 13.96, 13.98, 14.0, 14.02, 14.04, 14.06, 14.08, 14.1, 14.12, 14.14, 14.16, 14.18, 14.2, 14.22, 14.24, 14.25, 14.27, 14.29, 14.31, 14.33, 14.35, 14.37, 14.39, 14.41, 14.43, 14.45, 14.47, 14.49, 14.51, 14.53, 14.55, 14.57, 14.59, 14.61, 14.63, 14.65, 14.67, 14.69, 14.71, 14.73, 14.75, 14.76, 14.78, 14.8, 14.82, 14.84, 14.86, 14.88, 14.9, 14.92, 14.94, 14.96, 14.98, 15.0 };
    for (int i = 0; i < 256; i++) {
        opacityTransferFunction->AddPoint(opacityi[i], opcityv[i]);
    }

    vtkColorTransferFunction* colorTransferFunction = vtkColorTransferFunction::New(); \
        colorTransferFunction->AddRGBPoint(10 + 0 * 5, 71.0 / 255.0, 71.0 / 255.0, 219.0 / 255.0);
    colorTransferFunction->AddRGBPoint(10 + 0.143 * 5, 0.0, 0.0, 91.0 / 255.0);
    colorTransferFunction->AddRGBPoint(10 + 0.285 * 5, 0, 255.0 / 255.0, 255.0 / 255.0);
    colorTransferFunction->AddRGBPoint(10 + 0.429 * 5, 0, 127.0 / 255.0, 0);
    colorTransferFunction->AddRGBPoint(10 + 0.571 * 5, 255.0 / 255.0, 255.0 / 255.0, 0);
    colorTransferFunction->AddRGBPoint(10 + 0.714 * 5, 255.0 / 255.0, 96.0 / 255.0, 0);
    colorTransferFunction->AddRGBPoint(10 + 0.857 * 5, 107.0 / 255.0, 0, 0);
    colorTransferFunction->AddRGBPoint(10 + 1 * 5, 224.0 / 255.0, 76.0 / 255.0, 76.0 / 255.0);

    vtkVolumeProperty* volumeProperty = vtkVolumeProperty::New();
    volumeProperty->SetColor(colorTransferFunction);
    volumeProperty->SetScalarOpacity(opacityTransferFunction);
    volumeProperty->SetScalarOpacityUnitDistance(300);
    volumeProperty->ShadeOff();

    vtkVolume* volume = vtkVolume::New();
    volume->SetMapper(volumeMapper);
    volume->SetProperty(volumeProperty);

    vtkRenderer* ren = vtkRenderer::New();
    ren->AddVolume(volume);

    vtkRenderWindow* renwin = vtkRenderWindow::New();
    renwin->AddRenderer(ren);

    vtkRenderWindowInteractor* iren = vtkRenderWindowInteractor::New();
    iren->SetRenderWindow(renwin);

    renwin->Render();
    iren->Start();


    while (1);

}

void
IntersectUnstructuredTetra(Ray ray, int samples, const double stepSize,
    TransferFunction transfer, const double near, const double far,
    const int ncells,
    vtkUnstructuredGrid* triugrid,
    double rayTermination, unsigned char* pixelRGB) {

    double samplePos[3];
    double RGB_F[] = { 0, 0, 0, 0 };

    vtkPoints* points = triugrid->GetPoints();
    const char* arrayName = triugrid->GetPointData()->GetArrayName(0);
    vtkDataArray* weights = triugrid->GetPointData()->GetArray(arrayName);

    for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++) {
        samplePos[0] = ray.dir[0] * stepSize * sampleIndex + ray.orig[0] + ray.dir[0] * near;
        samplePos[1] = ray.dir[1] * stepSize * sampleIndex + ray.orig[1] + ray.dir[1] * near;
        samplePos[2] = ray.dir[2] * stepSize * sampleIndex + ray.orig[2] + ray.dir[2] * near;

        //cerr<<samplePos[0]<<samplePos[1]<<samplePos[2]<<endl;
        for (int id = 0; id < ncells; id++) {
            vtkCell* cell = triugrid->GetCell(id);
            vtkIdList* point_ids = cell->GetPointIds();

            vtkIdType pt1 = point_ids->GetId(0);
            vtkIdType pt2 = point_ids->GetId(1);
            vtkIdType pt3 = point_ids->GetId(2);
            vtkIdType pt4 = point_ids->GetId(3);

            double x1[3];
            double x2[3];
            double x3[3];
            double x4[3];
            double pt1w = 0;
            double pt2w = 0;
            double pt3w = 0;
            double pt4w = 0;
            points->GetPoint(pt1, x1);
            points->GetPoint(pt2, x2);
            points->GetPoint(pt3, x3);
            points->GetPoint(pt4, x4);

            pt1w = weights->GetTuple1(pt1);
            pt2w = weights->GetTuple1(pt2);
            pt3w = weights->GetTuple1(pt3);
            pt4w = weights->GetTuple1(pt4);
            double bcoords[4] = { 0,0,0,0 };


            vtkTetra::BarycentricCoords(samplePos, x1, x2, x3, x4, bcoords);
            if (bcoords[0] > 0 && bcoords[1] > 0 && bcoords[2] > 0 && bcoords[3] > 0) {


                unsigned char* RGB = new unsigned char[3];
                double opacity = 0;
                double c = pt1w * bcoords[0] + pt2w * bcoords[1] + pt3w * bcoords[2] + pt4w * bcoords[3];
                transfer.ApplyTransferFunctionTetra(c, RGB, opacity);
                double r = RGB[0];
                double g = RGB[1];
                double b = RGB[2];



                //alpha correction 
                double corrected = 1 - pow((1 - opacity), 500.0 / samples);

                RGB_F[0] = r / 255.0 * (1 - RGB_F[3]) * corrected + RGB_F[0];
                RGB_F[1] = g / 255.0 * (1 - RGB_F[3]) * corrected + RGB_F[1];
                RGB_F[2] = b / 255.0 * (1 - RGB_F[3]) * corrected + RGB_F[2];
                RGB_F[3] = (1 - RGB_F[3]) * corrected + RGB_F[3];

                if (RGB_F[3] > rayTermination) {
                    break;
                }

            }


        }



    }
    pixelRGB[0] = round(RGB_F[0] * 255);
    pixelRGB[1] = round(RGB_F[1] * 255);
    pixelRGB[2] = round(RGB_F[2] * 255);



}




void
RayCastingUnstructuredTetra(int windowSize, int samples, const int ncells,
    vtkUnstructuredGrid* triugrid, double rayTermination, unsigned char* data) {
    Camera cam = SetupCameraTetra();
    TransferFunction transfer = SetupTransferFunctionTetra();

    double stepSize = (cam.far - cam.near) / (samples - 1);
    double lookdir[] = { cam.focus[0] - cam.position[0],
                        cam.focus[1] - cam.position[1],
                        cam.focus[2] - cam.position[2] };
    Normalize(lookdir);

    double u[3];
    double v[3];

    CrossProduct(lookdir, cam.up, u);
    Normalize(u);

    CrossProduct(lookdir, u, v);
    Normalize(v);

    double delta = 2 * tan(cam.angle * M_PI / 180 / 2) / windowSize;
    double deltaX[] = { delta * u[0], delta * u[1], delta * u[2] };
    double deltaY[] = { delta * v[0], delta * v[1], delta * v[2] };



    unsigned char* pixelRGB = new unsigned char[3];
    for (int x = 0; x < windowSize; x++) {
        for (int y = 0; y < windowSize; y++) {

            Ray ray;
            ray.dir[0] = lookdir[0] + (2.0 * x + 1.0 - (double)windowSize) / 2.0 * deltaX[0] + (2.0 * y + 1.0 - (double)windowSize) / 2.0 * deltaY[0];
            ray.dir[1] = lookdir[1] + (2.0 * x + 1.0 - (double)windowSize) / 2.0 * deltaX[1] + (2.0 * y + 1.0 - (double)windowSize) / 2.0 * deltaY[1];
            ray.dir[2] = lookdir[2] + (2.0 * x + 1.0 - (double)windowSize) / 2.0 * deltaX[2] + (2.0 * y + 1.0 - (double)windowSize) / 2.0 * deltaY[2];
            ray.orig[0] = cam.position[0];
            ray.orig[1] = cam.position[1];
            ray.orig[2] = cam.position[2];

            IntersectUnstructuredTetra(ray, samples, stepSize, transfer, cam.near, cam.far, ncells, triugrid, rayTermination, pixelRGB);

            data[3 * x * windowSize + 3 * y] = pixelRGB[0];
            data[3 * x * windowSize + 3 * y + 1] = pixelRGB[1];
            data[3 * x * windowSize + 3 * y + 2] = pixelRGB[2];



        }
    }

    // cerr <<"delta"<<delta<<endl;
    // cerr<<"deltaX: ";
    // for(auto i : deltaX){
    //     cerr<<i<<" ";
    // }

    // cerr<<"ru: ";
    // for(auto i : u){
    //     cerr<<i<<" ";
    // }
    // cerr<<endl;

    // cerr<<"rv: ";
    // for(auto i : v){
    //     cerr<<i<<" ";
    // }
    // cerr<<endl;
    // cerr<<"rx: ";
    // for(auto i : deltaX){
    //     cerr<<i<<" ";
    // }
    // cerr<<endl;
    // cerr<<"ry: ";
    // for(auto i : deltaY){
    //     cerr<<i<<" ";
    // }
    // cerr<<endl;

    // cerr<<"ray origin: ";
    // for(auto i : cam.position){
    //     cerr<<i<<" ";
    // }
    // cerr<<endl;
    // cerr<<"look direction: ";
    // for(auto i : lookdir){
    //     cerr<<i<<" ";
    // }
    // cerr<<endl;



}


int UnstructuredRenderingTetra(int windowSize, int samples, double rayTermination) {

    vtkNew<vtkXMLUnstructuredGridReader> reader;
    reader->SetFileName("tetra1.vtu");
    reader->Update();
    if (reader->GetOutput() == NULL || reader->GetOutput()->GetNumberOfCells() == 0)
    {
        cerr << "Could not find input file." << endl;
        exit(EXIT_FAILURE);
    }
    vtkUnstructuredGrid* triugrid = (vtkUnstructuredGrid*)reader->GetOutput();
    const char* arrayName = triugrid->GetPointData()->GetArrayName(0);
    vtkDataArray* weights = triugrid->GetPointData()->GetArray(arrayName);

    // vtkUnstructuredGrid * triugrid = filter->GetOutput ();
    int ncells = triugrid->GetNumberOfCells();


    unsigned char* data = new unsigned char[3 * windowSize * windowSize];
    for (int i = 0; i < 3 * windowSize * windowSize; i += 3) {
        data[i] = 128;
        data[i + 1] = 0;
        data[i + 2] = 0;
    }

    double avgTime = 0;
    cerr << "Raycasting begins..." << endl;

    auto timerStart = std::chrono::high_resolution_clock::now();
    RayCastingUnstructuredTetra(windowSize, samples, ncells, triugrid, rayTermination, data);
    auto timerEnd = std::chrono::high_resolution_clock::now();
    avgTime += std::chrono::duration_cast<std::chrono::milliseconds>(timerEnd - timerStart).count();

    cerr << " Running raycasting times takes " << avgTime << " milliseconds" << endl;

    vtkNamedColors* colors = vtkNamedColors::New();
    vtkImageData* colorImage = vtkImageData::New();
    CreateImage(colorImage, data, windowSize);

    vtkImageMapper* imageMapper = vtkImageMapper::New();
    imageMapper->SetInputData(colorImage);
    imageMapper->SetColorWindow(windowSize);
    imageMapper->SetColorLevel(128);

    vtkActor2D* imageActor = vtkActor2D::New();
    imageActor->SetMapper(imageMapper);
    imageActor->SetPosition(windowSize / 2, windowSize / 2);

    vtkRenderer* ren = vtkRenderer::New();
    ren->AddActor2D(imageActor);

    vtkRenderWindow* renwin = vtkRenderWindow::New();
    renwin->SetSize(windowSize * 2, windowSize * 2);
    renwin->AddRenderer(ren);

    vtkRenderWindowInteractor* iren = vtkRenderWindowInteractor::New();
    iren->SetRenderWindow(renwin);

    renwin->Render();
    iren->Start();

    while (1);

}

void
buildTetra() {
    vtkNew<vtkNamedColors> colors;

    std::string filename = "tetra1.vtu";

    vtkNew<vtkPoints> points;
    points->InsertNextPoint(0, 0, 0);
    points->InsertNextPoint(1, 0, 0);
    points->InsertNextPoint(1, 1, 0);
    points->InsertNextPoint(0, 1, 1);

    vtkNew<vtkTetra> tetra;

    tetra->GetPointIds()->SetId(0, 0);
    tetra->GetPointIds()->SetId(1, 1);
    tetra->GetPointIds()->SetId(2, 2);
    tetra->GetPointIds()->SetId(3, 3);

    vtkNew<vtkCellArray> cellArray;
    cellArray->InsertNextCell(tetra);

    vtkNew<vtkUnstructuredGrid> unstructuredGrid;
    vtkDoubleArray* scalars = vtkDoubleArray::New();
    scalars->SetNumberOfComponents(1);
    scalars->SetNumberOfTuples(4);
    scalars->SetComponent(0, 0, 0);
    scalars->SetComponent(1, 0, 5);
    scalars->SetComponent(2, 0, 0);
    scalars->SetComponent(3, 0, 12);
    unstructuredGrid->SetPoints(points);
    unstructuredGrid->GetPointData()->SetScalars(scalars);
    unstructuredGrid->SetCells(VTK_TETRA, cellArray);


    // Write file
    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    writer->SetFileName(filename.c_str());
    writer->SetInputData(unstructuredGrid);
    writer->Write();


}



int main() {
    int windowSize = 100;
    int samples = 1000;
    bool twodTransfer = 0; //2d transfer function 
    double rayTermination = 0.99; //change thresholds here

    char* filename = "astro512_ascii.vtk";



    VolumeRender(windowSize, samples, rayTermination, twodTransfer, filename);


    // first build a tetrahedron, then render it 
    //buildTetra();
    //UnstructuredRenderingTetra(windowSize, samples, 1);



    //unstructured tetrahedron rendering with vtkOpenGLProjectedTetrahedraMapper and vtkVolume on astro64
    //UnstructuredRendering();



}