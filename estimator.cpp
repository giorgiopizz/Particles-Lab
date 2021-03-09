/*
c++ -o estimator estimator.cpp `root-config --cflags --glibs`
*/


#include <iostream>
#include <fstream>
#include <cmath>
//#include <math.h>
#include <string>
#include <sstream>

#include "TStyle.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TH1F.h"
#include "TAxis.h"
#include "TFitResult.h"
#include "TMatrixDSym.h"
#include "TMath.h"
#include "TRandom3.h"


using namespace std;


void set_parameters(TF1 * func){
    func->SetParameter(0,35);
    func->SetParameter(1,2);
    func->SetParameter(2,0);
    func->SetParameter(3,35);
    func->SetParameter(4,2);
}

void histo_populate(TH1F * h, int n_tot, TF1 * func){
    TRandom3 r = new TRandom();
    double width = (max-min)/bin;
    for(int i=0;i<bin;i++){
        nu = n_tot * func->Integrate(min+i*width,min+(i+1)*width);
        h->Fill(r->Poisson(nu));
    }
    return
}

int main(int argc, char** argv)
{


    TApplication* myApp = new TApplication("myApp", NULL, NULL);

    TF1 * func = new TF1("fun", "[0]*TMath::Exp(-x[0]/[1])+[2]*TMath::Exp(-x[0]/[3]) + [4]",0, 11);
    set_parameters();

    std::vector<double> tau;
    std::vector<double> chi_square;
    for(int i=0;i<2;i++){
        // generate each pseudoexperiment
        // in a single experiment we populate an histogram with the histo_populate
        // method, we then fit this histogram in order to estimate tau,
        double min=0, max=10;
        int bin = 10;
        TH1F *h = new TH1F("h", "example histogram",bin,min,max);
        set_parameters();
        histo_populate(h,10000,func);
        auto result = h->Fit(func, "S");
        tau.push_back(result->GetParameter(1));
        chi_square.push_back(result->Chi2());
  }
