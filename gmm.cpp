#include <iostream>
#include <random>
#include <ctime>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;

const double PI = 3.14159;
double getsum(double** data, int size, int dim);
double* calMean(double** data, int size, int dim);
double* calSD(double** data, int size, double* mean, int dim);
double* findMinMax(double** data, int size, int dim);
double fRand(double fMin, double fMax);
double** generateMix(int gmmnum, int dim);

double ran0to1();
double* generateDataset(int size);
void writeLoglike(double loglike);

double* getsumMulti(double** weightarr, int datasize, int dim);
double*  checkSumto1(double* weights, int size);

class Gaussian{
    public:
        Gaussian(double* mean, double* sd, int dimensions, double* pilist){
            mu = mean;
            sigma = sd;
            dim = dimensions;
            mix = pilist;
        }
        void setValues (double* mean, double* sd){
            mu = mean;
            sigma = sd;
        }
        double* calProbDensity(double* datum){
            double* result = new double[dim];
            for(int j=0; j<dim; j++){
                double u = (datum[j] - mu[j]) / abs(sigma[j]);
                result[j] = (1/(sqrt(2 * PI) * abs(sigma[j]))) * exp(-u* u / 2);
            }
            return result;
        }
        double* logProbDensity(double* datum){
            double* result = new double[dim];
            for(int j=0; j<dim; j++){
                //(-size(mu,1)/2*log(2*pi)-size(mu,1)/2*log(sig_sq)-0.5*sum((y-repmat(mu,1,size(y,2))).^2)./sig_sq);
                double r1 = -(dim/2)*log(2*mix[j]); //(-dim/2*log(2*pi))    //(dim*n/2)?
                double r2 = -(dim/2)*(log(pow(sigma[j],2))); //(-dim/2*log(sigma))
                double r3 = -(1/2) * pow((datum[j] - mu[j])/(sigma[j]),2);  //<-----?
                result[j] = r1+r2+r3;
            }
            return result;
        }
        void print(int index){
            cout << "Gaussian: " << index << " | ";
            for(int j=0; j<dim; j++){
                cout << "mean: " << mu[j] << ", ";
            }
            for(int j=0; j<dim; j++){
                cout << "sd: " << sigma[j] << " , ";
            }
            cout << endl;
        }
    private:
        double* mu; //mean
        double* sigma; //sigma square
        int dim;
        double* mix;

};

//GaussianMixture(2, arr, size, mix); num of gmm, data, datasize, portion
class GaussianMixture{
    public:
        GaussianMixture(int num, double** dataset, int datasetsize, int datadim){   
            gmmnum = num;
            dim = datadim;
            data = dataset;
            datasize = datasetsize;
            //generate a list of pi value that sum to one
            //cout << "generating gaussian mixture... "<< endl;
            mix = generateMix(gmmnum, dim);
            //find min and max datapt value
            double* minmax = findMinMax(dataset, datasetsize, dim); 
            mumin = minmax[0];
            mumax = minmax[1];
            sigmamin = 0;
            sigmamax = abs(minmax[1]-minmax[0])/gmmnum; 
            
            generateDistribution(); //set the vector<Gaussian>
        }
        //~GaussianMixture(); //// destructor????
        void generateDistribution();
        void EMstep();
        void EMloop();
        double getLoglike(){ return loglike; }
        void train();
        void printGaussianInfos();
        
    private:
        double** data; //a copy of datas
        int datasize; // length of the set
        int dim;//dimention of the dataset
        double mumin; //min of data
        double mumax; //max of data
        double sigmamin; //min sd
        double sigmamax; //max sd
        int gmmnum; // number of gmm
        vector<Gaussian> gmms; //save gmms
        double** mix; //pi portion
        double loglike;

};


void GaussianMixture::generateDistribution(){
    srand(time(NULL));
    vector<Gaussian> gmmlist;
    for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
        double* mulist = new double[dim];
        double* sigmalist = new double[dim];
        double* pilist = mix[gmmindex];
        for(int j=0; j<dim; j++){
            mulist[j] = fRand(mumin,mumax);
            sigmalist[j] = fRand(sigmamin,sigmamax);
        }

        Gaussian g1 = Gaussian(mulist, sigmalist, dim, pilist);
        gmmlist.push_back(g1);
    }
    //cout << "size check: " << gmmlist.size() << endl;
    gmms = gmmlist;
}

void GaussianMixture::EMloop(){
    int notconverged = 1;
    int iterationsize = 1000;
    double* likelihood = new double[iterationsize];
    likelihood[0] = -1e308;
    int counter = 1;
    //initalize the weight array with num of gaussians, 
    //for each datapoint in the gaussian model, filled with gammahat in different dimensions
    double*** gammahat = new double**[gmmnum]; //double[gmmindex][datapt][dim]
    //while(notconverged){ //go until convergence
        counter += 1; //increment counter
        if(counter > iterationsize){ //if we have filled up likelihood storage, make it twice as big
            double* newlikelihood = new double[iterationsize*2];
            copy(likelihood, likelihood + std::min(iterationsize, iterationsize*2), newlikelihood);
            delete[] likelihood;
            likelihood = newlikelihood;
            cout << "curr iteration: " << counter << endl; //periodically report progress
        }
        //Mixture of Gaussians Expectation step
        //loop over the gaussians
        for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
            gammahat[gmmindex] = new double*[datasize];
            //get responsibility numerators for A PARTICULAR gaussian (row) and for every data point (column)
            for(int i=0; i<datasize; i++){ //loop every data pt
                gammahat[gmmindex][i] = new double[dim];
                double* tempgamma = gmms[gmmindex].logProbDensity(data[i]);
                for(int j=0; j<dim; j++){
                    gammahat[gmmindex][i][dim] = mix[gmmindex][j]* exp(tempgamma[j]);
                    cout << "gammahat: "<< gammahat[gmmindex][i][dim]  << endl;
                }
            }
        }
        // sum over all gaussians for that data point
        double** den = new double*[datasize];
        for(int i=0; i<datasize;i++){
            den[i] = new double[dim];
            for(int j=0; j<dim; j++){
                double tempden = 0;
                for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
                    double tempgamma = gammahat[gmmindex][i][j];
                    tempden += tempgamma;
                }
                cout << "den: " <<tempden << endl;
                den[i][j] = tempden;

            }
        }

        //divide every numerator by the sum
        for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
            for(int i=0; i<datasize; i++){ 
                for(int j=0; j<dim; j++){
                    gammahat[gmmindex][i][dim] /= den[i][j];
                }
            }
        }



    //}

}

void GaussianMixture::EMstep(){
    cout << "---EM starting..." << endl;
    double*** weightarr = new double**[gmmnum];
    
    double loglikelihood = 0;
    double likelitest = 0;
    
    //in the e step for every data pt
    for(int i=0; i< datasize; i++){ 
        double* den = new double[dim];
        double denvalue = 0;
        //compute the weight for gaussians
        //compute denominator
        for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
            double* weight = gmms[gmmindex].calProbDensity(data[i]); 
            for(int j=0; j<dim; j++){
                double weightvalue = weight[j]*mix[gmmindex][j];
                denvalue += weightvalue;//compute denominator
                weightarr[gmmindex][i][j] = weightvalue;
            }
            for(int j=0; j<dim; j++){
                den[j] = denvalue;
                cout << "den value: " <<  den[j] << endl;
            }
        }

        double likelihood = 0;
        for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
            for(int j=0; j<dim; j++){
                weightarr[gmmindex][i][j] /= den[j];//normalize
                likelihood += weightarr[gmmindex][i][j]; //compute likelihood 
            }
        }
        loglikelihood += log(likelihood); //sum(log(pi*pdf(yi)for(gmm1-n))
        //cout <<"|| likelihood check: " << loglikelihood << endl;

    }


    cout << "|| -----log likelihood so far: "<< loglikelihood << endl;
    //cout << "|| -----log likelihood so far v2: "<< loglikelihood_v2 << endl;

    loglike = likelitest;

    //get the sum of weights (sum gamma)
    double** weightsum = new double*[gmmnum];
    for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
        for(int j=0;j<dim;j++){
            weightsum[gmmindex] = getsumMulti(weightarr[gmmindex], datasize, dim); 
        }
    }
 
    //loop through weight1 arr zip
    double** muhat = new double*[gmmnum];
    double** sdhat = new double*[gmmnum];
    double** pihat = new double*[gmmnum];
    //cal new mu(mean)
    for(int i=0; i<datasize; i++){
        for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
            for(int j=0;j<dim;j++){
                muhat[gmmindex][j] += weightarr[gmmindex][i][j]*data[i][j] / weightsum[gmmindex][j];
            }
        }
    }
    //cal new sigma(sd)
    for(int i=0; i<datasize; i++){
        for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
            for(int j=0;j<dim;j++){
                sdhat[gmmindex][j] += weightarr[gmmindex][i][j]*pow((data[i][j]-muhat[gmmindex][j]),2) / weightsum[gmmindex][j];
            }
        }     
    }
    //take sqrt of the sdhat
    for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
        for(int j=0; j<dim; j++){
            sdhat[gmmindex][j] = sqrt(sdhat[gmmindex][j]);
        }
    }

    //cal new mix(pi)
    for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
        for(int j=0; j<dim; j++){
            pihat[gmmindex][j] = weightsum[gmmindex][j]/datasize;
            cout << "check mix detail: "<< weightsum[gmmindex][j] << " " << datasize << " | ";
            cout << "check mix: " << pihat[gmmindex][j] << " ";
        }
        cout << endl;
    }

    //update the parameters
    for(int gmmindex=0; gmmindex<gmmnum; gmmindex++){
        mix[gmmindex] = pihat[gmmindex];
        gmms[gmmindex].setValues(muhat[gmmindex],sdhat[gmmindex]);//update the gaussian mu and sd
    }

    //check pi(mix) sum to 1
    for(int j=0;j<dim;j++){
        mix[j] = checkSumto1(mix[j], dim);
    }
    //print the two gaussians infos
    //printGaussianInfos();
    
}

void GaussianMixture::printGaussianInfos(){
    //cout << "--------print infos--------" << endl;
    for(int i=0; i<gmms.size(); i++){
        gmms[i].print(i);
        for(int j=0; j<dim; j++){
            cout << "mix: " << mix[i][j] << " ";   
        }
        cout << " | "<< endl;
    }  
}

int main(int argc, char *argv[]){ 
    // double arr[20] = {-2.46,-1.925,-1.308,-1.997 ,-1.573,-1.521,-2.393,-2.25,-1.967,-1.553,1.4580e+0
    // ,1.2760e+0,1.1520e+0,7.5050e-1,1.1990e+0,1.3110e+0,1.1120e-1 ,8.3650e-1,5.0340e-1,6.8690e-1 };
    // int size = 20;
    // int data_dim = 1;
    int size = 10;
    int dim = 2;
    // double arr[10][2] = 
    // {{-2.46,-1.925}
    // ,{-1.308,-1.997} 
    // ,{-1.573,-1.521}
    // ,{-2.393,-2.25}
    // ,{-1.967,-1.553}
    // ,{1.4580e+0,1.2760e+0}
    // ,{1.1520e+0,7.5050e-1}
    // ,{1.1990e+0,1.3110e+0}
    // ,{1.1120e-1 ,8.3650e-1}
    // ,{5.0340e-1,6.8690e-1}};
    cout << "generate data: " << endl;
    double** arr = new double*[size];
    for(int i=0; i<size; i++){
        arr[i] = new double[dim];
        for(int j=0; j<dim; j++){
            arr[i][j] = ran0to1();
            //cout << arr[i][j]<<" ";
        }
        //cout << " " << endl;
    }
    //double* arr = generateDataset(size);
    cout << "--- GMM --- " << endl;
    double* minmax = findMinMax(arr, size, dim); //for mean
    double* datamean = calMean(arr,size, dim); 
    double* datasd = calSD(arr, size, datamean, dim);
    //print relevent data infos
    // for(int j=0; j<dim; j++){
    //     cout << "cal data mean: " <<datamean[j] << endl;
    //     cout << "data standard div: "  << datasd[j] << endl;
    // }
    //cout << "data min: "<< minmax[0] << " | max: " << minmax[1] << endl;

    int iterations = 5;
    int gaussian_num = 4; 
    GaussianMixture gm = GaussianMixture(gaussian_num, arr, size, dim);
    cout << "---initial guess---" << endl;
    gm.printGaussianInfos();
    cout << "---print info ended---" << endl;
    cout << endl;
    gm.EMloop();

    // GaussianMixture best_gm = GaussianMixture(gaussian_num, arr, size, dim);
    // for(int i=0;i<iterations;i++){
    //     cout << "| --- iteration " << i << endl;
    //     gm.EMstep();
    //     if(gm.getLoglike() >= loglikelihood){
    //         best_gm = gm;
    //     }
    //     //cout << gm.getLoglike() << endl;
    //     writeLoglike(gm.getLoglike());
    // }
    // cout << "final--------" << endl;



    return 0;
}



//----------- UTILITY FUNCTION -----------------------------
double* getsumMulti(double** weightarr, int datasize, int dim){
    double* result = new double[dim];
    double sum = 0;
    for(int j=0;j<dim;j++){
        for(int i=0;i<datasize;i++){
            sum += weightarr[i][j];
        }
        result[j] = sum;
    }
    return result;
}
double* checkSumto1(double* weights, int size){
    double sum = 0.0;
    //take the sum
    for(int i=0; i<size; i++){
        sum += weights[i];
    }
    for(int i=0; i<size; i++){
        weights[i] /= sum;
        cout << "in checking fun... " <<weights[i] << endl;
    }
    return weights;
}
double** generateMix(int num, int dim){ //gausiaan num 
    double** mix = new double*[num];
    //create a 2d array with height num, width dim
    //fill the array with ran nums from 0-1
    for(int i=0;i<num; i++){
        mix[i] = new double[num];
        for(int j=0; j<dim; j++){
            //double ran = ran0to1(); //randomly initialize
            double ran = 1.0/(num); //initialize by even portion
            // cout << ran << endl;
            mix[i][j] = ran;
        }
    }
    /*
    //randomly initialize the pi
    //extract the every first/second/third etc dimnum of the outter gmmnum array 
    //[[dim1,dim2], [dim1,dim2],...] => [[dim1,dim1,...],[dim2,dim2...]]
    double** extractlist = new double*[dim];
    for(int j=0;j<dim; j++){
        extractlist[j] = new double[num];
        for(int i=0; i<num; i++){
            extractlist[j][i] = mix[i][j];
        }
    }
    for(int j=0;j<dim;j++){ 
        extractlist[j] = checkSumto1(extractlist[j], dim);
    }

    for(int j=0; j<dim; j++){
        for(int i=0;i<num;i++){
            mix[i][j] = extractlist[j][i];
        }
    }

    cout << "initial mix(pi): ";
    for(int i=0; i<num; i++){
        for(int j=0;j<dim;j++){
            cout << mix[i][j] << " ";
        }
        cout << " | ";
    }
    */

    return mix;
}
double getsum(double** arr, int size, int dim){
    double result = 0.0;
    for(int i=0; i< size; i++){
        for(int j=0; j<dim; j++){
            result += arr[i][j];
        }
    }
    return result;
}
double* calMean(double** data, int size, int dim){
    double result = 0.0;
    double* resultarr = new double[dim]; //each index for each dimension
    for(int j=0; j<dim; j++){
        double result = 0.0;
        for(int i=0; i<size; i++){
            result += data[i][j];
        }
        resultarr[j] = result/size;
    }
    return resultarr;
}
double* calSD(double** data, int size, double* mean, int dim){
    double** sd = new double*[size];
    for(int i=0; i<size; i++){
        sd[i] = new double[dim];
        for(int j=0; j<dim; j++){
            sd[i][j] = pow(data[i][j] - mean[j], 2);
        }
    }
    double* result = calMean(sd, size, dim);
    for(int j=0; j<dim; j++){
        result[j] = sqrt(result[j]);
    }
    return result;
}
double* findMinMax(double** data, int size, int dim){
    double min = data[0][0];
    double max = data[0][0];
    for(int i=0; i<size; i++){
        for(int j=0; j<dim; j++){
            if(data[i][j] >= max){
                max = data[i][j];
            }
            if(data[i][j] <= min){
                min = data[i][j];
            }
        }
    }
    double* result = new double[2]; //a new array just for the two num
    result[0] = min;
    result[1] = max;
    return result;
}
double fRand(double fMin, double fMax){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<float> dis(fMin*100, fMax*100);
    double ran = dis(gen)/100;
    //double f = (double)rand() / RAND_MAX;
    //return fMin + f * (fMax - fMin);
    return ran;
}
double ran0to1(){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<float> dis(1, 100);
    double ran = dis(gen)/100;
    return ran;
}

void writeLoglike(double loglike){
    string filename1 = "gmm-loglike-";
	string filename2 = ".csv";
	string filename = filename1+filename2;
    ofstream outputfile;
    outputfile.open(filename,fstream::app);
	outputfile << loglike << "\n";
	outputfile.close();
}

double* generateDataset(int size){
    double* dataset = new double[size];
    double mix = 0.5;
    vector<double> mu;
    mu.push_back(200.0);
    mu.push_back(10.0);
    vector<double> sigma;
    sigma.push_back(8);
    sigma.push_back(3);
    double count = 0;
    for(int i=0; i<size; i++){
        double ran = ran0to1();
        double ran1 = ran0to1();
        double whichtype = 0;
        if(ran<mix){
            count += 1;
            double whichtype = 1;

        }
        dataset[i] = ran1*sigma[whichtype] + mu[whichtype];
        //cout << dataset[i] << endl;
    }
    // double testmean = calMean(dataset, size);
    // cout << "test gene data mean: " << testmean << endl;
    // cout << "test gene data sd: " << calSD(dataset, size, testmean) << endl;
    // cout << "test gene data mix: " << count/size << endl;
    return dataset;
}





