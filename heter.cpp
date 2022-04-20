#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<sstream>
#include<cstring>
#include<algorithm>
#include<map>
#include<stdio.h>
#include<getopt.h>
#include<cmath>
#include<Eigen/Dense>
#include<set>
#include<unordered_set>
#include <omp.h>
#include <thread>
#include "bgen/bgen.h"
#include <boost/algorithm/string.hpp>
#include <boost/math/statistics/univariate_statistics.hpp>

#define ITMAX1 100
#define ITMAX2 50
#define EPS1 1.0e-9
#define EPS2 1.0e-8
#define FPMIN 1.0e-30
#define RESSIZE 14
#define GRIDNUM 10
using namespace std;
using namespace Eigen;
using namespace boost;
using namespace boost::math::statistics;

void cover()
{
	cout << endl;
	cout << "+=======================================+" << endl;
	cout << "|					|" << endl;
	cout << "|		heter			|" << endl;
	cout << "|		version 1.0.0		|" << endl;
	cout << "|					|" << endl;
	cout << "+=======================================+" << endl;
}

void help()
{
	cover();
	cout << "--bgen\t\t" << "Input genotype file in bgen format." << endl;
	cout << "--sample\t" << "Input sample file in SNPTEST sample file format. Missing data is coded as \"NA\"." << endl;
	cout << "--out\t\t" << "Output file name [default: heter]." << endl;
	cout << "--pheno\t\t" << "Specify the continous phenotype for analysis." << endl;
	cout << "--covs\t\t" << "Specify the covariate(s) to be adjusted in the analysis." << endl;
	cout << "--vc\t\t" << "Conduct the analysis with the variance component model." << endl;
	cout << "--mixed\t\t" << "Conduct the analysis with the linear mixed model." << endl;
	cout << "--reg\t\t" << "Conduct the analysis by the linear regression." << endl;
	cout << "--creg\t\t" << "Conduct the analysis by the chi-square regression." << endl;
	cout << "--lev\t\t" << "Conduct the analysis by the Levene's (Brown-Forsythe) test." << endl;
	cout << "--maf-min\t" << "Minimum minor allele frequency of the SNPs for the analysis [default: 0.01]." << endl;
	cout << "--thread-num\t" << "Number of threads on which the program will be running [default: thread number in your machine - 1]." << endl;
	cout << "--batch-size\t" << "Number of SNPs to be processed in a batch [default: 1000]." << endl;
}

class Dataset {

private:
const char* delimiter = " \t,";
const char* miss = "NA";
const double cof[6] = { 76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 };
int pheno_pos, covs_num, covs_C_num, covs_D_num, covs_D_col_num, X_col_num, sample_num, nomiss_sample_num, bgenSample_num, snp_num, batch_size, maf_filterred_num;
string pheno_name, pheno_type, bgen_file, sample_file, out_file, rs_file, arg[30], this_snp;
vector<string> covs_name, covs_type, covs_C_name, covs_D_name, SNPID, rsid, chromosome, allele1, allele2;
vector<int> covs_pos, covs_C_pos, covs_D_pos, position;
vector<bool> miss_flag;
vector<vector<string>> covs_D_val;
vector<vector<int>> covs_D_cnt;
VectorXd y, Py, r, G_mean, tau_grid;
VectorXi maf_flag;
MatrixXd X, P, Xt, batch_result;
double r2, reg_df, like0, tau1_start, tau1_end, tau2_start, tau2_end, tau3_start, tau3_end, grid_if, maf_min, freq_min, freq_max;
int tau1_num, tau2_num, tau3_num;
bool grid_flag = false;
set<string> rs_set;
	
public:
int thread_num, flag[30]{};
bool error_flag = false;
ofstream logFile;

Dataset(int argc, char *argv[]) 
{
	int opt, longindex, max_thread_num;
	const char *optstring = "";
	struct option long_options[] =
	{
		{ "bgen", required_argument,  NULL, 0},
		{ "sample", required_argument,  NULL, 1},
		{ "out", required_argument, NULL, 2},
		{ "pheno", required_argument, NULL, 3},
		{ "covs", required_argument, NULL, 4},
		{ "thread-num", required_argument, NULL, 5},
		{ "batch-size", required_argument, NULL, 6},
		{ "reg", no_argument, NULL, 7},
		{ "vc", no_argument, NULL, 8},
		{ "mixed", no_argument, NULL, 9},
		{ "rs-id", required_argument, NULL, 10},
		{ "tau1-start", required_argument, NULL, 11},
		{ "tau1-end", required_argument, NULL, 12},
		{ "tau1-num", required_argument, NULL, 13},
		{ "tau2-start", required_argument, NULL, 14},
		{ "tau2-end", required_argument, NULL, 15},
		{ "tau2-num", required_argument, NULL, 16},
		{ "tau3-start", required_argument, NULL, 17},
		{ "tau3-end", required_argument, NULL, 18},
		{ "tau3-num", required_argument, NULL, 19},
		{ "auto-grid", no_argument, NULL, 20},
		{ "grid-if", required_argument, NULL, 21},
		{ "rs-file", required_argument, NULL, 22},
		{ "creg", no_argument, NULL, 23},
		{ "lev", no_argument, NULL, 24},
		{ "maf-min", required_argument, NULL, 25},
		{ "help", no_argument, NULL, 26},
		{0, 0, 0, 0}
	};

	while ((opt = getopt_long(argc, argv, optstring, long_options, &longindex)) != -1)
	{
		switch (opt)
		{
		case 0: flag[0] = 1; arg[0] = optarg; bgen_file = optarg; break;
		case 1: flag[1] = 1; arg[1] = optarg; sample_file = optarg; break;
		case 2: flag[2] = 1; arg[2] = optarg; out_file = optarg; break;
		case 3: flag[3] = 1; arg[3] = optarg; pheno_name = optarg; break;
		case 4: 		
		{
			flag[4] = 1;
			covs_name.push_back(optarg); 
			for (int i = optind; i < argc; i++) 
			{
				if (argv[i][0] == '-') break;
				else covs_name.push_back(argv[i]);
			}
			covs_num = covs_name.size();
			break;
		}
		case 5: flag[5] = 1; arg[5] = optarg; thread_num = atoi(optarg); break;
		case 6: flag[6] = 1; arg[6] = optarg; batch_size = atoi(optarg); break;
		case 7: flag[7] = 1; break;
		case 8: flag[8] = 1; break;
		case 9: flag[9] = 1; break;
		case 10: flag[10] = 1; arg[10] = optarg; this_snp = optarg; break;
		case 11: flag[11] = 1; arg[11] = optarg; tau1_start = atof(optarg); break;
		case 12: flag[12] = 1; arg[12] = optarg; tau1_end = atof(optarg); break;
		case 13: flag[13] = 1; arg[13] = optarg; tau1_num = atoi(optarg); break;
		case 14: flag[14] = 1; arg[14] = optarg; tau2_start = atof(optarg); break;
		case 15: flag[15] = 1; arg[15] = optarg; tau2_end = atof(optarg); break;
		case 16: flag[16] = 1; arg[16] = optarg; tau2_num = atoi(optarg); break;
		case 17: flag[17] = 1; arg[17] = optarg; tau3_start = atof(optarg); break;
		case 18: flag[18] = 1; arg[18] = optarg; tau3_end = atof(optarg); break;
		case 19: flag[19] = 1; arg[19] = optarg; tau3_num = atoi(optarg); break;
		case 20: flag[20] = 1; break;
		case 21: flag[21] = 1; arg[21] = optarg; grid_if = atof(optarg); break;
		case 22: flag[22] = 1; arg[22] = optarg; rs_file = optarg; break;
		case 23: flag[23] = 1; break;
		case 24: flag[24] = 1; break;
		case 25: flag[25] = 1; arg[25] = optarg; maf_min = atof(optarg); freq_min = maf_min; freq_max = 1 - maf_min; break;
		case 26: flag[26] = 1; help(); exit(0); break;
		default: break;
		}
	}

	if (flag[2] == 0) out_file = "heter";
	if (flag[10] == 1) out_file = out_file + "." + this_snp;
	logFile.open(out_file + ".log", ios::out);
	cover();
	cout << "Options specified:" << endl;
	logFile << "Options specified:" << endl;
	for (int i = 0; i < 4; i++) 
	{
		if (flag[i] == 1) 
		{
			cout << "--" << long_options[i].name << " " << arg[i] << endl;
			logFile << "--" << long_options[i].name << " " << arg[i] << endl;
		}
	}
	if (flag[4] == 1) 
	{
		cout << "--" << long_options[4].name;
		logFile << "--" << long_options[4].name;
		for (int i = 0; i < covs_name.size(); i++) 
		{
			cout << " " << covs_name[i];
			logFile << " " << covs_name[i];
		}
	}
	cout << endl;
	logFile << endl;
	for (int i = 7; i < 10; i++)
	{
		if (flag[i] == 1) 
		{
			cout << "--" << long_options[i].name << endl;
			logFile << "--" << long_options[i].name << endl;
		}
	}
	for (int i = 23; i < 25; i++)
	{
		if (flag[i] == 1) 
		{
			cout << "--" << long_options[i].name << endl;
			logFile << "--" << long_options[i].name << endl;
		}
	}
	if (flag[25] == 1) 
	{
		cout << "--" << long_options[25].name << " " << arg[25] << endl;
		logFile << "--" << long_options[25].name << " " << arg[25] << endl;
	}
	for (int i = 5; i < 7; i++)
	{
		if (flag[i] == 1) 
		{
			cout << "--" << long_options[i].name << " " << arg[i] << endl;
			logFile << "--" << long_options[i].name << " " << arg[i] << endl;
		}
	}

	for (int i = 10; i < 20; i++)
	{
		if (flag[i] == 1) 
		{
			cout << "--" << long_options[i].name << " " << arg[i] << endl;
			logFile << "--" << long_options[i].name << " " << arg[i] << endl;
		}
	}
	for (int i = 20; i < 21; i++)
	{
		if (flag[i] == 1) 
		{
			cout << "--" << long_options[i].name << endl;
			logFile << "--" << long_options[i].name << endl;
		}
	}
	for (int i = 21; i < 23; i++)
	{
		if (flag[i] == 1) 
		{
			cout << "--" << long_options[i].name << " " << arg[i] << endl;
			logFile << "--" << long_options[i].name << " " << arg[i] << endl;
		}
	}

	if (flag[0] == 0)
	{
		cout << "Use --bgen to speficy the genotype file in bgen format." << endl;
		logFile << "Use --bgen to speficy the genotype file in bgen format." << endl;
		error_flag = true;
	}

	if (flag[1] == 0)
	{
		cout << "Use --sample to speficy the sample file in SNPTEST sample file format." << endl;
		logFile << "Use --sample to speficy the sample file in SNPTEST sample file format." << endl;
		error_flag = true;
	}

	if (flag[3] == 0)
	{
		cout << "Use --pheno to speficy the continuous phenotype for analysis." << endl;
		logFile << "Use --pheno to speficy the continuous phenotype for analysis." << endl;
		error_flag = true;
	}

	max_thread_num = std::thread::hardware_concurrency();
	if (flag[5] == 1) 
	{
		if (thread_num < 1 || thread_num > max_thread_num)
		{
			cout << "Error: --thread-num should be from 1 to " << max_thread_num << endl;
			logFile << "Error: --thread-num should be from 1 to " << max_thread_num << endl;
			error_flag = true;
		}
	}
	else if (max_thread_num > 1) thread_num = max_thread_num - 1;
	else thread_num = 1;

	if (flag[6] == 1)
	{
		if (batch_size < 1)
		{
			cout << "ERROR: --batch-size should be larger than 1." << endl;
			logFile << "ERROR: --batch-size should be larger than 1." << endl;
			error_flag = true;
		}
	}
	else batch_size = 1000;

	if (flag[7] + flag[8] + flag[9] + flag[23] + flag[24] > 1)
	{
		cout << "ERROR: specify one analysis at a time." << endl;
		logFile << "ERROR: specify one analysis at a time." << endl;
		error_flag = true;
	}
	if (flag[7] + flag[8] + flag[9] + flag[23] + flag[24] == 0)
	{
		cout << "ERROR: no analysis was specified." << endl;
		logFile << "ERROR: no analysis was specified." << endl;
		error_flag = true;
	}
	if (flag[11] == 1 || flag[12] == 1 || flag[13] == 1 || flag[14] == 1 || flag[15] == 1 || flag[16] == 1 || flag[17] == 1 || flag[18] == 1 || flag[19] == 1 || flag[20] == 1)
	{
		grid_flag= true;
		if (flag[11] == 1 && flag[12] == 1 && flag[13] == 1 && flag[14] == 1 && flag[15] == 1 && flag[16] == 1 && flag[17] == 1 && flag[18] == 1 && flag[19] == 1 && flag[20] == 0)
		{
			if (tau1_start <= 0)
			{
				cout << "ERROR: --tau1-start should be larger than 0." << endl;
				logFile << "ERROR: --tau1-start should be larger than 0." << endl;
				error_flag = true;;
			}
			if (tau1_end <= 0)
			{
				cout << "ERROR: --tau1-end should be larger than 0." << endl;
				logFile << "ERROR: --tau1-end should be larger than 0." << endl;
				error_flag = true;
			}
			if (tau1_num <= 0)
			{
				cout << "ERROR: --tau1-num should be larger than 0." << endl;
				logFile << "ERROR: --tau1-num should be larger than 0." << endl;
				error_flag = true;
			}
			if (tau1_num == 1 && tau1_start != tau1_end)
			{
				cout << "ERROR: --tau1-num is 1, but --tau1_start is not equal to --tau1-end." << endl;
				logFile << "ERROR: --tau1-num is 1, but --tau1_start is not equal to --tau1-end." << endl;
				error_flag = true;
			}
			if (tau2_num <= 0)
			{
				cout << "ERROR: --tau2-num should be larger than 0." << endl;
				logFile << "ERROR: --tau2-num should be larger than 0." << endl;
				error_flag = true;
			}
			if (tau2_num == 1 && tau2_start != tau2_end)
			{
				cout << "ERROR: --tau2-num is 1, but --tau2_start is not equal to --tau2-end." << endl;
				logFile << "ERROR: --tau2-num is 1, but --tau2_start is not equal to --tau2-end." << endl;
				error_flag = true;
			}
			if (tau3_start <= 0)
			{
				cout << "ERROR: --tau3-start should be larger than 0." << endl;
				logFile << "ERROR: --tau3-start should be larger than 0." << endl;
				error_flag = true;
			}
			if (tau3_end <= 0)
			{
				cout << "ERROR: --tau3-end should be larger than 0." << endl;
				logFile << "ERROR: --tau3-end should be larger than 0." << endl;
				error_flag = true;
			}
			if (tau3_num <= 0)
			{
				cout << "ERROR: --tau3-num should be larger than 0." << endl;
				logFile << "ERROR: --tau3-num should be larger than 0." << endl;
				error_flag = true;
			}
			if (tau3_num == 1 && tau3_start != tau3_end)
			{
				cout << "ERROR: --tau3-num is 1, but --tau3_start is not equal to --tau3-end." << endl;
				logFile << "ERROR: --tau3-num is 1, but --tau3_start is not equal to --tau3-end." << endl;
				error_flag = true;
			}
			if (tau1_start > tau1_end)
			{
				cout << "ERROR: --tau1-start should be smaller than --tau1-end." << endl;
				logFile << "ERROR: --tau1-start should be smaller than --tau1-end." << endl;
				error_flag = true;
			}
			if (tau2_start > tau2_end)
			{
				cout << "ERROR: --tau2-start should be smaller than --tau2-end." << endl;
				logFile << "ERROR: --tau2-start should be smaller than --tau2-end." << endl;
				error_flag = true;
			}
			if (tau3_start > tau3_end)
			{
				cout << "ERROR: --tau3-start should be smaller than --tau3-end." << endl;
				logFile << "ERROR: --tau3-start should be smaller than --tau3-end." << endl;
				error_flag = true;
			}
			if (tau1_start*tau3_start-tau2_start*tau2_start <= 0 ||  tau1_start*tau3_start-tau2_end*tau2_end <= 0)
			{
				cout << "ERROR: variance-covariance matrix [tau1 tau2; tau2 tau3] on the specified search grids should be positive definite, check the range of tau2." << endl;
				logFile << "ERROR: variance-covariance matrix [tau1 tau2; tau2 tau3] on the specified search grids should be positive definite, check the range of tau2." << endl;
				error_flag = true;
			}
		}
		else if  (flag[20] == 0)
		{
			cout << "ERROR: please specifiy values of --tau1-start, --tau1-end, --tau1-num, --tau2-start, --tau2-end, --tau2-num, --tau3-start, --tau3-end, --tau3-num for a grid search." << endl;
			logFile << "ERROR: please specifiyvalues of --tau1-start, --tau1-end, --tau1-num, --tau2-start, --tau2-end, --tau2-num, --tau3-start, --tau3-end, --tau3-num for a grid search." << endl;
			error_flag = true;
		}
		if (flag[10] == 0)
		{
			cout << "ERROR: grid search is available for the analysis of one selected SNP only, use --rs-id to specify the SNP." << endl;
			logFile << "ERROR: grid search is available for the analysis of one selected SNP only, use --rs-id to specify the SNP." << endl;
			error_flag = true;			
		}
		if (flag[23] == 1 ||  flag[24] == 0)
		{
			cout << "ERROR: grid search is available for the variance component or mixed model analysis only." << endl;
			logFile << "ERROR: grid search is available for the variance component or mixed model analysis only." << endl;
			error_flag = true;			
		}
	}
	if ((flag[21] == 1 && flag[23] == 1) || (flag[21] == 1 && flag[7] == 1) || (flag[21] == 1 && flag[24] == 1))
	{
		cout << "ERROR: --grid-if is available for the variance component or mixed model analysis only." << endl;
		logFile << "ERROR: --grid-if is available for the variance component or mixed model analysis only." << endl;
		error_flag = true;
	}
	if (flag[21] == 1)
	{
		if (grid_if <= 0)
		{
			cout << "ERROR: --grid-if should be larger than 0." << endl;
			logFile << "ERROR: --grid-if should be larger than 0." << endl;
			error_flag = true;
		}
	}
	else grid_if = 1e6;
	if (flag[25] == 1)
	{
		if (maf_min <= 0 || maf_min >=0.5)
		{
			cout << "ERROR: --maf-min should be larger than 0 and smaller than 0.5." << endl;
			logFile << "ERROR: --maf-min should be larger than 0 and smaller than 0.5." << endl;
			error_flag = true;
		}
	}
	else {maf_min = 0.01; freq_min = 0.01; freq_max = 0.99;}
}

~Dataset() {logFile.close();}

double gammln(double x) 
{
	double y, tmp, ser;
	if (x == 0.5) return(0.57236494292470009);
	else 
	{
		y = x;
		tmp = x + 5.5;
		tmp = tmp - (x + 0.5)*log(tmp);
		ser = 1.000000000190015;
		for (int i = 0; i < 6; i++)
		{
			y++;
			ser = ser + cof[i] / y;
		}
		return(-tmp + log(2.5066282746310005 * ser / x));
	}
}

void gser(double *gamser, double a, double x, double *gln) 
{
	double sum, del, ap;
	*gln = gammln(a);
	if (x <= 0.0) 
	{
		*gamser = 0.0;
		return;
	}
	else 
	{
		ap = a;
		del = 1.0 / a;
		sum = del;
		for (int i = 0; i < ITMAX1; i++) 
		{
			ap++;
			del = del * x / ap;
			sum = sum + del;
			if (fabs(del) < (fabs(sum) * EPS1)) 
			{
				*gamser = sum * exp(-x + a * log(x) - (*gln));
				return;
			}
		}
	*gamser = -99.99;
	return;
	}
}

void gcf(double *gammcf, double a, double x, double *gln) 
{
	double an, b, c, d, del, h;
	int i;
	*gln = gammln(a);
	b = x + 1.0 - a;
	c = 1.0 / FPMIN;
	d = 1.0 / b;
	h = d;
	for (i = 0; i < ITMAX1; i++) 
	{
		an = -i * (i-a);
		b = b + 2.0;
		d = an * d + b;
		if (fabs(d) < FPMIN) d = FPMIN;
		c = b + an /c;
		if (fabs(c) < FPMIN) c = FPMIN;
		d = 1.0 / d;
		del = d * c;
		h = h * del;
		if (fabs(del - 1.0) < EPS1) break;
	}
	if (i <= ITMAX1) *gammcf = exp(-x + a * log(x) - (*gln)) * h;
	else *gammcf = -99.99;
	return;
}

double gammq(double a, double x) {
	double gamser, gammcf, gln;
	if (x < 0.0 || a <= 0.0) return(-99.99);
	if (x < (a + 1.0)) 
	{
		gser(&gamser, a, x, &gln);
		return(1.0 - gamser);
	}
	else 
	{
		gcf(&gammcf, a, x, &gln);
		return(gammcf);
	}
}

double norm2sdf(double beta, double se) 
{
	return(gammq(0.5, beta * beta / se /se / 2));
}

double chi2sdf(double x, double df)
{
	if (df == 2)
	{
		return(exp(-x / 2));
	}
	else
	{
		return(gammq(df / 2.0, x / 2.0));
	}
}

void readrsfile() 
{
	ifstream rsfile;
	string thisline;
	vector<string> parsedline;
	cout << "Loading snp file [" + rs_file + "]... " << flush;
	logFile << "Loading snp file [" + rs_file + "]... " << flush;
	rsfile.open(rs_file.c_str(), ios::in);
	if(!rsfile)
	{
		cout << "ERROR: " << rs_file << " was not found!" << endl;
		logFile << "ERROR: " << rs_file << " was not found!" << endl;
		exit(0);
	}
	while (getline(rsfile, thisline, '\n'))
	{
		split(parsedline, thisline, is_any_of(delimiter));
		rs_set.insert(parsedline[0]);
	}
	rsfile.close();
	cout << "done." << endl;
	logFile << "done." << endl;
	cout << rs_set.size() << " SNPs loaded from snp file [" + rs_file + "]." << endl;
	logFile << rs_set.size() << " SNPs loaded from snp file [" + rs_file + "]." << endl;
}

void checkSampleFile() 
{
	string thisline;
	char * strtmp;
	int i, j, k, sample_col_num;
	vector<string> line1, line2, line;
	bool found;
	vector<set<string>> covs_D_set;
	vector<unordered_multiset<string>> covs_D_multiset;
	set<string>::iterator tmpiter;

	ifstream sampleFile(sample_file.c_str());
	if (!sampleFile.is_open())
	{
		cout << "ERROR: Unable to open " << sample_file  << "!" << endl;
		logFile << "ERROR: Unable to open " << sample_file  << "!" << endl;
		logFile.close();
		exit(0);
	}

	getline(sampleFile, thisline, '\n');
	split(line1, thisline, is_any_of(delimiter));
	getline(sampleFile, thisline, '\n');
	split(line2, thisline, is_any_of(delimiter));
	sample_col_num = line1.size();

	found = false;
	for (i = 0; i < sample_col_num; i++) 
	{
		strtmp = (char *) line1[i].c_str();
		if (strcmp(strtmp, (char *) pheno_name.c_str()) == 0) 
		{
			pheno_type = line2[i];
			pheno_pos = i;
			found = true;
		}
	}
	if (!found) 
	{
		cout << "ERROR: Unable to find phenotype " << pheno_name  << " in sample file!" << endl;
		logFile << "ERROR: Unable to find phenotype " << pheno_name  << " in sample file!" << endl;
		logFile.close();
		exit(0);
	}
	strtmp = (char *) pheno_type.c_str();
	if ((strcmp(strtmp, "P") != 0) && (strcmp(strtmp, "C") != 0)) 
	{
		cout << "ERROR: Phenotype " << pheno_name  << " is of type " <<  pheno_type << ", which is unrecognized! ";
		logFile << "ERROR: Phenotype " << pheno_name  << " is of type " <<  pheno_type << ", which is unrecognized! ";
		cout << "Only continuous phenotype of type P or C is supported!" << endl;
		logFile << "Only continuous phenotype of type P or C is supported!" << endl;
		logFile.close();
		exit(0);
	}

	for (j = 0; j < covs_num; j++) 
	{
		found = false;
		for (i = 0; i < sample_col_num; i++) 
		{
			strtmp = (char *) line1[i].c_str();
			if (strcmp(strtmp, (char *) covs_name[j].c_str()) == 0) 
			{
				covs_type.push_back(line2[i]);
				covs_pos.push_back(i);
				found = true;
			}
		} 
		if (!found) 
		{
			cout << "ERROR: Unable to find covariate " << covs_name[j]  << " in the sample file!" << endl;
			logFile << "ERROR: Unable to find covariate " << covs_name[j]  << " in the sample file!" << endl;
			logFile.close();
			exit(0);
		}
	}
		
	for (j = 0; j < covs_num; j++) 
	{
		strtmp = (char *) covs_type[j].c_str();
		if (strcmp(strtmp, "C") == 0) 
		{
			covs_C_name.push_back(covs_name[j]);
			covs_C_pos.push_back(covs_pos[j]);
		}
		else if (strcmp(strtmp, "D") == 0) 
		{
			covs_D_name.push_back(covs_name[j]);
			covs_D_pos.push_back(covs_pos[j]);
		}
		else 
		{
			cout << "ERROR: Covariate " << covs_name[j]  << " is of type " <<  covs_type[j] << ", which is unrecognized!" << endl;
			logFile << "ERROR: Covariate " << covs_name[j]  << " is of type " <<  covs_type[j] << ", which is unrecognized!" << endl;
			logFile.close();
			exit(0);
		}
	}
	covs_C_num = covs_C_name.size();
	covs_D_num = covs_D_name.size();

	sample_num = 0;
	nomiss_sample_num = 0;
	covs_D_set.resize(covs_D_num);
	covs_D_multiset.resize(covs_D_num);
	covs_D_val.resize(covs_D_num);
	covs_D_cnt.resize(covs_D_num);
	while (getline(sampleFile, thisline, '\n')) 
	{
		miss_flag.push_back(false);
		split(line, thisline, is_any_of(delimiter));
		if (strcmp(miss, (char *) line[pheno_pos].c_str()) == 0) miss_flag[sample_num] = true;
		for (j = 0; j < covs_C_num; j++) 
		{
			if (strcmp(miss, (char *) line[covs_C_pos[j]].c_str()) == 0) miss_flag[sample_num] = true;
		}
		for (j = 0; j < covs_D_num; j++) 
		{
			if (strcmp(miss, (char *) line[covs_D_pos[j]].c_str()) == 0) miss_flag[sample_num] = true;
		}
		if (!miss_flag[sample_num]) 
		{
			for (j = 0; j < covs_D_num; j++) 
			{
				covs_D_set[j].insert(line[covs_D_pos[j]]);
				covs_D_multiset[j].insert(line[covs_D_pos[j]]);
			}
			nomiss_sample_num++;
		}
		sample_num++;
	}
	sampleFile.close();

	covs_D_col_num = 0;
	for (j = 0; j < covs_D_num; j++) 
	{
		for (tmpiter = covs_D_set[j].begin(); tmpiter != covs_D_set[j].end(); tmpiter++) 
		{
			covs_D_val[j].push_back(*tmpiter);
			covs_D_cnt[j].push_back(covs_D_multiset[j].count(*tmpiter));
		}
		covs_D_col_num += covs_D_cnt[j].size() - 1;
	}
	cout << sample_num << " samples read in " << sample_file << ", " << nomiss_sample_num << " samples having non-missing phenotype and covariate(s)." << endl;
	logFile << sample_num << " samples read in " << sample_file << ", " << nomiss_sample_num << " samples having non-missing phenotype and covariate(s)." << endl;
}

void readSampleFile() 
{
	int i, j, k, l, m;
	string thisline;
	vector<string> line;
	X_col_num = covs_C_num + covs_D_col_num + 1;
	y.resize(nomiss_sample_num);
	X.resize(nomiss_sample_num, X_col_num);
	ifstream sampleFile(sample_file.c_str());
	getline(sampleFile, thisline, '\n');
	getline(sampleFile, thisline, '\n');
	k = 0;
	for (i = 0; i < sample_num; i++) 
	{
		getline(sampleFile, thisline, '\n');
		split(line, thisline, is_any_of(delimiter));
		if (!miss_flag[i]) 
		{
			y[k] = atof(line[pheno_pos].c_str());
			l = 0;
			for (j = 0; j < covs_C_num; j++) 
			{
				X(k, l) = atof(line[covs_C_pos[j]].c_str());
				l++;
			}
			for (j = 0; j < covs_D_num; j++) 
			{
				for (m = 1; m < covs_D_cnt[j].size(); m++) 
				{
					X(k, l) = 0;
					if (strcmp((char *) line[covs_D_pos[j]].c_str(), (char *) covs_D_val[j][m].c_str()) == 0) X(k, l) = 1;
					l++;
				}
			}
			X(k, l) = 1;
			k++;
		}
	}
	sampleFile.close();
}

void summPhenoCovs() 
{
	int i, j, k;
	double y_min, y_max, y_mean, y_sd;
	vector<double> covs_C_min, covs_C_max, covs_C_mean, covs_C_sd;
	covs_C_min.resize(covs_C_num);
	covs_C_max.resize(covs_C_num);
	covs_C_mean.resize(covs_C_num);
	covs_C_sd.resize(covs_C_num);

	y_min = y.minCoeff();
	y_max = y.maxCoeff();
	y_mean = y.mean();
	y_sd = sqrt((y.squaredNorm() - nomiss_sample_num * y_mean * y_mean) / (nomiss_sample_num - 1));
	for (i = 0; i < covs_C_num; i++) 
	{
		covs_C_min[i] = (X.col(i)).minCoeff();
		covs_C_max[i] = (X.col(i)).maxCoeff();
		covs_C_mean[i] = (X.col(i)).mean();
		covs_C_sd[i] = sqrt(((X.col(i)).squaredNorm() - nomiss_sample_num * covs_C_mean[i] * covs_C_mean[i]) / (nomiss_sample_num - 1));
	}

	cout << "Phenotype (N = " << nomiss_sample_num << "):"  << endl;
	logFile << "Phenotype (N = " << nomiss_sample_num << "):" << endl;
	cout <<  "name	type	min	max	mean	sd"  << endl;
	logFile << "name	type	min	max	mean	sd" << endl;
	cout << pheno_name << "	" << pheno_type << "	" << y_min << "	" << y_max << "	" << y_mean << "	" << y_sd << endl;
	logFile << pheno_name << "	" << pheno_type << "	" << y_min << "	" << y_max << "	" << y_mean << "	" << y_sd << endl;
	if (covs_C_num > 0) 
	{
		cout << "Continuous covariate(s) (N = " << nomiss_sample_num << "):"  << endl;	
		logFile << "Continuous covariate(s) (N = " << nomiss_sample_num << "):" << endl;
		cout <<  "name	type	min	max	mean	sd"  << endl;	
		logFile << "name	type	min	max	mean	sd" << endl;
		for (j = 0; j < covs_C_num; j++) 
		{
			cout << covs_C_name[j] << "	C	" << covs_C_min[j] << "	" << covs_C_max[j] << "	" << covs_C_mean[j] << "	" << covs_C_sd[j] << endl;
			logFile << covs_C_name[j] << "	C	" << covs_C_min[j] << "	" << covs_C_max[j] << "	" << covs_C_mean[j] << "	" << covs_C_sd[j] << endl;
		}
	}
	if (covs_D_num > 0) 
	{
		cout << "Discrete covariate(s) (N = " << nomiss_sample_num << "):"  << endl;
		logFile << "Discrete covariate(s) (N = " << nomiss_sample_num << "):" << endl;
		cout <<  "name	type	level number: value(count)"  << endl;	
		logFile <<  "name	type	level number: value(count)"  << endl;	
		for (j = 0; j < covs_D_num; j++) 
		{
			cout << covs_D_name[j] << "	D	" << covs_D_val[j].size() << ": ";
			logFile << covs_D_name[j] << "	D	" << covs_D_val[j].size() << ": ";
			for (k = 0; k < covs_D_val[j].size(); k++) 
			{
				cout << covs_D_val[j][k] << "(" <<  covs_D_cnt[j][k] << ")" << " ";
				logFile << covs_D_val[j][k] << "(" <<  covs_D_cnt[j][k] << ")" << " ";
			}
			cout << endl;
			logFile << endl;
		}
	}
}

void Ini() 
{
	Xt = X.transpose();
	P = X * (Xt * X).inverse();
	Py = P * (Xt * y);
	r = y - Py;
	r2 = r.dot(r);
	reg_df = nomiss_sample_num - X_col_num - 1;
	like0 = nomiss_sample_num * log(r2 / nomiss_sample_num) + nomiss_sample_num;
}


void TauIni(double freq, double beta2, double beta3, double* tau2, double* tau3)
{
	if (beta3 <= 0)
	{
		*tau2 = beta2 + (1.0 + freq) * beta3;
		*tau3 = -beta3;
	}
	else
	{
		*tau2 = beta2;
		*tau3 = beta3;
	}
	if (*tau3 < 0.000001) *tau3 = 0.000001;
}

void NetwonRaphson_d(VectorXd* g, VectorXd* gg, VectorXd* s, VectorXd* d, ArrayXd* deno, VectorXd* delta_d)
{
	VectorXd f(3);
	MatrixXd H(3, 3);
	ArrayXd one_deno, one_deno2, two_g_deno, g2_deno;
	double d1, d2, d3, l1_tau1, l1_tau2, l1_tau3, l2_tau1, l2_tau2, l2_tau3, l_tau1, l_tau2, l_tau3, l1_2_tau1_tau1, l1_2_tau1_tau2, l1_2_tau1_tau3, l1_2_tau2_tau2, l1_2_tau2_tau3, l1_2_tau3_tau3;
	double l2_2_tau1_tau1, l2_2_tau1_tau2, l2_2_tau1_tau3, l2_2_tau2_tau2, l2_2_tau2_tau3, l2_2_tau3_tau3, l_2_tau1_tau1, l_2_tau1_tau2, l_2_tau1_tau3, l_2_tau2_tau2, l_2_tau2_tau3, l_2_tau3_tau3;
	d1 = (*d)(0);
	d2 = (*d)(1);
	d3 = (*d)(2);
	one_deno = 1 / *deno;
	l1_tau1 = one_deno.sum();
	two_g_deno = (2.0 * *g).array() * one_deno;
	l1_tau2 = two_g_deno.sum();
	g2_deno = (*gg).array() * one_deno;
	l1_tau3 = g2_deno.sum();
	one_deno2 = one_deno * one_deno;
	l2_tau1 = - ((*s).array() * one_deno2).sum();
	l2_tau2 = - ((*s).array() * two_g_deno * one_deno).sum();
	l2_tau3 = - ((*s).array() * g2_deno * one_deno).sum();
	l_tau1 = l1_tau1 + l2_tau1;
	l_tau2 = l1_tau2 + l2_tau2;
	l_tau3 = l1_tau3 + l2_tau3;
	f(0) = 2.0 * d1 * l_tau1;
	f(1) = 2.0 * d2 * l_tau1 + d3 * l_tau2;
	f(2) = d2 * l_tau2 + 2.0 * d3 * l_tau3;
	l1_2_tau1_tau1 = - (one_deno2).sum();
	l1_2_tau1_tau2 = - (one_deno * two_g_deno).sum();	
	l1_2_tau1_tau3 = - (one_deno * g2_deno).sum();	
	l1_2_tau2_tau2 = - (two_g_deno * two_g_deno).sum();
	l1_2_tau2_tau3 = - (two_g_deno * g2_deno).sum();	
	l1_2_tau3_tau3 = - (g2_deno * g2_deno).sum();
	l2_2_tau1_tau1 = 2.0 *  ((*s).array() * one_deno * one_deno2).sum();
	l2_2_tau1_tau2 = 2.0 *  ((*s).array() * two_g_deno * one_deno2).sum();
	l2_2_tau1_tau3 = 2.0 *  ((*s).array() * g2_deno * one_deno2).sum();
	l2_2_tau2_tau2 = 2.0 *  ((*s).array() * two_g_deno * two_g_deno *  one_deno).sum();
	l2_2_tau2_tau3 = 2.0 *  ((*s).array() * two_g_deno * g2_deno * one_deno).sum();
	l2_2_tau3_tau3 = 2.0 *  ((*s).array() *  g2_deno *  g2_deno * one_deno).sum();
	l_2_tau1_tau1 = l1_2_tau1_tau1 + l2_2_tau1_tau1;
	l_2_tau1_tau2 = l1_2_tau1_tau2 + l2_2_tau1_tau2;
	l_2_tau1_tau3 = l1_2_tau1_tau3 + l2_2_tau1_tau3;
	l_2_tau2_tau2 = l1_2_tau2_tau2 + l2_2_tau2_tau2;
	l_2_tau2_tau3 = l1_2_tau2_tau3 + l2_2_tau2_tau3;
	l_2_tau3_tau3 = l1_2_tau3_tau3 + l2_2_tau3_tau3;
	H(0, 0) = 2.0 * l_tau1 + 4.0 * d1 * d1 * l_2_tau1_tau1;
	H(0, 1) = 4.0 * d1 * d2 * l_2_tau1_tau1 + 2.0 * d1 * d3 * l_2_tau1_tau2;
	H(1, 0) = H(0, 1);
	H(0, 2) = 2.0 * d1 * d2 * l_2_tau1_tau2 + 4.0 * d1 * d3 * l_2_tau1_tau3;
	H(2, 0) = H(0, 2);
	H(1, 1) = 2.0 * l_tau1 + 4.0 * d2 * d2 * l_2_tau1_tau1 + 4.0 * d2 * d3 * l_2_tau1_tau2 + d3 * d3 * l_2_tau2_tau2;
	H(1, 2) = 2.0 * d2 * d2 * l_2_tau1_tau2 + 4.0 * d2 * d3 * l_2_tau1_tau3 + l_tau2 + d2 * d3 * l_2_tau2_tau2 + 2.0 * d3 * d3 * l_2_tau2_tau3;
	H(2, 1) = H(1, 2);
	H(2, 2) = d2 * d2 * l_2_tau2_tau2 + 4.0 * d2 * d3 * l_2_tau2_tau3 + 2.0 *  l_tau3 + 4.0 * d3 * d3 * l_2_tau3_tau3;
	*delta_d = H.inverse() * f;
}

void NetwonRaphson_tau(VectorXd* g, VectorXd* gg, VectorXd* s, VectorXd* tau, ArrayXd* deno, VectorXd* delta_tau)
{
	VectorXd f(3);
	MatrixXd H(3, 3);
	ArrayXd one_deno, one_deno2, two_g_deno, g2_deno;
	double tau1, tau2, tau3, l1_tau1, l1_tau2, l1_tau3, l2_tau1, l2_tau2, l2_tau3, l_tau1, l_tau2, l_tau3, l1_2_tau1_tau1, l1_2_tau1_tau2, l1_2_tau1_tau3, l1_2_tau2_tau2, l1_2_tau2_tau3, l1_2_tau3_tau3;
	double l2_2_tau1_tau1, l2_2_tau1_tau2, l2_2_tau1_tau3, l2_2_tau2_tau2, l2_2_tau2_tau3, l2_2_tau3_tau3, l_2_tau1_tau1, l_2_tau1_tau2, l_2_tau1_tau3, l_2_tau2_tau2, l_2_tau2_tau3, l_2_tau3_tau3;
	tau1 = (*tau)(0);
	tau2 = (*tau)(1);
	tau3 = (*tau)(2);
	one_deno = 1 / *deno;
	l1_tau1 = one_deno.sum();
	two_g_deno = (2.0 * *g).array() * one_deno;
	l1_tau2 = two_g_deno.sum();
	g2_deno = (*gg).array() * one_deno;
	l1_tau3 = g2_deno.sum();
	one_deno2 = one_deno * one_deno;
	l2_tau1 = - ((*s).array() * one_deno2).sum();
	l2_tau2 = - ((*s).array() * two_g_deno * one_deno).sum();
	l2_tau3 = - ((*s).array() * g2_deno * one_deno).sum();
	l_tau1 = l1_tau1 + l2_tau1;
	l_tau2 = l1_tau2 + l2_tau2;
	l_tau3 = l1_tau3 + l2_tau3;
	f(0) = l_tau1;
	f(1) = l_tau2;
	f(2) = l_tau3;
	l1_2_tau1_tau1 = - (one_deno2).sum();
	l1_2_tau1_tau2 = - (one_deno * two_g_deno).sum();	
	l1_2_tau1_tau3 = - (one_deno * g2_deno).sum();	
	l1_2_tau2_tau2 = - (two_g_deno * two_g_deno).sum();
	l1_2_tau2_tau3 = - (two_g_deno * g2_deno).sum();	
	l1_2_tau3_tau3 = - (g2_deno * g2_deno).sum();
	l2_2_tau1_tau1 = 2.0 *  ((*s).array() * one_deno * one_deno2).sum();
	l2_2_tau1_tau2 = 2.0 *  ((*s).array() * two_g_deno * one_deno2).sum();
	l2_2_tau1_tau3 = 2.0 *  ((*s).array() * g2_deno * one_deno2).sum();
	l2_2_tau2_tau2 = 2.0 *  ((*s).array() * two_g_deno * two_g_deno *  one_deno).sum();
	l2_2_tau2_tau3 = 2.0 *  ((*s).array() * two_g_deno * g2_deno * one_deno).sum();
	l2_2_tau3_tau3 = 2.0 *  ((*s).array() *  g2_deno *  g2_deno * one_deno).sum();
	l_2_tau1_tau1 = l1_2_tau1_tau1 + l2_2_tau1_tau1;
	l_2_tau1_tau2 = l1_2_tau1_tau2 + l2_2_tau1_tau2;
	l_2_tau1_tau3 = l1_2_tau1_tau3 + l2_2_tau1_tau3;
	l_2_tau2_tau2 = l1_2_tau2_tau2 + l2_2_tau2_tau2;
	l_2_tau2_tau3 = l1_2_tau2_tau3 + l2_2_tau2_tau3;
	l_2_tau3_tau3 = l1_2_tau3_tau3 + l2_2_tau3_tau3;
	H(0, 0) = l_2_tau1_tau1;
	H(0, 1) = l_2_tau1_tau2;
	H(1, 0) = H(0, 1);
	H(0, 2) = l_2_tau1_tau3;
	H(2, 0) = H(0, 2);
	H(1, 1) = l_2_tau2_tau2;
	H(1, 2) = l_2_tau2_tau3;
	H(2, 1) = H(1, 2);
	H(2, 2) = l_2_tau3_tau3;
	*delta_tau = H.inverse() * f;
}

void reg1snp(VectorXd* g, VectorXd* gg, VectorXd* s, double* like1, VectorXd* result)
{
	VectorXd Pg, r_g;
	ArrayXd deno;
	double Pg2, gy, gPy, g2, k, sigma2, beta_snp, se_snp, p_snp, r2_snp, gs, ggs, a1, a2, a3, d, beta1, beta2, beta3, chi2_het, p_het, r2_het, freq, like2;
	*gg = (*g).cwiseAbs2();
	freq = (*g).sum() / nomiss_sample_num;
	*g = (*g).array() - freq;
	freq /= 2.0;
	*gg = (*gg).array() - (*gg).sum() / nomiss_sample_num;
	g2 = (*g).dot(*g);
	Pg = P * (Xt * *g);
	Pg2 = Pg.dot(Pg);
	gy = (*g).dot(y);
	gPy = (*g).dot(Py);
	k = g2 - Pg2;
	beta_snp = (gy - gPy) / k;
	sigma2 = (r2 - beta_snp * beta_snp * k) /  reg_df;
	r2_snp = beta_snp * beta_snp * k / r2;
	se_snp = sqrt(sigma2 / k);
	p_snp = norm2sdf(beta_snp, se_snp);
	r_g = r - (*g - Pg) * beta_snp;
	*s = r_g.cwiseAbs2();
	gs = (*g).dot(*s);
	ggs = (*gg).dot(*s);
	a1 = 4.0 * g2;
	a2 = 2.0 * (*g).dot(*gg);
	a3 = (*gg).dot(*gg);
	d = a1 * a3 - a2 * a2;
	beta1 = (*s).sum() / nomiss_sample_num;
	beta2 = (a3 * 2.0 * gs - a2 * ggs) / d;
	beta3 = (a1 * ggs - a2 * 2.0 * gs) / d;
	*like1 = nomiss_sample_num * log((*s).sum() / nomiss_sample_num) + nomiss_sample_num;
	deno = beta1 + (2.0 * beta2 * *g + beta3 * *gg).array();
	like2 = (deno.log()).sum() + ((*s).array() / deno).sum();
	chi2_het = *like1 - like2;
	if (chi2_het < 0) chi2_het = 0;
	p_het = chi2sdf(chi2_het, 2);
	r2_het = 1 - exp(-chi2_het / nomiss_sample_num);
	(*result)(0) = freq;
	(*result)(1) = beta_snp;
	(*result)(2) = se_snp;
	(*result)(3) = p_snp;
	(*result)(4) = r2_snp;
	(*result)(5) = beta1;
	(*result)(6) = beta2;
	(*result)(7) = beta3;
	(*result)(8) = chi2_het;
	(*result)(9) = p_het;
	(*result)(10) = r2_het;
}

void vc1snp(VectorXd* g, VectorXd* gg, VectorXd* s, double tau1, double tau2, double tau3, double like1, VectorXd* result)
{
	VectorXd d(3), delta_d(3), this_d(3);
	bool update_flag;
	int iter_flag, i, j;
	ArrayXd deno, this_deno;
	double chi2_het, p_het, r2_het, this_like, last_like, lambda;
	d(2) = sqrt(tau3);
	d(1) = tau2 / d(2);
	d(0) = sqrt(abs(tau1 - d(1) * d(1)));
	deno = tau1 + (2.0 * tau2 * *g + tau3 * *gg).array();
	last_like = (deno.log()).sum() + ((*s).array() / deno).sum();
//	cout << endl << tau1 << ", " << tau2 << ", " << tau3<< ", " << like1 << ", " << like1 - last_like << endl;
	iter_flag = 0;
	for (i = 0; i < ITMAX2 && iter_flag == 0; i++)
	{
		NetwonRaphson_d(g, gg, s, &d, &deno, &delta_d);
		lambda = 1.0;
		update_flag = false;
		j = 0;
		while (!update_flag && iter_flag == 0 && j < ITMAX2)
		{
			this_d = d - lambda * delta_d;
			this_deno = this_d(0) * this_d(0) + this_d(1) * this_d(1) + (2.0 * (this_d(1) * this_d(2)) * *g + (this_d(2) * this_d(2)) * *gg).array();
			this_like = (this_deno.log()).sum() + ((*s).array()  / this_deno).sum();
			if (last_like - this_like > 0)
			{
				update_flag = true;
				d = this_d;
				tau1 = d(0) *  d(0) + d(1) * d(1);
				tau2 = d(1) *  d(2);
				tau3 = d(2) *  d(2);
				deno = this_deno;
				if (last_like - this_like < 0.01) iter_flag = 1;
//				cout << endl << tau1 << ", " << tau2 << ", " << tau3<< ", " << this_like << ", " << last_like - this_like << endl;
				last_like = this_like;	
			}
			else if (((lambda * delta_d.array()).abs()).sum() / ((d.array()).abs()).sum() <  EPS2) iter_flag = 2;
			else lambda /= 2.0;	
			j ++;		
		}
	}
	chi2_het = like1 - last_like;
	if (chi2_het < 0) chi2_het = 0;
	p_het = (chi2sdf(chi2_het, 1) + chi2sdf(chi2_het, 2)) / 2.0;
	r2_het = 1 - exp(-chi2_het / nomiss_sample_num);
	(*result)(5) = tau1;
	(*result)(6) = tau2;
	(*result)(7) = tau3;
	(*result)(8) = chi2_het;
	(*result)(9) = p_het;
	(*result)(10) = r2_het;
	(*result)(11) = iter_flag;
}

void mixed1snp(VectorXd* g, VectorXd* gg, VectorXd* s, double tau1, double tau2, double tau3, double like1, VectorXd* result)
{
	VectorXd d(3), delta_d(3), this_d(3), GLS_beta, r_g;
	MatrixXd GLS_X, GLS_Xt, GLS_inv;
	bool update_flag;
	int iter_flag, i, j;
	ArrayXd deno, this_deno;
	double beta_snp, se_snp, p_snp, r2_snp, chi2_het, p_het, r2_het, this_like, last_like, lambda, GLS_r2;
	GLS_Xt.resize(X_col_num + 1, nomiss_sample_num);
	beta_snp = (*result)(1);
	se_snp =  (*result)(2);
	p_snp = (*result)(3);
	r2_snp = (*result)(4);
	d(2) = sqrt(tau3);
	d(1) = tau2 / d(2);
	d(0) = sqrt(abs(tau1 - d(1) * d(1)));
	deno = tau1 + (2.0 * tau2 * *g + tau3 * *gg).array();
	last_like = (deno.log()).sum() + ((*s).array() / deno).sum();
//	cout << endl << tau1 << ", " << tau2 << ", " << tau3<< ", " << like1 << ", " << like1 - last_like << endl;
	iter_flag = 0;
	for (i = 0; i < ITMAX2 && iter_flag == 0; i++)
	{
		NetwonRaphson_d(g, gg, s, &d, &deno, &delta_d);
		lambda = 1.0;
		update_flag = false;
		j = 0;
		while (!update_flag && iter_flag == 0 && j < ITMAX2)
		{
			this_d = d - lambda * delta_d;
			this_deno = this_d(0) * this_d(0) + this_d(1) * this_d(1) + (2.0 * (this_d(1) * this_d(2)) * *g + (this_d(2) * this_d(2)) * *gg).array();
			this_like = (this_deno.log()).sum() + ((*s).array() / this_deno).sum();
			if (last_like - this_like > 0)
			{
				update_flag = true;
				d = this_d;
				tau1 = d(0) *  d(0) + d(1) * d(1);
				tau2 = d(1) *  d(2);
				tau3 = d(2) *  d(2);
				deno = this_deno;
				if (last_like - this_like < 0.01) iter_flag = 1;
				GLS_Xt.block(0, 0, X_col_num, nomiss_sample_num) = Xt;
				GLS_Xt.row(X_col_num) = (*g).transpose();
				GLS_X = GLS_Xt.transpose();
				for (int j = 0; j < nomiss_sample_num; j++) GLS_Xt.col(j) = (GLS_Xt.col(j)).array() / deno(j);
				GLS_inv = (GLS_Xt * GLS_X).inverse();
				GLS_beta = GLS_inv * (GLS_Xt * y);
				r_g = y - GLS_X * GLS_beta;
				*s = r_g.cwiseAbs2();
				beta_snp = GLS_beta(X_col_num);
				se_snp = sqrt(GLS_inv(X_col_num, X_col_num));
				p_snp = norm2sdf(beta_snp, se_snp);
//				cout << endl << tau1 << ", " << tau2 << ", " << tau3<< ", " << this_like << ", " << last_like - this_like << endl;
				last_like = this_like;	
			}
			else if (((lambda * delta_d.array()).abs()).sum() / ((d.array()).abs()).sum() <  EPS2) iter_flag = 2;
			else lambda /= 2.0;
			j ++;			
		}
	}
	r2_snp = 1 - exp(-beta_snp * beta_snp / se_snp / se_snp / nomiss_sample_num);
	chi2_het = like1 - last_like;
	if (chi2_het < 0) chi2_het = 0;
	p_het = (chi2sdf(chi2_het, 1) + chi2sdf(chi2_het, 2)) / 2.0;
	r2_het = 1 - exp(-chi2_het / nomiss_sample_num);
	(*result)(1) = beta_snp;
	(*result)(2) = se_snp;
	(*result)(3) = p_snp;
	(*result)(4) = r2_snp;
	(*result)(5) = tau1;
	(*result)(6) = tau2;
	(*result)(7) = tau3;
	(*result)(8) = chi2_het;
	(*result)(9) = p_het;
	(*result)(10) = r2_het;
	(*result)(11) = iter_flag;
}

void creg1snp(VectorXd* g, VectorXd* gg, VectorXd* s, double tau1, double tau2, double tau3, double like1, VectorXd* result)
{
	VectorXd tau(3), delta_tau(3), this_tau(3);
	bool update_flag;
	int iter_flag, i, j;
	ArrayXd deno, this_deno;
	double chi2_het, p_het, r2_het, this_like, last_like, lambda;
	tau(2) = tau3;
	tau(1) = tau2;
	tau(0) = tau1;
	deno = tau1 + (2.0 * tau2 * *g + tau3 * *gg).array();
	last_like = (deno.log()).sum() + ((*s).array() / deno).sum();
//	cout << endl << tau1 << ", " << tau2 << ", " << tau3<< ", " << like1 << ", " << like1 - last_like << endl;
	iter_flag = 0;
	for (i = 0; i < ITMAX2 && iter_flag == 0; i++)
	{
		NetwonRaphson_tau(g, gg, s, &tau, &deno, &delta_tau);
		lambda = 1.0;
		update_flag = false;
		j = 0;
		while (!update_flag && iter_flag == 0 && j < ITMAX2)
		{
			this_tau = tau - lambda * delta_tau;
			this_deno = this_tau(0) + (2.0 * this_tau(1) * *g + this_tau(2) * *gg).array();
			this_like = (this_deno.log()).sum() + ((*s).array()  / this_deno).sum();
			if (last_like - this_like > 0)
			{
				update_flag = true;
				tau = this_tau;
				tau1 = tau(0);
				tau2 = tau(1);
				tau3 = tau(2);
				deno = this_deno;
				if (last_like - this_like < 0.01) iter_flag = 1;
//				cout << endl << tau1 << ", " << tau2 << ", " << tau3<< ", " << this_like << ", " << last_like - this_like << endl;
				last_like = this_like;	
			}
			else if (((lambda * delta_tau.array()).abs()).sum() / ((tau.array()).abs()).sum() <  EPS2) iter_flag = 2;
			else lambda /= 2.0;	
			j ++;		
		}
	}
	chi2_het = like1 - last_like;
	if (chi2_het < 0) chi2_het = 0;
	p_het = chi2sdf(chi2_het, 2);
	r2_het = 1 - exp(-chi2_het / nomiss_sample_num);
	(*result)(5) = tau1;
	(*result)(6) = tau2;
	(*result)(7) = tau3;
	(*result)(8) = chi2_het;
	(*result)(9) = p_het;
	(*result)(10) = r2_het;
	(*result)(11) = iter_flag;
}

void lev1snp(VectorXd* g, int g0_num, int g1_num, int g2_num, VectorXd* result)
{
	VectorXd Pg, r_g, gt, gt2, z0, z1, z2;
	z0.resize(g0_num);
	z1.resize(g1_num);
	z2.resize(g2_num);
	double Pg2, gy, gPy, g2, k, sigma2, beta_snp, se_snp, p_snp, r2_snp, freq, m0, m1, m2, a, a0, a1, a2, chi2_lev, p_lev, n_sum, d_sum;
	int df_lev = 2;
	gt2 = (*g).cwiseAbs2();
	freq = (*g).sum() / nomiss_sample_num;
	gt = (*g).array() - freq;
	freq /= 2.0;
	gt2 = gt2.array() - gt2.sum() / nomiss_sample_num;
	g2 = gt.dot(gt);
	Pg = P * (Xt * gt);
	Pg2 = Pg.dot(Pg);
	gy = gt.dot(y);
	gPy = gt.dot(Py);
	k = g2 - Pg2;
	beta_snp = (gy - gPy) / k;
	sigma2 = (r2 - beta_snp * beta_snp * k) /  reg_df;
	r2_snp = beta_snp * beta_snp * k / r2;
	se_snp = sqrt(sigma2 / k);
	p_snp = norm2sdf(beta_snp, se_snp);
	r_g = r - (gt - Pg) * beta_snp;
	int g0_idx = 0;
	int g1_idx = 0;
	int g2_idx = 0;
	for (int i = 0; i < nomiss_sample_num; i++)
	{
		if ((*g)(i) == 0) {z0(g0_idx) = r_g(i); g0_idx++;}
		else if ((*g)(i) == 1) {z1(g1_idx) = r_g(i); g1_idx++;}
		else {z2(g2_idx) = r_g(i); g2_idx++;}
	}

	if (g0_num > 0) 
	{
		m0 = median(z0.begin(), z0.end());
		z0 = (z0.array() - m0).abs();
		a0 = z0.sum() / g0_num;
	}
	else 
	{
		df_lev--; 
		a0 = 0;
	}
	if (g1_num > 0)  
	{
		m1 = median(z1.begin(), z1.end());
		z1 = (z1.array() - m1).abs();
		a1 = z1.sum() / g1_num;
	}
	else 
	{
		df_lev--; 
		a1 = 0;
	}
	if (g2_num > 0)  
	{
		m2 = median(z2.begin(), z2.end());
		z2 = (z2.array() - m2).abs();
		a2 = z2.sum() / g2_num;
	}
	else 
	{
		df_lev--; 
		a2 = 0;
	}
	a = (a0 * g0_num + a1 * g1_num + a2 * g2_num) / nomiss_sample_num;
	n_sum = 0;
	d_sum = 0;
	if (g0_num > 0) {n_sum += g0_num * (a0 - a) * (a0 - a); d_sum += ((z0.array() - a0).abs2()).sum();}
	if (g1_num > 0) {n_sum += g1_num * (a1 - a) * (a1 - a); d_sum += ((z1.array() - a1).abs2()).sum();}
	if (g2_num > 0) {n_sum += g2_num * (a2 - a) * (a2 - a); d_sum += ((z2.array() - a2).abs2()).sum();}
	if (df_lev == 0) 
	{
		chi2_lev = 0; 
		p_lev = 1;
	}
	else 
	{
		chi2_lev = (nomiss_sample_num - df_lev - 1) * n_sum / d_sum;
		p_lev = chi2sdf(chi2_lev, df_lev);
	}
	(*result)(0) = freq;
	(*result)(1) = beta_snp;
	(*result)(2) = se_snp;
	(*result)(3) = p_snp;
	(*result)(4) = r2_snp;
	(*result)(5) = g0_num;
	(*result)(6) = g1_num;
	(*result)(7) = g2_num;
	(*result)(8) = chi2_lev;
	(*result)(9) = p_lev;
	(*result)(10) = df_lev;
}

void analysis() 
{
	ofstream outFile;
	int analyzed_snp_num;
	string result_file;
	if (flag[7] == 1) result_file = out_file + ".reg";
	else if (flag[8] == 1) result_file = out_file + ".vc";
	else if (flag[9] == 1) result_file = out_file + ".mixed";
	else if (flag[23] == 1) result_file = out_file + ".creg";
	outFile.open(result_file, ios::out);
	MatrixXd G;
	G.resize(nomiss_sample_num, batch_size);
	batch_result.resize(RESSIZE, batch_size);
	SNPID.resize(batch_size);
	rsid.resize(batch_size);
	chromosome.resize(batch_size);
	allele1.resize(batch_size);
	allele2.resize(batch_size);
	position.resize(batch_size);
	maf_flag.resize(batch_size);
	struct bgen_file* bgen;
	struct bgen_metafile* mf;
	struct bgen_partition const* partition;
	struct bgen_variant const* vm;
	struct bgen_genotype* vg;
	bgen = bgen_file_open(bgen_file.c_str());
	if(bgen == NULL)
	{
		cout << "ERROR: Unable to open " << bgen_file  << "!" << endl;
		logFile << "ERROR: Unable to open " << bgen_file  << "!" << endl;
		logFile.close();
		exit(0);		
	}
	if(bgen_file_nsamples(bgen) != sample_num)
	{
		cout << "ERROR: Sample number in " <<  bgen_file <<  " is " << bgen_file_nsamples(bgen) << ", which does not match the sample number " << sample_num <<  " in file " << sample_file << "!" << endl;
		logFile <<  "ERROR: Sample number in " <<  bgen_file <<  " is " << bgen_file_nsamples(bgen) << ", which does not match the sample number " << sample_num <<  " in file " << sample_file << "!" << endl;
		logFile.close();
		exit(0);		
	}
	snp_num = bgen_file_nvariants(bgen);
	cout << sample_num << " samples and " << snp_num << " variants found in " <<  bgen_file << "." << endl;
	logFile << sample_num << " samples and " << snp_num << " variants found in " <<  bgen_file << "." << endl;

	int batch_num = ceil(double(snp_num) / batch_size);
	mf = bgen_metafile_create(bgen, (bgen_file + ".metafile").c_str(), 1, 0);
	partition = bgen_metafile_read_partition(mf, 0);
	double *probs = new double[sample_num * 3];
	if (flag[7] == 1) 
	{
		cout << "Linear regression started." << endl;
		logFile << "Linear regression started." << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het" << endl;
	}
	else if (flag[8] == 1) 
	{
		cout << "Variance component analysis started." << endl;
		logFile << "Variance component analysis started." << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
	}
	else if (flag[9] == 1) 
	{
		cout << "Linear mixed model analysis started." << endl;
		logFile << "Linear mixed model analysis started." << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
	}
	else if (flag[23] == 1) 
	{
		cout << "Chi-square regression started." << endl;
		logFile << "Chi-square regression started." << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
	}
	maf_filterred_num = 0;
	analyzed_snp_num = 0;

	for (int batch_idx = 1; batch_idx <= batch_num; ++ batch_idx)
	{
		int snp_start = (batch_idx - 1) * batch_size;
		int snp_end = batch_idx * batch_size;
		if (batch_idx == batch_num)
		{
			snp_end = snp_num;
		}
	cout << " scanning SNP " << snp_start + 1  << " to SNP " << snp_end << "..." << flush;
	logFile << " scanning SNP " << snp_start + 1  << " to SNP " << snp_end << "..." << flush;
	int col_idx, col_max;
	col_idx = 0;
	for (uint32_t i = snp_start; i < snp_end; i++) 
	{
		vm = bgen_partition_get_variant(partition, i);
		SNPID[col_idx].assign(bgen_string_data(vm->id), bgen_string_length(vm->id));
		rsid[col_idx].assign(bgen_string_data(vm->rsid), bgen_string_length(vm->rsid));
		chromosome[col_idx].assign(bgen_string_data(vm->chrom), bgen_string_length(vm->chrom));
		position[col_idx] = vm->position;
		allele1[col_idx].assign(bgen_string_data(vm->allele_ids[0]), bgen_string_length(vm->allele_ids[0]));
		allele2[col_idx].assign(bgen_string_data(vm->allele_ids[1]), bgen_string_length(vm->allele_ids[1]));
		if (vm->nalleles != 2)
		{
			cout << SNPID[col_idx]  << " has " << vm->nalleles << " alleles, skipped!" << endl;
			logFile << SNPID[col_idx]  << " has " << vm->nalleles << " alleles, skipped!" << endl;
			continue;
		}
		vg = bgen_file_open_genotype(bgen, vm->genotype_offset);
		if (bgen_genotype_ncombs(vg) != 3)
		{
			cout << SNPID[col_idx]  << " has " << bgen_genotype_ncombs(vg) << " genotypes, skipped!" << endl;
			logFile << SNPID[col_idx]  << " has " << bgen_genotype_ncombs(vg) << " genotypes, skipped!" << endl;
			continue;
		}

		bgen_genotype_read(vg, probs);
		int row_idx= 0;
		for (std::size_t j = 0; j < sample_num; j++) 
		{
			if (!miss_flag[j]) 
			{
				G(row_idx, col_idx) = 0;
				for (std::size_t l = 1; l < 3; l++)
				{
					G(row_idx, col_idx) += probs[j * 3 + l] * l;
				}
				row_idx++;
			}
		}
		bgen_genotype_close(vg);
		col_idx++;
	}
	col_max = col_idx;
	if (flag[7] == 1) 
	{
		#pragma omp parallel for
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			VectorXd g, gg, s, result(RESSIZE);
			double like1, freq;
			g = G.col(col_idx);
			freq = g.sum() / nomiss_sample_num / 2.0;
			if (freq >= freq_min && freq <= freq_max)
			{ 
				maf_flag(col_idx) = 0;
				reg1snp(&g, &gg, &s, &like1, &result);
				batch_result.col(col_idx) = result;
			}
			else maf_flag(col_idx) = 1;
		}
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			if (maf_flag(col_idx) == 0)
			{
				outFile << chromosome[col_idx] << " " <<  SNPID[col_idx] << " " << rsid[col_idx] << " " << position[col_idx] << " " << allele1[col_idx] << " " <<  allele2[col_idx];
				outFile << " " << batch_result(0, col_idx)  << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
				outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
				outFile << " " << batch_result(8, col_idx)  << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << endl;
				analyzed_snp_num++;
			}
			else maf_filterred_num++;
		}
	}
	else if (flag[8] == 1) 
	{
		#pragma omp parallel for
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			VectorXd g, gg, s, result(RESSIZE), reg_result(RESSIZE);
			double chi2_reg, chi2_vc, beta1, beta2, beta3, tau1, tau2, tau3, like1, freq;
			g = G.col(col_idx);
			freq = g.sum() / nomiss_sample_num / 2.0;
			if (freq >= freq_min && freq <= freq_max)
			{ 
				maf_flag(col_idx) = 0;
				reg1snp(&g, &gg, &s, &like1, &result);
				beta1 = result(5);
				beta2 = result(6);
				beta3 = result(7);
				chi2_reg = result(8);
				tau1 = beta1;
				reg_result = result;
				TauIni(result(0), beta2, beta3, &tau2, &tau3);
				vc1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &result);
				chi2_vc =  result(8);
				if (chi2_reg - chi2_vc < grid_if || beta3 > 0) batch_result.col(col_idx) = result;
				else
				{
					VectorXd ml_result(RESSIZE);
					int grid_num = GRIDNUM;
					double tau2_start = beta2 + beta3;
					double tau2_end = beta2 + 2.0 * beta3;	
					double chi2_max = -1;
					tau3 = -beta3;
					if (tau3 < 0.000001) tau3 = 0.000001;
					for (int grid_idx = 0; grid_idx < grid_num; grid_idx++) 
					{
						result = reg_result;
						tau2 = (tau2_end - tau2_start) / (grid_num - 1) * grid_idx + tau2_start;
						vc1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &result);
						if (result(8) > chi2_max) 
						{
							chi2_max = result(8);
							ml_result = result;
						}
					}
					batch_result.col(col_idx) = ml_result;
				}
			}
			else maf_flag(col_idx) = 1;
		}
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			if (maf_flag(col_idx) == 0)
			{
				outFile << chromosome[col_idx] << " " <<  SNPID[col_idx] << " " << rsid[col_idx] << " " << position[col_idx] << " " << allele1[col_idx] << " " <<  allele2[col_idx];
				outFile << " " << batch_result(0, col_idx)  << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
				outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
				outFile << " " << batch_result(8, col_idx)  << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << " " << batch_result(11, col_idx) << endl;
				analyzed_snp_num++;
			}
			else maf_filterred_num++;
		}
	}
	else if (flag[9] == 1) 
	{
		#pragma omp parallel for
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			VectorXd g, gg, s, result(RESSIZE), reg_result(RESSIZE);
			double chi2_reg, chi2_mixed, beta1, beta2, beta3, tau1, tau2, tau3, like1, freq;
			g = G.col(col_idx);
			freq = g.sum() / nomiss_sample_num / 2.0;
			if (freq >= freq_min && freq <= freq_max)
			{ 
				maf_flag(col_idx) = 0;
				reg1snp(&g, &gg, &s, &like1, &result);
				beta1 = result(5);
				beta2 = result(6);
				beta3 = result(7);
				chi2_reg = result(8);
				tau1 = beta1;
				reg_result = result;
				TauIni(result(0), beta2, beta3, &tau2, &tau3);
				mixed1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &result);
				chi2_mixed =  result(8);
				if (chi2_reg - chi2_mixed < grid_if || beta3 > 0) batch_result.col(col_idx) = result;
				else
				{
					VectorXd ml_result(RESSIZE);
					int grid_num = GRIDNUM;
					double tau2_start = beta2 + beta3;
					double tau2_end = beta2 + 2.0 * beta3;	
					double chi2_max = -1;
					tau3 = -beta3;
					if (tau3 < 0.000001) tau3 = 0.000001;
					for (int grid_idx = 0; grid_idx < grid_num; grid_idx++) 
					{
						result = reg_result;
						tau2 = (tau2_end - tau2_start) / (grid_num - 1) * grid_idx + tau2_start;
						mixed1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &result);
						if (result(8) > chi2_max) 
						{
							chi2_max = result(8);
							ml_result = result;
						}
					}
					batch_result.col(col_idx) = ml_result;
				}
			}
			else maf_flag(col_idx) = 1;
		}
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			if (maf_flag(col_idx) == 0)
			{
				outFile << chromosome[col_idx] << " " <<  SNPID[col_idx] << " " << rsid[col_idx] << " " << position[col_idx] << " " << allele1[col_idx] << " " <<  allele2[col_idx];
				outFile << " " << batch_result(0, col_idx)  << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
				outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
				outFile << " " << batch_result(8, col_idx)  << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << " " << batch_result(11, col_idx) << endl;
				analyzed_snp_num++;
			}
			else maf_filterred_num++;
		}
	}
	else if (flag[23] == 1) 
	{
		#pragma omp parallel for
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			VectorXd g, gg, s, result(RESSIZE), reg_result(RESSIZE);
			double beta1, beta2, beta3, tau1, tau2, tau3, like1, freq;
			g = G.col(col_idx);
			freq = g.sum() / nomiss_sample_num / 2.0;
			if (freq >= freq_min && freq <= freq_max)
			{ 
				maf_flag(col_idx) = 0;
				reg1snp(&g, &gg, &s, &like1, &result);
				beta1 = result(5);
				beta2 = result(6);
				beta3 = result(7);
				tau1 = beta1;
				reg_result = result;
				creg1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &result);
				batch_result.col(col_idx) = result;
			}
			else maf_flag(col_idx) = 1;
		}
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			if (maf_flag(col_idx) == 0)
			{
				outFile << chromosome[col_idx] << " " <<  SNPID[col_idx] << " " << rsid[col_idx] << " " << position[col_idx] << " " << allele1[col_idx] << " " <<  allele2[col_idx];
				outFile << " " << batch_result(0, col_idx)  << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
				outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
				outFile << " " << batch_result(8, col_idx)  << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << " " << batch_result(11, col_idx) << endl;
				analyzed_snp_num++;
			}
			else maf_filterred_num++;
		}
	}
	cout << " done." << endl;
	logFile << " done." << endl;
	}
	delete [] probs;
	bgen_partition_destroy(partition);
	bgen_metafile_close(mf);
	bgen_file_close(bgen);
	outFile.close();
	cout <<  maf_filterred_num << " SNPs having MAFs < " << maf_min << ", " << analyzed_snp_num << " SNPs analyzed, the results were saved to " << result_file << "." << endl;
	logFile << maf_filterred_num << " SNPs having MAFs < " << maf_min << ", " << analyzed_snp_num << " SNPs analyzed, the results were saved to " << result_file << "." << endl;
}

void analysis1snp() 
{
	ofstream outFile;
	bool snp_found = false;
	string result_file, this_rsid;
	int grid_num;
	batch_size = 1;
	if (flag[7] == 1) result_file = out_file + ".reg";
	else if (flag[8] == 1) result_file = out_file + ".vc";
	else if (flag[9] == 1) result_file = out_file + ".mixed";
	else if (flag[23] == 1) result_file = out_file + ".creg";
	outFile.open(result_file, ios::out);
	MatrixXd G, tau_grid;
	G.resize(nomiss_sample_num, batch_size);
	SNPID.resize(batch_size);
	rsid.resize(batch_size);
	chromosome.resize(batch_size);
	allele1.resize(batch_size);
	allele2.resize(batch_size);
	position.resize(batch_size);
	struct bgen_file* bgen;
	struct bgen_metafile* mf;
	struct bgen_partition const* partition;
	struct bgen_variant const* vm;
	struct bgen_genotype* vg;
	bgen = bgen_file_open(bgen_file.c_str());
	if(bgen == NULL)
	{
		cout << "ERROR: Unable to open " << bgen_file  << "!" << endl;
		logFile << "ERROR: Unable to open " << bgen_file  << "!" << endl;
		logFile.close();
		exit(0);		
	}
	if(bgen_file_nsamples(bgen) != sample_num)
	{
		cout << "ERROR: Sample number in " <<  bgen_file <<  " is " << bgen_file_nsamples(bgen) << ", which does not match the sample number " << sample_num <<  " in file " << sample_file << "!" << endl;
		logFile <<  "ERROR: Sample number in " <<  bgen_file <<  " is " << bgen_file_nsamples(bgen) << ", which does not match the sample number " << sample_num <<  " in file " << sample_file << "!" << endl;
		logFile.close();
		exit(0);		
	}
	snp_num = bgen_file_nvariants(bgen);
	cout << endl << sample_num << " samples and " << snp_num << " variants found in " <<  bgen_file << "." << endl;;
	logFile << endl << sample_num << " samples and " << snp_num << " variants found in " <<  bgen_file << "." << endl;
	mf = bgen_metafile_create(bgen, (bgen_file + ".metafile").c_str(), 1, 0);
	partition = bgen_metafile_read_partition(mf, 0);
	double *probs = new double[sample_num * 3];
	int col_idx = 0;

	for (int i = 0; i < snp_num; ++ i)
	{
		vm = bgen_partition_get_variant(partition, i);
		this_rsid.assign(bgen_string_data(vm->rsid), bgen_string_length(vm->rsid));
		if (this_rsid != this_snp) continue;
		SNPID[col_idx].assign(bgen_string_data(vm->id), bgen_string_length(vm->id));
		rsid[col_idx].assign(bgen_string_data(vm->rsid), bgen_string_length(vm->rsid));
		chromosome[col_idx].assign(bgen_string_data(vm->chrom), bgen_string_length(vm->chrom));
		position[col_idx] = vm->position;
		allele1[col_idx].assign(bgen_string_data(vm->allele_ids[0]), bgen_string_length(vm->allele_ids[0]));
		allele2[col_idx].assign(bgen_string_data(vm->allele_ids[1]), bgen_string_length(vm->allele_ids[1]));
		if (vm->nalleles != 2)
		{
			cout << SNPID[col_idx]  << " has " << vm->nalleles << " alleles, skipped!" << endl;
			logFile << SNPID[col_idx]  << " has " << vm->nalleles << " alleles, skipped!" << endl;
			continue;
		}
		vg = bgen_file_open_genotype(bgen, vm->genotype_offset);
		if (bgen_genotype_ncombs(vg) != 3)
		{
			cout << SNPID[col_idx]  << " has " << bgen_genotype_ncombs(vg) << " genotypes, skipped!" << endl;
			logFile << SNPID[col_idx]  << " has " << bgen_genotype_ncombs(vg) << " genotypes, skipped!" << endl;
			continue;
		}
		snp_found = true;
		bgen_genotype_read(vg, probs);
		int row_idx= 0;
		for (std::size_t j = 0; j < sample_num; j++) 
		{
			if (!miss_flag[j]) 
			{
				G(row_idx, col_idx) = 0;
				for (std::size_t l = 1; l < 3; l++)
				{
					G(row_idx, col_idx) += probs[j * 3 + l] * l;
				}
				row_idx++;
			}
		}
		bgen_genotype_close(vg);
	}
	if (!snp_found)
	{
		cout << "ERROR: " << this_snp << " was not found in " << bgen_file << "!" << endl;
		logFile << "ERROR: " << this_snp << " was not found in " << bgen_file << "!" << endl;
		logFile.close();
		exit(0);
	}

	if (flag[7] == 1) 
	{
		cout << endl << "Regression analysis of " << this_snp << " started." << endl;
		logFile << endl << "Regression analysis of " << this_snp << " started."  << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het" << endl;
		col_idx = 0; 
		VectorXd g, gg, s, result(RESSIZE);
		double like1;
		g = G.col(col_idx);
		reg1snp(&g, &gg, &s, &like1, &result);
		batch_result.col(col_idx) = result;
		outFile << chromosome[col_idx] << " " <<  SNPID[col_idx] << " " << rsid[col_idx] << " " << position[col_idx] << " " << allele1[col_idx] << " " <<  allele2[col_idx];
		outFile << " " << batch_result(0, col_idx)  << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
		outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
		outFile << " " << batch_result(8, col_idx) << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << endl;
	}
	else if (flag[8] == 1 && grid_flag && flag[20] == 0) 
	{
		grid_num = tau1_num * tau2_num * tau3_num;
		tau_grid.resize(3, grid_num);
		batch_result.resize(RESSIZE, grid_num);
		cout << endl << "Grid search with variance component analysis of " << this_snp << " started."  << endl;
		logFile << endl << "Grid search with variance component analysis of " << this_snp << " started."  << endl;
		logFile << "chrom SNPID rsid position alleleA alleleB tau1_ini tau2_ini tau3_ini freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
		VectorXd g, gg, s, reg_result(RESSIZE);
		double like1;
		g = G.col(0);
		reg1snp(&g, &gg, &s, &like1, &reg_result);
		#pragma omp parallel for
		for (col_idx = 0; col_idx < grid_num; col_idx++) 
		{
			VectorXd this_result(RESSIZE);
			int tau1_idx = col_idx / (tau2_num * tau3_num);
			int tau2_idx = (col_idx - tau1_idx * tau2_num * tau3_num) / tau3_num;
			int tau3_idx = col_idx -  tau1_idx * tau2_num * tau3_num - tau2_idx * tau3_num;
			double tau1, tau2, tau3;
			if (tau1_num == 1) tau1 = tau1_start;
			else tau1 = (tau1_end - tau1_start) / (tau1_num - 1) * tau1_idx + tau1_start;
			if (tau2_num == 1) tau2 = tau2_start;
			else tau2 = (tau2_end - tau2_start) / (tau2_num - 1) * tau2_idx + tau2_start;
			if (tau3_num == 1) tau3 = tau3_start;
			else tau3 = (tau3_end - tau3_start) / (tau3_num - 1) * tau3_idx + tau3_start;
			this_result = reg_result;
			if (tau3 < 0.000001) tau3 = 0.000001;
			vc1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &this_result);
			batch_result.col(col_idx) = this_result;
			tau_grid(0, col_idx) =  tau1;
			tau_grid(1, col_idx) =  tau2;
			tau_grid(2, col_idx) =  tau3;
		}
		double chi2_max = -1;
		int chi2_idx = -1;
		for (col_idx = 0; col_idx < grid_num; col_idx++) 
		{
			logFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0] << " " << tau_grid(0, col_idx) << " " << tau_grid(1, col_idx) << " " << tau_grid(2, col_idx);
			logFile << " " << batch_result(0, col_idx) << " " << batch_result(1, col_idx) << " " << batch_result(2, col_idx) << " " << batch_result(3, col_idx);
			logFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx) << " " << batch_result(6, col_idx) << " " << batch_result(7, col_idx);  
			logFile << " " << batch_result(8, col_idx) << " " << batch_result(9, col_idx) << " " << batch_result(10, col_idx) << " " << batch_result(11, col_idx) << endl;
			if (batch_result(8, col_idx) > chi2_max)
			{
				chi2_max = batch_result(8, col_idx);
				chi2_idx = col_idx;
			}
		}
		outFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0];
		outFile << " " << batch_result(0, chi2_idx) << " " << batch_result(1, chi2_idx)  << " " << batch_result(2, chi2_idx)  << " " << batch_result(3, chi2_idx);
		outFile << " " << batch_result(4, chi2_idx) << " " << batch_result(5, chi2_idx)  << " " << batch_result(6, chi2_idx)  << " " << batch_result(7, chi2_idx);  
		outFile << " " << batch_result(8, chi2_idx) << " " << batch_result(9, chi2_idx)  << " " << batch_result(10, chi2_idx)  << " " << batch_result(11, chi2_idx) << endl;
	}
	else if (flag[8] == 1 && grid_flag && flag[20] == 1) 
	{
		cout << endl << "Auto-grid search with variance component analysis of " << this_snp << " started."  << endl;
		logFile << endl << "Auto-grid search with variance component analysis of " << this_snp << " started."  << endl;
		logFile << "chrom SNPID rsid position alleleA alleleB tau1_ini tau2_ini tau3_ini freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
		VectorXd g, gg, s, reg_result(RESSIZE), result(RESSIZE);
		double beta1, beta2, beta3, like1;
		g = G.col(0);
		reg1snp(&g, &gg, &s, &like1, &result);
		beta1 = result(5);
		beta2 = result(6);
		beta3 = result(7);
		reg_result = result;
		if (beta3 > 0) 
		{
			grid_num = 1;
			tau_grid.resize(3, grid_num);
			batch_result.resize(RESSIZE, grid_num);
			result = reg_result;
			if (beta3 < 0.000001) beta3 = 0.000001;
			vc1snp(&g, &gg, &s, beta1, beta2, beta3, like1, &result);
			batch_result.col(0) = result;
			tau_grid(0, 0) =  beta1;
			tau_grid(1, 0) =  beta2;
			tau_grid(2, 0) =  beta3;
			col_idx = 0;
			logFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0] << " " << tau_grid(0, col_idx) << " " << tau_grid(1, col_idx) << " " << tau_grid(2, col_idx);
			logFile << " " << batch_result(0, col_idx) << " " << batch_result(1, col_idx) << " " << batch_result(2, col_idx) << " " << batch_result(3, col_idx);
			logFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx) << " " << batch_result(6, col_idx) << " " << batch_result(7, col_idx);  
			logFile << " " << batch_result(8, col_idx) << " " << batch_result(9, col_idx) << " " << batch_result(10, col_idx) << " " << batch_result(11, col_idx) << endl;
			outFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0];
			outFile << " " << batch_result(0, col_idx) << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
			outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
			outFile << " " << batch_result(8, col_idx) << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << " " << batch_result(11, col_idx) << endl;
		}
		else
		{
			grid_num = GRIDNUM;
			tau_grid.resize(3, grid_num);
			batch_result.resize(RESSIZE, grid_num);
			tau2_start = beta2 + beta3;
			tau2_end = beta2 + 2.0 * beta3;			
			#pragma omp parallel for
			for (col_idx = 0; col_idx < grid_num; col_idx++) 
			{
				VectorXd this_result(RESSIZE);
				double tau1, tau2, tau3;
				tau1 = beta1;
				tau2 = (tau2_end - tau2_start) / (grid_num - 1) * col_idx + tau2_start;
				tau3 = -beta3;
				if (tau3 < 0.000001) tau3 = 0.000001;
				this_result = reg_result;
				vc1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &this_result);
				batch_result.col(col_idx) = this_result;
				tau_grid(0, col_idx) =  tau1;
				tau_grid(1, col_idx) =  tau2;
				tau_grid(2, col_idx) =  tau3;
			}
			double chi2_max = -1;
			int chi2_idx = -1;
			for (col_idx = 0; col_idx < grid_num; col_idx++) 
			{
				logFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0] << " " << tau_grid(0, col_idx) << " " << tau_grid(1, col_idx) << " " << tau_grid(2, col_idx);
				logFile << " " << batch_result(0, col_idx) << " " << batch_result(1, col_idx) << " " << batch_result(2, col_idx) << " " << batch_result(3, col_idx);
				logFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx) << " " << batch_result(6, col_idx) << " " << batch_result(7, col_idx);  
				logFile << " " << batch_result(8, col_idx) << " " << batch_result(9, col_idx) << " " << batch_result(10, col_idx) << " " << batch_result(11, col_idx) << endl;
				if (batch_result(8, col_idx) > chi2_max)
				{
					chi2_max = batch_result(8, col_idx);
					chi2_idx = col_idx;
				}
			}
			outFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0];
			outFile << " " << batch_result(0, chi2_idx) << " " << batch_result(1, chi2_idx)  << " " << batch_result(2, chi2_idx)  << " " << batch_result(3, chi2_idx);
			outFile << " " << batch_result(4, chi2_idx) << " " << batch_result(5, chi2_idx)  << " " << batch_result(6, chi2_idx)  << " " << batch_result(7, chi2_idx);  
			outFile << " " << batch_result(8, chi2_idx) << " " << batch_result(9, chi2_idx)  << " " << batch_result(10, chi2_idx)  << " " << batch_result(11, chi2_idx) << endl;
		}
	}
	else if (flag[8] == 1)
	{
		cout << endl << "Variance component analysis of " << this_snp << " started."  << endl;
		logFile << endl << "Variance component analysis of " << this_snp << " started."  << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
		col_idx = 0; 
		grid_num = 1;
		VectorXd g, gg, s, result(RESSIZE);
		double tau1, tau2, tau3, beta1, beta2, beta3, like1;
		batch_result.resize(RESSIZE, grid_num);
		g = G.col(col_idx);
		reg1snp(&g, &gg, &s, &like1, &result);
		beta1 = result(5);
		beta2 = result(6);
		beta3 = result(7);
		tau1 = beta1;
		TauIni(result(0), beta2, beta3, &tau2, &tau3);
		vc1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &result);
		batch_result.col(col_idx) = result;
		outFile << chromosome[col_idx] << " " <<  SNPID[col_idx] << " " << rsid[col_idx] << " " << position[col_idx] << " " << allele1[col_idx] << " " <<  allele2[col_idx];
		outFile << " " << batch_result(0, col_idx)  << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
		outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
		outFile << " " << batch_result(8, col_idx)  << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << " " << batch_result(11, col_idx) << endl;
	}
	else if (flag[9] == 1 && grid_flag && flag[20] == 0) 
	{
		grid_num = tau1_num * tau2_num * tau3_num;
		tau_grid.resize(3, grid_num);
		batch_result.resize(RESSIZE, grid_num);
		cout << endl << "Grid search with mixed model analysis of " << this_snp << " started."  << endl;
		logFile << endl << "Grid search with mixed model analysis of " << this_snp << " started."  << endl;
		logFile << "chrom SNPID rsid position alleleA alleleB tau1_ini tau2_ini tau3_ini freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
		VectorXd g, gg, s, reg_result(RESSIZE);
		double like1;
		g = G.col(0);
		reg1snp(&g, &gg, &s, &like1, &reg_result);
		#pragma omp parallel for
		for (col_idx = 0; col_idx < grid_num; col_idx++) 
		{
			VectorXd this_result(RESSIZE);
			int tau1_idx = col_idx / (tau2_num * tau3_num);
			int tau2_idx = (col_idx - tau1_idx * tau2_num * tau3_num) / tau3_num;
			int tau3_idx = col_idx -  tau1_idx * tau2_num * tau3_num - tau2_idx * tau3_num;
			double tau1, tau2, tau3;
			if (tau1_num == 1) tau1 = tau1_start;
			else tau1 = (tau1_end - tau1_start) / (tau1_num - 1) * tau1_idx + tau1_start;
			if (tau2_num == 1) tau2 = tau2_start;
			else tau2 = (tau2_end - tau2_start) / (tau2_num - 1) * tau2_idx + tau2_start;
			if (tau3_num == 1) tau3 = tau3_start;
			else tau3 = (tau3_end - tau3_start) / (tau3_num - 1) * tau3_idx + tau3_start;
			this_result = reg_result;
			if (tau3 < 0.000001) tau3 = 0.000001;
			mixed1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &this_result);
			batch_result.col(col_idx) = this_result;
			tau_grid(0, col_idx) =  tau1;
			tau_grid(1, col_idx) =  tau2;
			tau_grid(2, col_idx) =  tau3;
		}
		double chi2_max = -1;
		int chi2_idx = -1;
		for (col_idx = 0; col_idx < grid_num; col_idx++) 
		{
			logFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0] << " " << tau_grid(0, col_idx) << " " << tau_grid(1, col_idx) << " " << tau_grid(2, col_idx);
			logFile << " " << batch_result(0, col_idx) << " " << batch_result(1, col_idx) << " " << batch_result(2, col_idx) << " " << batch_result(3, col_idx);
			logFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx) << " " << batch_result(6, col_idx) << " " << batch_result(7, col_idx);  
			logFile << " " << batch_result(8, col_idx) << " " << batch_result(9, col_idx) << " " << batch_result(10, col_idx) << " " << batch_result(11, col_idx) << endl;
			if (batch_result(8, col_idx) > chi2_max)
			{
				chi2_max = batch_result(8, col_idx);
				chi2_idx = col_idx;
			}
		}
		outFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0];
		outFile << " " << batch_result(0, chi2_idx) << " " << batch_result(1, chi2_idx)  << " " << batch_result(2, chi2_idx)  << " " << batch_result(3, chi2_idx);
		outFile << " " << batch_result(4, chi2_idx) << " " << batch_result(5, chi2_idx)  << " " << batch_result(6, chi2_idx)  << " " << batch_result(7, chi2_idx);  
		outFile << " " << batch_result(8, chi2_idx) << " " << batch_result(9, chi2_idx)  << " " << batch_result(10, chi2_idx)  << " " << batch_result(11, chi2_idx) << endl;
	}
	else if (flag[9] == 1 && grid_flag && flag[20] == 1) 
	{
		cout << endl << "Auto-grid search with mixed model analysis of " << this_snp << " started."  << endl;
		logFile << endl << "Auto-grid search with mixed model analysis of " << this_snp << " started."  << endl;
		logFile << "chrom SNPID rsid position alleleA alleleB tau1_ini tau2_ini tau3_ini freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
		VectorXd g, gg, s, reg_result(RESSIZE), result(RESSIZE);
		double beta1, beta2, beta3, like1;
		g = G.col(0);
		reg1snp(&g, &gg, &s, &like1, &result);
		beta1 = result(5);
		beta2 = result(6);
		beta3 = result(7);
		reg_result = result;
		if (beta3 > 0) 
		{
			grid_num = 1;
			tau_grid.resize(3, grid_num);
			batch_result.resize(RESSIZE, grid_num);
			result = reg_result;
			if (beta3 < 0.000001) beta3 = 0.000001;
			mixed1snp(&g, &gg, &s, beta1, beta2, beta3, like1, &result);
			batch_result.col(0) = result;
			tau_grid(0, 0) =  beta1;
			tau_grid(1, 0) =  beta2;
			tau_grid(2, 0) =  beta3;
			col_idx = 0;
			logFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0] << " " << tau_grid(0, col_idx) << " " << tau_grid(1, col_idx) << " " << tau_grid(2, col_idx);
			logFile << " " << batch_result(0, col_idx) << " " << batch_result(1, col_idx) << " " << batch_result(2, col_idx) << " " << batch_result(3, col_idx);
			logFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx) << " " << batch_result(6, col_idx) << " " << batch_result(7, col_idx);  
			logFile << " " << batch_result(8, col_idx) << " " << batch_result(9, col_idx) << " " << batch_result(10, col_idx) << " " << batch_result(11, col_idx) << endl;
			outFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0];
			outFile << " " << batch_result(0, col_idx) << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
			outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
			outFile << " " << batch_result(8, col_idx) << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << " " << batch_result(11, col_idx) << endl;
		}
		else
		{
			grid_num = GRIDNUM;
			tau_grid.resize(3, grid_num);
			batch_result.resize(RESSIZE, grid_num);
			tau2_start = beta2 + beta3;
			tau2_end = beta2 + 2.0 * beta3;			
			#pragma omp parallel for
			for (col_idx = 0; col_idx < grid_num; col_idx++) 
			{
				VectorXd this_result(RESSIZE);
				double tau1, tau2, tau3;
				tau1 = beta1;
				tau2 = (tau2_end - tau2_start) / (grid_num - 1) * col_idx + tau2_start;
				tau3 = -beta3;
				if (tau3 < 0.000001) tau3 = 0.000001;
				this_result = reg_result;
				mixed1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &this_result);
				batch_result.col(col_idx) = this_result;
				tau_grid(0, col_idx) =  tau1;
				tau_grid(1, col_idx) =  tau2;
				tau_grid(2, col_idx) =  tau3;
			}
			double chi2_max = -1;
			int chi2_idx = -1;
			for (col_idx = 0; col_idx < grid_num; col_idx++) 
			{
				logFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0] << " " << tau_grid(0, col_idx) << " " << tau_grid(1, col_idx) << " " << tau_grid(2, col_idx);
				logFile << " " << batch_result(0, col_idx) << " " << batch_result(1, col_idx) << " " << batch_result(2, col_idx) << " " << batch_result(3, col_idx);
				logFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx) << " " << batch_result(6, col_idx) << " " << batch_result(7, col_idx);  
				logFile << " " << batch_result(8, col_idx) << " " << batch_result(9, col_idx) << " " << batch_result(10, col_idx) << " " << batch_result(11, col_idx) << endl;
				if (batch_result(8, col_idx) > chi2_max)
				{
					chi2_max = batch_result(8, col_idx);
					chi2_idx = col_idx;
				}
			}
			outFile << chromosome[0] << " " <<  SNPID[0] << " " << rsid[0] << " " << position[0] << " " << allele1[0] << " " <<  allele2[0];
			outFile << " " << batch_result(0, chi2_idx) << " " << batch_result(1, chi2_idx)  << " " << batch_result(2, chi2_idx)  << " " << batch_result(3, chi2_idx);
			outFile << " " << batch_result(4, chi2_idx) << " " << batch_result(5, chi2_idx)  << " " << batch_result(6, chi2_idx)  << " " << batch_result(7, chi2_idx);  
			outFile << " " << batch_result(8, chi2_idx) << " " << batch_result(9, chi2_idx)  << " " << batch_result(10, chi2_idx)  << " " << batch_result(11, chi2_idx) << endl;
		}
	}
	else if (flag[9] == 1)
	{
		cout << endl << "Mixed model analysis of " << this_snp << " started."  << endl;
		logFile << endl << "Mixed model analysis of " << this_snp << " started."  << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
		col_idx = 0; 
		grid_num = 1;
		VectorXd g, gg, s, result(RESSIZE);
		double tau1, tau2, tau3, beta1, beta2, beta3, like1;
		batch_result.resize(RESSIZE, grid_num);
		g = G.col(col_idx);
		reg1snp(&g, &gg, &s, &like1, &result);
		beta1 = result(5);
		beta2 = result(6);
		beta3 = result(7);
		tau1 = beta1;
		TauIni(result(0), beta2, beta3, &tau2, &tau3);
		mixed1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &result);
		batch_result.col(col_idx) = result;
		outFile << chromosome[col_idx] << " " <<  SNPID[col_idx] << " " << rsid[col_idx] << " " << position[col_idx] << " " << allele1[col_idx] << " " <<  allele2[col_idx];
		outFile << " " << batch_result(0, col_idx)  << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
		outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
		outFile << " " << batch_result(8, col_idx)  << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << " " << batch_result(11, col_idx) << endl;
	}

	delete [] probs;
	bgen_partition_destroy(partition);
	bgen_metafile_close(mf);
	bgen_file_close(bgen);
	outFile.close();
	if (grid_flag)
	{
		cout << "The maximum likelihood result was save to " << result_file << "." << endl;
		logFile << "The maximum likelihood result was save to " << result_file << "." << endl;
		cout << "Full grid-search results are available in " << out_file << ".log" << "." << endl;
		logFile << "Full grid-search results are available in " << out_file << ".log" << "." << endl;
	}
	else
	{
		cout << "The result was save to " << result_file << "." << endl;
		logFile << "The result was save to " << result_file << "." << endl;
	}
}

void analysisNsnps() 
{
	ofstream outFile;
	int analyzed_snp_num;
	string result_file, this_rsid;
	bool last_snp, batch_filled;
	if (flag[7] == 1) result_file = out_file + ".reg";
	else if (flag[8] == 1) result_file = out_file + ".vc";
	else if (flag[9] == 1) result_file = out_file + ".mixed";
	else if (flag[23] == 1) result_file = out_file + ".creg";
	outFile.open(result_file, ios::out);
	MatrixXd G;
	G.resize(nomiss_sample_num, batch_size);
	batch_result.resize(RESSIZE, batch_size);
	SNPID.resize(batch_size);
	rsid.resize(batch_size);
	chromosome.resize(batch_size);
	allele1.resize(batch_size);
	allele2.resize(batch_size);
	position.resize(batch_size);
	struct bgen_file* bgen;
	struct bgen_metafile* mf;
	struct bgen_partition const* partition;
	struct bgen_variant const* vm;
	struct bgen_genotype* vg;
	bgen = bgen_file_open(bgen_file.c_str());
	if(bgen == NULL)
	{
		cout << "ERROR: Unable to open " << bgen_file  << "!" << endl;
		logFile << "ERROR: Unable to open " << bgen_file  << "!" << endl;
		logFile.close();
		exit(0);		
	}
	if(bgen_file_nsamples(bgen) != sample_num)
	{
		cout << "ERROR: Sample number in " <<  bgen_file <<  " is " << bgen_file_nsamples(bgen) << ", which does not match the sample number " << sample_num <<  " in file " << sample_file << "!" << endl;
		logFile <<  "ERROR: Sample number in " <<  bgen_file <<  " is " << bgen_file_nsamples(bgen) << ", which does not match the sample number " << sample_num <<  " in file " << sample_file << "!" << endl;
		logFile.close();
		exit(0);		
	}
	snp_num = bgen_file_nvariants(bgen);
	int batch_num = ceil(double(snp_num) / batch_size);
	cout << endl << sample_num << " samples and " << snp_num << " variants found in " <<  bgen_file << "." << endl;;
	logFile << endl << sample_num << " samples and " << snp_num << " variants found in " <<  bgen_file << "." << endl;

	mf = bgen_metafile_create(bgen, (bgen_file + ".metafile").c_str(), 1, 0);
	partition = bgen_metafile_read_partition(mf, 0);
	double *probs = new double[sample_num * 3];
	if (flag[7] == 1) 
	{
		cout << "Regression analysis started." << endl;
		logFile << "Regression analysis started." << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het" << endl;
	}
	else if (flag[8] == 1) 
	{
		cout << "Variance component analysis started." << endl;
		logFile << "Variance component analysis started." << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
	}
	else if (flag[9] == 1) 
	{
		cout << "Mixed model analysis started." << endl;
		logFile << "Mixed model analysis started." << endl;
		outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp tau1 tau2 tau3 chi2_het p_het r2_het convergence_flag" << endl;
	}
	analyzed_snp_num = 0;

	last_snp = false;
	int i = 0;
	while (!last_snp)
	{
	int col_idx, col_max;
	col_idx = 0;
	batch_filled = false;
	while (!batch_filled && !last_snp) 
	{
		int batch_idx = i / batch_size;
		if (batch_idx * batch_size == i)
		{
			int snp_start = batch_idx * batch_size;
			int snp_end = (batch_idx + 1) * batch_size;
			if (batch_idx == batch_num - 1) snp_end = snp_num;
			if (batch_idx != 0)
			{
				cout << " done." << endl;
				logFile << " done." << endl;
			}
			cout << " scanning SNP " << snp_start + 1  << " to SNP " << snp_end << "..." << flush;
			logFile << " scanning SNP " << snp_start + 1  << " to SNP " << snp_end << "..." << flush;
		}
		vm = bgen_partition_get_variant(partition, i);
		i ++;
		if (i == snp_num) last_snp = true;
		this_rsid.assign(bgen_string_data(vm->rsid), bgen_string_length(vm->rsid));
		if (rs_set.find(this_rsid) == rs_set.end()) continue;
		SNPID[col_idx].assign(bgen_string_data(vm->id), bgen_string_length(vm->id));
		rsid[col_idx].assign(bgen_string_data(vm->rsid), bgen_string_length(vm->rsid));
		chromosome[col_idx].assign(bgen_string_data(vm->chrom), bgen_string_length(vm->chrom));
		position[col_idx] = vm->position;
		allele1[col_idx].assign(bgen_string_data(vm->allele_ids[0]), bgen_string_length(vm->allele_ids[0]));
		allele2[col_idx].assign(bgen_string_data(vm->allele_ids[1]), bgen_string_length(vm->allele_ids[1]));
		if (vm->nalleles != 2)
		{
			cout << SNPID[col_idx]  << " has " << vm->nalleles << " alleles, skipped!" << endl;
			logFile << SNPID[col_idx]  << " has " << vm->nalleles << " alleles, skipped!" << endl;
			continue;
		}
		vg = bgen_file_open_genotype(bgen, vm->genotype_offset);
		if (bgen_genotype_ncombs(vg) != 3)
		{
			cout << SNPID[col_idx]  << " has " << bgen_genotype_ncombs(vg) << " genotypes, skipped!" << endl;
			logFile << SNPID[col_idx]  << " has " << bgen_genotype_ncombs(vg) << " genotypes, skipped!" << endl;
			continue;
		}

		bgen_genotype_read(vg, probs);
		int row_idx= 0;
		for (std::size_t j = 0; j < sample_num; j++) 
		{
			if (!miss_flag[j]) 
			{
				G(row_idx, col_idx) = 0;
				for (std::size_t l = 1; l < 3; l++)
				{
					G(row_idx, col_idx) += probs[j * 3 + l] * l;
				}
				row_idx++;
			}
		}
		bgen_genotype_close(vg);
		col_idx++;
		if (col_idx == batch_size) batch_filled = true;
	}
	col_max = col_idx;
	if (flag[7] == 1) 
	{
		#pragma omp parallel for
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			VectorXd g, gg, s, result(RESSIZE);
			double like1;
			g = G.col(col_idx);
			reg1snp(&g, &gg, &s, &like1, &result);
			batch_result.col(col_idx) = result;
		}
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			outFile << chromosome[col_idx] << " " <<  SNPID[col_idx] << " " << rsid[col_idx] << " " << position[col_idx] << " " << allele1[col_idx] << " " <<  allele2[col_idx];
			outFile << " " << batch_result(0, col_idx)  << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
			outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
			outFile << " " << batch_result(8, col_idx) << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << endl;
			analyzed_snp_num++;
		}
	}
	else if (flag[8] == 1) 
	{
		#pragma omp parallel for
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			VectorXd g, gg, s, result(RESSIZE), reg_result(RESSIZE);
			double chi2_reg, chi2_vc, beta1, beta2, beta3, tau1, tau2, tau3, like1;
			g = G.col(col_idx);
			reg1snp(&g, &gg, &s, &like1, &result);
			beta1 = result(5);
			beta2 = result(6);
			beta3 = result(7);
			chi2_reg = result(8);
			tau1 = beta1;
			reg_result = result;
			TauIni(result(0), beta2, beta3, &tau2, &tau3);
			vc1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &result);
			chi2_vc =  result(8);
			if (chi2_reg - chi2_vc < grid_if || beta3 > 0) batch_result.col(col_idx) = result;
			else
			{
				VectorXd ml_result(RESSIZE);
				int grid_num = GRIDNUM;
				double tau2_start = beta2 + beta3;
				double tau2_end = beta2 + 2.0 * beta3;	
				double chi2_max = -1;
				tau3 = -beta3;
				if (tau3 < 0.000001) tau3 = 0.000001;
				for (int grid_idx = 0; grid_idx < grid_num; grid_idx++) 
				{
					result = reg_result;
					tau2 = (tau2_end - tau2_start) / (grid_num - 1) * grid_idx + tau2_start;
					vc1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &result);
					if (result(8) > chi2_max) 
					{
						chi2_max = result(8);
						ml_result = result;
					}
				}
				batch_result.col(col_idx) = ml_result;
			}
		}
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			outFile << chromosome[col_idx] << " " <<  SNPID[col_idx] << " " << rsid[col_idx] << " " << position[col_idx] << " " << allele1[col_idx] << " " <<  allele2[col_idx];
			outFile << " " << batch_result(0, col_idx)  << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
			outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
			outFile << " " << batch_result(8, col_idx)  << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << " " << batch_result(11, col_idx) << endl;
			analyzed_snp_num++;
		}
	}
	else if (flag[9] == 1) 
	{
		#pragma omp parallel for
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			VectorXd g, gg, s, result(RESSIZE), reg_result(RESSIZE);
			double chi2_reg, chi2_mixed, beta1, beta2, beta3, tau1, tau2, tau3, like1;
			g = G.col(col_idx);
			reg1snp(&g, &gg, &s, &like1, &result);
			beta1 = result(5);
			beta2 = result(6);
			beta3 = result(7);
			chi2_reg = result(8);
			tau1 = beta1;
			reg_result = result;
			TauIni(result(0), beta2, beta3, &tau2, &tau3);
			mixed1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &result);
			chi2_mixed =  result(8);
			if (chi2_reg - chi2_mixed < grid_if || beta3 > 0) batch_result.col(col_idx) = result;
			else
			{
				VectorXd ml_result(RESSIZE);
				int grid_num = GRIDNUM;
				double tau2_start = beta2 + beta3;
				double tau2_end = beta2 + 2.0 * beta3;	
				double chi2_max = -1;
				tau3 = -beta3;
				if (tau3 < 0.000001) tau3 = 0.000001;
				for (int grid_idx = 0; grid_idx < grid_num; grid_idx++) 
				{
					result = reg_result;
					tau2 = (tau2_end - tau2_start) / (grid_num - 1) * grid_idx + tau2_start;
					mixed1snp(&g, &gg, &s, tau1, tau2, tau3, like1, &result);
					if (result(8) > chi2_max) 
					{
						chi2_max = result(8);
						ml_result = result;
					}
				}
				batch_result.col(col_idx) = ml_result;
			}
		}
		for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
		{
			outFile << chromosome[col_idx] << " " <<  SNPID[col_idx] << " " << rsid[col_idx] << " " << position[col_idx] << " " << allele1[col_idx] << " " <<  allele2[col_idx];
			outFile << " " << batch_result(0, col_idx)  << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
			outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);  
			outFile << " " << batch_result(8, col_idx)  << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx)  << " " << batch_result(11, col_idx) << endl;
			analyzed_snp_num++;
		}
	}
	cout << " done." << endl;
	logFile << " done." << endl;
	}
	delete [] probs;
	bgen_partition_destroy(partition);
	bgen_metafile_close(mf);
	bgen_file_close(bgen);
	outFile.close();
	cout << analyzed_snp_num << " SNPs analyzed, the results were saved to " << result_file << "." << endl;
	logFile << analyzed_snp_num << " SNPs analyzed, the results were saved to " << result_file << "." << endl;
}

double genotype(double p0, double p1, double p2)
{
	double genotype = 0.0;
	if (p1 > p0 && p1 > p2) genotype = 1.0;
	if (p2 > p0 && p2 > p1) genotype = 2.0;
	return genotype;
}

void Levene()
{
	ofstream outFile;
	int analyzed_snp_num, g0_num, g1_num, g2_num;
	string result_file;
	result_file = out_file + ".lev";
	outFile.open(result_file, ios::out);
	MatrixXd G;
	MatrixXi G_num;
	G.resize(nomiss_sample_num, batch_size);
	G_num.resize(3, batch_size);
	batch_result.resize(RESSIZE, batch_size);
	SNPID.resize(batch_size);
	rsid.resize(batch_size);
	chromosome.resize(batch_size);
	allele1.resize(batch_size);
	allele2.resize(batch_size);
	position.resize(batch_size);
	maf_flag.resize(batch_size);
	struct bgen_file* bgen;
	struct bgen_metafile* mf;
	struct bgen_partition const* partition;
	struct bgen_variant const* vm;
	struct bgen_genotype* vg;
	bgen = bgen_file_open(bgen_file.c_str());
	if(bgen == NULL)
	{
		cout << "ERROR: Unable to open " << bgen_file  << "!" << endl;
		logFile << "ERROR: Unable to open " << bgen_file  << "!" << endl;
		logFile.close();
		exit(0);		
	}
	if(bgen_file_nsamples(bgen) != sample_num)
	{
		cout << "ERROR: Sample number in " <<  bgen_file <<  " is " << bgen_file_nsamples(bgen) << ", which does not match the sample number " << sample_num <<  " in file " << sample_file << "!" << endl;
		logFile <<  "ERROR: Sample number in " <<  bgen_file <<  " is " << bgen_file_nsamples(bgen) << ", which does not match the sample number " << sample_num <<  " in file " << sample_file << "!" << endl;
		logFile.close();
		exit(0);		
	}
	snp_num = bgen_file_nvariants(bgen);
	cout << sample_num << " samples and " << snp_num << " variants found in " <<  bgen_file << "." << endl;
	logFile << sample_num << " samples and " << snp_num << " variants found in " <<  bgen_file << "." << endl;

	int batch_num = ceil(double(snp_num) / batch_size);
	mf = bgen_metafile_create(bgen, (bgen_file + ".metafile").c_str(), 1, 0);
	partition = bgen_metafile_read_partition(mf, 0);
	double *probs = new double[sample_num * 3];
	cout << "Levene's (Brown-Forsythe) test started." << endl;
	logFile << "Levene's (Brown-Forsythe) test started." << endl;
	outFile << "chrom SNPID rsid position alleleA alleleB freqB beta_snp se_snp p_snp r2_snp N_AA N_AB N_BB chi2_het p_het df_het" << endl;
	maf_filterred_num = 0;
	analyzed_snp_num = 0;

	for (int batch_idx = 1; batch_idx <= batch_num; ++ batch_idx)
	{
		int snp_start = (batch_idx - 1) * batch_size;
		int snp_end = batch_idx * batch_size;
		if (batch_idx == batch_num)
		{
			snp_end = snp_num;
		}
	cout << " scanning SNP " << snp_start + 1  << " to SNP " << snp_end << "..." << flush;
	logFile << " scanning SNP " << snp_start + 1  << " to SNP " << snp_end << "..." << flush;
	int col_idx, col_max;
	col_idx = 0;
	for (uint32_t i = snp_start; i < snp_end; i++) 
	{
		vm = bgen_partition_get_variant(partition, i);
		SNPID[col_idx].assign(bgen_string_data(vm->id), bgen_string_length(vm->id));
		rsid[col_idx].assign(bgen_string_data(vm->rsid), bgen_string_length(vm->rsid));
		chromosome[col_idx].assign(bgen_string_data(vm->chrom), bgen_string_length(vm->chrom));
		position[col_idx] = vm->position;
		allele1[col_idx].assign(bgen_string_data(vm->allele_ids[0]), bgen_string_length(vm->allele_ids[0]));
		allele2[col_idx].assign(bgen_string_data(vm->allele_ids[1]), bgen_string_length(vm->allele_ids[1]));
		if (vm->nalleles != 2)
		{
			cout << SNPID[col_idx]  << " has " << vm->nalleles << " alleles, skipped!" << endl;
			logFile << SNPID[col_idx]  << " has " << vm->nalleles << " alleles, skipped!" << endl;
			continue;
		}
		vg = bgen_file_open_genotype(bgen, vm->genotype_offset);
		if (bgen_genotype_ncombs(vg) != 3)
		{
			cout << SNPID[col_idx]  << " has " << bgen_genotype_ncombs(vg) << " genotypes, skipped!" << endl;
			logFile << SNPID[col_idx]  << " has " << bgen_genotype_ncombs(vg) << " genotypes, skipped!" << endl;
			continue;
		}

		bgen_genotype_read(vg, probs);
		int row_idx = 0;
		g0_num = 0;
		g1_num = 0;
		g2_num = 0;
		for (std::size_t j = 0; j < sample_num; j++) 
		{
			if (!miss_flag[j]) 
			{
				G(row_idx, col_idx) = genotype(probs[j * 3], probs[j * 3 +1], probs[j * 3 +2]);
				if (G(row_idx, col_idx) == 0) g0_num++;
				else if (G(row_idx, col_idx) == 1) g1_num++;
				else g2_num++;
				row_idx++;
			}
		}
		G_num(0, col_idx) = g0_num;
		G_num(1, col_idx) = g1_num;
		G_num(2, col_idx) = g2_num;
		bgen_genotype_close(vg);
		col_idx++;
	}
	col_max = col_idx;
	#pragma omp parallel for
	for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
	{
		double freq;
		VectorXd g, result(RESSIZE);
		g = G.col(col_idx);
		freq = g.sum() / nomiss_sample_num / 2.0;
		if (freq >= freq_min && freq <= freq_max)
		{ 
			maf_flag(col_idx) = 0;
			lev1snp(&g, G_num(0, col_idx), G_num(1, col_idx), G_num(2, col_idx), &result);
			batch_result.col(col_idx) = result;
		}
		else maf_flag(col_idx) = 1;
	}
	for (uint32_t col_idx = 0; col_idx < col_max; col_idx++) 
	{
		if (maf_flag(col_idx) == 0)
		{
//			cout  << endl << G_num(0, col_idx) << '\t' << G_num(1, col_idx) << '\t' << G_num(2, col_idx) << '\t' << G_num(0, col_idx) + G_num(1, col_idx) + G_num(2, col_idx);
			outFile << chromosome[col_idx] << " " <<  SNPID[col_idx] << " " << rsid[col_idx] << " " << position[col_idx] << " " << allele1[col_idx] << " " <<  allele2[col_idx];
			outFile << " " << batch_result(0, col_idx) << " " << batch_result(1, col_idx)  << " " << batch_result(2, col_idx)  << " " << batch_result(3, col_idx);
			outFile << " " << batch_result(4, col_idx) << " " << batch_result(5, col_idx)  << " " << batch_result(6, col_idx)  << " " << batch_result(7, col_idx);
			outFile << " " << batch_result(8, col_idx)  << " " << batch_result(9, col_idx)  << " " << batch_result(10, col_idx) << endl;  
			analyzed_snp_num++;
		}
		else maf_filterred_num++;
	}
//	cout  << endl ;
	cout << " done." << endl;
	logFile << " done." << endl;
	}
	delete [] probs;
	bgen_partition_destroy(partition);
	bgen_metafile_close(mf);
	bgen_file_close(bgen);
	outFile.close();
	cout <<  maf_filterred_num << " SNPs having MAFs < " << maf_min << ", " << analyzed_snp_num << " SNPs analyzed, the results were saved to " << result_file << "." << endl;
	logFile << maf_filterred_num << " SNPs having MAFs < " << maf_min << ", " << analyzed_snp_num << " SNPs analyzed, the results were saved to " << result_file << "." << endl;
}

};

int main(int argc, char ** argv)
{
	time_t current_time;
	char * current_time_str;
	Dataset dat(argc, argv);
	if (dat.error_flag)
	{
		dat.logFile.close();
		exit(0);
	}
	stringstream ss;
	ss << dat.thread_num;
	setenv("OMP_NUM_THREADS", ss.str().c_str(), 1);
	omp_set_num_threads(dat.thread_num);
	cout << "The program will be running on " << dat.thread_num << " thread(s)." << endl;
	dat.logFile << "The program will be running on " << dat.thread_num << " thread(s)." << endl;
	current_time = time(0);
	current_time_str = ctime(&current_time);
	cout << endl << "Start time: " << current_time_str;
	dat.logFile << endl << "Start time: " << current_time_str;
	dat.checkSampleFile();
	dat.readSampleFile();
	dat.summPhenoCovs();
	dat.Ini();
	if (dat.flag[24] == 1) dat.Levene();
	else if (dat.flag[10] == 1) dat.analysis1snp();
	else if (dat.flag[22] == 1) 
	{
		dat.readrsfile();
		dat.analysisNsnps();
	}
	else dat.analysis();
	current_time = time(0);
	current_time_str = ctime(&current_time);
	cout << "End time: " << current_time_str << endl;
	dat.logFile << "End time: " << current_time_str << endl;
	return 0;
}
