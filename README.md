# heter
"heter" is a program for conducting genome-wide variance quantitative trait locus (vQTL) analysis of quantitative traits with genotype data in bgen format. It comes with the analyses based on linear mixed model, variance component model, regression, chi-square resression as well as the Leven's (Brown-Forsythe) test.

## Author
Gang Shi (gshi@xidian.edu.cn)

## Dependencies
+ [EIGEN] (https://eigen.tuxfamily.org)
+ [Intel MKL] (https://software.intel.com/)
+ [Boost] (https://www.boost.org/)
+ [zlib] (https://www.zlib.net/)
+ [bgen] (https://github.com/limix/bgen)

## Installation
The archive includes an executable binary "heter" pre-compiled under CentOS8.0 (x86-64). To compile from source code, edit Makefile to point EIGEN_PATH, BGEN and MKL_PATH to your own locations of EIGEN3, BGEN and MKL, then type "make".

## Usage
Type "./heter --help" from the command line to display program options of heter:

    --bgen		Input genotype file in bgen format.
    --sample	Input sample file in SNPTEST sample file format. Missing data is coded as "NA".
    --out		Output file name [default: heter].
    --pheno		Specify the continous phenotype for analysis.
    --covs		Specify the covariate(s) to be adjusted in the analysis.
    --vc		Conduct the analysis with the variance component model.
    --mixed		Conduct the analysis with the linear mixed model.
    --reg		Conduct the analysis by the linear regression.
    --creg		Conduct the analysis by the chi-square regression.
    --lev		Conduct the analysis by the Levene's (Brown-Forsythe) test.
    --maf-min	Minimum MAF of the SNPs for the analysis [default: 0.01].
    --thread-num	Number of threads on which the program will be running [default: thread number in your machine - 1].
    --batch-size	Number of SNPs to be processed in a batch [default: 1000].
    
## Citation
Shi G. Genome-wide variance quantitative trait locus analysis of blood pressures in UK Biobank. In submission.
