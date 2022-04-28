# heter
"heter" is a program for conducting genome-wide variance quantitative trait locus (vQTL) analysis. It comes with the analyses based on a linear mixed model, variance component model, linear regression, chi-square resression as well as Levene's (Brown-Forsythe) test.

## Author
Gang Shi (gshi@xidian.edu.cn)

## Dependencies
+ [EIGEN] (https://eigen.tuxfamily.org)
+ [Intel MKL] (https://software.intel.com/)
+ [Boost] (https://www.boost.org/)
+ [bgen] (https://github.com/limix/bgen)

## Installation
The archive includes an executable binary "heter" pre-compiled under CentOS8.0 (x86-64). To compile from source code, edit Makefile to point EIGEN_PATH, BGEN_PATH and MKL_PATH to your own locations of EIGEN3, BGEN and MKL, then type "make".

## Usage
Type "./heter --help" from the command line to display the program options:

    --bgen		Input imputed genotype file in bgen format.
    --sample	Input sample file in SNPTEST sample file format. Missing data is coded as "NA".
    --out		Output file name [default: heter].
    --pheno		Specify the continous phenotype for analysis.
    --covs		Specify the covariate(s) to be adjusted in the analysis.
    --vc		Conduct the analysis with the variance component model.
    --mixed		Conduct the analysis with the linear mixed model.
    --reg		Conduct the analysis with the linear regression.
    --creg		Conduct the analysis with the chi-square regression.
    --lev		Conduct the analysis with the Levene's (Brown-Forsythe) test.
    --maf-min	Minimum minor allele frequency of SNPs for the analysis [default: 0.01].
    --thread-num	Number of threads on which the program will be running [default: thread number in your machine - 1].
    --batch-size	Number of SNPs to be processed in a batch [default: 1000].
    
## Citation
Shi G. Novel analysis suggests small interaction effects in blood pressure traits. In submission.
