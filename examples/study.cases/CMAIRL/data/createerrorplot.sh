#!/bin/bash


for i in {1..10}
do
    run=${i}
    outfile="policies_l2error_${i}.eps"

    policies="_korali_result_reg_01_${run}-t-0.5 _korali_result_reg_01_${run}-t-0.25 _korali_result_reg_01_${run}-t-0.125 _korali_result_reg_01_${run}-t-0.0625 _korali_result_reg_01_${run}-t-0.0"

    obsfiles="observations-vracer-reg-1.csv observations-vracer-reg-2.csv observations-vracer-reg-3.csv observations-vracer-reg-4.csv observations-vracer-reg-4.csv observations-vracer-reg-5.csv observations-vracer-reg-6.csv observations-vracer-reg-7.csv observations-vracer-reg-8.csv observations-vracer-reg-9.csv observations-vracer-reg-10.csv" 

    python comparepolicy.py --policies $policies --obsfile $obsfiles --outfile $outfile

done
