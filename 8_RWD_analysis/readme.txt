Before analysis, please download RXNORM.csv at: https://drive.google.com/file/d/1Xnz6b7pK_-BFHCCZhcxCb1fDn9Yo7JtM/view?usp=share_link,
and put it into code/mapping folder

1) PD_pre_drug.py-------checking drug prevalence and build dictionary to save each patient's drug historical records; please run it first.
2) PD_pre_demo.py--------building PD patient cohort and generate index date
3) PD_cohort_build-------based on inclusion and exclusion criteria, build cohort for each drug
4) PD_PSMatching.py------PSM (performing propensity score matching)
5) PD_compute_HR.py------compute Hazard ratio
