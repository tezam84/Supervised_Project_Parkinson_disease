# Detection_Parkinson_disease

Parkinson’s Disease (PD) is a degeneration of the brain leading to cognitive problems. 
It’s one of the most frequent neurodegenerative diseases and it affects more than 10 million globally. 
It causes trembling, stiffness, and issues with walking, balancing, and coordinating. 
Symptoms come gradually with sleep difficulties, depression, exhaustion, and memory problems.
Yet, the diagnosis is still difficult to evaluate, especially in its early stages.
PD disease manifests itself differently in each individual.
In this project, the goal is to demonstrate if a correlation exists between PD diagnosing positively and the patient’s ability to speak distinctly.
More info here: www.parkinson.org 

It’s a supervised learning problem with labeled data (we know the input and output). Involves building a model to estimate or predict an output based on one or more inputs.
We use classification. The most popular algorithms are :
Random Forest
KNN
AdaBoost
Decision Tree
Naive Bayes

It’s a dataset composed of biomedical voice measurements from 50 people 
There 6 recordings per patient
195 rows of data and 23 features for the columns. 
The features are numeric except for the first column “name” 
Let’s go deeper.
Patients number 7, 10, 13, 17, 42, 43, 49, and 50 do not have PD (8 patients),  and others do.

In the first column “name” listed 
“phon_R01_S01_1” stands for the first recording of the first patient
“phon_R01_S01_2” stands for the second recording of the first patient etc…

MDVP: Fo(Hz) - Average vocal fundamental frequency 
MDVP: Fhi(Hz) - Maximum vocal fundamental frequency 
MDVP: Flo(Hz) - Minimum vocal fundamental frequency 
MDVP: Jitter(%),
MDVP: Jitter(Abs),
MDVP: RAP,
MDVP: PPQ
Jitter:DDP - Several measures of variation in fundamental frequency MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude NHR,HNR -

Health status of the subject 1 = unhealthy (PD)
0 = healthy 

RPDE, D2 - Two nonlinear dynamical complexity measures 
DFA - Signal fractal scaling exponent 
spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation 


