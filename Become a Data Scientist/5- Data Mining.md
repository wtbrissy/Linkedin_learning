## 5. Data Mining 
### Table of Contents 
- [Preliminaries](#preliminaries)
- [Data Reduction](#data-reduction)
- [Clustering](#clustering)
- [Classification](#classification)
- [Anomalies Detection](#anomalies-detection)
- [Regression Analysis](#regression-analyosis)
- [Sequential Patterns](#sequential-patterns)
- [Text Mining](#text-mining)
- [Online DS Resourses](#online-ds-resourses)

### Studying List 
- Algorithm - Classical Staticists 
	- [ ] k-means classification 
	- [ ] k-nearest neighbors 
	- [ ] Hierarchical clustering
 - Algorithm - Mechine Learning 
	- [ ]   Hidden Markov models  
	- [ ]   Support vector regression 
	- [ ]   Random forests 
	- [ ]   LASSO regression 

### Preliminaries 
- Data minining prerequisites 
	- Simplify 
		- Reduce noise in data
		- Reduce dimersionality
		- Find important variables or combinations 
	- Groups 
		- Clustering 
		-  Classification 
		-  Association analysis 
		-  Anomaly detection 
	-  Predict scores 
		-  Find variables that can be used to predict outcomes 
		-  Use regression models
-  Algorithm prerequisite
	-  Classical statistics
		-  Methods based on familiar statistics, typically transparent, possible to calculate by hand
			-  Linear regression 
			-  k-means classification 
			-  k-nearest neighbors 
			-  Hierarchical clustering 
	-  Machine leanring 
		-  Complex methods, often opaque, require substanial computing power
			-  Hidden Markov models  
			-  Support vector regression 
			-  Random forests 
			-  LASSO regression 
	-  Apriori algorithm 
		-  Market based analysis 
	-  Word counts
		-  Using work stems, handling stop words, and comparing frequencies 

### Data Reduction 
Data reduction helps to simplify the dataset and focus on variables or constructs that are most likely to carry meaning and least likely to carry noise
- Partical constraints 
	- Storage 
	- Memory 
	- Time 
- Statistical
	- Avoid multicollineartiy 
	- Get increased degress of freedom 
	- Avoid overfitting

Reduce Noise | Focus on Patterns| Easier to Interpret
---|---|---|
Reduce distractions and meaningless information|Make regularities easier to see by 'zoom out'|Is easier to interpret and use larger patterns

- Algorithms for data reduction 

	Linear mthods 
	- Stright lines through data
	- Use linear equations
	- Methods
		- **Principal component analysis(PCA)**
			- Rotation: with mulltiple components, rotated solutions can be easier to interpret
			- Factor analysis: factors are closely related but based on a different theory
			- Interpretaility: for human use, the ability to interpret is critical; less so for machine learning
		- Reduce number of variables 
		- Maximize variablility in lower-dimensional space


	Non-linear methods
	- For high-dimensional manifolds
	- Use complex equations 
	- Methods
		- Useful for non-linear manifolds 
			- Kernel PCA 
			- Isomap
			- locally linear embeeding 
			- Maximum variance unfolding
		- Used in computer vision, etc.
		- Diffult to interpret

### Clustering 
- Goals of Clustering 
	- Clusters ar pragmatic groupings 
	- They allow limited interchangeability 
	- They vary by purpose, data, and algorithm 
- Clustering data 
	- Distance between points 
		- Euclidean distance 
		- Connectivity models 
		- Hierarchical diagrams 
		- Joining or splitting 
		- Converx clusters
		- Slow for big data 
	- Distance from centroid
		- Defines centre point by mean vector 
		- K-means
		- Convex clusters fof similar size 
		- Must choose K
	- Density of data
		- Connected dense regions in k-dimensons 
		- Can model nonconvex clusters and clusters of different sizes
		- Can ignore outliners 
		- Hard to describe 
	- Distribution models
		- Clusters- models as statistical distributions 
		- Multivariate normal 
		- Prone to overfitting 

	![](https://github.com/wtbrissy/Linkedin_learning/blob/main/Images/Pasted%20image%2020211117075910.png)
	
	[Clustering in Python](https://github.com/wtbrissy/Linkedin_learning/blob/main/Become%20a%20Data%20Scientist/Codes/Clustering%20in%20Python.ipynb)
	
	### Classification
	- Classification details
		- Categories already exist 
		- Question is where to put new cases 
		- Models are based on variables in dataset 
	- Classification goals
		- Complements clustering 
		- Supervised vs. unsupervised learning 
		- Limited to data provided to algorithm 
	- Algortihms for Classificaiton 
		- *k*-Nearest Neighbors(*k*-NN)
			- Find the k cases in multidimensional space closest to the new one
			- Take a majority vote on their category 
		- Native Bayes
			- Begin with overall probabilites of group membership
			- Adjust probability with each new piece of infromation
			- Use Bayes's theorem
		- Decision tress
			- Find variables and values that best split cases at multiple levels
			- Follow largest branches to leaf at end(final category)
		- Random forests
			- A collectin of decision trees
			- Randomly select cases and variables or features 
			- More reliable and less prone to overfitting
		- Support vector machines(SVM)
			- Use the ''kernel trick" to find a hyperplane that cleanly separates two groups 
		- Artifical neural networks(ANN)
			- Multiple layers of equations and weights to model nonlinear outcomes 
	- How to choose the right model
		- **Human-in-the-Loop**	
		If human make descisions using principles from resultes, then use*** transparent methods like decision trees or naive Bayes***.
		_ **Black Box Models**
		If the algorithm directly controls the outcome and accuracy is paramount, opaque methods like SVM and ANN are acceptable.
		
		![](https://github.com/wtbrissy/Linkedin_learning/blob/main/Images/Pasted%20image%2020211117092303.png)
		
		[Classification in Pyton](https://github.com/wtbrissy/Linkedin_learning/blob/main/Become%20a%20Data%20Scientist/Codes/Classification%20in%20Python.ipynb)
		
### Anomaly Detection
- Types of outliers 
	- Univariate outliers
		- Variance/SD: cases are serveral standard deviations aways; 
		- Quartiles: distances are based on the interquartile range(IQR) most common
		- Experience: common standards are used for unusual scores 
	- Bivariate and multivariate outliers 
		- Distance measures: many choices, but usually ignore 2D visualization 
		- Bivariate normal distribution: ellipses over scatterplots
		- Density plots
	- Categorical outliers
		- Distance measures: Euclidean distance from centre of dataset 
			- Mahalanobis distance
			- Robust measures of distance 
		- Density measures: local density of data in multdimensional space
			- More flexable and robust 
			- Harder to describe 
- Impacts of outliers 
	- Distorted statistics
	- Distorted relationships
	- Misleading results 
	- Failure to generalize 
- What to do with outliers 
	- Delete: remove the cases if few in number
	- Transform: take logarithm or square the socres to make distribution symmetrical 
	- Robust: use methods that are not strongly influenced by outliers 
- Algorithms for Anomaly Detection 
	- Visual or numerical analysis 
	- Means-based or robust methods 
	- Sensitivity vs. interpretability 

![](https://github.com/wtbrissy/Linkedin_learning/blob/main/Images/Pasted%20image%2020211117113026.png)

[Anomaly Detection in Python](https://github.com/wtbrissy/Linkedin_learning/blob/main/Become%20a%20Data%20Scientist/Codes/Anomaly%20Detection%20in%20Python.ipynb)

### Association Analysis 
- Association analysis goals 
	- Frequent itemsets 
		- Calculate 'supprot' for combinations of items
		- supp(X) = X/T
			- The proportion of transactions T that contain itemset X
			- Set minimum level: minsup
	- Rule generation 
		- if/then statement
		- Calculate 'confidence' or conditional probability 
		- Conf(X->Y) = supp(X u Y)/supp(X)
			- if X occurs, then what is the conditional probaility of Y?
			- Set minimun level: minconf
- Association analysis algorithms
	- Apriori
		- Calculate support for single-item itemsets
		- if supp < minsup, ten remve itemset
		- Then expand to two item itemsets, etc..
	- Eclat
	- FP-growth
	- RElim
	- SaM
	- JIM

[Association Analysis with Apriori in Python](https://github.com/wtbrissy/Linkedin_learning/blob/main/Become%20a%20Data%20Scientist/Codes/Association%20Analysis%20with%20Apriori%20in%20Python.ipynb)

### Regression Analysis 
- Regression analysis goal 
	- Slope and intercept 
	- Corrlelation 
	- Require normality
	- Gives fit and requires linearity 
- Regression analysis algorithms 
	- Classical methods
		- Based on means and squared deviations from predicted values 
			- Simultaneous entry
			- Blocked entry
			- Stepwise entry
			- Nonlinear method
	- Modern methods
		- Alternative methods for calculating distance and for choosing between predictors 
			- LASSO regression 
			- LARS
			- RFE
			- SVR

### Sequential Patterns
- Sequence mining goals 	
	- Temporal associations 
		- Similar to association analysis 
		- Events that go togeter 
		- Order of event important 
- Sequence mining algorithms 
	- GSP(Generalized Sequential Pattern)
		- Similar to apriori association analysis but observes order
		- Uses sliding window for simultaneity 
	- SPADE
		- Sequential Pattern Discovery using Equivalence classes 
		- Fewer database scans by using intersecting ID-lists 
	- FreeSpan 
	- HMM(Hidden Markov models)
		- Qualitatively distinct patterns of behavior 
		- Easy to test specific hypotheses

[Sequence Mining in Python](https://github.com/wtbrissy/Linkedin_learning/blob/main/Become%20a%20Data%20Scientist/Codes/Sequence%20Mining%20in%20Python.ipynb)

### Text Mining 
- Text mining goals 
	- Assessing authorship and voice 
	- Clustering groups of respondents 
	- Sentiment analysis in social media 

- Text mining algorithms 
	- Focus on Meaning 
	- Bag of Words 

### Online DS Resourses
- https://www.kdnuggets.com/
- https://www.kaggle.com/

[Certificate Of Completion Data Science Foundations Data Mining](https://github.com/wtbrissy/Linkedin_learning/blob/main/Certificates/Certificate%20Of%20Completion%20Data%20Science%20Foundations%20Data%20Mining.pdf)