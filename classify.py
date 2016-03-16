import csv
import collections as clt
import random
from sklearn import svm
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metric import classification_report

#ORIGINAL dataset features

#iyear,imonth,iday,approxdate,extended,resolution,country,country_txt,region,region_txt,provstate,city,
# latitude,longitude,specificity,vicinity,location,summary,crit1,crit2,crit3,doubtterr,alternative,alternative_txt,
# multiple,success,suicide,attacktype1,attacktype1_txt,attacktype2,attacktype2_txt,attacktype3,attacktype3_txt,targtype1,
# targtype1_txt,targsubtype1,targsubtype1_txt,corp1,target1,natlty1,natlty1_txt,targtype2,targtype2_txt,targsubtype2,
# targsubtype2_txt,corp2,target2,natlty2,natlty2_txt,targtype3,targtype3_txt,targsubtype3,targsubtype3_txt,corp3,target3,
# natlty3,natlty3_txt,gname,gsubname,gname2,gsubname2,gname3,gsubname3,motive,guncertain1,guncertain2,guncertain3,nperps,
# nperpcap,claimed,claimmode,claimmode_txt,claim2,claimmode2,claimmode2_txt,claim3,claimmode3,claimmode3_txt,compclaim,
# weaptype1,weaptype1_txt,weapsubtype1,weapsubtype1_txt,weaptype2,weaptype2_txt,weapsubtype2,weapsubtype2_txt,weaptype3,
# weaptype3_txt,weapsubtype3,weapsubtype3_txt,weaptype4,weaptype4_txt,weapsubtype4,weapsubtype4_txt,weapdetail,nkill,
# nkillus,nkillter,nwound,nwoundus,nwoundte,property,propextent,propextent_txt,propvalue,propcomment,ishostkid,nhostkid,
# nhostkidus,nhours,ndays,divert,kidhijcountry,ransom,ransomamt,ransomamtus,ransompaid,ransompaidus,ransomnote,hostkidoutcome,
# hostkidoutcome_txt,nreleased,addnotes,scite1,scite2,scite3,dbsource,INT_LOG,INT_IDEO,INT_MISC,INT_ANY,related

#PRUNED dataset features

#iyear,imonth,iday,extended,country,country_txt,region,region_txt,provstate,city,specificity,vicinity,crit1,crit2,crit3,doubtterr,
# multiple,success,suicide,attacktype1,attacktype1_txt,targtype1,targtype1_txt,targsubtype1,targsubtype1_txt,natlty1,natlty1_txt,gname,guncertain1,
# weaptype1,weaptype1_txt,weapsubtype1,weapsubtype1_txt,weaptype2,ransom

#FINAL dataset features

#1970..........2014, 1...12, 1...31,extended, country1 country2...country3, region1 region2...region3, specficity, vicinity, crit1, crti2, crit3, doubterr,
#multiple,success,suicide,attacktype1,attacktype1_txt,targtype1,targtype1_txt,targsubtype1,targsubtype1_txt,natlty1,natlty1_txt,gname,guncertain1,
# weaptype1,weaptype1_txt,weapsubtype1,weapsubtype1_txt,weaptype2,ransom  

#**********************Data Cleaning to obtain the input vector X and label Y . **********************##

#The data set had modly ordinal and categorical features. To overcome this I converted each feature into multi-dimensional binary variables (dimensionality equal to the total possible
#number of values for that feature)

X = []
Y=[]
yearL = []
monthL = []
dayL = []
countryL = []
regionL = []
attacktype1L = []
targettype1L = []
targetsubtype1L = []
natlty1L = []
weaptype1L = []
weapsubtype1L = []
yearDict = clt.defaultdict(int)
groupDict = clt.defaultdict(int)
countryDict = clt.defaultdict(int)
regionDict = clt.defaultdict(int)
attacktype1Dict = clt.defaultdict(int)
targettype1Dict = clt.defaultdict(int)
targetsubtype1Dict = clt.defaultdict(int)
natlty1Dict = clt.defaultdict(int)
weaptype1Dict = clt.defaultdict(int)
weapsubtype1Dict = clt.defaultdict(int)
groupCount = 0
countryCount = 0
regionCount = 0
attacktype1Count = 0
targettype1Count = 0
targetsubtype1Count = 0
natlty1Count = 0
weaptype1Count = 0
weapsubtype1Count = 0
countryDictName = clt.defaultdict(int)
regionDictName = clt.defaultdict(int)
attacktype1DictName = clt.defaultdict(int)
targettype1DictName = clt.defaultdict(int)
targetsubtype1DictName = clt.defaultdict(int)
natlty1DictName = clt.defaultdict(int)
groupL = []
weaptype1DictName = clt.defaultdict(int)
weapsubtype1DictName = clt.defaultdict(int)
with open('dataset_pruned.csv') as dataset:
	read = csv.reader(dataset, delimiter=',', quotechar='|')
	next(dataset)
	for i in read:
		yearInt = int(i[0])
		yearL.append(yearInt)
		yearDict[yearInt]=yearInt-1970
		monthInt = int(i[1])
		monthL.append(monthInt)
		dayInt = int(i[2])
		dayL.append(dayInt)
		countryDictName[int(i[4])] = i[5]
		countryL.append(int(i[4]))
		if int(i[4]) not in countryDict:
			countryDict[int(i[4])]= countryCount
			countryCount+=1
		regionDictName[int(i[6])] = i[7]
		regionL.append(int(i[6]))
		if int(i[6]) not in regionDict:
			regionDict[int(i[6])]= regionCount
			regionCount+=1
		attacktype1DictName[int(i[19])] = i[20]
		if int(i[19]) not in attacktype1Dict:
			attacktype1Dict[int(i[19])]= attacktype1Count
			attacktype1Count+=1
		attacktype1L.append(int(i[19]))
		if int(i[21]) not in targettype1Dict:
			targettype1Dict[int(i[21])]= targettype1Count
			targettype1Count+=1
		targettype1DictName[int(i[21])] = i[22]
		targettype1L.append(int(i[21]))
		try:
			targetsubtype1DictName[int(i[23])] = i[24]
			targetsubtype1L.append(int(i[23]))
			if int(i[23]) not in targetsubtype1Dict:
				targetsubtype1Dict[int(i[23])]= targetsubtype1Count
				targetsubtype1Count+=1
		except:
			targetsubtype1DictName[-1] = 'unk'
			targetsubtype1L.append(-1)
			if -1 not in targetsubtype1Dict:
				targetsubtype1Dict[-1]= targetsubtype1Count
				targetsubtype1Count+=1

		try:
			natlty1DictName[int(i[25])] = i[26]
			natlty1L.append(int(i[25]))
			if int(i[25]) not in natlty1Dict:
				natlty1Dict[int(i[25])]= natlty1Count
				natlty1Count+=1
		except:
			natlty1DictName[-1] = 'unk'
			natlty1L.append(-1)
			if -1 not in natlty1Dict:
				natlty1Dict[-1]= natlty1Count
				natlty1Count+=1			
		gname = i[27]
		groupL.append(gname)
		if gname not in groupDict:
			groupDict[gname]= groupCount
			groupCount+=1
		weaptype1DictName[int(i[29])] = i[30]
		weaptype1L.append(int(i[29]))
		if int(i[29]) not in weaptype1Dict:
			weaptype1Dict[int(i[29])]= weaptype1Count
			weaptype1Count+=1
		try:
			weapsubtype1DictName[int(i[31])] = i[32]
			weapsubtype1L.append(int(i[31]))
			if int(i[31]) not in weapsubtype1Dict:
				weapsubtype1Dict[int(i[31])]= weapsubtype1Count
				weapsubtype1Count+=1
		except:
			weapsubtype1DictName[-1] = 'unk'
			weapsubtype1L.append(-1)
			if -1 not in weapsubtype1Dict:
				weapsubtype1Dict[-1]= weapsubtype1Count
				weapsubtype1Count+=1


	yearL = list(set(yearL))
	monthL = list(set(monthL))
	dayL = list(set(dayL))
	groupL = list(set(groupL))
	countryL = list(set(countryL))
	regionL = list(set(regionL))
	attacktype1L = list(set(attacktype1L))
	targettype1L = list(set(targettype1L))
	targetsubtype1L = list(set(targetsubtype1L))
	natlty1L = list(set(natlty1L))
	weaptype1L = list(set(weaptype1L))	
	weapsubtype1L = list(set(weapsubtype1L))
	

	dataset.seek(0)
	read = csv.reader(dataset, delimiter=',', quotechar='|')
	next(dataset)



	for j in read:
		x = []
		yearVector = [0]*(len(yearL)+1)
		monthVector = [0]*13
		dayVector = [0]*32
		countryVector = [0]*(max(countryL)+1)
		regionVector = [0]*(max(regionL)+1)
		attacktype1Vector = [0]*(max(attacktype1L)+1)
		targettype1vector = [0]*(max(targettype1L)+1)
		targetsubtype1Vector = [0]*(max(targetsubtype1L)+1)
		natlty1Vector = [0]*(max(natlty1L)+1)
		weaptype1Vector = [0]*(max(weaptype1L)+1)
		weapsubtype1Vector = [0]*(max(weapsubtype1L)+1)

		yearVector[yearDict[int(j[0])]] = 1
		monthVector[int(j[1])] = 1
		dayVector[int(j[2])] = 1
		countryVector[countryDict[int(j[4])]] = 1
		regionVector[regionDict[int(j[6])]] = 1
		attacktype1Vector[attacktype1Dict[int(j[19])]] = 1
		try:
			targetsubtype1Vector[targettype1Dict[int(j[23])]] = 1
		except:
			targetsubtype1Vector[targettype1Dict[-1]] = 1
		try:
			natlty1Vector[natlty1Dict[int(j[25])]] = 1
		except:
			natlty1Vector[natlty1Dict[-1]] = 1
		weaptype1Vector[weaptype1Dict[int(j[29])]] = 1
		try:
			weapsubtype1Vector[weapsubtype1Dict[int(j[31])]] = 1
		except:
			weapsubtype1Vector[weapsubtype1Dict[-1]] = 1

		x= yearVector+monthVector+dayVector+countryVector+regionVector+attacktype1Vector+targetsubtype1Vector+natlty1Vector+weaptype1Vector+weapsubtype1Vector
		x.append(int(j[3]))
		try:
			x.append(int(j[10]))
		except:
			x.append(-1)
		try:
			x.append(int(j[11]))
		except:
			x.append(-1)
		try:
			x.append(int(j[12]))
		except:
			x.append(-1)
		try:
			x.append(int(j[13]))
		except:
			x.append(-1)
		try:
			x.append(int(j[14]))
		except:
			x.append(-1)
		try:
			x.append(int(j[15]))
		except:
			x.append(-1)
		try:
			x.append(int(j[16]))
		except:
			x.append(-1)
		try:
			x.append(int(j[17]))
		except:
			x.append(-1)
		try:
			x.append(int(j[18]))
		except:
			x.append(-1)
		try:
			x.append(int(j[28]))
		except:
			x.append(-1)
		try:
			x.append(int(j[34]))
		except:
			x.append(-1)
		X.append(x)
		Y.append(groupDict[j[27]])

	#**********************Classification using Multi-Class SVM **********************##



	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

	myClassifier = svm.SVC() #uses RBF Kernel by default
	myClassifier.fit(X_train,y_train)

	predicted = myClassifier.predict(X_test)

	print "**********ACCURACY OF THE CLASSIFIER*************"

	print accuracy_score(y_test,predicted)

	
	print "**********PREDICTED LABELS FOR TEST DATA*************"
	
	for i in predicted:
		print groupDict[i]





	








	




		

