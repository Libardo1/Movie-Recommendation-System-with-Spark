import sys
import csv
import math
import pyspark
sc=pyspark.SparkContext("local")

rd=sys.argv[1]
td=sys.argv[2]

ratings=sc.textFile(rd)
ratings=ratings.map(lambda x: x.split(",")).map(lambda x:(x[0],(x[1],x[2])))

test=sc.textFile(td)
test=test.map(lambda x: x.split(",")).filter(lambda x: x[0]!='userId')\
         .map(lambda x: (x[0],x[1]))

raw=ratings.filter(lambda x: x[0]!='userId')\
        .map(lambda x: ((x[0],x[1][0]),x[1][1]))

test_label=test.map(lambda x: ((x[0],x[1]),'test'))
train_test=raw.leftOuterJoin(test_label)

# train data
# (userid,(movieid,rating))
rating=train_test.filter(lambda x: x[1][1]!='test')\
   .map(lambda x: (x[0][0],(x[0][1],x[1][0])))

# test data
# (userid,(movieid,rating))
test_rating=train_test.filter(lambda x: x[1][1]=='test')\
   .map(lambda x: (x[0][0],(x[0][1],x[1][0])))

# user_avg
# (userid,user_avg)
user_avg=rating.map(lambda x: (x[0],(float(x[1][1]),1)))\
       .reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1])).map(lambda x: (x[0],x[1][0]/x[1][1]))

# movie,(userid,rating)
corate=rating.map(lambda x: (x[1][0],(x[0],x[1][1])))

# corate_pair (movie, ((user1,rating1),(useri,ratingi)) remove self-self pair
corate_pair=corate.join(corate).filter(lambda x: x[1][0]!=x[1][1])

# corate_p (user1,useri),(rating1,1)
corate_p=corate_pair.map(lambda x: ((x[1][0][0],x[1][1][0]),(float(x[1][0][1]),1)))

# avg_corate
# ((user1,useri),corate_avg_1)
avg_corate=corate_p.reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1]))\
           .map(lambda x:(x[0],x[1][0]/x[1][1]))

# corate_rating
# (user1,useri),(rating1,ratingi)
corate_rating=corate_pair.map(lambda x:((x[1][0][0],x[1][1][0]),(float(x[1][0][1]),float(x[1][1][1]))))

# pearson similarity
# (user1,useri),similarity
similarity=corate_rating.join(avg_corate).map(lambda x: (x[0], ((x[1][0][0]-x[1][1]),(x[1][0][1]-x[1][1]))))\
                        .map(lambda x: (x[0],(x[1][0]*x[1][1], math.pow(x[1][0],2),math.pow(x[1][1],2))))\
                        .reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2]))\
                        .map(lambda x: (x[0], x[1][0]/(math.sqrt(x[1][1])*math.sqrt(x[1][2])+0.0000001)))

# user1,(useri,similarity)
user_similarity=similarity.map(lambda x: (x[0][0],(x[0][1],x[1])))

# useri_rating
# (useri,movie),rating
useri_rating=rating.map(lambda x: ((x[0],x[1][0]),x[1][1]))

### get top n nearest neighbour for test users/movies
# user1,[(ui,ri),(ui,si),(ui,si),(ui,si),(ui,si)]
user_similarity_k=user_similarity.sortBy(lambda x: math.fabs(x[1][1]), ascending=False)\
               .groupByKey().map(lambda x: (x[0],list(x[1]))).map(lambda x: (x[0],x[1][:330]))

# prediction
# pred_part1: ((user1,movie1),part1)
def f(x): return x
pred_part1=test.join(user_similarity_k).map(lambda x: ((x[0],x[1][0]),x[1][1])).flatMapValues(f)\
                .map(lambda x: ((x[1][0],x[0][1]),(x[0][0],x[1][1])))\
                .join(useri_rating).map(lambda x: ((x[0][0],x[1][0][0]),(x[0][1],x[1][0][1],float(x[1][1]))))\
                .join(avg_corate).map(lambda x: ((x[0][1],x[1][0][0]),((x[1][0][2]-x[1][1])*x[1][0][1],math.fabs(x[1][0][1]))))\
                .reduceByKey(lambda x,y: ((x[0]+y[0]),(x[1]+y[1]))).map(lambda x: (x[0],x[1][0]/(x[1][1]+0.0000001)))

# user1,(movie1,[(ui,si).(),(),(),()])
# (user1,movie1),[(ui,si).(),(),(),()]
# (user1,movie1),(ui,si) *5
# (ui,movie1),(user1,si)
# (ui,movie1),((user1,si),ri)
# (ui,user1),(movie1,si,ri)
# (ui,user1),((movie1,si,ri),avg_ri)
# (user1,movie1),((si-avg_ri)*si,|si|)
# aggragate --> (user1,movie1),(numerator,demoninator)
# (user1,movie1),(numerator/denominator)

# pred_part2:(user1,movie1),user1_avg
pred_part2= test.join(user_avg).map(lambda x: ((x[0],x[1][0]),x[1][1]))

prediction_1= pred_part2.leftOuterJoin(pred_part1).filter(lambda x: x[1][1]==None)\
                        .map(lambda x: (x[0],x[1][0]))

predcition_2=pred_part2.leftOuterJoin(pred_part1).filter(lambda x: x[1][1]!=None)\
                       .map(lambda x:(x[0],x[1][0]+x[1][1]))
                       
prediction=prediction_1.union(predcition_2).map(lambda x: [x[0],x[1]])

small=prediction.filter(lambda x: x[1]<0).map(lambda x: (x[0],0))
large=prediction.filter(lambda x: x[1]>5).map(lambda x: (x[0],5))   
medium=prediction.filter(lambda x: x[1]<5 and x[1]>0).map(lambda x: (x[0],x[1]))   
final_prediction=small.union(large).union(medium).sortBy(lambda x:(int(x[0][0]),int(x[0][1])),ascending=True)\
                      .map(lambda x: (int(x[0][0]),int(x[0][1]),x[1]))

title=[('UserId','MovieId','Pred_rating')]
title=sc.parallelize(title)
output=title.union(final_prediction)
output.take(5)

def removeBracket(data):
    return ','.join(str(i) for i in data)
    
output.map(removeBracket).repartition(1).saveAsTextFile('Chong_Li_result_task2')

###### evaluation #########
Diff=test_rating.map(lambda x:((x[0],x[1][0]),float(x[1][1]))).join(final_prediction)\
           .map(lambda x:(x[0],math.fabs(x[1][0]-x[1][1])))

RMSE=Diff.map(lambda x:('RMSE',(pow(x[1],2),1)))\
         .reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1]))\
         .map(lambda x: (x[0],str(math.sqrt(x[1][0]/x[1][1]))))

c1=0
c2=0
c3=0
c4=0
c5=0
for row in Diff.collect():
    if row[1]<1 and row[1]>=0: c1+=1
    elif row[1]<2 and row[1]>=1: c2+=1
    elif row[1]<3 and row[1]>=2: c3+=1
    elif row[1]<4 and row[1]>=3: c4+=1
    elif row[1]>=4: c5+=1

print ">=0 and <1: ", c1 
print ">=1 and <2: ", c2 
print ">=2 and <3: ", c3 
print ">=3 and <4: ", c4 
print ">=4: ", c5 
for i in RMSE.collect():
    print 'RMSE = ', i[1]
