# detection-of-cyberbullying-on-social-media-using-machine-learing-
Cyberbullying is a growing issue in the digital age, impacting both teenagers and adults, often leading to severe consequences such as depression and suicide. This study focuses on the detection of cyberbullying in text data using Natural Language Processing (NLP) and Machine Learning techniques. Utilizing data from two sources
#!/usr/bin/envpython
"""Django'scommand-lineutilityforadministrativetasks.""" 
import os
importsys
def main():
"""Run administrative 
tasks."""os.environ.setdefault('DJANGO_SETTINGS_MODULE', 
'detection_of_cyberbullying.settings')
try:
fromdjango.core.managementimportexecute_from_command_line 
except ImportError as exc:
raiseImportError( 
"Couldn'timportDjango.Areyousureit'sinstalledand"
) from exc 
execute_from_command_line(sys.argv)
ifname =="main": 
#Checkforenvironmentvariables
required_env_vars=['DJANGO_SETTINGS_MODULE','SECRET_KEY', 
'DATABASE_URL']
missing_vars=[varforvarinrequired_env_varsifnotos.getenv(var)] if 
missing_vars:
print(f"Warning:Missingenvironmentvariables:{','.join(missing_vars)}")
#Provideahelpfulmessageforcommoncommands if 
len(sys.argv) > 1:
command=sys.argv[1] 
ifcommand=='runserver':
print("StartingDjangodevelopmentserver...") 
elif command == 'migrate':
print("Applyingdatabasemigrations...") 
elif command == 'createsuperuser':
print("Creatingasuperuser.Followtheprompts.") 
elif command == 'startapp':
iflen(sys.argv)>2: 
print(f"CreatinganewDjangoapp:{sys.argv[2]}") else: 
print("Error:Pleasespecifyanappname.")
main()
REMOTE USER:
fromdjango.db.modelsimportCount 
from django.db.models import Q
fromdjango.shortcutsimportrender,redirect,get_object_or_404 
import datetime
importopenpyxl
importwarnings 
warnings.filterwarnings('ignore') 
import pandas as pd 
importnumpyasnp
importmatplotlib.pyplotasplt import 
seaborn as sns
fromsklearn.model_selectionimporttrain_test_split,GridSearchCV 
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.naive_bayes import 
MultinomialNBfromsklearn.ensembleimportRandomF 
orestClassifier
fromsklearn.metricsimportaccuracy_score,confusion_matrix, classification_report 
importre
importpandasaspd 
fromsklearn.ensembleimportVotingClassifier
#Createyour viewshere.
from Remote_User.models import 
ClientRegister_Model,Tweet_Message_model,Tweet_Prediction_model,detection_ 
ratio_model,detection_accuracy_model
deflogin(request): 
ifrequest.method=="POST"and'submit1'inrequest.POST:
username = request.POST.get('username')
password=request.POST.get('password') 
try:
enter = 
ClientRegister_Model.objects.get(username=username,password=password)
request.session["userid"]=enter.id 
return redirect('Search_DataSets')
except:
pass
return render(request,'RUser/login.html') 
def Add_DataSet_Details(request):
returnrender(request,'RUser/Add_DataSet_Details.html',{"excel_data":''})
defRegister1(request): 
ifrequest.method =="POST":
username=request.POST.get('username') 
email = request.POST.get('email') 
password=request.POST.get('password') 
phoneno = request.POST.get('phoneno') 
country = request.POST.get('country') 
state = request.POST.get('state')
city = request.POST.get('city') 
ClientRegister_Model.objects.create(username=username, email=email,
password=password,phoneno=phoneno,
country=country,state=state,city=city)
returnrender(request,'RUser/Register1.html') 
else:
returnrender(request,'RUser/Register1.html')
def ViewYourProfile(request): 
userid=request.session['userid'] 
obj=ClientRegister_Model.objects.get(id=userid)
returnrender(request,'RUser/ViewYourProfile.html',{'object':obj})
defSearch_DataSets(request): 
ifrequest.method =="POST":
Tweet_Message=request.POST.get('keyword') 
df = pd.read_csv("./train_tweets.csv") 
df.head()
offensive_tweet=df[df.label==1] 
offensive_tweet.head() 
normal_tweet = df[df.label == 0] 
normal_tweet.head() 
#OffensiveWordclouds
from os import pathfrom 
PIL import Image
fromwordcloudimportWordCloud,STOPWORDS,ImageColorGenerator 
text = "".join(review for review in offensive_tweet) 
wordcloud=WordCloud(max_font_size=50,max_words=100,
background_color="white").generate(text) 
fig = plt.figure(figsize=(20, 6))
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off")
#plt.show() 
#distributions
df_Stat=df[['label','tweet']].groupby('label').count().reset_index()
df_Stat.columns=['label','count'] 
df_Stat['percentage']=(df_Stat['count']/df_Stat['count'].sum())*100 df_Stat
defprocess_tweet(tweet):
return"".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])","", tweet.lower()).split())
df['processed_tweets']=df['tweet'].apply(process_tweet) 
df.head()
#Asthisdataset ishighlyimbalancewehavetobalancethisbyoversampling 
cnt_non_fraud = df[df['label'] == 0]['processed_tweets'].count() 
df_class_fraud = df[df['label'] == 1] 
df_class_nonfraud=df[df['label']== 0] 
df_class_fraud_oversample=df_class_fraud.sample(cnt_non_fraud,
replace=True)
df_oversampled=pd.concat([df_class_nonfraud, 
df_class_fraud_oversample], axis=0)
print('Random over-sampling:') 
print(df_oversampled['label'].value_counts()) 
# Split data into training and test sets
fromsklearn.model_selectionimporttrain_test_split X
= df_oversampled['processed_tweets'] 
y=df_oversampled['label']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, 
stratify=None)
fromsklearn.feature_extraction.textimportCountVectorizer,
TfidfTransformer
count_vect = CountVectorizer(stop_words='english') 
transformer=TfidfTransformer(norm='l2',sublinear_tf=True) 
x_train_counts = count_vect.fit_transform(X_train) 
x_train_tfidf = transformer.fit_transform(x_train_counts) 
print(x_train_counts.shape)
print(x_train_tfidf.shape)
x_test_counts = count_vect.transform(X_test) 
x_test_tfidf=transformer.transform(x_test_counts)
models= []
# SVMModel
fromsklearnimport svm
lin_clf = svm.LinearSVC() 
lin_clf.fit(x_train_tfidf, y_train) 
predict_svm=lin_clf.predict(x_test_tfidf)
svm_acc=accuracy_score(y_test,predict_svm)*100 
print("SVM ACCURACY")
print(svm_acc) 
models.append(('svm', lin_clf))
detection_accuracy_model.objects.create(names="SVM",ratio=svm_acc)
fromsklearn.metricsimportconfusion_matrix,f1_score 
print(confusion_matrix(y_test, predict_svm)) 
print(classification_report(y_test, predict_svm))
#classifier=VotingClassifier(models) 
##classifier.fit(X_train, y_train) 
#y_pred = classifier.predict(X_test)
review_data=[Tweet_Message] 
vector1=count_vect.transform(review_data).toarray() 
predict_text = lin_clf.predict(vector1)
pred=str(predict_text).replace("[","") 
pred1 = pred.replace("]", "")
prediction=int(pred1) if 
prediction == 0:
val= 'NonOffensive orNonCyberbullying'
elifprediction==1: 
val='OffensiveorCyberbullying'
Tweet_Prediction_model.objects.create(Tweet_Message=Tweet_Message,Pre 
diction_Type=val)
returnrender(request,'RUser/Search_DataSets.html',{'objs':val}) 
return render(request, 'RUser/Search_DataSets.html')
SERVICE PROVIDER:
from django.db.models importCount, Avg 
fromdjango.shortcutsimportrender,redirect 
from django.db.models import Count 
fromdjango.db.modelsimportQ import 
datetime
importxlwt 
fromdjango.httpimportHttpResponse 
import warnings 
warnings.filterwarnings('ignore') 
import pandas as pd 
importnumpyasnp 
importmatplotlib.pyplotasplt import 
seaborn as sns
fromsklearn.model_selectionimporttrain_test_split,GridSearchCV 
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.naive_bayes import 
MultinomialNBfromsklearn.ensembleimportRandomF 
orestClassifier
fromsklearn.metricsimportaccuracy_score,confusion_matrix, classification_report 
importre
importpandasaspd
#Createyour viewshere.
from Remote_User.models import 
ClientRegister_Model,Tweet_Message_model,Tweet_Prediction_model,detection_ 
ratio_model,detection_accuracy_model
defserviceproviderlogin(request): 
ifrequest.method== "POST":
admin = request.POST.get('username') 
password=request.POST.get('password')
if admin == "Admin" and password =="Admin": 
detection_accuracy_model.objects.all().delete() 
return redirect('View_Remote_Users')
return render(request,'SProvider/serviceproviderlogin.html') 
def Find_Cyberbullying_Prediction_Ratio(request):
detection_ratio_model.objects.all().delete()
ratio =""
kword= 'NonOffensiveorNonCyberbullying'
print(kword) 
obj=Tweet_Prediction_model.objects.all().filter(Q(Prediction_Type=kword)) 
obj1 = Tweet_Prediction_model.objects.all()
count = obj.count(); 
count1=obj1.count(); 
ratio=(count/count1)*100 if 
ratio != 0:
detection_ratio_model.objects.create(names=kword,ratio=ratio)
ratio1="" 
kword1='OffensiveorCyberbullying'print(kword1)
obj1=Tweet_Prediction_model.objects.all().filter(Q(Prediction_Type=kword1)) obj11
= Tweet_Prediction_model.objects.all() 
count1 = obj1.count(); 
count11=obj11.count(); 
ratio1=(count1/count11)*100 if
ratio1 != 0: 
detection_ratio_model.objects.create(names=kword1,ratio=ratio1)
obj=detection_ratio_model.objects.all() 
returnrender(request,'SProvider/Find_Cyberbullying_Prediction_Ratio.html',
{'objs':obj})
def View_Remote_Users(request): 
obj=ClientRegister_Model.objects.all()
returnrender(request,'SProvider/View_Remote_Users.html',{'objects':obj})
defViewTrendings(request): 
topic =
Tweet_Prediction_model.objects.values('topics').annotate(dcount=Count('topics')). 
order_by('-dcount')
returnrender(request,'SProvider/ViewTrendings.html',{'objects':topic})
defcharts(request,chart_type): 
chart1 =
detection_ratio_model.objects.values('names').annotate(dcount=Avg('ratio')) 
return render(request,"SProvider/charts.html", {'form':chart1,
'chart_type':chart_type})
defcharts1(request,chart_type): 
chart1 =
detection_accuracy_model.objects.values('names').annotate(dcount=Avg('ratio')) 
return render(request,"SProvider/charts1.html", {'form':chart1,
'chart_type':chart_type}) 
defView_Cyberbullying_Predict_Type(request):
obj =Tweet_Prediction_model.objects.all() 
returnrender(request,'SProvider/View_Cyberbullying_Predict_Type.html',
{'list_objects':obj})
deflikeschart(request,like_chart): 
charts
=detection_accuracy_model.objects.values('names').annotate(dcount=Avg('ratio')) 
return render(request,"SProvider/likeschart.html", {'form':charts,
'like_chart':like_chart}) 
defDownload_Cyber_Bullying_Prediction(request):
response=HttpResponse(content_type='application/ms-excel') # 
decide file name
response['Content-Disposition'] = 'attachment; 
filename="Cyberbullying_Predicted_DataSets.xls"'
#creatingworkbook 
wb=xlwt.Workbook(encoding='utf-8') # 
adding sheet 
ws=wb.add_sheet("sheet1") #
Sheet header, first row 
row_num = 0 
font_style=xlwt.XFStyle() 
# headers are bold 
font_style.font.bold = True
#writer=csv.writer(response) 
obj=Tweet_Prediction_model.objects.all() 
data =obj# dummy method to fetch data. for 
my_row in data:
row_num= row_num+ 1 
ws.write(row_num,0,my_row.Tweet_Message,font_style) 
ws.write(row_num,1,my_row.Prediction_Type,font_style)
wb.save(response) 
return response
def train_model(request): 
detection_accuracy_model.objects.all().delete()
df=pd.read_csv("./train_tweets.csv") 
df.head()
offensive_tweet=df[df.label==1] 
offensive_tweet.head() 
normal_tweet = df[df.label == 0] 
normal_tweet.head() 
#OffensiveWordclouds
from os import pathfrom 
PIL import Image
fromwordcloudimportWordCloud,STOPWORDS,ImageColorGenerator text
= "".join(review for review in offensive_tweet) 
wordcloud=WordCloud(max_font_size=50,max_words=100,
background_color="white").generate(text) 
fig = plt.figure(figsize=(20, 6))
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off")
#plt.show() 
#distributions
df_Stat=df[['label','tweet']].groupby('label').count().reset_index() 
df_Stat.columns = ['label', 'count'] 
df_Stat['percentage']=(df_Stat['count']/df_Stat['count'].sum())*100 
df_Stat
defprocess_tweet(tweet):
return"".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])","", 
tweet.lower()).split())
df['processed_tweets']=df['tweet'].apply(process_tweet) 
df.head()
#Asthisdataset ishighlyimbalancewehavetobalancethisbyoversampling 
cnt_non_fraud = df[df['label'] == 0]['processed_tweets'].count() 
df_class_fraud = df[df['label'] == 1] 
df_class_nonfraud=df[df['label']== 0] 
df_class_fraud_oversample=df_class_fraud.sample(cnt_non_fraud,
replace=True) 
df_oversampled=pd.concat([df_class_nonfraud,df_class_fraud_oversample],
axis=0)
print('Random over-sampling:') 
print(df_oversampled['label'].value_counts()) 
# Split data into training and test sets
fromsklearn.model_selectionimporttrain_test_split X
= df_oversampled['processed_tweets'] 
y=df_oversampled['label']
X_train,X_test,y_train,y_test =train_test_split(X,y, test_size=0.2,
stratify=None) 
fromsklearn.feature_extraction.textimportCountVectorizer,TfidfTransformer 
count_vect = CountVectorizer(stop_words='english') 
transformer=TfidfTransformer(norm='l2',sublinear_tf=True)
x_train_counts = count_vect.fit_transform(X_train) 
x_train_tfidf = transformer.fit_transform(x_train_counts) 
print(x_train_counts.shape)
print(x_train_tfidf.shape)
x_test_counts = count_vect.transform(X_test) 
x_test_tfidf=transformer.transform(x_test_counts)
#SVM Model
from sklearn import svm 
lin_clf=svm.LinearSVC() 
lin_clf.fit(x_train_tfidf, y_train) 
predict_svm=lin_clf.predict(x_test_tfidf)
svm_acc=accuracy_score(y_test,predict_svm)*100 
print("SVM ACCURACY")
print(svm_acc) 
detection_accuracy_model.objects.create(names="SVM",ratio=svm_acc)
fromsklearn.metricsimportconfusion_matrix,f1_score 
print(confusion_matrix(y_test,predict_svm)) 
print(classification_report(y_test, predict_svm))
#LogisticRegressionModel 
fromsklearn.linear_modelimportLogisticRegression 
logreg = LogisticRegression(random_state=42)
# Building Logistic RegressionModel 
logreg.fit(x_train_tfidf, y_train) 
predict_log=logreg.predict(x_test_tfidf) 
logistic=accuracy_score(y_test,predict_log)*100 
print("Logistic Accuracy")
print(logistic)detection_accuracy_model.objects.create(names="Logistic 
Regression",
ratio=logistic) 
fromsklearn.metricsimportconfusion_matrix,f1_score 
fromsklearn.metricsimportconfusion_matrix,f1_score 
print(confusion_matrix(y_test,predict_log)) 
print(classification_report(y_test, predict_log))
fromsklearn.naive_bayesimport MultinomialNB
NB = MultinomialNB() 
NB.fit(x_train_tfidf, y_train) 
predict_nb=NB.predict(x_test_tfidf)
naivebayes=accuracy_score(y_test,predict_nb)*100 
print("Naive Bayes")
print(naivebayes) 
detection_accuracy_model.objects.create(names="Naive Bayes",
ratio=naivebayes) 
print(confusion_matrix(y_test,predict_nb)) 
print(classification_report(y_test, predict_nb))
#Test Data Set 
df_test=pd.read_csv("./test_tweets.csv") 
df_test.head()
df_test.shape
df_test['processed_tweets']=df_test['tweet'].apply(process_tweet) 
df_test.head()
X = df_test['processed_tweets'] 
x_test_counts=count_vect.transform(X) 
x_test_tfidf=transformer.transform(x_test_counts) 
df_test['predict_nb'] = NB.predict(x_test_tfidf) 
df_test[df_test['predict_nb'] == 1] 
df_test['predict_svm'] = NB.predict(x_test_tfidf) 
#df_test['predict_rf'] = model.predict(x_test_tfidf) 
df_test.head()
file_name = 
'Predictions.csv'df_test.to_csv(file_name, 
index=False)
obj=detection_accuracy_model.objects.all()
return render(request,'SProvider/train_model.html', {'objs': 
obj,'svmcm':confusion_matrix(y_test,predict_svm),'lrcm':confusion_matrix(y_test, 
predict_log),'nbcm':confusion_matrix(y_test,predict_nb)})
