#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary libraries required to start the EDA for Lead Scoring

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import precision_recall_curve
#sns.set(style="whitegrid")

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns',120)
pd.set_option("display.max_rows",None)
pd.set_option('display.width',None)
pd.options.display.float_format = '{:.3f}'.format
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
pd.options.mode.chained_assignment = None

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Read the dataset
leads_data = pd.read_csv("Leads.csv")
leads_data.head() 


# In[4]:


# let's look at the statistics
leads_data.describe()


# In[5]:


#checking the shape of the dataset
leads_data.shape


# In[6]:


# Checking datatype for the dataset
leads_data.info()


# In[7]:


#Checking null values in each columns
leads_data.isnull().sum().sort_values(ascending=False)


# In[8]:


# Checking for duplicates
print(leads_data.duplicated().sum())


# In[11]:


# Finding the null value percentage
perc_missing_df=100*(leads_data.isna().mean()).sort_values(ascending=False)
perc_missing_df


# In[12]:


### Dropping the columns with more than 40% Null Values
col_to_drop = perc_missing_df[perc_missing_df>=39].index.to_list()
leads_data.drop(labels=col_to_drop, axis=1, inplace=True)
print("dropped columns ", col_to_drop)

# Check null percentage to verify

100*(leads_data.isna().mean()).sort_values(ascending=False)


# In[13]:


# Dropping `Lead Number` and `Prospect ID` as they are not needed in EDA

leads_data.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[14]:


def categorical_col_plot(col):
    """ For a given categorical col, plat count plot against target variable.  """
    plt.rc('xtick', labelsize=8) 
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    for c, ax in zip([col], axs.ravel()):
        s1=sns.countplot(x=c, data=leads_data, ax=ax)
        s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
        s2=sns.countplot(x=leads_data[c], data=leads_data, hue=leads_data.Converted)
        s2.set_xticklabels(s1.get_xticklabels(),rotation=90)


# ##### 1) Specialization

# In[15]:


categorical_col_plot('Specialization')


# In[16]:


# Since the Specialization column has reasonably distributed data in general and high number of converted leads
#     we will keep this column and impute the missing with e.g. "Unspecified"
leads_data['Specialization'] = leads_data['Specialization'].replace(np.nan, 'Unspecified')


# In[17]:


categorical_col_plot('Tags')


# In[18]:


# Most of the values are 'Will revert after reading the email' and also is mostly Converted, 

leads_data['Tags']=leads_data['Tags'].replace(np.nan, 'Unspecified')


# In[19]:


leads_data['Tags'].value_counts()


# In[20]:


leads_data['Tags'] = leads_data['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized', 'switched off',
                                      'Already a student',
                                       'Not doing further education',
                                       'invalid number',
                                       'wrong number given',
                                       'Interested  in full time MBA'],
                                                           "Others")


# #### 3) What matters most to you in choosing a course 

# In[21]:


categorical_col_plot('What matters most to you in choosing a course')


# In[22]:


#Dropping the following columns
leads_data=leads_data.drop('What matters most to you in choosing a course',axis=1)


# #### 4) What is your current occupation

# In[23]:


categorical_col_plot('What is your current occupation')


# In[24]:


# Majority of values are 'Unemployed'
leads_data['What is your current occupation']=leads_data['What is your current occupation'].replace(np.nan,'Unemployed')


# In[25]:


categorical_col_plot('Country')


# In[26]:


# More than 90% values are for India and hence not a useful column, hence dropping it
leads_data=leads_data.drop('Country',axis=1)


# In[27]:


categorical_col_plot('TotalVisits')


# In[28]:


# Has just 1.48% null rows, hence we remove the rows with null values without impacting the overall datasize
leads_data = leads_data[~pd.isnull(leads_data['TotalVisits'])]


# In[29]:


# Has just 1.48% null rows, hence we remove the rows with null values without impacting the overall datasize
leads_data = leads_data[~pd.isnull(leads_data['Page Views Per Visit'])]


# In[30]:


categorical_col_plot('Last Activity')


# In[31]:


# Has just 1.1% null rows, hence we remove the rows with null values without impacting the overall datasize
leads_data = leads_data[~pd.isnull(leads_data['Last Activity'])]


# In[32]:


# Observed above some categories in the "Last Activity" have very few records, so grouping these into into "others"
leads_data['Last Activity'] = leads_data['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                               'Had a Phone Conversation', 
                                                               'Approached upfront',
                                                               'View in browser link Clicked',       
                                                               'Email Marked Spam',                  
                                                               'Email Received','Visited Booth in Tradeshow',
                                                               'Resubscribed to emails'],
                                                           "Others")


# In[33]:


categorical_col_plot('Lead Source')


# In[34]:


# Has just 0.3% null rows, hence we remove the rows with null values without impacting the overall datasize
leads_data = leads_data[~pd.isnull(leads_data['Lead Source'])]


# In[35]:


# Replace 'google' with 'Google'
leads_data['Lead Source'] = leads_data['Lead Source'].replace(['google'], 'Google')


# In[36]:


# Observed above categories in the "Lead Score" have very few rows, so grouping these
leads_data['Lead Source'] = leads_data['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 
                                                               'Pay per Click Ads', 'Press_Release','Social Media', 
                                                               'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home',
                                                               'youtubechannel'],
                                                           "Others")


# In[37]:


# find other categorical variables
other_cate_columns = ['Lead Origin',
 'Do Not Email',
 'Do Not Call',
 'Search',
 'Magazine',
 'Newspaper Article',
 'X Education Forums',
 'Newspaper',
 'Digital Advertisement',
 'Through Recommendations',
 'Receive More Updates About Our Courses',
 'Update me on Supply Chain Content',
 'Get updates on DM Content',
 'I agree to pay the amount through cheque',
 'A free copy of Mastering The Interview',
 'Last Notable Activity']


# In[38]:


# Plotting the categorical variables
fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(11, 20))
plt.subplots_adjust(left=0.1,
                    bottom=0.0,
                    right=0.9,
                    top=2.5,
                    wspace=0.4,
                    hspace=0.4)
for col, ax in zip(other_cate_columns, axs.ravel()):
    figsize=(5, 3)
    plt.title(col)
    s1=sns.countplot(x=col, data=leads_data, ax=ax, hue=leads_data.Converted)
    s1.set_xticklabels(s1.get_xticklabels(), rotation=45)


# In[39]:


# Noticed above categories in the "Last Notable Activity" have very few records, so grouping these into into "others" 
leads_data['Last Notable Activity'] = leads_data['Last Notable Activity'].replace(['Had a Phone Conversation',
                                                                       'Email Marked Spam',
                                                                         'Unreachable',
                                                                         'Unsubscribed',
                                                                         'Email Bounced',                                                                    
                                                                       'Resubscribed to emails',
                                                                       'View in browser link Clicked',
                                                                       'Approached upfront', 
                                                                       'Form Submitted on Website', 
                                                                       'Email Received'], 'Other')


# In[40]:


#dropping columns which are not needed.
leads_data = leads_data.drop(['Search','Magazine','Newspaper Article','X Education Forums',
                            'Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                            'Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque',
                            'Do Not Call'], 1)
leads_data.columns


# In[41]:


100*(leads_data.isna().mean()).sort_values(ascending=False)


# In[42]:


# Size of rows after cleanup
(len(leads_data.index)/9240)*100


# In[43]:


leads_data.describe()


# In[44]:


# Observing Correlation

plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(leads_data.corr(), annot=True,cmap="BrBG", robust=True,linewidth=0.1, vmin=-1 )
plt.show()


# In[ ]:





# In[45]:


total = len(leads_data["Converted"])
plt.figure(figsize = [6, 6])
plt.title("Converted Vs Non-Converted - distribution")
leads_data["Converted"].value_counts().plot.pie(autopct = lambda x : '{:.2f}%\n({:.0f})'.format(x, total*x/100), colors = ["lightblue", "yellow"])

plt.show()


# In[46]:


plt.figure(figsize=(3, 3))
sns.distplot(leads_data['TotalVisits'])
plt.show()


# In[47]:


plt.figure(figsize=(3, 3))
sns.distplot(leads_data['Total Time Spent on Website'])
plt.show()


# In[48]:


plt.figure(figsize=(3, 3))
sns.distplot(leads_data['Page Views Per Visit'])
plt.show()


# In[49]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Page Views Per Visit vs TotalVisits",fontsize=16)
sns.regplot(data=leads_data,y="TotalVisits",x="Page Views Per Visit",fit_reg=True, line_kws={"color": "red"})
plt.xlabel("Page Views Per Visit")
plt.show()


# In[50]:


vars =  ['Do Not Email', 'A free copy of Mastering The Interview']

def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

leads_data[vars] = leads_data[vars].apply(binary_map)


# In[51]:


leads_data.head()


# #### Creating Dummy variables for the categorical features:

# In[52]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy = pd.get_dummies(leads_data[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization', 
                                   'What is your current occupation', 'Tags', 'Last Notable Activity']], drop_first=True)

# Adding the results to the master dataframe
leads_data = pd.concat([leads_data, dummy], axis=1)


# In[53]:


# Drop original categorical vars now 
leads_data = leads_data.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization', 
                                   'What is your current occupation', 'Tags', 'Last Notable Activity'],1)


# In[54]:


leads_data.shape


# In[55]:


# Checking for outliers in the continuous variables
num_leads_data = leads_data[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']]


# In[56]:


# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
num_leads_data.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# In[57]:


def outliers_plot_numerical_data(df):
    """ Method to boxplot a given numerical column to check outliers """
    plt.figure(figsize=[22,11])
    plt.subplots_adjust(wspace=0.4,hspace=0.5)

    for i,j in enumerate(df.columns):
        plt.subplot(2,2,i+1)

        sns.boxplot(y=num_leads_data[j])

        plt.suptitle("\n Boxplot for outliers check",fontsize=20,color="green")
        plt.ylabel(None)
        plt.title(j, fontsize=15, color='red')

outliers_plot_numerical_data(num_leads_data)


# In[58]:



def outlier_treatment(df, cols):
    """ Caps the outliers for the numerical columns using IQR range"""
    for i in cols:
        q1 = df[i].describe()["25%"]
        q3 = df[i].describe()["75%"]
        IQR = q3 - q1

        upper_bound = q3 + 1.5*IQR
        lower_bound = q1 - 1.5*IQR

        # capping upper_bound
        df[i] = np.where(df[i] > upper_bound, upper_bound,df[i])

        # flooring lower_bound
        df[i] = np.where(df[i] < lower_bound, lower_bound,df[i])

        
outlier_treatment(leads_data, ["TotalVisits","Page Views Per Visit"])        


# In[59]:


num_leads_data = leads_data[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']]
num_leads_data.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# In[60]:


outliers_plot_numerical_data(num_leads_data)


# In[61]:



# Putting predictor variables to X
X = leads_data.drop('Converted', axis=1)

# Putting Target variables to y
y = leads_data["Converted"]


# In[62]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[63]:


# using standard scaler for scaling the numerical column data
scaler = StandardScaler()

X_train[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']])


# In[64]:


X_train.shape


# In[65]:


X_train.describe()


# In[66]:


logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=15)             
rfe = rfe.fit(X_train, y_train)


# In[67]:


# RFE Ranking
rfe_ranking = pd.DataFrame({'rank' : rfe.ranking_, 'support': rfe.support_, 'features' : X_train.columns}).sort_values(by='rank',ascending=True)
rfe_ranking


# In[68]:


# columns selected by RFE
rfe_col = X_train.columns[rfe.support_]
rfe_col


# In[69]:


# columns not selected by RFE
X_train.columns[~rfe.support_]


# In[70]:


def vif(X_train):
    """Takes a training dataframe and calculates VIF for the feature columns in desc VIFs """
    vif = pd.DataFrame()
    X = X_train
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif


# In[71]:


# Creating X_train dataframe with variables selected
X_train_rfe = X_train[rfe_col]

# Adding a constant 
X_train_sm = sm.add_constant(X_train_rfe)

# Create a fitted model
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial()).fit() 

logm1.params


# In[72]:


X_train_rfe.shape


# In[73]:



print(logm1.summary())
X_train_sm =  X_train_sm.drop(['const'], axis=1)
vif(X_train_sm)


# In[74]:



X_train_sm = sm.add_constant(X_train_sm)
X_train_sm = X_train_sm.drop(['Last Activity_SMS Sent'], axis=1)
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial()).fit() 
print(logm1.summary())
X_train_sm =  X_train_sm.drop(['const'], axis=1)
vif(X_train_sm)


# In[75]:


X_train_sm = sm.add_constant(X_train_sm)
y_train_pred = logm1.predict(X_train_sm).values.reshape(-1)


# In[76]:


# Creating a dataframe with the actual Converted flag
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[77]:


# Creating new column 'predicted' with 1
y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[78]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_pred_final["Converted"], y_train_pred_final["predicted"])
print(confusion)


# In[80]:


# Overall accuracy.
print(metrics.accuracy_score(y_train_pred_final["Converted"], y_train_pred_final["predicted"]))


# In[81]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[82]:


# Let's see the sensitivity of our logistic regression model
print("Sensitivity :",TP / float(TP+FN))


# In[83]:


# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# In[84]:


# Calculate false postive rate - predicting conversion when customer does not have converted
print(FP/ float(TN+FP))


# In[85]:


# positive predictive value 
print (TP / float(TP+FP))


# In[86]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[87]:


# UDF to draw ROC curve 
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[88]:


fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final["Converted"], y_train_pred_final["Converted_prob"], drop_intermediate = False )


# In[89]:


# Drawing ROC curve for Train Set
draw_roc(y_train_pred_final["Converted"], y_train_pred_final["Converted_prob"])


# In[90]:


# Creating columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final['Converted_prob'].map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[91]:


cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final["Converted"], y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[92]:


# plotting the accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[93]:


y_train_pred_final['final_predicted'] = y_train_pred_final['Converted_prob'].map( lambda x: 1 if x > 0.3 else 0)

# deleting the unwanted columns from dataframe
y_train_pred_final.drop([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],axis = 1, inplace = True) 
y_train_pred_final.head()


# In[94]:


# Checking the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final["Converted"], y_train_pred_final["final_predicted"]))


# In[95]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[96]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[97]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[98]:


# Let us calculate specificity
TN / float(TN+FP)


# In[99]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[100]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[101]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[102]:


y_train_pred_final.head()


# In[103]:


confusion3 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion3


# In[104]:


TP = confusion3[1,1] # true positive 
TN = confusion3[0,0] # true negatives
FP = confusion3[0,1] # false positives
FN = confusion3[1,0] # false negatives


# In[105]:


# Let's see the Recall Or sensitivity of our logistic regression model
TP / float(TP+FN)


# In[106]:


# Let us calculate specificity
TN / float(TN+FP)


# In[107]:


# Calculate false postive rate - predicting churn
print(FP/ float(TN+FP))


# In[108]:


# Precision Or Positive predictive value 
print (TP / float(TP+FP))


# In[109]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[110]:


# Creating precision-recall tradeoff curve
#y_train_pred_final['Converted'], y_train_pred_final['final_predicted']
p, r, thresholds = precision_recall_curve(y_train_pred_final['Converted'], y_train_pred_final['Converted_prob'])


# In[111]:


# plot precision-recall tradeoff curve
plt.plot(thresholds, p[:-1], "g-", label="Precision")
plt.plot(thresholds, r[:-1], "r-", label="Recall")



#plt.axvline(x=0.41, color='teal',linewidth = 0.55, linestyle='--')
plt.legend(loc='lower left')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')

plt.show()


# In[112]:


# scaling the numerical column data
X_test[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']] = scaler.fit_transform(X_test[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']])


# In[113]:


X_test_sm = sm.add_constant(X_test)
X_test_sm = X_test_sm[X_train_sm.columns]
X_test_sm.head()


# In[114]:


## Making predictions on the test set


# In[115]:


# making prediction using final model
y_test_pred = logm1.predict(X_test_sm)


# In[116]:


# Changing to dataframe of predicted probability
y_test_pred = pd.DataFrame(y_test_pred)
y_test_pred.head()


# In[117]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
y_test_df.head()


# In[118]:


# Putting Prospect ID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[119]:


# Removing index for both dataframes to append them side by side 
y_test_pred.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[120]:


# Appending y_test_df and y_test_pred
y_pred_final = pd.concat([y_test_df, y_test_pred],axis=1)
y_pred_final.head()


# In[121]:



y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})


y_pred_final = y_pred_final.reindex(['Prospect ID','Converted','Converted_Prob'], axis=1)

y_pred_final.head()


# In[122]:



y_pred_final['final_predicted'] = y_pred_final['Converted_Prob'].map(lambda x: 1 if x > 0.3 else 0)
y_pred_final.head()


# In[123]:


fpr, tpr, thresholds = metrics.roc_curve(y_pred_final["Converted"], y_pred_final["Converted_Prob"], drop_intermediate = False )

draw_roc(y_pred_final["Converted"], y_pred_final["Converted_Prob"])


# In[124]:


confusion5 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final['final_predicted'])


# In[125]:


TP = confusion5[1,1] # true positive 
TN = confusion5[0,0] # true negatives
FP = confusion5[0,1] # false positives
FN = confusion5[1,0] # false negatives


# In[126]:


# Overall accuracy.
accuracy = (TN+TP)/(TN+TP+FN+FP)
accuracy


# In[127]:


# sensitivity of our logistic regression model
TP / float(TP+FN)


# In[128]:


# specificity
TN / float(TN+FP)


# In[129]:


# Calculating false postive rate - predicting churn
print(FP/ float(TN+FP))


# In[130]:


# +ve predictive value 
print (TP / float(TP+FP))


# In[131]:


# -ve predictive value
print (TN / float(TN+ FN))


# In[ ]:




