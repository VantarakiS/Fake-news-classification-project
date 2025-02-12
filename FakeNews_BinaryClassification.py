import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer
from matplotlib import cm
import pickle  # For saving and loading embeddings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, Doc2Vec
#from gensim.models.doc2vec import TaggedDocument
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
tqdm.pandas()  # For progress bars
import os
from sklearn.metrics import ConfusionMatrixDisplay
from nltk.tokenize import word_tokenize
# Download NLTK tokenizer resources (if you haven't already)
nltk.download('punkt')
import re
import string
from nltk.stem import PorterStemmer 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import opinion_lexicon
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import randint
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import KFold, HalvingGridSearchCV




#reproducibility
random.seed(42)
np.random.seed(42)
LIAR = pd.read_csv("C:/Users/sotin/Downloads/liar_dataset/LIARfull.tsv", sep='\t')

LIAR.head()

#headers_missing I have to put them. Searching online the headers are:

LIAR = pd.read_csv("C:/Users/sotin/Downloads/liar_dataset/LIARfull.tsv", sep='\t',
                         names=['ID', 'label', 'statement', 'subject(s)', 'speaker', 'speaker job title', 'state info', 
                                'party affiliation', 'barely true counts', 'false counts', 'half true counts', 
                                'mostly true counts', 'pants onfire counts', 'venue'])


LIAR.head()
LIAR.tail()
LIAR.info()
LIAR.describe()
print(LIAR.shape)

LIAR.drop(['ID'], axis = 1, inplace= True)

print(LIAR.columns)

print('lenght of data is', len(LIAR))
print(LIAR.dtypes)

print(np.sum(LIAR.isnull().any(axis=1)))

print('Count of columns in the data is:  ', len(LIAR.columns))

print('Count of rows in the data is:  ', len(LIAR))


#check dupes

current=len(LIAR)
print('Rows of data before Delecting ', current)

LIAR=LIAR.drop_duplicates()
now=len(LIAR)
print('Rows of data before Delecting ', now)

diff=current-now
print('Duplicated rows are ', diff)


print(LIAR.isnull().sum()) #labels has no na value so it is complete


LIAR.replace('', np.nan, inplace=True)

LIAR = LIAR.dropna(subset=['label'])



LIAR['venue']= LIAR['venue'].replace(np.nan, 'Unknown')
LIAR["speaker job title"]= LIAR["speaker job title"].replace(np.nan, 'Unknown')
LIAR["state info"]= LIAR["state info"].replace(np.nan, 'Unknown')
LIAR['party affiliation'] = LIAR['party affiliation'].replace(np.nan, 'Unknown')
print(LIAR.isnull().sum())



num_cols = ['barely true counts', 'false counts', 'half true counts', 
                                'mostly true counts', 'pants onfire counts']

cate_cols = LIAR.columns.drop('label').drop(num_cols)

print(LIAR[cate_cols].apply(lambda x: x.nunique(), axis=0))



#histo of labels 

sns.countplot(data= LIAR, x = "label")
plt.show()

# Create a figure for the pie chart
plt.figure(figsize=(8, 8))
# Plot the pie chart
LIAR["label"].value_counts().head(7).plot(
    kind='pie', 
    autopct='%1.1f%%'
)
# Add a legend to the pie chart
plt.ylabel("") #no show count due to text overlap
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Label Distribution")
plt.show()


print(LIAR["label"].value_counts())

#each label words
data1=LIAR[LIAR['label']=='barely-true']
d =data1['statement']
string_ = []
for t in d:
    string_.append(t)
string_ = pd.Series(string_).map(str)
string_=str(string_)
wordcloud = WordCloud(width=1500, height=700,max_font_size=250, background_color ='white').generate(string_)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


data1=LIAR[LIAR['label']=='half-true']
d =data1['statement']
string_ = []
for t in d:
    string_.append(t)
string_ = pd.Series(string_).map(str)
string_=str(string_)
wordcloud = WordCloud(width=1500, height=700,max_font_size=250, background_color ='black').generate(string_)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


data1=LIAR[LIAR['label']=='mostly-true']
d =data1['statement']
string_ = []
for t in d:
    string_.append(t)
string_ = pd.Series(string_).map(str)
string_=str(string_)
wordcloud = WordCloud(width=1500, height=700,max_font_size=250, background_color ='yellow').generate(string_)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

data1=LIAR[LIAR['label']=='true']
d =data1['statement']
string_ = []
for t in d:
    string_.append(t)
string_ = pd.Series(string_).map(str)
string_=str(string_)
wordcloud = WordCloud(width=1500, height=700,max_font_size=250, background_color ='purple').generate(string_)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

data1=LIAR[LIAR['label']=='false']
d =data1['statement']
string_ = []
for t in d:
    string_.append(t)
string_ = pd.Series(string_).map(str)
string_=str(string_)
wordcloud = WordCloud(width=1500, height=700,max_font_size=250, background_color ='pink').generate(string_)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


data1=LIAR[LIAR['label']=='pants-fire']
d =data1['statement']
string_ = []
for t in d:
    string_.append(t)
string_ = pd.Series(string_).map(str)
string_=str(string_)
wordcloud = WordCloud(width=1500, height=700,max_font_size=250, background_color ='navy').generate(string_)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


#statements preproc

LIAR['statement']=LIAR['statement'].str.lower()
LIAR['statement'].tail()

stopwords_list = stopwords.words('english')
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))

STOPWORDS = set(stopwords.words('english'))
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
LIAR["statement"] = LIAR["statement"].apply(lambda text: cleaning_stopwords(text))
LIAR["statement"].head()


import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
LIAR["statement"] = LIAR["statement"].apply(lambda x: cleaning_punctuations(x))
LIAR["statement"].tail()

#sooooon to son problem of missinterpretation of word soon son
def cleaning_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)
LIAR["statement"] = LIAR["statement"].apply(lambda x: cleaning_repeating_char(x))
LIAR["statement"].tail()

def cleaning_email_urls(text):
    pattern = r'([a-zA-Z0-9._%-]+@[a-zA-Z0-9._%-]+\.[a-zA-Z]{2,})|((www\.[^\s]+)|(https?://[^\s]+))'
    #return re.sub('@[^\s]+', ' ', text)
    return re.sub(pattern, ' ', text)
LIAR["statement"] = LIAR["statement"].apply(lambda x: cleaning_email_urls(x))
LIAR["statement"].tail()


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
LIAR["statement"] = LIAR["statement"].apply(lambda x: cleaning_numbers(x))
LIAR["statement"].tail()


LIAR["statement"] = LIAR["statement"].apply(lambda x: word_tokenize(x))
LIAR["statement"].head()
print(np.sum(LIAR.isnull().any(axis=1)))


st = nltk.PorterStemmer()
def stemming_on_text(data):
    
    return [st.stem(word) for word in data]

LIAR["statement"] = LIAR["statement"].apply(lambda x: stemming_on_text(x))
LIAR["statement"].head()


lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    
    return [lm.lemmatize(word) for word in data]

LIAR["statement"] = LIAR["statement"].apply(lambda x: lemmatizer_on_text(x))
LIAR["statement"].head()

words = [word for statements in LIAR["statement"] for word in statements]
lenght_of_each_sentence = [len(statements) for statements in LIAR["statement"]]
vocabulary  = sorted(list(set(words)))
print("There are %s words in total, with vocabulary size of %s" % (len(words), len(vocabulary)))


counts_of_words = Counter(words)
counts_of_words.most_common(25)


words = []
counts = []
for letter, count in counts_of_words.most_common(25):
    words.append(letter)
    counts.append(count)
colors = cm.rainbow(np.linspace(0, 2, 12))
plt.rcParams['figure.figsize'] = (20, 10)

plt.title('Top 25 frequently words in news statement text')
plt.xlabel('Count')
plt.ylabel('Words')
plt.barh(words, counts, color=colors)
plt.show()

#subject preproc

LIAR["subject(s)"].head()
LIAR["subject(s)"]= LIAR["subject(s)"].str.replace(",", " ")

LIAR["subject(s)"] = LIAR["subject(s)"].astype(str)

subjectTokenize = []
for sen in LIAR["subject(s)"]:
    subjectTokenize.append(word_tokenize(sen))


filteredsubjects = []
for words in subjectTokenize:
    stopWords = set(stopwords.words('english'))
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    filteredsubjects.append(wordsFiltered)


ps = PorterStemmer() 
index = 0    
for words in filteredsubjects:
    subjects=""
    for w in words: 
        subjects=subjects+ps.stem(w)+" "
    LIAR.at[index, "subject(s)"] = subjects
    index += 1


LIAR["dummp"]=LIAR["subject(s)"]
LIAR["dummp"]=LIAR["dummp"].str.strip()
LIAR.loc[LIAR["dummp"].str.contains('job|worker'), 'dummp'] = 'jobs'
LIAR.loc[LIAR["dummp"].str.contains('hous'), 'dummp'] = 'budget'
LIAR.loc[LIAR["dummp"].str.contains('county-budget'), 'dummp'] = 'budget'
LIAR.loc[LIAR["dummp"].str.contains('federal-budget'), 'dummp'] = 'budget'
LIAR.loc[LIAR["dummp"].str.contains('state-budget|city-budget'), 'dummp'] = 'budget'
LIAR.loc[LIAR["dummp"].str.contains('state-fin'), 'dummp'] = 'budget'
LIAR.loc[LIAR["dummp"].str.contains('edu'), 'dummp'] = 'education'
LIAR.loc[LIAR["dummp"].str.contains('economi|incom|tax|debt|market-regul|financial-regul|trade|small-busi'), 'dummp'] = 'economy'
LIAR.loc[LIAR["dummp"].str.contains('militari|veteran'), 'dummp'] = 'military'
LIAR.loc[LIAR["dummp"].str.contains('government-effici|city-govern|county-govern|government-regul|supreme-court|state'), 'dummp'] = 'government'

LIAR.loc[LIAR["dummp"].str.contains('health-car|medicar|abort|public-health'), 'dummp'] = 'health-care'
LIAR.loc[LIAR["dummp"].str.contains('crime|gun|public-safeti|legal-issu|terror|homeland-secur'), 'dummp'] = 'crime'

LIAR.loc[LIAR["dummp"].str.contains('climate-chang|environ|anim'), 'dummp'] = 'environment'

LIAR.loc[LIAR["dummp"].str.contains('foreign-polici|voting-record|congress|elect|politics'), 'dummp'] = 'politics'

LIAR.loc[LIAR["dummp"].str.contains('children|immigr|women|popul|poverti|social-secur|religion'), 'dummp'] = 'social'
LIAR.loc[~LIAR["dummp"].str.contains('jobs|budget|education|economy|military|government|health-care|crime|environment|politics|social'), 'dummp'] = 'other'
LIAR["dummp"].value_counts()

LIAR["subject(s)"]=LIAR["dummp"]
LIAR["subject(s)"].value_counts()


LIAR=LIAR.drop(columns=['dummp'])
jt=[]
jt=LIAR["subject(s)"].unique()
for i in jt:
    print(i)

# Bar plot for the top 12 subjects
plt.figure(figsize=(10, 6))  # Create a new figure
LIAR["subject(s)"].value_counts().head(12).plot(kind='bar', color='skyblue')
plt.title('Top 12 Subjects - Bar Plot')
plt.xlabel('Subjects')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout
plt.show()  # Show and clear the plot

# Pie chart for the top 12 subjects
plt.figure(figsize=(8, 8))  # Create a new figure
LIAR["subject(s)"].value_counts().head(12).plot(
    kind='pie', autopct='%1.1f%%', startangle=90, legend=True
)
plt.title('Top 12 Subjects - Pie Chart')
plt.legend(bbox_to_anchor=(1, 1))  # Position legend outside the plot
plt.tight_layout()  # Adjust layout
plt.show()  # Show and clear the plot


#speaker
LIAR["speaker"].head()
# Create a figure for the pie chart
plt.figure(figsize=(8, 8))
# Plot the pie chart
LIAR["speaker"].value_counts().head(10).plot(
    kind='pie', 
    autopct='%1.1f%%'
)
# Add a legend to the pie chart
plt.ylabel("") #no show count due to text overlap
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Top Speakers")
plt.show()

# Create a figure for the pie chart
plt.figure(figsize=(8, 8))
# Plot the pie chart
LIAR["label"].value_counts().tail(10).plot(
    kind='pie', 
    autopct='%1.1f%%'
)
# Add a legend to the pie chart
plt.ylabel("") #no show count due to text overlap
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Least common speakers")
plt.show()
LIAR["speaker"].value_counts().head(50)

speaker_label = LIAR.groupby(['speaker', 'label']).size().reset_index(name='count')
speaker_label = speaker_label.sort_values(by='count', ascending=False)

speaker_label.head()

top_speakers_per_label = speaker_label.groupby('label', group_keys=False).apply(
    lambda x: x.nlargest(5, 'count')
)
# Plot top 5 speakers for each label
for label, group in top_speakers_per_label.groupby('label'):
    plt.figure(figsize=(15, 10))
    sns.barplot(
        data=group,
        x='count',
        y='speaker',  # Ensure the column name is correct
        #palette="viridis"  # Optional: adds color variety
    )
    plt.title(f"Top 5 Speakers for Label: {label}", fontsize=16)
    plt.xlabel("Number of Statements", fontsize=14)
    plt.ylabel("Speakers", fontsize=14)
    plt.tight_layout()
    plt.show()


#job preproc
LIAR["speaker job title"] .head()

LIAR["speaker job title"] = LIAR["speaker job title"].apply(str)
jobTokenize = []
for sen in LIAR["speaker job title"]:
    jobTokenize.append(word_tokenize(sen))

from nltk.corpus import stopwords
filteredjobs = []
for words in jobTokenize:
    stopWords = set(stopwords.words('english'))
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    filteredjobs.append(wordsFiltered)


from nltk.stem import PorterStemmer 
ps = PorterStemmer() 

index = 0    
for words in filteredjobs:

    job=""
    for w in words: 
        job=job+ps.stem(w)+" "
    LIAR.at[index, "speaker job title"] = job
    index += 1


LIAR["dummp"]=LIAR["speaker job title"]
LIAR["dummp"]=LIAR["dummp"].str.strip()
LIAR.loc[LIAR["dummp"].str.contains('repres'), 'dummp'] = 'U.S. representative'
LIAR.loc[LIAR["dummp"].str.contains('governor'), 'dummp'] = 'state representative'
LIAR.loc[LIAR["dummp"].str.contains('state'), 'dummp'] = 'state representative'
LIAR.loc[LIAR["dummp"].str.contains('congressman'), 'dummp'] = 'state representative'
LIAR.loc[LIAR["dummp"].str.contains('senat'), 'speaker'] = 'state representative'
LIAR.loc[LIAR["dummp"].str.contains('congresswoman'), 'dummp'] = 'state representative'
LIAR.loc[LIAR["dummp"].str.contains('deleg'), 'dummp'] = 'state representative'
LIAR.loc[LIAR["dummp"].str.contains('mayor'), 'dummp'] = 'state representative'
LIAR.loc[LIAR["dummp"].str.contains('presid'), 'dummp'] = 'president'
LIAR.loc[LIAR["dummp"].str.contains('director'), 'dummp'] = 'office director'
LIAR.loc[LIAR["dummp"].str.contains('group'), 'dummp'] = 'company'
LIAR.loc[LIAR["dummp"].str.contains('chairman'), 'dummp'] = 'company'
LIAR.loc[LIAR["dummp"].str.contains('program'), 'dummp'] = 'company'
LIAR.loc[LIAR["dummp"].str.contains('counti'), 'dummp'] = 'government'
LIAR.loc[LIAR["dummp"].str.contains('attorney'), 'dummp'] = 'government'
LIAR.loc[LIAR["dummp"].str.contains('govern'), 'dummp'] = 'government'
LIAR.loc[LIAR["dummp"].str.contains('media'), 'dummp'] = 'media'
LIAR.loc[LIAR["dummp"].str.contains('blog'), 'dummp'] = 'media'
LIAR.loc[LIAR["dummp"].str.contains('show'), 'dummp'] = 'media'
LIAR.loc[LIAR["dummp"].str.contains('host'), 'dummp'] = 'media'
LIAR.loc[LIAR["dummp"].str.contains('radio'), 'dummp'] = 'media'
LIAR.loc[LIAR["dummp"].str.contains('tv'), 'dummp'] = 'media'
LIAR.loc[LIAR["dummp"].str.contains('unknown'), 'dummp'] = 'unknown'
LIAR.loc[~LIAR["dummp"].str.contains('state representative|president|office director|company|U.S. representative|government|media|unknown'), 'dummp'] = 'other'
LIAR["dummp"].value_counts()

LIAR["speaker job title"]=LIAR["dummp"]
LIAR["speaker job title"].value_counts()

LIAR=LIAR.drop(columns=['dummp'])
jt=[]
jt=LIAR["speaker job title"].unique()
for i in jt:
    print(i)


# Create a figure for the pie chart
plt.figure(figsize=(8, 8))
# Plot the pie chart
LIAR["speaker job title"].value_counts().head(10).plot(
    kind='bar', 
)
# Add a legend to the pie chart
plt.ylabel("") #no show count due to text overlap
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Most common speaker's job")
plt.show()

#state
LIAR["state info"].head()

# Create a figure for the pie chart
plt.figure(figsize=(8, 8))
# Plot the pie chart
LIAR["state info"].value_counts().head(10).plot(
    kind='pie', 
    autopct='%1.1f%%'
)
# Add a legend to the pie chart
plt.ylabel("") #no show count due to text overlap
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Most common states")
plt.show()
LIAR["state info"].value_counts().tail(10).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend(bbox_to_anchor=(1, 1))

# Create a figure for the pie chart
plt.figure(figsize=(8, 8))
# Plot the pie chart
LIAR["state info"].value_counts().tail(10).plot(
    kind='pie', 
    autopct='%1.1f%%'
)
# Add a legend to the pie chart
plt.ylabel("") #no show count due to text overlap
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Least common states")
plt.show()
LIAR["state info"].value_counts().head(50)


#party preproc
LIAR['party affiliation'].head()

party=[]
party=LIAR['party affiliation'].unique()
for i in party:
    print(i)



LIAR["party affiliation"]= LIAR["party affiliation"].replace('none', 'Unknown')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('activist', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('organization', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('libertarian', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('journalist', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('columnist', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('state-official', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('business-leader', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('talk-show-host', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('government-body', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('newsmaker', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('county-commissioner', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('constitution-party', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('labor-leader', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('education-official', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('tea-party-member', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('green', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('liberal-party-canada', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('Moderate', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('democratic-farmer-labor', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('ocean-state-tea-party-action', 'Other')
LIAR["party affiliation"]= LIAR["party affiliation"].replace('independent', 'Other')
party=[]
party=LIAR['party affiliation'].unique()
for i in party:
    print(i)


def check_dist(dataset):
  sns.countplot(x='party affiliation', data=LIAR, palette='hls')
check_dist(LIAR)


# Create a figure for the pie chart
plt.figure(figsize=(8, 8))
# Plot the pie chart
LIAR["party affiliation"].value_counts().head(10).plot(
    kind='pie',
    autopct='%1.1f%%',       # Show percentages
 
)
# Add a legend to the pie chart
plt.ylabel("") #no show count due to text overlap
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Most common speaker's party")
plt.show()


#histo of flags PER PERSONA
plt.figure(figsize=(12, 6))
sns.histplot(data=LIAR[LIAR['speaker']=='barack-obama'], x='label')
plt.title("Distribution of Obama's statements")
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()



plt.figure(figsize=(12, 6))
sns.histplot(data=LIAR[LIAR['speaker']=='donald-trump'], x='label')
plt.title("Distribution of donald-trump's statements")
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()


political_idio = LIAR.groupby(['party affiliation', 'label']).size().reset_index(name='count')
political_idio.head()

plt.figure(figsize=(15, 10))
sns.barplot(
    data=political_idio,
    x='count',
    y='party affiliation',
    hue='label',
    dodge=False
)
plt.xscale('log')
plt.title("Political affiliation for Each Label")
plt.xlabel("Number of Statements in log scale")
plt.ylabel("Polotical affiliation")
plt.legend(title="label")
plt.tight_layout()
plt.show()



#venue preproc
LIAR["venue"].head()
LIAR["venue"] = LIAR["venue"].apply(str)

vanueTokenize = []
for sen_tex in LIAR['venue']:
    vanueTokenize.append(word_tokenize(sen_tex))


filteredVenues = []
for words in vanueTokenize:
    stopWords = set(stopwords.words('english'))
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    filteredVenues.append(wordsFiltered)


ps = PorterStemmer()
index = 0    
for words in filteredVenues:
    vn=""
    for w in words: 
        vn=vn+ps.stem(w)+" "
    LIAR.at[index, 'venue'] = vn
    index += 1


LIAR["dummp"]=LIAR["venue"]
LIAR["dummp"]=LIAR["dummp"].str.strip()
LIAR.loc[LIAR["dummp"].str.contains('confer|press|speech|interview|debate|broadcast|meet|opinion|statement|letter|ralli'), 'dummp'] = 'interview'
LIAR.loc[LIAR["dummp"].str.contains('campaign|ad|flier|commerci|mailer|panel|billboard'), 'dummp'] = 'ad'
LIAR.loc[LIAR["dummp"].str.contains('facebook|imag|media|meme|tweet|email|e-email|forum|blog|twitter'), 'dummp'] = 'social media'

LIAR.loc[LIAR["dummp"].str.contains('abc|articl|news|cnn|msnbc|book|journal|hbo|fox|column|newslett'), 'dummp'] = 'news'
LIAR.loc[LIAR["dummp"].str.contains('websit|web'), 'dummp'] = 'website'

LIAR.loc[LIAR["dummp"].str.contains('show'), 'dummp'] = 'show'

LIAR.loc[LIAR["dummp"].str.contains('unknown'), 'dummp'] = 'unknown'
LIAR.loc[~LIAR["dummp"].str.contains('interview|ad|social media|news|website|show|unknown'), 'dummp'] = 'other'
LIAR["dummp"].value_counts()


LIAR["venue"]=LIAR["dummp"]
LIAR["venue"].value_counts()



LIAR=LIAR.drop(columns=['dummp'])
v=[]
v=LIAR['venue'].unique()
for i in v:
    print(i)


def check_dist(dataset):
    sns.countplot(x='venue', data=LIAR, palette='hls')
check_dist(LIAR)


# Create a figure for the pie chart
plt.figure(figsize=(8, 8))
# Plot the pie chart
LIAR["venue"].value_counts().head(10).plot(
    kind='pie',
    autopct='%1.1f%%',       # Show percentages
 
)
# Add a legend to the pie chart
plt.ylabel("") #no show count due to text overlap
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Most common speaker's party")
plt.show()



#numeric

num=LIAR[['barely true counts', 'false counts',
       'half true counts', 'mostly true counts', 'pants onfire counts']]
num=num.fillna(np.nan)
num.head()

num.hist(figsize=(15,12),bins = 20, color="#107009AA")
plt.title("Features Distribution")
plt.show()

#14 Features extraction from the "Statement of the news"
#glove
# Download pre-trained GloVe embeddings (link provided below)
#!wget https://nlp.stanford.edu/data/glove.6B.zip -P glove_data/
#!unzip glove_data/glove.6B.zip -d glove_data/


LIAR['statement'] = LIAR['statement'].fillna('')

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3),
    max_features =5000)

Get_Vec= word_vectorizer.fit_transform(LIAR['statement'].astype('str'))
Get_Vec= Get_Vec.toarray()


vocab1 = word_vectorizer.get_feature_names_out()

tfFeatures_vect=pd.DataFrame(np.round(Get_Vec, 1), columns=vocab1)
tfFeatures_vect.head()

# Save TF-IDF Embeddings
with open('tfidf_embeddings.pkl', 'wb') as f:
    pickle.dump(Get_Vec, f)

# Save as DataFrame
tfFeatures_vect.to_csv('tfidf_embeddings.csv', index=False)
print("TF-IDF DataFrame saved as 'tfidf_embeddings.csv'.")

#BOW
# Initialize CountVectorizer
bow_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')
  # Limit vocabulary to 5000 most frequent words
bow_features = bow_vectorizer.fit_transform(LIAR['statement'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x))


# Save BoW Embeddings
with open('bow_embeddings.pkl', 'wb') as f:
    pickle.dump(bow_features, f)


#Word2Vec
w2v_skipgram = Word2Vec(sentences=LIAR['statement'], vector_size=300, window=5, min_count=2, sg=1, epochs=10)

# Train Word2Vec with CBOW (sg=0)
w2v_cbow = Word2Vec(sentences=LIAR['statement'], vector_size=300, window=5, min_count=2, sg=0, epochs=10)
# Generate embeddings for each statement using Skip-gram
LIAR['Word2Vec_SkipGram'] = LIAR['statement'].apply(
    lambda x: np.mean([w2v_skipgram.wv[word] for word in x if word in w2v_skipgram.wv], axis=0) if x else np.zeros(300)
)

# Generate embeddings for each statement using CBOW
LIAR['Word2Vec_CBOW'] = LIAR['statement'].apply(
    lambda x: np.mean([w2v_cbow.wv[word] for word in x if word in w2v_cbow.wv], axis=0) if x else np.zeros(300)
)


# Ensure all rows in 'Word2Vec_SkipGram' are lists/arrays
LIAR['Word2Vec_SkipGram'] = LIAR['Word2Vec_SkipGram'].apply(
    lambda x: x if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.zeros(300)
)

# Ensure all rows in 'Word2Vec_SkipGram' are lists/arrays
LIAR['Word2Vec_CBOW'] = LIAR['Word2Vec_CBOW'].apply(
    lambda x: x if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.zeros(300)
)

# Save Word2Vec models and embeddings
w2v_skipgram.save('word2vec_skipgram.model')
w2v_cbow.save('word2vec_cbow.model')

with open('word2vec_skipgram_embeddings.pkl', 'wb') as f:
    pickle.dump(LIAR['Word2Vec_SkipGram'].tolist(), f)

with open('word2vec_cbow_embeddings.pkl', 'wb') as f:
    pickle.dump(LIAR['Word2Vec_CBOW'].tolist(), f)



#glove
# Load pre-trained GloVe embeddings
glove_model = {}
with open("C:/Users/sotin/Downloads/glove.6B (1)\glove.6B.300d.txt", "r", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_model[word] = vector



def get_glove_embedding(tokens, glove_model, embedding_dim=300):  # Set to 300
    if not isinstance(tokens, list):  # Ensure tokens is a list
        return np.zeros(embedding_dim)
    vectors = [glove_model[word] for word in tokens if word in glove_model]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)  # Fixed length fallback
LIAR['Glove_Embedding'] = LIAR['statement'].apply(
    lambda x: get_glove_embedding(x, glove_model, embedding_dim=300)
)
embedding_shapes = [len(embedding) if isinstance(embedding, np.ndarray) else 0 for embedding in LIAR['Glove_Embedding']]
print(set(embedding_shapes))  # Should output {300}
X_glove = np.array(LIAR['Glove_Embedding'].tolist())
print(X_glove.shape)  # Should output (number_of_samples, 300)




LIAR['Glove_Embedding'] = LIAR['statement'].apply(lambda x: get_glove_embedding(x, glove_model))

# Save GloVe embeddings
with open('glove_embeddings.pkl', 'wb') as f:
    pickle.dump(LIAR['Glove_Embedding'].tolist(), f)


# Convert to DataFrame
bow_df = pd.DataFrame(bow_features.toarray(), columns=bow_vectorizer.get_feature_names_out())

# Save as DataFrame
bow_df.to_csv('bow_embeddings.csv', index=False)
print("Bag of Words DataFrame saved as 'bow_embeddings.csv'.")


# Convert to DataFrame
word2vec_skipgram_df = pd.DataFrame(LIAR['Word2Vec_SkipGram'].to_list())
word2vec_cbow_df = pd.DataFrame(LIAR['Word2Vec_CBOW'].to_list())

# Save as DataFrames
word2vec_skipgram_df.to_csv('word2vec_skipgram_embeddings.csv', index=False)
word2vec_cbow_df.to_csv('word2vec_cbow_embeddings.csv', index=False)
print("Word2Vec DataFrames saved as 'word2vec_skipgram_embeddings.csv' and 'word2vec_cbow_embeddings.csv'.")


# Convert to DataFrame
glove_df = pd.DataFrame(LIAR['Glove_Embedding'].to_list())

# Save as DataFrame
glove_df.to_csv('glove_embeddings.csv', index=False)
print("GloVe DataFrame saved as 'glove_embeddings.csv'.")


cleanup_nums = {"venue":     {'interview': 0, 'ad': 1, 'social media': 2,'news': 3, 'website': 4, 'show': 5, 'unknown': -1 , 'other' : 6},
                "speaker job title": {'state representative': 0, 'president': 1, 'office director': 2, 'company': 3,
                                  'U.S. representative': 4, 'government': 5, 'media':6, 'unknown': -1, 'other' : 7 },
                "party affiliation":     {'republican': 0, 'democrat': 1, 'Unknown': -1,'Other': 2},
                "subject(s)": {'jobs':0, 'military':1, 'education':2, 'economy':3, 'government':4, 'health-care':5, 
                               'crime':6, 'environment':7, 'budget':8, 'politics':9,'social':10, 'other':11}
              }


LIAR.replace(cleanup_nums, inplace=True)
LIAR.head()


x = pd.Categorical(LIAR['speaker'])               
LIAR['speaker']=x.codes
x = pd.Categorical(LIAR['state info'])               
LIAR['state info']=x.codes

LIAR.head()


# Map Label to Binary Classification (True/False)
label_map = {
    'mostly-true': 'True',  # true
    'true': 'True',  # mostly-true
    'half-true': 'False',  # half-true
    'false': 'False',  # false
    'barely-true': 'False',  # barely-true
    'pants-fire': 'False',  # pants-fire
}

LIAR = LIAR.dropna(subset=['label'])

LIAR['Binary_Label'] = LIAR['label'].replace(label_map)

# Encode binary labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(LIAR['Binary_Label'])  # 1 for True, 0 for False

# Convert class names to strings for classification report
class_names = list(map(str, label_encoder.classes_))

#for some reason i kept getting an error so I used it two times to get over it
LIAR = LIAR.dropna(subset=['label'])

sns.countplot(data= LIAR, x = y)
plt.show()



X_tfidf = Get_Vec
X_bow = bow_features.toarray()
X_glove = np.array(LIAR['Glove_Embedding'].tolist())
X_word2vec_sk = np.array(LIAR['Word2Vec_SkipGram'].tolist())
X_word2vec_cb = np.array(LIAR['Word2Vec_CBOW'].tolist())



# Ensure reproducibility
np.random.seed(42)


# Function to balance data
def balance_data(X, y, method='SMOTE'):
    if method == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    else:
        raise ValueError("Invalid method. Choose from 'SMOTE'")
    return X_res, y_res

# Assuming embeddings from earlier
embedding_methods = {
    "TF-IDF": X_tfidf,
    "BoW": X_bow,
    "Word2Vec_sk": X_word2vec_sk,
    "Word2Vec_cb": X_word2vec_cb,
    "GloVe": X_glove
}

# Feature Selection Methods
def apply_feature_selection(method, X_train, y_train, X_test, embedding_name):
    if method == 'PCA':
        pca = PCA(n_components=0.95)  # Preserve 95% variance
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)
    elif method == 'MutualInfo':
        selector = SelectKBest(score_func=mutual_info_classif, k=100)
        X_train_reduced = selector.fit_transform(X_train, y_train)
        X_test_reduced = selector.transform(X_test)
    elif method == 'Chi-Square': #Chi squere takes only positive values while glove and word2vec can be negative
        # Transform features to be non-negative
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        selector = SelectKBest(score_func=chi2, k=100)
        X_train_reduced = selector.fit_transform(X_train, y_train)
        X_test_reduced = selector.transform(X_test)
   
        
    elif method == 'Tree-Based':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        selector = SelectFromModel(model, prefit=True)
        X_train_reduced = selector.transform(X_train)
        X_test_reduced = selector.transform(X_test)
    else:
        raise ValueError("Invalid feature selection method")
    return X_train_reduced, X_test_reduced

# Feature selection methods
feature_methods = ['Original', 'PCA', 'MutualInfo', 'Chi-Square', 'Tree-Based']

# Classifiers and randomized search configurations
classifiers = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
        "params": {
            "C": [ 10, 100],  # Regularization strength
            "solver": ['lbfgs'],  # Solvers
            "penalty": ['l2'],  # Regularization type
        },
    },

    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight="balanced", criterion = "entropy"),
        "params": {
            "n_estimators": [100],  # Random range for number of trees
            "max_depth": [None],  # Tree depth
            "min_samples_split": [2],
              "min_samples_leaf":[1]  # Minimum samples per split
              # Use entropy for node splits
        },
    },

}

nltk.download('opinion_lexicon')


# Predefined positive and negative word lists (from opinion_lexicon)
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# Function to count positive/negative words
def count_sentiment_words(tokens, word_set):
    return sum(1 for token in tokens if token in word_set)



# Add the new features to the dataset
LIAR['Positive_Word_Count'] = LIAR['statement'].apply(lambda x: count_sentiment_words(x, positive_words))
LIAR['Negative_Word_Count'] = LIAR['statement'].apply(lambda x: count_sentiment_words(x, negative_words))
metadata = LIAR[['Positive_Word_Count', 'Negative_Word_Count', 'venue', 'speaker job title', 'party affiliation', 'subject(s)']].values


# Results storage
results = []
conf_matrix_list_of_arrays = []

# Create results directory
os.makedirs("results", exist_ok=True)

# KFold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
i=0
for train, test in kf.split(LIAR):
    i=i+1
    print("KFold Split ",i )
    print("%s %s" % (train, test))
    print(' \n')


# Training and evaluation loop
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(y)):
    print(f"Starting Fold {fold_idx + 1}...")
    print("%s %s" % (train_idx, test_idx))

    for embedding_name, embedding_data in embedding_methods.items():
        
        X_train_emb, X_test_emb = embedding_data[train_idx], embedding_data[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Split metadata
        X_train_metadata, X_test_metadata = metadata[train_idx], metadata[test_idx]

         # Combine embeddings and metadata
        X_train = np.hstack([X_train_emb, X_train_metadata])
        X_test = np.hstack([X_test_emb, X_test_metadata])
        # Get feature names
        #feature_names = get_feature_names(embedding_name, X_train_metadata, vectorizer)

        # Apply rebalancing
        X_train_balanced, y_train_balanced = balance_data(X_train, y_train, method='SMOTE')

        for feature_method in feature_methods:
            print(f"Embedding: {embedding_name}, Feature Method: {feature_method}, Fold: {fold_idx + 1}")

            # Apply feature selection/reduction
            if feature_method == 'Original':
                X_train_fs, X_test_fs = X_train_balanced, X_test
            else:
                X_train_fs, X_test_fs = apply_feature_selection(feature_method, X_train_balanced, y_train_balanced, X_test, embedding_name)

            for clf_name, clf_info in classifiers.items():
                halving_grid = HalvingGridSearchCV(
                    estimator=clf_info['model'],
                    param_grid=clf_info['params'],
                    scoring="roc_auc",
                    factor=2,
                    cv=3,
                    n_jobs=-1,
                    verbose=1,
                )
                halving_grid.fit(X_train_fs, y_train_balanced)
                best_model = halving_grid.best_estimator_
                                
                # Predictions
                y_pred = best_model.predict(X_test_fs)
                y_prob = best_model.predict_proba(X_test_fs)[:, 1] if hasattr(best_model, "predict_proba") else None

                 # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
                
                
                # ROC Curve
                if y_prob is not None:
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

                    # Plot and Save ROC Curve
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
                    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
                    plt.title(f"ROC Curve: {embedding_name} | {clf_name} | {feature_method} | Fold {fold_idx + 1}")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.legend(loc="lower right")
                    plt.savefig(f"results/ROC_curve_{embedding_name}_{clf_name}_{feature_method}_fold_{fold_idx + 1}.png")
                    plt.close()
                    #plt.show()

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                
                conf_matrix_list_of_arrays.append(cm)
                
                print(cm)
                # Plot and Save Confusion Matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap="Blues")
                plt.title(f"Confusion Matrix: {embedding_name} | {clf_name} | {feature_method} | Fold {fold_idx + 1}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                #plt.show()
                plt.savefig(f"results/confusion_matrix_{embedding_name}_{clf_name}_{feature_method}_fold_{fold_idx + 1}.png")
                plt.close()

                # Store results
                results.append({
                    "Fold": fold_idx + 1,
                    "Embedding": embedding_name,
                    "Feature Method": feature_method,
                    "Classifier": clf_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "ROC AUC": roc_auc,
                    "Classification Report": classification_report(y_test, y_pred, target_names=class_names),
                })

print('\n')
print('Average Confusion Matrix')
aa = np.mean(conf_matrix_list_of_arrays, axis=0)

aaa = np.ceil(aa)

b=pd.DataFrame(aaa)
b=b.astype(int)
labels =['False','True']

c=np.array(b)


# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=c, display_labels=labels)
disp.plot(ax=ax, cmap="Blues", colorbar=True)

# Set labels for better visualization
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)

plt.show()


print('\n')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
mean_metrics = {metric: np.mean([result[metric] for result in results if metric in result]) for metric in metrics}

for metric, mean_value in mean_metrics.items():
    print(f"Mean {metric}: {mean_value:.2f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv('classification_results_multiclass.csv', index=False)

print("Results saved to classification_results_multiclass.csv.")

