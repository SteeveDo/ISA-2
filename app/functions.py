import pandas as pd
import spacy
import gensim
from gensim.models import Word2Vec
import unidecode
import unicodedata
import re
import os
import numpy as np
from sklearn import cluster
from sklearn.cluster import KMeans
import pickle
from dotenv import load_dotenv
import json

load_dotenv()
#télécharger le modèle de spacy qui permet de travailler avec le texte français
num_features = int(os.getenv("NUM_FEATURES")) #NLP CONFIG
ticket_file=os.getenv("TICKET_FILE")
solution_file=os.getenv("SOLUTION_FILE")
log_messages_file=os.getenv("LOG_MESSAGES_FILE")

nlp = spacy.load('fr_core_news_sm')
all_stopwords = nlp.Defaults.stop_words # récuperer les stopsWord (le, la, les, un, ...)

model_w2v = gensim.models.Word2Vec.load('WE_models/w2v_cbow_300D_Réseaux_21_11_22')

#Ecriture du message dans un fichier temporaire
def create_file(message,ticket_file=ticket_file,log_messages_file=log_messages_file):


    f=open(ticket_file, "a", encoding="utf8")
    f.write(f"Ticket\n{message}")
    f.close()

    f=open(log_messages_file,"a",encoding="utf8")
    f.write(f"{message}\n")
    f.close()

#Suppression du fichier contenant le message
def delete_file(ticket_file=ticket_file):

    f=open(ticket_file, "a", encoding="utf8")
    f.truncate(0)

# Normalisation du texte en utilisant les techniques du NLP
def normalisation_text(ticket_file=ticket_file):

    df=pd.read_csv(ticket_file,header=0)

    list_ligne_prix=df["Ticket"].values
    texts = []

    sentences = []
    i=0
    index_sent=0
    
    print("Parsing sentences from training set")
    
    for ligne_prix in list_ligne_prix :
        ligne_prix = str(ligne_prix.lower()) # mettre les lignes de prix en miniscule
        # supprimer les stop words:
        lst=[]
        for token in ligne_prix.split():
            if token.lower() not in all_stopwords: 
                lst.append(token)        
        ligne_prix= ' '.join(lst)
        
        doc = nlp(ligne_prix) # cette instruction permet de manipuler les lignes du prix en utilisant Spacy
        
        #lemmatization en utilisant spacy: permet de récuperer la racine des mots
        lemma_list = []
        for token in doc:
            if token not in all_stopwords:
                if token.lemma_ != " ":
                    lemma_list.append(token.lemma_)
        
        
        #remplace accents (é ---> e par exemple):
        without_accents = []
        for token in lemma_list:
            try:
                token = str(token, 'utf-8')         #unicode() a été remplacé par str()
            except (TypeError, NameError): 
                pass
            token = unicodedata.normalize('NFD', token) 
            token = token.encode('ascii', 'ignore')
            token = token.decode("utf-8")
            token = re.sub('[^A-Za-z0-9]+', ' ', token) 
            if token != " ":
                without_accents.append(token)

        
        #Supprimer les punctuation:
        not_punct_list = []
        for word in without_accents:
            punct = nlp.vocab[word]
            if punct.is_punct == False:
                not_punct_list.append(word)
        #print(not_punct_list)
        
        if len(not_punct_list)>0:
            sentences.append(not_punct_list)
        else:
            index_sent= i
            sentences.append(not_punct_list)
        #print(sentences)
        i=i+1
    texts+=sentences
    return texts


    # calculer le wordEmbedding des lignes de prix

# average all word vectors in a paragraph
def featureVecMethod(words, model_w2v=model_w2v, num_features=num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model_w2v.wv.index_to_key)
    
    for word in  words:
        #print(word)
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model_w2v.wv[word])
            
                
    featureVec = np.divide(featureVec, nwords)
    return featureVec

# calculate the average feature vector
def getAvgFeatureVecs(texts, model_w2v=model_w2v, num_features=num_features):
    counter = 0
    
    textFeatureVecs = np.zeros((len(texts),num_features),dtype="float32")# 6647*300
    #print(texts)
    #print(textFeatureVecs)
    for text in texts:
        #print(text)
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("text %d of %d"%(counter,len(texts)))
            
        textFeatureVecs[counter] = featureVecMethod(text, model_w2v, num_features)
        counter = counter+1
        
    return textFeatureVecs

def getRecommandations(DataVecs,ticket_file=ticket_file):
    
    df=pd.read_csv(ticket_file,header=0)
    if(len(DataVecs)>0):
        # Télécharger les models du clustering selon l'unité
        pickled_model = pickle.load(open('Model_Cluster_Réseau.pickle_21_11_2022', 'rb'))
        #predire le Cluster
        labels = pickled_model.predict(DataVecs)
        # dataframe contient les numéro du custers
        df_result_label = pd.DataFrame(labels)
        df_result_label.rename({0: 'Cluster'}, axis=1, inplace=True)
        # ajouter les numéro de clusters aux données de nouveau DQE.
        df_result_label = df_result_label.join(df)
        return df_result_label
    else:
        return None #Voir avec EHAB ce cas

def get_solutions(df_result_label):

    df_Sol = pd.read_csv(solution_file, sep=';')

    #res=df_Sol.loc[df_Sol.Cluster==165]

    df_result = pd.merge(df_result_label, df_Sol)

    df_final = df_result[["Ticket", "Nom_Cluster_Sol"]]

    result=df_final.rename(columns = {'Nom_Cluster_Sol':'Solutions'})

    return result.to_dict('records')