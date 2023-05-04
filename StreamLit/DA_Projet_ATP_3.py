# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:55:13 2023

@author: egern
"""



import pandas as pd 
import numpy as np 
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt 
import io
import plotly.express as px
from bokeh.plotting import figure, output_notebook, show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC


import streamlit as st
from PIL import Image

image = Image.open('meilleurs-joueurs-tennis.jpg')
st.image(image)

st.title(' :sunglasses: Paris Sportifs :tennis:')

st.markdown("*OBJECTIF : battre* :muscle: *les algorithmes des bookmakers sur l’estimation de la probabilité d’une équipe gagnant* :trophy: *un match* :moneybag: :moneybag: :moneybag:")


df = pd.read_csv('atp_data.csv')
df['Date'] = pd.to_datetime(df['Date'])


df_origine = pd.read_csv('atp_data.csv')
df_origine['Date'] = pd.to_datetime(df_origine['Date'])

st.sidebar.title("Paris Sportifs")

pages = ["Projet", "Jeu de données", "Dataviz", "Pre-processing","Modelisation", "Conclusion et perspectives"]


page = st.sidebar.radio("Sommaire", pages)

with st.sidebar:
    st.write("Auteurs:"
             "- Tarik AMAROUCHE "
             "- Euriel GERNIGON "
             "- Quentin LEFFRAY "
             "- Xin SU")
    
 ### Homogénéisation des données par la suppression des lignes où nous ne detenons pas l'information de côte de la part des deux bookmakers 

df_origine = df_origine.dropna(axis = 0, how = 'all', subset = ['PSW'])
df_origine = df_origine.dropna(axis = 0, how = 'all', subset = ['B365W'])

### Remplacement des valeurs NaN des W/LSets par 0 car uniquement des matchs Walkover(sans match joué) et Retired (match commencé mais sans set terminé pour le gagnant). 1 exception pour un match terminé, marginal donc remplacé par 0

df_origine['Wsets'] = df_origine['Wsets'].fillna(0)
df_origine['Lsets'] = df_origine['Lsets'].fillna(0)
df_origine.reset_index(drop=True, inplace=True)

### Création des colonnes permettant de savoir si un joueur ayant un meilleur ELO / Rang / Favori pour Pinnacle / Favori pour Bet365 gagne
df_origine['ELO_Predicable_Win'] = (df_origine['elo_winner'] - df_origine['elo_loser']).apply(lambda x: True if  x > 0  else False)
df_origine['Rank_Predicable_Win'] = (df_origine['WRank'] - df_origine['LRank']).apply(lambda x: True if  x < 0  else False)
df_origine['Pinnacle_Predicable_Win'] = (df_origine['PSW'] - df_origine['PSL']).apply(lambda x: True if  x < 0  else False)
df_origine['B365_Predicable_Win'] = (df_origine['B365W'] - df_origine['B365L']).apply(lambda x: True if  x < 0  else False)

### value_counts() pour chacune des variables crées, permettant de déterminer une certaine cohérence dans les données

### Création de la colonne permettant de visualiser des écarts standardisés pour réaliser les graphiques
df_origine['ELO_Diff'] = df_origine['elo_winner'] - df_origine['elo_loser']

df_origine['ELO_Diff_Gagnant'] = pd.cut(x = df_origine['ELO_Diff'],
                                bins = [-1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000],
                                labels = ['-1000 / -500','-500 / - 400', '-400 / - 300', '-300 / -200', '-200 / -100', '-100 / 0', 
                                          '0 / 100', '100 / 200', '200 / 300', '300 / 400', '400 / 500', '500 / 1000'])


### Vérification graphique de stratégie (Pari sur le mieux classé à l'ELO / Favori pour le bookmaker sur les deux bookmakers)

roi_favori_ps=100*(df_origine.PSW[df_origine['PSW']<df_origine['PSL']].sum()-len(df_origine))/len(df_origine)
roi_meilleur_rang_ps=100*(df_origine.PSW[df_origine['elo_winner']>df_origine['elo_loser']].sum()-len(df_origine))/len(df_origine)

roi_favori_365=100*(df_origine.B365W[df_origine['B365W']<df_origine['B365L']].sum()-len(df_origine))/len(df_origine)
roi_meilleur_rang_365=100*(df_origine.B365W[df_origine['elo_winner']>df_origine['elo_loser']].sum()-len(df_origine))/len(df_origine)

# Calcul du ROI maximum possible
max_roi_ps = 100*(df_origine.PSW.sum()-len(df_origine))/len(df_origine)
max_roi_365 = 100*(df_origine.B365W.sum()-len(df_origine))/len(df_origine)

### Stockage dans une variable de la proba_elo, transformée en %

proba_elo_en_pct = df_origine['proba_elo']*100

# On réattribue les colonnes Winner/Loser au profit d'un J1/J2 pour enlever le biais de connaitre le résultat à l'avance

df_origine['Joueur_1'] = np.where(np.random.rand(len(df_origine)) > 0.5, df_origine['Winner'], df_origine['Loser'])
df_origine['Joueur_2'] = np.where(df_origine['Joueur_1'] == df_origine['Winner'], df_origine['Loser'], df_origine['Winner'])

### ANONIMISATION DES DONNÉES

# Réaffectation de la côte à chaque joueur et non plus au couple Winner/Loser

df_origine['PSJ1'] = np.where(df_origine['Winner'] == df_origine['Joueur_1'], df_origine['PSW'], df_origine['PSL'])
df_origine['PSJ2'] = np.where(df_origine['Winner'] == df_origine['Joueur_1'], df_origine['PSL'], df_origine['PSW'])

# Réaffectation de l'ELO à chaque joueur et non plus au couple Winner/Loser

df_origine['elo_J1'] = np.where(df_origine['Winner'] == df_origine['Joueur_1'], df_origine['elo_winner'], df_origine['elo_loser'])
df_origine['elo_J2'] = np.where(df_origine['Winner'] == df_origine['Joueur_1'], df_origine['elo_loser'], df_origine['elo_winner'])

# Proba ELO pour chaque joueur

df_origine['proba_elo_J1'] = np.where(df_origine['Winner'] == df_origine['Joueur_1'], df_origine['proba_elo'], 1 - df_origine['proba_elo'])
df_origine['proba_elo_J2'] = np.where(df_origine['Winner'] == df_origine['Joueur_1'], 1 - df_origine['proba_elo'], df_origine['proba_elo'])


### CREATION DE VARIABLES EXPLICATIVES

# Créer les colonnes qui calcule le pourcentage de victoires pour chaque joueur

win= df_origine['Winner'].value_counts()
matchs_played = df_origine[['Winner', 'Loser']].stack().value_counts()

winrate = (win / matchs_played).round(2)

df_origine['winrate_J1'] = df_origine['Joueur_1'].map(winrate)
df_origine['winrate_J2'] = df_origine['Joueur_2'].map(winrate)

# Créer les colonnes qui calcule le pic ELO pour chaque joueur

maxima_elo = df_origine.groupby('Loser')['elo_loser'].max()

df_origine['max_elo_J1'] = df_origine['Joueur_1'].map(maxima_elo)
df_origine['max_elo_J2'] = df_origine['Joueur_2'].map(maxima_elo)

df_origine[['winrate_J1', 'winrate_J2']] = df_origine[['winrate_J1', 'winrate_J2']].fillna(0) ### Suppression des valeurs nulles (ici par manque de données pour le calcul du winrate)
df_origine[['max_elo_J1', 'max_elo_J2']] = df_origine[['max_elo_J1', 'max_elo_J2']].fillna(1500)

### CREATION DE LA TARGET

df_origine['Target'] = np.where(df_origine['Joueur_1'] == df_origine['Winner'], 0, 1)

df_final = df_origine.drop(['ATP', 'Comment', 'Location', 'Tournament', 'Series','Round', 'Best of', 'B365W', 'B365L', 'Wsets', 'Lsets', 'WRank', 
              'LRank', 'B365_Predicable_Win','Rank_Predicable_Win', 'ELO_Diff', 'ELO_Diff_Gagnant', 'ELO_Predicable_Win',
              'Pinnacle_Predicable_Win', 'Winner', 'Loser', 'Date', 'Court', 'PSW', 'PSL', 'elo_winner', 'elo_loser', 'proba_elo'],
               axis=1)




# Séparation du jeu de données en deux DataFrame (dont un pour sortir la cible)

feats = df_final.drop(['Target', 'Joueur_1', 'Joueur_2'], axis = 1)

target = df_final['Target']

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.25, shuffle = False)

# Standardisation des valeurs numériques

sc = StandardScaler()

col_num = ['PSJ1', 'PSJ2', 'elo_J1', 'elo_J2', 'proba_elo_J1', 'proba_elo_J2', 'winrate_J1', 'winrate_J2', 'max_elo_J1', 'max_elo_J2']

X_train[col_num] = sc.fit_transform(X_train[col_num])
X_test[col_num] = sc.transform(X_test[col_num])

#Encoding sur les variables explicatives/catégorielles (ici textuelles) avec un .replace

X_train['Surface'].replace(['Hard', 'Clay', 'Grass', 'Carpet'], [0, 1, 2, 3], inplace=True)
X_test['Surface'].replace(['Hard', 'Clay', 'Grass', 'Carpet'], [0, 1, 2, 3], inplace=True)
   
    

if page == pages[0]:
    st.header("Projet")
    
    st.write("Nous allons donc essayer de surperformer de les algorithmes des bookmakers, ici sur une base de données de matchs de tennis masculins de 2000 à 2018. Pour cela, nous allons mettre en pratique les différentes techniques vues lors des modules et MasterClass (nettoyage des données, visualisation, expérimentation de modèles de machine learning)")
    st.write("L'intérêt ici est donc économique puisque l'on va chercher à trouver le meilleur Return On Investment (ROI)en fonction des données qui sont à notre disposition. Plus que la recherche d'obtenir un ROI, nous allons essayer à travers différents modèles de machine learning de l'optimiser.")
    
    st.subheader("Quelques définitions")
    
    st.markdown("- **ELO** : système de classement par point, qui permet ensuite de ressortir des statistiques en fonction des écarts entre les deux adversaires, indépendamment de la surface sur laquelle les matchs sont joués (ex : un écart de 100 points implique une chance statistique de 64% pour celui ayant le plus de 100 points). Ce système de points est actualisé avant chaque match mais n'est pas réinitialisé à chaque année calendaire (différent donc de l'ATP Race). Tout joueur entrant sur le circuit professionnel commence avec 1 500 points.")
    st.markdown("- **Rang** : permet de classifier les joueurs en fonction de leur ELO à une date donnée.")
    
    

    
if page == pages[1]:
    st.header("Jeu de données")    


    st.subheader("Les variables")   
    st.markdown("le jeu de données se trouve [ici](https://www.kaggle.com/datasets/edouardthomas/atp-matches-dataset)")
        
    st.markdown("Description des colonnes : [data_atp](https://docs.google.com/spreadsheets/d/1YCB3vDHN-c4CQ8w0ZihWmrFHJQlP1QRYAaunqr0Uzu0/edit#gid=0)")


    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)       
    
    
    st.subheader("Extrait du Dataframe d'origine")   
    st.table(df.head(7))
    
    st.markdown("Nous voyons donc que le dataset contient différentes données relatives au circuit professionnel de tennis (tournoi, surface, etc.) ainsi que des côtes relatives à certains bookmakers.")
    


    st.subheader("Taille du Dataframe") 
    st.markdown("44 708 lignes x 23 colonnes") 
    st.markdown("44 708 matchs") 
    st.markdown("1020 joueurs") 

    st.subheader("Gestion des doublons") 
    st.write( "Pas de doublon")

    st.subheader("Gestion des valeurs manquantes")     
        
    st.dataframe(df.isna().sum().loc[lambda x: x> 0].sort_values(ascending = False))  
    
    
    st.markdown("- PSW/PSL : Suppression des lignes où les données conernant les matchs les plus anciens (entre 2000 et 2004) car elles ne disposent pas de côte sur Pinnacle")
    st.markdown("- B365W/B365L : Suppression des lignes où les données étaient manquantes (pas besoin de l'appliquer sur les colonnes PSL/B365L puisque les valeurs manquantes fonctionnaient par paire avec PSW/B365W) :")
    st.markdown("- Wset/Lset : Remplacement les valeurs manquantes pour ces variables par des 0 puisque cela concerne uniquement des matchs avec abandon d'un des joueurs durant le match ou avant le match")
        
    st.subheader("Ajout des nouvelles variables")    
    
    
    df = df.dropna(axis = 0, how = 'all', subset = ['PSW'])
    df = df.dropna(axis = 0, how = 'all', subset = ['B365W'])


    df['Wsets'] = df['Wsets'].fillna(0)
    df['Lsets'] = df['Lsets'].fillna(0)

    df['ELO_Predicable_Win'] = (df['elo_winner'] - df['elo_loser']).apply(lambda x: True if  x > 0  else False)
    df['Rank_Predicable_Win'] = (df['WRank'] - df['LRank']).apply(lambda x: True if  x < 0  else False)
    df['Pinnacle_Predicable_Win'] = (df['PSW'] - df['PSL']).apply(lambda x: True if  x < 0  else False)
    df['B365_Predicable_Win'] = (df['B365W'] - df['B365L']).apply(lambda x: True if  x < 0  else False)

    df['ELO_Diff'] = df['elo_winner'] - df['elo_loser']

    df['ELO_Diff_Gagnant'] = pd.cut(x = df['ELO_Diff'],
                                bins = [-1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000],
                                labels = ['-1000 / -500','-500 / - 400', '-400 / - 300', '-300 / -200', '-200 / -100', '-100 / 0', 
                                          '0 / 100', '100 / 200', '200 / 300', '300 / 400', '400 / 500', '500 / 1000'])
 
    st.markdown("Nous avons crée les colonnes suivantes, permettant de ressortir un résultat booléen afin de déterminer si le plus haut ELO / Rang / Favori du match pour Pinnacle et Bet365 gagne le match :")
    st.markdown("- ELO_Predicable_Win : Booléen qui détermine si un joueur qui a un meilleur Elo gagne")
    st.markdown("- Rank_Predicable_Win : Booléen qui détermine si un joueur qui a un meilleur rang (Rank) gagne")
    st.markdown("- Pinnacle_Predicable_Win : Booléen qui détermine si un joueur favori selon Pinnacle gagne ")
    st.markdown("- B365_Predicable_Win : Booléen qui détermine si un joueur favori selon B365 gagne ")
    st.markdown("- ELO_Diff : écart d'Elo entre les joueurs")
    st.markdown("- Winrate : calcule le pourcentage de victoires pour chaque joueur")
    st.markdown("- Max_elo : qui détermine le pic Elo de chaque joueur")
    st.markdown("- Variable cible : le gagnant du match")        
        
        
        
elif page == pages[2]:
    st.header("Data Visualisation")
    
    st.markdown("- Langage utilisé : Python")
    st.markdown("- Librairies utilisées : Pandas, NumPy, Matplotlib, Seaborn, Sklearn")
    

    if st.checkbox('Top des 15 meilleurs joueurs'):        
        winners = df.loc[(df["Round"] == 'The Final')]['Winner']
        winners.head()

        sns.barplot(y=winners.value_counts().head(15).index, x=winners.value_counts().head(15));
        plt.title(label = 'Nombre de tournois gagnés par joueur entre 2000 et 2018');
        
        st.pyplot(plt)
               
        st.markdown("GOAT : Roger Federer")
    
    if st.checkbox('Matrice de corrélation'):
        image2 = Image.open('matrice_corre.png')
        st.image(image2)
        
        
        
        st.markdown("On remarque une faible corrélation entre les variables entre elles, excepté entre les côtes  (PSW/B365W | PSL/B365L | Proba_Elo/Elo_Winner)")
    
    if st.checkbox('Résultats en fonction des nouvelles variables créées'):
        image3 = Image.open('plot4.png')
        st.image(image3)
        
        st.markdown("On constate qu'il existe une relation entre les 3 variables que sont l'ELO / le rang / le favori pour les bookmakers (la plus forte variation est sur le couple Pinnacle/Rang qui représente un écart de 4,2%). On voit donc que les bookmakers sont marginalement des meilleurs indicateurs sur le gagnant d'un match que l'ELO/Rang. Cela peut s'expliquer notamment par le retour à la compétition de joueurs de très haut niveau après une blessure (n'ayant pas participé à des tournois, ils ont perdu en ELO donc en rang). Cependant le modèle de Pinnacle semble plus performant que l'ELO prédire le gagnant d'un match") 
                                                                                                                                                                                                                                                                                                                                                                                                                                        
    
    
    if st.checkbox('Répartition des victoires en fonction de la différence d ELO'):
    
        image4 = Image.open('plot_Elo.png')
        st.image(image4)
    
        st.markdown("Ce graphe nous montre la distribution des gagnants en fonction de l'écart d'ELO des deux joueurs. On remarque que le plupart des matchs sont joués entre des écarts allant de -100 a 200, ce qui laisse supposer des matchs entre des joueurs de même niveau et forme physique (puisque l'ELO mesure l'état de forme également puisque actualisé avant chaque tournoi) plutôt serrés et donc difficilement pronosticables. (1 match sur 3, que ce soit en fonction de l'ELO/Rang/ Favori des bookmakers, se solde par la victoire de l'outsider")
    
    
    if st.checkbox('Vérification graphique de stratégie (Pari sur le mieux classé à l ELO / Favori pour le bookmaker sur les deux bookmakers)'):
        
        image5 = Image.open('strategie.png')
        st.image(image5)
        
        st.markdown("Quels sont mes bénéfices si je parie 1 € sur chaque match avec une stratégie donnée (Meilleur ELO / Favori pour le BM) ?")
        st.markdown("Même si 2 matchs sur 3 sont remportés par le favori, si nous misons sur chaque match, nous sommes quoi qu'il arrive perdant au final (exprimés en %).")
        st.markdown("Cela s'explique par les côtes des favoris qui sont mathématiquement plus basses que celles des outsiders.")
    
   
    


    if st.checkbox('Pari sur le mieux classé à l ELO / Favori pour le bookmaker sur les deux bookmakers'):
        
        image6 = Image.open('bar.png')
        st.image(image6)
        
    
        st.markdown("Ici c'est le ROI maximal que nous pourrions atteindre si nous misions sur tous les matchs avec une accuracy à 100% . On constate que Pinnacle est bien mieux placé avec un ROI à 92% alors que Bet365 dépasse à peine les 80%. Nous allons donc nous concentrer sur Pinnacle, sur lequel il semble plus simple de 'battre le bookmaker' à la faveur de côte généralement supérieures à celles de Bet365")
    
    
    
    
    if st.checkbox('Probabilité de victoire en fonction de l écart d ELO du favori'):
        image7 = Image.open('droite.png')
        st.image(image7)


        st.markdown("Ici nous avons la représentation graphique de la probabilité que le favori a de gagner en fonction de son écart d'élo, à partir de la base de données fournie")

elif page == pages[3]:
    st.header("Pre Processing")
    
    st.subheader("Méthode 1")
    st.write("Choisir un joueur de manière aléatoire parmi les deux colonnes Winner/Loser")


    st.markdown("- **Selection aléatoire d'un Joueur et création de la variable cible**")
    
    code0 = ''' # Création de la variable random picking, permettant une classification binaire.
    # On ajoute une colonne Player au Dataframe pour stocker le joueur principal de chaque match.
    # cette nouvelle colonne permet d'assigner de manière aléatoire le joueur principal de chaque match en choisissant soit le vainqueur soit le perdant
    
    select_player = np.where(np.random.rand(len(df)) > 0.5, df['Winner'], df['Loser']) 
    
    # On ajoute une colonne Target pour stocker la variable cible binaire (0 si le joueur choisi a perdu, 1 sinon)
    df['Target'] = np.where((select_player  == df['Winner'] &  df['Winner'] != 0), 0, 1)
    
     '''
    st.code(code0, language='python') 
    
    st.markdown("- **Suppression de certaines données**")
    code6 = '''# On retire des colonnes pour alléger la base de données afin d'avoir un retour du modèle plus rapide

df = df.drop(['ATP', 'Comment', 'Location', 'Tournament', 'Series','Round', 'Best of', 'B365W', 'B365L', 'Wsets', 'Lsets', 'WRank', 
              'LRank', 'B365_Predicable_Win','Rank_Predicable_Win', 'ELO_Diff', 'ELO_Diff_Gagnant', 'ELO_Predicable_Win',
              'Pinnacle_Predicable_Win', 'Winner', 'Loser', 'Date', 'Court', 'PSW', 'PSL', 'elo_winner', 'elo_loser', 'proba_elo'],
               axis=1)
    '''
    st.code(code6, language='python')
    

    st.markdown("- **Slicing entre les variables explicatives et la variable cible**")
    code8 = '''
    
    feats = df_final.drop(['Target'], axis = 1)

    target = df_final['Target']
     
    '''
    st.code(code8, language='python')
    
    st.markdown("- **Séparation du jeu d'entrainement et du jeu de test**")
    code9 = ''' # Répartition 75/25
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.25, shuffle = False)
    
    '''
    st.code(code9, language='python')
    
    st.markdown("- **Standardisation/Encodage**")
    st.markdown("Nous avons standardisé les données numériques à l'aide de StandardScaler de la librairie scikitlearn")
    
    code4 = '''sc = StandardScaler()
     col_num = ['PSJ1', 'PSJ2', 'elo_J1', 'elo_J2', 'proba_elo_J1', 'proba_elo_J2', 'winrate_J1', 'winrate_J2', 'max_elo_J1', 'max_elo_J2']
     X_train[col_num] = sc.fit_transform(X_train[col_num])
     X_test[col_num] = sc.transform(X_test[col_num])'''
    st.code(code4, language='python')
    
    st.markdown("Nous avons ensuite géré les données textuelles du jeu de données en passant par un .replace plutôt que par un LabelEncoder ou un OneHotEncoder :")
    code5 = '''  X_train['Surface'].replace(['Hard', 'Clay', 'Grass', 'Carpet'], [0, 1, 2, 3], inplace=True)
X_test['Surface'].replace(['Hard', 'Clay', 'Grass', 'Carpet'], [0, 1, 2, 3], inplace=True)'''  
    st.code(code5, language='python') 
    
    st.subheader("Méthode 2")
    
    st.write("Nouvelle approche : anonymiser chaque match et essayer d'enrichir notre dataset avec les informations en notre possession")

    st.markdown("- **Anonymisation du Dataset**")
    
    code0 = '''            
    # On réattribue les colonnes Winner/Loser au profit d'un J1/J2 pour enlever le biais de connaitre le résultat à l'avance

df['Joueur_1'] = np.where(np.random.rand(len(df)) > 0.5, df['Winner'], df['Loser'])
df['Joueur_2'] = np.where(df['Joueur_1'] == df['Winner'], df['Loser'], df['Winner'])
    
    
    '''
    st.code(code0, language='python')      
    
    code1 = ''' ### ANONYMISATION DES DONNÉES
# Réaffectation de la côte à chaque joueur et non plus au couple Winner/Loser
df['PSJ1'] = np.where(df['Winner'] == df['Joueur_1'], df['PSW'], df['PSL'])
df['PSJ2'] = np.where(df['Winner'] == df['Joueur_1'], df['PSL'], df['PSW'])

# Réaffectation de l'ELO à chaque joueur et non plus au couple Winner/Loser

df['elo_J1'] = np.where(df['Winner'] == df['Joueur_1'], df['elo_winner'], df['elo_loser'])
df['elo_J2'] = np.where(df['Winner'] == df['Joueur_1'], df['elo_loser'], df['elo_winner'])

# Proba ELO pour chaque joueur

df['proba_elo_J1'] = np.where(df['Winner'] == df['Joueur_1'], df['proba_elo'], 1 - df['proba_elo'])
df['proba_elo_J2'] = np.where(df['Winner'] == df['Joueur_1'], 1 - df['proba_elo'], df['proba_elo'])
'''    
    st.code(code1, language='python') 

    st.write("Nous avons retenu le poids de 0.5 pour que le jeu de données soit le plus équitablement réparti entre une victoire du joueur 1 et une victoire du joueur 2 (nous avons fait les tests sur des nombres extrêmes tel que 0.1 ou 0.9, cela conduisait à une performance de modèle supérieure à 0.95 mais aussi à un biais trop important, le modèle comprenant que la majorité des vainqueurs se trouvait dans l'une ou l'autre colonne, faussant ainsi l'analyse).")

    
    st.markdown("- **Création de nouvelles variables**")
        
    code2 = '''### CREATION DE VARIABLES EXPLICATIVES

# Créer les colonnes qui calcule le pourcentage de victoires pour chaque joueur

win= df['Winner'].value_counts()
matchs_played = df[['Winner', 'Loser']].stack().value_counts()

winrate = (win / matchs_played).round(2)

df['winrate_J1'] = df['Joueur_1'].map(winrate)
df['winrate_J2'] = df['Joueur_2'].map(winrate)

# Créer les colonnes qui calcule le pic ELO pour chaque joueur

maxima_elo = df.groupby('Loser')['elo_loser'].max()

df['max_elo_J1'] = df['Joueur_1'].map(maxima_elo)
df['max_elo_J2'] = df['Joueur_2'].map(maxima_elo)
'''
    st.code(code2, language='python') 
    
    
    
    st.markdown("- **Construction de la variable cible**")
    st.write("Nous sommes partis sur la construction d'une variable choisissant un joueur aléatoire dans un match, permettant de savoir si nous avons trouvé ou non le gagnant du match, qui sera notre variable cible")
    
    code3 = '''
    ### CREATION DE LA TARGET

    df['Target'] = np.where(df['Joueur_1'] == df['Winner'], 0, 1)
    '''
    st.code(code3, language='python') 
   
    st.write("Nous avons ensuite enlevé toutes les colonnes qui n'avaient ici selon nous pas d'intérêt pour l'entrainement du modèle afin de pouvoir gagner en performances lors de ce dit entrainement.")
    
    st.subheader("Jeu de données final")
    
    df = df.dropna(axis = 0, how = 'all', subset = ['PSW'])
    df = df.dropna(axis = 0, how = 'all', subset = ['B365W'])


    df['Wsets'] = df['Wsets'].fillna(0)
    df['Lsets'] = df['Lsets'].fillna(0)
    
    df['Joueur_1'] = np.where(np.random.rand(len(df)) > 0.5, df['Winner'], df['Loser'])
    df['Joueur_2'] = np.where(df['Joueur_1'] == df['Winner'], df['Loser'], df['Winner'])

    df['ELO_Predicable_Win'] = (df['elo_winner'] - df['elo_loser']).apply(lambda x: True if  x > 0  else False)
    df['Rank_Predicable_Win'] = (df['WRank'] - df['LRank']).apply(lambda x: True if  x < 0  else False)
    df['Pinnacle_Predicable_Win'] = (df['PSW'] - df['PSL']).apply(lambda x: True if  x < 0  else False)
    df['B365_Predicable_Win'] = (df['B365W'] - df['B365L']).apply(lambda x: True if  x < 0  else False)

    df['ELO_Diff'] = df['elo_winner'] - df['elo_loser']

    df['ELO_Diff_Gagnant'] = pd.cut(x = df['ELO_Diff'],
                             bins = [-1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000],
                             labels = ['-1000 / -500','-500 / - 400', '-400 / - 300', '-300 / -200', '-200 / -100', '-100 / 0', 
                                       '0 / 100', '100 / 200', '200 / 300', '300 / 400', '400 / 500', '500 / 1000'])



    df['PSJ1'] = np.where(df['Winner'] == df['Joueur_1'], df['PSW'], df['PSL'])
    df['PSJ2'] = np.where(df['Winner'] == df['Joueur_1'], df['PSL'], df['PSW'])

    df['elo_J1'] = np.where(df['Winner'] == df['Joueur_1'], df['elo_winner'], df['elo_loser'])
    df['elo_J2'] = np.where(df['Winner'] == df['Joueur_1'], df['elo_loser'], df['elo_winner'])

    df['proba_elo_J1'] = np.where(df['Winner'] == df['Joueur_1'], df['proba_elo'], 1 - df['proba_elo'])
    df['proba_elo_J2'] = np.where(df['Winner'] == df['Joueur_1'], 1 - df['proba_elo'], df['proba_elo'])

    win= df['Winner'].value_counts()    
    matchs_played = df[['Winner', 'Loser']].stack().value_counts()

    winrate = (win / matchs_played).round(2)

    df['winrate_J1'] = df['Joueur_1'].map(winrate)
    df['winrate_J2'] = df['Joueur_2'].map(winrate)

    maxima_elo = df.groupby('Loser')['elo_loser'].max()

    df['max_elo_J1'] = df['Joueur_1'].map(maxima_elo)
    df['max_elo_J2'] = df['Joueur_2'].map(maxima_elo)
    
    df[['winrate_J1', 'winrate_J2']] = df[['winrate_J1', 'winrate_J2']].fillna(0)
    df[['max_elo_J1', 'max_elo_J2']] = df[['max_elo_J1', 'max_elo_J2']].fillna(1500)
    df['Target'] = np.where(df['Joueur_1'] == df['Winner'], 0, 1)
    
    df['Surface'].replace(['Hard', 'Clay', 'Grass', 'Carpet'], [0, 1, 2, 3], inplace=True)
    
    df = df.drop(['ATP', 'Comment', 'Location', 'Tournament', 'Series','Round', 'Best of', 'B365W', 'B365L', 'Wsets', 'Lsets', 'WRank', 
              'LRank', 'B365_Predicable_Win','Rank_Predicable_Win', 'ELO_Diff', 'ELO_Diff_Gagnant', 'ELO_Predicable_Win',
              'Pinnacle_Predicable_Win', 'Winner', 'Loser', 'Date', 'Court', 'PSW', 'PSL', 'elo_winner', 'elo_loser', 'proba_elo'],
               axis=1)
    
    st.dataframe(df.head(10))
    
    
    st.markdown("- **Séparation du jeu de données**")
    code10 ='''
# Séparation du jeu de données en deux DataFrame (dont un pour sortir la cible)

feats = df.drop(['Target', 'Joueur_1', 'Joueur_2'], axis = 1)

target = df['Target']'''

    st.code(code10, language='python')
    
    
elif page == pages[4] : 
    
    st.header("Modélisation")
    
    st.markdown("- But : On cherche à prédire l'issue d'un match entre un joueur J1 et un joueur J2 -> classification")
   

    
    def train_model(model_choisi) : 
         
         if model_choisi == 'RandomForest' :
             model = RandomForestClassifier()
         elif model_choisi == 'Decision Tree' :
             model = DecisionTreeClassifier()
         elif model_choisi == 'KNN' :
             model = KNeighborsClassifier()
         elif model_choisi == 'Regression Log' :
             model = LogisticRegression()
         model.fit(X_train, y_train)
         score = model.score(X_test, y_test)
         return score
    model_choisi = st.selectbox(label = "Choix de mon modèle", options = ['RandomForest', 'Decision Tree', 'KNN', 'Regression Log']) 
    
    st.write("Score test", train_model(model_choisi))
   
    
     
    
    if st.checkbox('Best Model'): 
        reglog = LogisticRegression()
        reglog.fit(X_train, y_train)

        st.write('Score sur ensemble train', reglog.score(X_train, y_train))
        st.write('Score sur ensemble test', reglog.score(X_test, y_test))
    
        st.write("- Metrics de performances obtenues")
    
        imagef = Image.open('cross_tab.png')
        st.image(imagef)
    
        st.write("- Meilleurs paramètres")
    
        codef = '''# définir les hyperparamètres à tester
param_grid = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]}

# définir le modèle
model = LogisticRegression(solver='liblinear')

# définir la grille de recherche
grid_search = GridSearchCV(model, param_grid, cv=4, n_jobs=-1, error_score='raise')

# ajuster la grille de recherche aux données
try:
    grid_search.fit(X_train, y_train)
except Exception as e:
    print('Error: ', e)

# afficher les meilleurs paramètres et score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)'''

        st.code(codef, language='python')  
        st.write("Best parameters:  {'C': 1, 'penalty': 'l1'}")
        st.write("Best score:  0.7106818557535233")
     
        st.write("- Résultats")
        st.write("Nos résultats sont donc meilleurs ici qu'en random picking (plus 0.2 points) par l'anonymisation des données et l'ajout de données explicatives. On est donc théoriquement au dessus des 66 % recherchés pour essayer de battre les bookmakers.")
        
        
    if st.checkbox('ROI'):
        
        code_roi = '''
        param_grid = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]}
        model = LogisticRegression(solver='liblinear')
        grid_search = GridSearchCV(model, param_grid, cv=4, n_jobs=-1, error_score='raise')
        try:
            grid_search.fit(X_train, y_train)
        except ValueError as e:
            st.write('Error: ', e)
    

        
        def allocation_paris(base_cagnotte=0, mise_min=10, mise_max=1000, sureté=0.99):
            y_pred_proba = grid_search.predict_proba(X_test) 
            cagnotte = base_cagnotte
            mise_totale = 0
    
            for i, probas in enumerate(y_pred_proba):
                cotes_J1 = df['PSJ1'].loc[i+24257]
                cotes_J2 = df['PSJ2'].loc[i+24257]
                
                if probas[0]>sureté:
                    if y_test.loc[i+24257]==0:
                        st.write('parier {}€ sur victoire Joueur 1 -'.format(round(mise_min+(mise_max-mise_min)*(probas[0]-sureté)/(1-sureté))) , 'cote à {} -'.format(cotes_J1), 'GAGNÉ - cagnotte : {}€'.format(cagnotte + round((mise_min+(mise_max-mise_min)*(probas[0]-sureté)/(1-sureté))*(cotes_J1))))
                        cagnotte += round((mise_min+(mise_max-mise_min)*(probas[0]-sureté)/(1-sureté))*(cotes_J1))
                    else:
                        st.write('parier {}€ sur victoire Joueur 1 -'.format(round(mise_min+(mise_max-mise_min)*(probas[0]-sureté)/(1-sureté))) , 'cote à {} -'.format(cotes_J1), 'PERDU - cagnotte : {}€'.format(cagnotte - round(mise_min+(mise_max-mise_min)*(probas[0]-sureté)/(1-sureté))))
                        cagnotte -= round(mise_min+(mise_max-mise_min)*(probas[0]-sureté)/(1-sureté))
                    mise_totale += round(mise_min+(mise_max-mise_min)*(probas[0]-sureté)/(1-sureté))
                
                if probas[1]>sureté:
                    if y_test.loc[i+24257]==1:
                        st.write('parier {}€ sur victoire Joueur 2 -'.format(round(mise_min+(mise_max-mise_min)*(probas[1]-sureté)/(1-sureté))) , 'cote à {} -'.format(cotes_J2), 'GAGNÉ - cagnotte : {}€'.format(cagnotte + round((mise_min+(mise_max-mise_min)*(probas[1]-sureté)/(1-sureté))*(cotes_J2))))
                        cagnotte += round((mise_min+(mise_max-mise_min)*(probas[1]-sureté)/(1-sureté))*(cotes_J2))
                    else:
                        st.write('parier {}€ sur victoire Joueur 2 -'.format(round(mise_min+(mise_max-mise_min)*(probas[1]-sureté)/(1-sureté))) , 'cote à {} -'.format(cotes_J2),'PERDU - cagnotte : {}€'.format(cagnotte - round(mise_min+(mise_max-mise_min)*(probas[1]-sureté)/(1-sureté))))
                        cagnotte -= round(mise_min+(mise_max-mise_min)*(probas[1]-sureté)/(1-sureté))
                    mise_totale += round(mise_min+(mise_max-mise_min)*(probas[1]-sureté)/(1-sureté))
                            
            st.write("La mise totale a été de ", mise_totale, '€')

        allocation_paris()
        
        plt.figure(figsize = (10,14))
        pd.Series(grid_search.best_estimator_.coef_[0], X_test.columns).sort_values(ascending=False).plot(kind='barh');
        '''
        
        st.code(code_roi, language='python')  
        imageroi = Image.open('ROI.png')
        st.image(imageroi)
        
        
        
        
        
        
        
        
elif page == pages[5] : 
    
    st.header("Conclusion")
    st.markdown("Ce projet nous a permis de travailler sur les différentes étapes d'un projet de Data Science (exploration et nettoyage des données, détermination et création des variables explicatives, machine learning, présentation des résultats).")
    
    st.markdown("Peut-on finalement battre les bookmakers ? Il est difficile de répondre de manière définitive à cette question car les bookmakers ont des années d'expérience et des modèles sophistiqués pour prédire les résultats de matchs de tennis.")
   
   
    
    st.subheader("Perspectives")
    st.markdown("- Enrichir notre dataset avec des statistiques de matchs plus poussées (nombre d'aces, de coups gagnants, la durée du match précédent, informations complémentaires sur les joueurs...)") 
    st.markdown("- Créer d'autres variables comme un ELO par surface, le nombre de sets joués au match précédent pour connaître la forme d'un joueur")
    st.markdown("- Réduire le nombre de matchs sur lequel faire les paris (par exemple se limiter aux tournois du Grand Chelem, qui propose 117 matchs par tournoi afin d'avoir un autre calcul de ROI).")
    st.markdown("- Réaliser une analyse sensorielle PCA sur les variables numériques.")
   

