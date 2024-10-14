##########################################
# Projet : FER - Facial-Emotion-Recognition
# Auteur : Stéphane Meurisse
# Contact : stephane.meurisse@example.com
# Site Web : https://www.codeandcortex.fr
# LinkedIn : https://www.linkedin.com/in/st%C3%A9phane-meurisse-27339055/
# Date : 14 octobre 2024
##########################################

# pip install streamlit
# pip install opencv-python-headless fer pandas matplotlib altair xlsxwriter scikit-learn numpy
# pip install tensorflow
# pip install tensorflow-metal -> pour Mac M2
# pip install yt_dlp
# pip install vl-convert-python
# FFmpeg -> attention sous Mac la procédure d'installation sous MAC nécessite "Homebrew"


import streamlit as st
import subprocess
import os
import pandas as pd
import numpy as np
from collections import Counter
from fer import FER
import cv2
from yt_dlp import YoutubeDL
import altair as alt
from collections import defaultdict

# Fonction pour vider le cache
def vider_cache():
    st.cache_data.clear()
    st.write("Cache vidé systématiquement au lancement du script")

# Appeler la fonction de vidage du cache au début du script
vider_cache()

# Fonction pour définir le répertoire de travail
def definir_repertoire_travail():
    repertoire = st.text_input("Définir le répertoire de travail", "", key="repertoire_travail")
    if not repertoire:
        st.write("Veuillez spécifier un chemin valide.")
        return ""
    repertoire = repertoire.strip()
    repertoire = os.path.abspath(repertoire)
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)
        st.write(f"Le répertoire a été créé : {repertoire}")
    else:
        st.write(f"Le répertoire existe déjà : {repertoire}")
    return repertoire

# Suppression des repertoires
def supprimer_repertoires_images(repertoire_1fps, repertoire_25fps):
    if os.path.exists(repertoire_1fps):
        st.write("Suppression du répertoire d'images 1fps...")
        subprocess.call(['rm', '-r', repertoire_1fps])
    if os.path.exists(repertoire_25fps):
        st.write("Suppression du répertoire d'images 25fps...")
        subprocess.call(['rm', '-r', repertoire_25fps])

# Fonction pour télécharger la vidéo avec yt-dlp
def telecharger_video(url, repertoire):
    video_path = os.path.join(repertoire, 'video.mp4')
    if os.path.exists(video_path):
        st.write(f"La vidéo est déjà présente dans le répertoire : {video_path}")
        return video_path
    st.write(f"Téléchargement de la vidéo à partir de {url}...")
    ydl_opts = {'outtmpl': video_path, 'format': 'best'}
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    st.write(f"Téléchargement terminé : {video_path}")
    return video_path

# Fonction pour extraire une image par seconde en utilisant FFmpeg
def extraire_image_ffmpeg(video_path, repertoire, seconde):
    image_path = os.path.join(repertoire, f"image_1s_{seconde}.jpg")
    if os.path.exists(image_path):
        st.write(f"L'image existe déjà : {image_path}")
        return image_path
    st.write(f"Extraction d'une image à la seconde {seconde}...")
    cmd = ['ffmpeg', '-ss', str(seconde), '-i', video_path, '-frames:v', '1', '-q:v', '2', image_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        st.write(f"Erreur FFmpeg : {result.stderr.decode('utf-8')}")
        return None
    return image_path

# Fonction pour extraire 25 images dans une seconde en utilisant FFmpeg
def extraire_images_25fps_ffmpeg(video_path, repertoire, seconde):
    images_extraites = []
    for frame in range(25):
        image_path = os.path.join(repertoire, f"image_25fps_{seconde}_{frame}.jpg")
        if os.path.exists(image_path):
            images_extraites.append(image_path)
            continue
        time = seconde + frame * (1 / 25)
        cmd = ['ffmpeg', '-ss', str(time), '-i', video_path, '-frames:v', '1', '-q:v', '2', image_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            st.write(f"Erreur FFmpeg à {time} seconde : {result.stderr.decode('utf-8')}")
            break
        images_extraites.append(image_path)
    st.write(f"Nombre d'images extraites à 25fps : {len(images_extraites)}")
    return images_extraites

# Fonction d'analyse d'émotion et d'annotation d'une image

def analyser_et_annoter_image(image_path, detector):
    if image_path is None:
        st.write(f"Aucune image extraite pour le chemin : {image_path}")
        return {}
    image = cv2.imread(image_path)
    if image is None:
        st.write(f"Impossible de lire l'image : {image_path}")
        return {}
    resultats = detector.detect_emotions(image)
    if resultats:
        for result in resultats:
            (x, y, w, h) = result["box"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            emotions = result['emotions']
            for idx, (emotion, score) in enumerate(emotions.items()):
                text = f"{emotion}: {score:.4f}"
                cv2.putText(image, text, (x, y + h + 20 + (idx * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(image_path, image)
        return resultats[0]['emotions']
    else:
        st.write(f"Aucune émotion détectée dans l'image {image_path}")
        return {}

# Calcul de l'émotion dominante par moyenne des scores
def emotion_dominante_par_moyenne(emotions_list):
    if emotions_list:
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  # Liste des émotions à traiter
        moyenne_emotions = {emotion: np.mean([emo.get(emotion, 0) for emo in emotions_list])  # Utilisation de .get
                            for emotion in emotions}
        emotion_dominante = max(moyenne_emotions, key=moyenne_emotions.get)
        return moyenne_emotions, emotion_dominante
    return {}, "Aucune émotion"


# Calcul de l'émotion dominante par somme des scores
def emotion_dominante_par_somme(emotions_list):
    if emotions_list:
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  # Liste des émotions à traiter
        somme_emotions = {emotion: np.sum([emo.get(emotion, 0) for emo in emotions_list])  # Utilisation de .get
                          for emotion in emotions}
        emotion_dominante = max(somme_emotions, key=somme_emotions.get)
        return somme_emotions, emotion_dominante
    return {}, "Aucune émotion"


# Calcul de l'émotion dominante par mode (émotion la plus fréquente)
def emotion_dominante_par_mode(emotions_list):
    if emotions_list:
        # Liste des émotions à traiter
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Initialiser le compteur avec toutes les émotions pour qu'elles apparaissent même avec fréquence 0
        emotions_dominantes = Counter({emotion: 0 for emotion in emotions})

        for emotion_dict in emotions_list:
            # Vérifier que le dictionnaire d'émotions n'est pas vide
            if emotion_dict:
                # Pour chaque frame, trouver l'émotion dominante
                emotion_max = max(emotion_dict, key=emotion_dict.get)
                # Augmenter le compteur pour cette émotion
                emotions_dominantes[emotion_max] += 1

        # Retourner directement le Counter avec les fréquences des émotions
        return emotions_dominantes, emotions_dominantes.most_common(1)[0][0]

    return Counter({emotion: 0 for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}), "Aucune émotion"


### Ajout Variance
# Ajout pour calculer la moyenne et la variance des probabilités de chaque émotion sur les 25 frames
def moyenne_et_variance_par_emotion(emotions_25fps_list):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    resultats = {}

    # Calculer la moyenne et la variance pour chaque émotion
    for emotion in emotions:
        # Récupérer les scores de l'émotion sur les 25 frames
        emotion_scores = [emotion_dict.get(emotion, 0) for emotion_dict in emotions_25fps_list]

        # Calculer la moyenne et la variance
        moyenne = np.mean(emotion_scores)
        variance = np.var(emotion_scores)

        resultats[emotion] = {'moyenne': moyenne, 'variance': variance}

    return resultats
### Fin Variance

# Fonction principale pour gérer le processus
def analyser_video(video_url, start_time, end_time, repertoire_travail):
    st.write(f"Analyse de la vidéo entre {start_time} et {end_time} seconde(s)")

    repertoire_1fps = os.path.join(repertoire_travail, "images_annotées_1s")
    repertoire_25fps = os.path.join(repertoire_travail, "images_annotées_25fps")

    # Appel à la fonction de suppression des répertoires avant de recréer les images
    supprimer_repertoires_images(repertoire_1fps, repertoire_25fps)

    # Ensuite, recréer les répertoires vides
    os.makedirs(repertoire_1fps, exist_ok=True)
    os.makedirs(repertoire_25fps, exist_ok=True)

    video_path = telecharger_video(video_url, repertoire_travail)

    detector = FER(mtcnn=True)

    results_1fps_25fps = []
    emotion_dominante_1fps_results = []
    emotion_dominante_somme_results = []
    emotion_dominante_moyenne_results = []
    emotion_dominante_mode_results = []

    for seconde in range(start_time, end_time + 1):
        st.write(f"Extraction et annotation de l'image à la seconde {seconde}...")
        image_1s_path = extraire_image_ffmpeg(video_path, repertoire_1fps, seconde)
        emotions_1fps = analyser_et_annoter_image(image_1s_path, detector)

        st.write(f"Extraction et annotation des images à 25fps pour la seconde {seconde}...")
        images_25fps = extraire_images_25fps_ffmpeg(video_path, repertoire_25fps, seconde)
        emotions_25fps_list = [analyser_et_annoter_image(image_path, detector) for image_path in images_25fps]

        if emotions_1fps:
            results_1fps_25fps.append({'Seconde': seconde, 'Frame': '1fps', **emotions_1fps})

        if emotions_25fps_list:
            for idx, emotions in enumerate(emotions_25fps_list):
                results_1fps_25fps.append({'Seconde': seconde, 'Frame': f'25fps_{idx}', **emotions})

        if emotions_1fps:
            emotion_dominante_1fps = max(emotions_1fps, key=emotions_1fps.get)
            emotion_dominante_1fps_results.append({
                'Seconde': seconde,
                'Emotion_dominante_1fps': emotion_dominante_1fps,
                **emotions_1fps
            })

        somme_emotions, emotion_dominante_somme = emotion_dominante_par_somme(emotions_25fps_list)
        emotion_dominante_somme_results.append({
            'Seconde': seconde,
            'Emotion_dominante_25fps_somme': emotion_dominante_somme,
            **somme_emotions
        })

        moyenne_emotions, emotion_dominante_moyenne = emotion_dominante_par_moyenne(emotions_25fps_list)
        emotion_dominante_moyenne_results.append({
            'Seconde': seconde,
            'Emotion_dominante_25fps_moyenne': emotion_dominante_moyenne,
            **moyenne_emotions
        })

        mode_emotions, emotion_dominante_mode = emotion_dominante_par_mode(emotions_25fps_list)
        emotion_dominante_mode_results.append({
            'Seconde': seconde,
            'Emotion_dominante_25fps_mode': emotion_dominante_mode,
            **mode_emotions  # Ajoute toutes les émotions et leur fréquence
        })


    df_emotions = pd.DataFrame(results_1fps_25fps)
    st.write("#### Analyse des émotions image par image (1fps et 25fps)")
    st.dataframe(df_emotions)

    # Extraire la partie numérique de la frame pour le tri
    df_emotions['Frame_Index'] = df_emotions['Frame'].apply(
        lambda x: int(x.split('_')[1]) if '25fps' in x else None
    )

    # Création du Streamgraph avec Frame_Index sur l'axe des abscisses
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    # Utilisation des frames 25fps
    df_streamgraph = df_emotions[df_emotions['Frame'].str.contains('25fps')].melt(
        id_vars=['Frame_Index'],
        value_vars=emotions,
        var_name='Emotion',
        value_name='Score'
    )

    # Création du streamgraph
    streamgraph = alt.Chart(df_streamgraph).mark_area().encode(
        x=alt.X('Frame_Index:Q', title='Frames (0 à 24)'),
        y=alt.Y('Score:Q', title='Score des émotions', stack='center'),
        color=alt.Color('Emotion:N', title='Émotion'),
        tooltip=['Frame_Index', 'Emotion', 'Score']
    ).properties(
        title='Streamgraph des émotions par frame',
        width=800,
        height=400
    )

    st.write("#### Streamgraph des émotions")
    st.altair_chart(streamgraph, use_container_width=True)

    # Création du Line Chart avec Frame_Index sur l'axe des abscisses
    line_chart = alt.Chart(df_streamgraph).mark_line().encode(
        x=alt.X('Frame_Index:Q', title='Frames (0 à 24)'),
        y=alt.Y('Score:Q', title='Score des émotions'),
        color=alt.Color('Emotion:N', title='Émotion'),
        tooltip=['Frame_Index', 'Emotion', 'Score']
    ).properties(
        title="Évolution des scores d'émotions par frame",
        width=800,
        height=400
    )

    st.write("#### Line chart des émotions")
    st.altair_chart(line_chart, use_container_width=True)

    # Sauvegarde du graph Streamgraph
    streamgraph.save(os.path.join(repertoire_travail, 'streamgraph_emotions.png'))
    streamgraph.save(os.path.join(repertoire_travail, 'streamgraph_emotions.html'))

    # Sauvegarde du graph Line Chart
    line_chart.save(os.path.join(repertoire_travail, 'linechart_emotions.png'))
    line_chart.save(os.path.join(repertoire_travail, 'linechart_emotions.html'))

    df_emotion_dominante_1fps = pd.DataFrame(emotion_dominante_1fps_results)
    st.write("#### Résultat de l'émotion - score le plus élevé pour 1fps")
    st.dataframe(df_emotion_dominante_1fps)

    df_emotion_dominante_somme = pd.DataFrame(emotion_dominante_somme_results)
    st.write("#### Résultat de l'émotion dominante par somme (25fps)")
    st.dataframe(df_emotion_dominante_somme)

    df_emotion_dominante_moyenne = pd.DataFrame(emotion_dominante_moyenne_results)
    st.write("#### Résultat de l'émotion dominante par moyenne (25fps)")
    st.dataframe(df_emotion_dominante_moyenne)

    st.markdown("""
    #### Explication du calcul de la moyenne :
    - Pour chaque émotion, les scores sont additionnés sur toutes les 25 images extraites dans une seconde.
    - On calcule ensuite la moyenne de chaque émotion. Par exemple, si les scores pour l'émotion 'angry' sont : 
    0.3, 0.4, 0.5 sur 3 images, la moyenne sera (0.3 + 0.4 + 0.5) / 3.
    """)

    df_emotion_dominante_mode = pd.DataFrame(emotion_dominante_mode_results)
    st.write("#### Résultat de l'émotion dominante par mode (25fps)")
    st.dataframe(df_emotion_dominante_mode)

    st.markdown("""
    #### Explication du calcul du mode :
    - Pour chaque image, l'émotion avec le score le plus élevé est déterminée.
    - Ensuite, l'émotion qui apparaît le plus souvent parmi ces 25 images est définie comme l'émotion dominante par mode.
    """)

### Ajout Variance
    # Vérifier que stats_par_emotion est bien généré à ce stade
    stats_par_emotion = moyenne_et_variance_par_emotion(emotions_25fps_list)

    # Vérifier si les données sont bien présentes
    if stats_par_emotion:
        # Convertir les résultats en DataFrame pour affichage
        # On crée une DataFrame à partir des résultats de la fonction 'moyenne_et_variance_par_emotion'
        df_stats = pd.DataFrame(stats_par_emotion).T.reset_index()  # T pour transposer les colonnes en lignes
        df_stats.columns = ['Emotion', 'Moyenne', 'Variance']  # Assigner explicitement les noms des colonnes

        # Afficher la DataFrame des moyennes et variances
        st.write("#### Tableau des moyennes et variances des émotions sur les 25 frames")
        st.dataframe(df_stats)

        # Création du graphique combinant Moyenne et Variance
        st.write("#### Graphique des moyennes et variances des émotions sur les 25 frames")

        # Barres pour les moyennes
        moyenne_bar = alt.Chart(df_stats).mark_bar().encode(
            x=alt.X('Emotion:N', title='Émotion'),
            y=alt.Y('Moyenne:Q', title='Moyenne des probabilités'),
            color=alt.Color('Emotion:N', legend=None)
        )

        # Points pour les variances
        variance_point = alt.Chart(df_stats).mark_circle(size=100, color='red').encode(
            x=alt.X('Emotion:N', title='Émotion'),
            y=alt.Y('Variance:Q', title='Variance des probabilités'),
            tooltip=['Emotion', 'Variance']
        )

        # Superposer les deux graphiques
        graphique_combine = alt.layer(moyenne_bar, variance_point).resolve_scale(
            y='independent'  # Permet d'avoir des échelles indépendantes pour Moyenne et Variance
        ).properties(
            width=600,
            height=400,
        )

        # Affichage du graphique
        st.altair_chart(graphique_combine, use_container_width=True)
    else:
        st.write("Aucune donnée disponible pour les moyennes et variances.")

    # Markdown sous le graphique
    st.markdown("""
    #### Interprétation de la variance et de la moyenne :
    - **Variance** : Calculer la variance pour chaque émotion permet de voir quelle émotion fluctue le plus sur les 25 frames.
    Une émotion avec une variance élevée signifie qu'elle varie fortement d'une frame à l'autre.
    - **Moyenne** : Utiliser la variance combinée avec la moyenne donne une idée plus précise de la stabilité de chaque émotion. 
    Une émotion qui a une **haute moyenne** et une **faible variance** serait un bon indicateur d'une émotion présente de manière stable et élevée sur toute la séquence.
    """)
### Fin Variance

    with pd.ExcelWriter(os.path.join(repertoire_travail, "resultats_emotions_complet.xlsx")) as writer:
        df_emotions.to_excel(writer, sheet_name="Emotions Image par Image", index=False)
        df_emotion_dominante_1fps.to_excel(writer, sheet_name="Emotion Dominante 1fps", index=False)
        df_emotion_dominante_somme.to_excel(writer, sheet_name="Emotion Dominante Somme", index=False)
        df_emotion_dominante_moyenne.to_excel(writer, sheet_name="Emotion Dominante Moyenne", index=False)
        df_emotion_dominante_mode.to_excel(writer, sheet_name="Emotion Dominante Mode", index=False)

    st.write("Analyse terminée, résultats exportés dans un fichier Excel.")


# Interface Streamlit
st.title("Analyse des émotions : 1 fps vs 25 fps vs somme - moyenne - mode - variance")
st.markdown("<h6 style='text-align: center;'>www.codeandcortex.fr</h5>", unsafe_allow_html=True)

# Utilisation dans Streamlit
st.subheader("Définir le répertoire de travail")
repertoire_travail = definir_repertoire_travail()

video_url = st.text_input("URL de la vidéo à analyser", "", key="video_url")

start_time = st.number_input("Temps de départ de l'analyse (en secondes)", min_value=0, value=0, key="start_time")
end_time = st.number_input("Temps d'arrivée de l'analyse (en secondes)", min_value=start_time, value=start_time + 1,
                           key="end_time")


####
if st.button("Lancer l'analyse"):
    if video_url and repertoire_travail:
        # Vider le cache avant de lancer l'analyse
        st.cache_data.clear()

        # Lancer l'analyse après avoir vidé le cache
        analyser_video(video_url, start_time, end_time, repertoire_travail)
    else:
        st.write("Veuillez définir le répertoire de travail et l'URL de la vidéo.")
####
