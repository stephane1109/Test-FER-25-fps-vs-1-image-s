# pip install streamlit
# pip install opencv-python-headless fer pandas matplotlib altair xlsxwriter scikit-learn numpy tensorflow
# pip install tensorflow
# pip install tensorflow-metal -> pour Mac M2
# pip install yt_dlp

import streamlit as st
import subprocess
import os
import pandas as pd
import numpy as np
from collections import Counter
from fer import FER
import cv2
from yt_dlp import YoutubeDL


# Fonction pour définir le répertoire de travail
import streamlit as st
import os

# Fonction pour définir le répertoire de travail
def definir_repertoire_travail():
    # Récupérer le chemin spécifié par l'utilisateur
    repertoire = st.text_input("Définir le répertoire de travail", "", key="repertoire_travail")

    # Vérifier si l'utilisateur a bien spécifié un chemin
    if not repertoire:
        st.write("Veuillez spécifier un chemin valide.")
        return ""

    # Nettoyer le chemin spécifié (éliminer les espaces superflus)
    repertoire = repertoire.strip()

    # Si le chemin n'est pas absolu, le transformer en chemin absolu
    repertoire = os.path.abspath(repertoire)

    # Vérifier si le répertoire existe déjà
    if not os.path.exists(repertoire):
        # Si non, créer le répertoire
        os.makedirs(repertoire)
        st.write(f"Le répertoire a été créé : {repertoire}")
    else:
        st.write(f"Le répertoire existe déjà : {repertoire}")

    # Retourner le chemin absolu
    return repertoire



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
        return resultats[0]['emotions']
    else:
        st.write(f"Aucune émotion détectée dans l'image {image_path}")
        return {}

# Calcul de l'émotion dominante par moyenne des scores
def emotion_dominante_par_moyenne(emotions_list):
    if emotions_list:
        moyenne_emotions = {emotion: np.mean([emo[emotion] for emo in emotions_list])
                            for emotion in emotions_list[0].keys()}
        emotion_dominante = max(moyenne_emotions, key=moyenne_emotions.get)
        return moyenne_emotions, emotion_dominante
    return {}, "Aucune émotion"

# Calcul de l'émotion dominante par somme des scores
def emotion_dominante_par_somme(emotions_list):
    if emotions_list:
        somme_emotions = {emotion: np.sum([emo[emotion] for emo in emotions_list])
                          for emotion in emotions_list[0].keys()}
        emotion_dominante = max(somme_emotions, key=somme_emotions.get)
        return somme_emotions, emotion_dominante
    return {}, "Aucune émotion"

# Calcul de l'émotion dominante par mode (émotion la plus fréquente)
def emotion_dominante_par_mode(emotions_list):
    if emotions_list:
        emotions_dominantes = [max(emotion, key=emotion.get) for emotion in emotions_list]
        emotion_dominante = Counter(emotions_dominantes).most_common(1)[0][0]
        return Counter(emotions_dominantes), emotion_dominante
    return {}, "Aucune émotion"

# Fonction principale pour gérer le processus
def analyser_video(video_url, start_time, end_time, repertoire_travail):
    st.write(f"Analyse de la vidéo entre {start_time} et {end_time} seconde(s)")

    repertoire_1fps = os.path.join(repertoire_travail, "images_annotées_1s")
    repertoire_25fps = os.path.join(repertoire_travail, "images_annotées_25fps")
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

        # Première dataframe : Analyse des émotions 1fps et 25fps image par image
        if emotions_1fps:
            results_1fps_25fps.append({'Seconde': seconde, 'Frame': '1fps', **emotions_1fps})

        if emotions_25fps_list:
            for idx, emotions in enumerate(emotions_25fps_list):
                results_1fps_25fps.append({'Seconde': seconde, 'Frame': f'25fps_{idx}', **emotions})

        # Deuxième dataframe : Résultat de l'émotion dominante pour 1 fps
        if emotions_1fps:
            emotion_dominante_1fps = max(emotions_1fps, key=emotions_1fps.get)
            emotion_dominante_1fps_results.append({
                'Seconde': seconde,
                'Emotion_dominante_1fps': emotion_dominante_1fps,
                **emotions_1fps
            })

        # Troisième dataframe : Résultat de l'émotion dominante par somme
        somme_emotions, emotion_dominante_somme = emotion_dominante_par_somme(emotions_25fps_list)
        emotion_dominante_somme_results.append({
            'Seconde': seconde,
            'Emotion_dominante_25fps_somme': emotion_dominante_somme,
            **somme_emotions
        })

        # Quatrième dataframe : Résultat de l'émotion dominante par moyenne
        moyenne_emotions, emotion_dominante_moyenne = emotion_dominante_par_moyenne(emotions_25fps_list)
        emotion_dominante_moyenne_results.append({
            'Seconde': seconde,
            'Emotion_dominante_25fps_moyenne': emotion_dominante_moyenne,
            **moyenne_emotions
        })

        # Cinquième dataframe : Résultat de l'émotion dominante par mode
        mode_emotions, emotion_dominante_mode = emotion_dominante_par_mode(emotions_25fps_list)
        emotion_dominante_mode_results.append({
            'Seconde': seconde,
            'Emotion_dominante_25fps_mode': emotion_dominante_mode,
            **mode_emotions
        })

    # Première dataframe : Analyse des émotions image par image
    df_emotions = pd.DataFrame(results_1fps_25fps)
    st.write("**Analyse des émotions image par image (1fps et 25fps)**")
    st.dataframe(df_emotions)

    # Deuxième dataframe : Résultats de l'émotion dominante pour 1 fps
    df_emotion_dominante_1fps = pd.DataFrame(emotion_dominante_1fps_results)
    st.write("**Résultat de l'émotion dominante pour 1fps**")
    st.dataframe(df_emotion_dominante_1fps)

    # Troisième dataframe : Résultat de l'émotion dominante par somme
    df_emotion_dominante_somme = pd.DataFrame(emotion_dominante_somme_results)
    st.write("**Résultat de l'émotion dominante par somme (25fps)**")
    st.dataframe(df_emotion_dominante_somme)

    # Quatrième dataframe : Résultat de l'émotion dominante par moyenne
    df_emotion_dominante_moyenne = pd.DataFrame(emotion_dominante_moyenne_results)
    st.write("**Résultat de l'émotion dominante par moyenne (25fps)**")
    st.dataframe(df_emotion_dominante_moyenne)

    # Explication du calcul de la moyenne
    st.markdown("""
    **Explication du calcul de la moyenne :**
    - Pour chaque émotion, les scores sont additionnés sur toutes les 25 images extraites dans une seconde.
    - On calcule ensuite la moyenne de chaque émotion. Par exemple, si les scores pour l'émotion 'angry' sont : 
    0.3, 0.4, 0.5 sur 3 images, la moyenne sera (0.3 + 0.4 + 0.5) / 3.
    """)

    # Cinquième dataframe : Résultat de l'émotion dominante par mode
    df_emotion_dominante_mode = pd.DataFrame(emotion_dominante_mode_results)
    st.write("**Résultat de l'émotion dominante par mode (25fps)**")
    st.dataframe(df_emotion_dominante_mode)

    # Explication du calcul du mode
    st.markdown("""
    **Explication du calcul du mode :**
    - Pour chaque image, l'émotion avec le score le plus élevé est déterminée.
    - Ensuite, l'émotion qui apparaît le plus souvent parmi ces 25 images est définie comme l'émotion dominante par mode.
    """)

    # Enregistrement de toutes les dataframes dans un seul fichier CSV
    with pd.ExcelWriter(os.path.join(repertoire_travail, "resultats_emotions_complet.xlsx")) as writer:
        df_emotions.to_excel(writer, sheet_name="Emotions Image par Image", index=False)
        df_emotion_dominante_1fps.to_excel(writer, sheet_name="Emotion Dominante 1fps", index=False)
        df_emotion_dominante_somme.to_excel(writer, sheet_name="Emotion Dominante Somme", index=False)
        df_emotion_dominante_moyenne.to_excel(writer, sheet_name="Emotion Dominante Moyenne", index=False)
        df_emotion_dominante_mode.to_excel(writer, sheet_name="Emotion Dominante Mode", index=False)

    st.write("Analyse terminée, résultats exportés dans un fichier Excel.")

# Interface Streamlit
st.title("Analyse des émotions : 1 fps vs 25 fps vs somme - moyenne - mode")
st.markdown("<h6 style='text-align: center;'>www.codeandcortex.fr</h5>", unsafe_allow_html=True)

# Utilisation dans Streamlit
st.subheader("Définir le répertoire de travail")
repertoire_travail = definir_repertoire_travail()

video_url = st.text_input("URL de la vidéo à analyser", "", key="video_url")

start_time = st.number_input("Temps de départ de l'analyse (en secondes)", min_value=0, value=0, key="start_time")
end_time = st.number_input("Temps d'arrivée de l'analyse (en secondes)", min_value=start_time, value=start_time + 1, key="end_time")

if st.button("Lancer l'analyse"):
    if video_url and repertoire_travail:
        analyser_video(video_url, start_time, end_time, repertoire_travail)
    else:
        st.write("Veuillez définir le répertoire de travail et l'URL de la vidéo.")
