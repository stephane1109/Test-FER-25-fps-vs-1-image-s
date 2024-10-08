# pip install streamlit
# pip install opencv-python-headless fer pandas matplotlib altair xlsxwriter scikit-learn numpy tensorflow
# pip install tensorflow
# pip install tensorflow-metal -> pour Mac M2
# pip install yt_dlp

import streamlit as st
import subprocess
import os
import pandas as pd
from fer import FER
import cv2
from yt_dlp import YoutubeDL


# Fonction pour créer le répertoire de travail
def definir_repertoire_travail():
    repertoire = st.text_input("Définir le répertoire de travail", "", key="repertoire_travail")
    if repertoire and not os.path.exists(repertoire):
        os.makedirs(repertoire)
    return repertoire


# Fonction pour télécharger la vidéo avec yt-dlp
def telecharger_video(url, repertoire):
    video_path = os.path.join(repertoire, 'video.mp4')  # Nom de fichier fixe pour éviter les doublons
    if os.path.exists(video_path):
        st.write(f"La vidéo est déjà présente dans le répertoire : {video_path}")
        return video_path

    st.write(f"Téléchargement de la vidéo à partir de {url}...")
    ydl_opts = {
        'outtmpl': video_path,
        'format': 'best'
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    st.write(f"Téléchargement terminé : {video_path}")
    return video_path


# Fonction pour extraire une image par seconde en utilisant FFmpeg
def extraire_image_ffmpeg(video_path, repertoire, seconde):
    image_path = os.path.join(repertoire, f"image_1s_{seconde}.jpg")

    # Si l'image existe déjà, on la réutilise
    if os.path.exists(image_path):
        st.write(f"L'image existe déjà : {image_path}")
        return image_path

    st.write(f"Extraction d'une image à la seconde {seconde}...")
    cmd = [
        'ffmpeg', '-ss', str(seconde), '-i', video_path, '-frames:v', '1',
        '-q:v', '2', image_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        st.write(f"Erreur FFmpeg : {result.stderr.decode('utf-8')}")
        return None
    return image_path


# Fonction pour extraire 25 images dans une seconde en utilisant FFmpeg
def extraire_images_25fps_ffmpeg(video_path, repertoire, seconde):
    images_extraites = []
    for frame in range(25):  # Extraire 25 images dans la seconde
        image_path = os.path.join(repertoire, f"image_25fps_{seconde}_{frame}.jpg")

        # Si l'image existe déjà, on la réutilise
        if os.path.exists(image_path):
            images_extraites.append(image_path)
            continue

        time = seconde + frame * (1 / 25)  # Calculer le temps pour chaque frame
        cmd = [
            'ffmpeg', '-ss', str(time), '-i', video_path, '-frames:v', '1',
            '-q:v', '2', image_path
        ]
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
        image_annotée = annoter_image(image, resultats)
        cv2.imwrite(image_path, image_annotée)  # Sauvegarder l'image annotée
        return resultats[0]['emotions']  # Retourner les émotions détectées
    else:
        st.write(f"Aucune émotion détectée dans l'image {image_path}")
        return {}


# Fonction pour annoter une image avec des émotions et un cadre vert autour du visage
def annoter_image(image, resultats_emotions):
    for result in resultats_emotions:
        (x, y, w, h) = result["box"]  # Coordonnées du cadre autour du visage
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dessiner un cadre vert

        # Annotation des émotions sur l'image
        for idx, (emotion, score) in enumerate(result["emotions"].items()):
            text = f"{emotion}: {score:.2f}"
            cv2.putText(image, text, (x, y - 10 - (idx * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image


# Fonction pour calculer l'émotion dominante automatiquement
def emotion_dominante_auto(emotions):
    if emotions:
        return max(emotions, key=emotions.get)
    return "Aucune émotion"


# Fonction principale pour gérer le processus
def analyser_video(video_url, start_time, end_time, repertoire_travail):
    st.write(f"Analyse de la vidéo entre {start_time} et {end_time} seconde(s)")

    # Créer les répertoires pour les images annotées
    repertoire_1fps = os.path.join(repertoire_travail, "images_annotées_1s")
    repertoire_25fps = os.path.join(repertoire_travail, "images_annotées_25fps")
    os.makedirs(repertoire_1fps, exist_ok=True)
    os.makedirs(repertoire_25fps, exist_ok=True)

    # Téléchargement de la vidéo avec yt-dlp
    video_path = telecharger_video(video_url, repertoire_travail)

    # Initialiser le détecteur FER
    detector = FER(mtcnn=True)

    # Liste pour stocker les résultats
    results = []

    # Parcours du temps de start_time à end_time
    for seconde in range(start_time, end_time + 1):
        # Extraction et analyse pour une image par seconde
        st.write(f"Extraction et annotation de l'image à la seconde {seconde}...")
        image_1s_path = extraire_image_ffmpeg(video_path, repertoire_1fps, seconde)
        emotions_1fps = analyser_et_annoter_image(image_1s_path, detector)

        # Extraction et analyse pour 25 images par seconde
        st.write(f"Extraction et annotation des images à 25fps pour la seconde {seconde}...")
        images_25fps = extraire_images_25fps_ffmpeg(video_path, repertoire_25fps, seconde)
        emotions_25fps_list = [analyser_et_annoter_image(image_path, detector) for image_path in images_25fps]

        # Calcul de l'émotion dominante pour les 25 images
        st.write("Calcul de l'émotion dominante pour les 25 images...")
        if emotions_25fps_list:
            emotions_sommees = pd.DataFrame(emotions_25fps_list).sum(axis=0)
            emotion_dominante_25fps = emotions_sommees.idxmax()
        else:
            emotions_sommees = pd.Series()
            emotion_dominante_25fps = "Aucune émotion détectée"

        # Calculer l'émotion dominante automatique pour 1fps et 25fps
        emotion_dominante_1fps_auto = emotion_dominante_auto(emotions_1fps)
        emotion_dominante_25fps_auto = emotion_dominante_auto(emotions_sommees.to_dict())

        # Ajout des résultats à la liste
        result = {
            'Seconde': seconde,
            'Emotion_dominante_1fps_auto': emotion_dominante_1fps_auto,
            'Emotion_dominante_25fps (Calculée)': emotion_dominante_25fps,
            'Emotion_dominante_25fps_auto': emotion_dominante_25fps_auto,
        }

        # Ajout des émotions séparément pour 1fps et 25fps
        if emotions_1fps:
            for emotion, score in emotions_1fps.items():
                result[f'1fps_{emotion}'] = f"{score:.2f}"  # Affiche deux chiffres après la virgule
        else:
            for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                result[f'1fps_{emotion}'] = "0.00"

        if not emotions_sommees.empty:
            for emotion, score in emotions_sommees.items():
                result[f'25fps_{emotion}'] = f"{score:.2f}"  # Affiche deux chiffres après la virgule
        else:
            for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                result[f'25fps_{emotion}'] = "0.00"

        results.append(result)

    # Création du DataFrame pour stocker les résultats
    df_resultats = pd.DataFrame(results)

    # Affichage des résultats
    st.write("Résultats de l'analyse des émotions :")
    st.dataframe(df_resultats)

    # Export des résultats au format CSV
    csv_path = os.path.join(repertoire_travail, "resultats_emotions.csv")
    df_resultats.to_csv(csv_path, index=False)

    st.write(f"Résultats exportés dans le fichier : {csv_path}")


# Interface Streamlit
st.title("Test FER (Facial Emotion Recognition) : 1 fps vs 25 fps vs emotion auto")
st.markdown("<h6 style='text-align: center;'>www.codeandcortex.fr</h5>", unsafe_allow_html=True)

# Définition du répertoire de travail
repertoire_travail = definir_repertoire_travail()

# URL de la vidéo
video_url = st.text_input("URL de la vidéo à analyser", "", key="video_url")

# Sélection du temps de départ et du temps d'arrivée
start_time = st.number_input("Temps de départ de l'analyse (en secondes)", min_value=0, value=0, key="start_time")
end_time = st.number_input("Temps d'arrivée de l'analyse (en secondes)", min_value=start_time, value=start_time + 1,
                           key="end_time")

# Bouton pour lancer l'analyse
if st.button("Lancer l'analyse"):
    if video_url and repertoire_travail:
        analyser_video(video_url, start_time, end_time, repertoire_travail)
    else:
        st.write("Veuillez définir le répertoire de travail et l'URL de la vidéo.")

