import os
import subprocess
import time
import whisper
import threading
import json
import requests
import sys

# --- Configuration Globale ---
DURATION = 15          # Durée de chaque segment audio (en secondes)
MODEL_NAME = "small"   # Modèle Whisper pour la précision
LANGUAGE = "fr"        # Langue cible

# Fichiers tampons pour la lecture et l'enregistrement asynchrone (Double Buffering)
FILE_A = "live_segment_A.mp3"
FILE_B = "live_segment_B.mp3"


# --- Fonction 1: Récéption de l'URL du Live (Intégration de la requête API) ---
def get_live_url(room_id):
    """
    Récupère l'URL de flux audio 'Audio Only' (ao) à partir du Room ID de TikTok 
    en utilisant une requête API.
    """
    print(f"📡 Tentative de récupération de l'URL pour le Room ID : {room_id}")

    # L'API publique de TikTok
    api_url = f"https://webcast.tiktok.com/webcast/room/info/?aid=1988&room_id={room_id}"
    
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status() 
        json_response = response.json()
        
        if json_response.get('status_code') == 0:
            
            # 1. Accès au bloc contenant les URLs des streams
            pull_data_block = json_response['data']['stream_url']['live_core_sdk_data']['pull_data']
            
            # 2. Désérialiser la chaîne JSON contenue dans 'stream_data'
            stream_data_json_string = pull_data_block['stream_data']
            stream_data = json.loads(stream_data_json_string) # <-- La correction clé!
            
            # 3. Naviguer dans l'objet désérialisé pour obtenir l'URL FLV Audio Only
            # stream_data['data']['ao']['main'] est maintenant un dictionnaire!
            stream_url = stream_data['data']['ao']['main']['flv']
            
            print(f"✅ URL du live récupérée.")
            return stream_url
        else:
            reason = json_response.get('data', {}).get('reason', 'Statut non disponible ou terminé.')
            print(f"❌ Erreur API TikTok (Status: {json_response.get('status_code')}). Raison: {reason}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de connexion lors de l'appel API: {e}")
        return None
    except KeyError:
        print(f"❌ Erreur: Clé non trouvée. La structure JSON a peut-être changé.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Erreur de décodage JSON dans 'stream_data': {e}")
        return None
    
# --- Fonction 2: Transcription Asynchrone (Thread) ---
def transcribe_segment(filename, segment_number, model):
    """Effectue la transcription d'un fichier audio donné dans un thread séparé."""
    
    # Vérifie si le fichier existe et a une taille non nulle
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print(f"⚠️ (T{segment_number}) Fichier vide ou inexistant. Saut de la transcription.")
        # Nettoyage si le fichier est vide (peut arriver si le stream se coupe exactement au début)
        if os.path.exists(filename): os.remove(filename) 
        return

    print(f"🧠 (T{segment_number}) Transcription en cours...")
    
    try:
        # Transcrit le segment en spécifiant la langue
        result = model.transcribe(filename, language=LANGUAGE)
        
        # Affichage du résultat
        transcription = result["text"].strip()
        
        if transcription:
             print(f"📝 (T{segment_number}) Transcription: **{transcription}**")
        else:
             print(f"🔇 (T{segment_number}) Silence ou pas de parole détectée.")
    
    except Exception as e:
        print(f"❌ (T{segment_number}) Erreur de transcription: {e}")
        
    # Nettoyage
    if os.path.exists(filename):
        os.remove(filename)
        
# --- Fonction Principale ---
def live_transcriber(room_id):
    """Boucle principale pour la capture et la transcription continue."""
    
    # 0. Initialisation
    try:
        model = whisper.load_model(MODEL_NAME)
        print(f"✅ Modèle Whisper '{MODEL_NAME}' chargé.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement de Whisper. Veuillez vérifier l'installation: {e}")
        return

    # 1. Récupération de l'URL du Live
    stream_url = get_live_url(room_id)
    if not stream_url:
        return
        
    print(f"🎉 Début de la transcription continue pour le Room ID {room_id}...")

    # 2. Boucle de Capture et Transcription Asynchrone
    segment_count = 0
    current_thread = None 

    try:
        while True:
            segment_count += 1
            
            # Alternance des fichiers pour le double tampon
            OUTPUT_FILENAME = FILE_A if segment_count % 2 != 0 else FILE_B
            
            # Vérification de sécurité : S'assurer que le thread précédent est terminé
            if current_thread is not None and current_thread.is_alive():
                print(f"⏳ (T{segment_count-1}) Transcription toujours en cours. Attente forcée pour éviter la perte d'audio...")
                current_thread.join() # Bloque jusqu'à la fin de la transcription précédente
            
            # --- Enregistrement (Bloque pendant DURATION secondes) ---
            start_time = time.time()
            
            # Commande FFmpeg : extrait exactement DURATION secondes
            ffmpeg_cmd = [
                "ffmpeg", 
                "-y",                   
                "-t", str(DURATION),    
                "-i", stream_url,       
                OUTPUT_FILENAME
            ]

            print(f"\n--- Segment #{segment_count} ({DURATION}s) : Enregistrement en cours... ---")
            
            try:
                # Execution de l'enregistrement (bloque le thread principal 15s)
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                record_time = time.time() - start_time
                print(f"✅ (T{segment_count}) Enregistrement terminé en {round(record_time, 2)}s.")
            except subprocess.CalledProcessError:
                print("\n🚨 Échec de FFmpeg. Le flux est probablement terminé ou l'URL a expiré.")
                break 

            # --- Lancement de la Transcription en Arrière-plan (ASYNCHRONE) ---
            current_thread = threading.Thread(target=transcribe_segment, args=(OUTPUT_FILENAME, segment_count, model))
            current_thread.start()
            
            # La boucle revient immédiatement au début pour enregistrer le segment suivant.

    except KeyboardInterrupt:
        print("\n👋 Processus arrêté par l'utilisateur. Nettoyage...")
        if current_thread and current_thread.is_alive():
            print("⏳ Attente de la fin de la dernière transcription (max 10s)...")
            current_thread.join(timeout=10) 
            
    except Exception as e:
        print(f"\n❌ Une erreur inattendue s'est produite : {e}")
        
    # --- Nettoyage Final ---
    for f in [FILE_A, FILE_B]:
        if os.path.exists(f):
            os.remove(f)

    print("\nProcessus de capture en continu terminé.")

