import os
import subprocess
import time
import whisper
import threading
import json
import requests
import logging
from datetime import datetime
from pathlib import Path
import re
from typing import Optional, Callable, Dict, Any

# --- Configuration Globale ---
DURATION = 15
MODEL_NAME = "small"
LANGUAGE = "fr"
FFMPEG_TIMEOUT = 20
API_RETRY_DELAY = 5
API_MAX_RETRIES = 3

FILE_A = "live_segment_A.mp3"
FILE_B = "live_segment_B.mp3"

# --- Configuration Logging ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def _setup_logger(name: str) -> logging.Logger:
    """Configure le logger pour le module."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        log_file = LOG_DIR / f"tiktok_transcriber_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

logger = _setup_logger(__name__)

# --- Classe pour les Métriques ---
class TranscriptionMetrics:
    """Suit les statistiques de transcription en thread-safe."""
    
    def __init__(self):
        self.total_segments = 0
        self.successful_transcriptions = 0
        self.failed_transcriptions = 0
        self.silent_segments = 0
        self.start_time = None
        self.lock = threading.Lock()
    
    def record_segment(self, success: bool = True, silent: bool = False) -> None:
        """Enregistre un segment traité."""
        with self.lock:
            self.total_segments += 1
            if silent:
                self.silent_segments += 1
            elif success:
                self.successful_transcriptions += 1
            else:
                self.failed_transcriptions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques actuelles."""
        with self.lock:
            elapsed = time.time() - self.start_time if self.start_time else 0
            total = max(self.total_segments, 1)
            success_rate = (self.successful_transcriptions / total * 100)
            
            return {
                "total_segments": self.total_segments,
                "successful": self.successful_transcriptions,
                "failed": self.failed_transcriptions,
                "silent": self.silent_segments,
                "elapsed_time": round(elapsed, 2),
                "success_rate": f"{success_rate:.1f}%"
            }
    
    def reset(self) -> None:
        """Réinitialise les métriques."""
        with self.lock:
            self.total_segments = 0
            self.successful_transcriptions = 0
            self.failed_transcriptions = 0
            self.silent_segments = 0
            self.start_time = None

# --- Classe Principale ---
class TikTokLiveTranscriber:
    """Transcripteur de live TikTok avec support asynchrone et callbacks."""
    
    def __init__(
        self,
        room_id: str,
        model: str = MODEL_NAME,
        language: str = LANGUAGE,
        duration: int = DURATION,
        ffmpeg_timeout: int = FFMPEG_TIMEOUT,
        on_transcription: Optional[Callable[[str, int], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[Dict], None]] = None
    ):
        """
        Initialise le transcripteur.
        
        Args:
            room_id: ID de la room TikTok
            model: Modèle Whisper à utiliser (tiny, base, small, medium, large)
            language: Code de langue (ex: 'fr', 'en')
            duration: Durée de chaque segment en secondes
            ffmpeg_timeout: Timeout FFmpeg en secondes
            on_transcription: Callback(transcription, segment_number) appelé à chaque transcription
            on_error: Callback(error_message) appelé en cas d'erreur
            on_complete: Callback(stats_dict) appelé à la fin
        """
        self.room_id = room_id
        self.model_name = model
        self.language = language
        self.duration = duration
        self.ffmpeg_timeout = ffmpeg_timeout
        
        self.on_transcription = on_transcription
        self.on_error = on_error
        self.on_complete = on_complete
        
        self.model = None
        self.metrics = TranscriptionMetrics()
        self.is_running = False
        self._stop_event = threading.Event()
        self._current_thread = None
        self._transcription_thread = None
    
    def _validate_room_id(self, room_id: str) -> bool:
        """Valide le format du Room ID TikTok."""
        if not room_id or not isinstance(room_id, str):
            logger.error("Room ID invalide : doit être une chaîne non vide.")
            return False
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', room_id):
            logger.error(f"Room ID contient des caractères invalides: {room_id}")
            return False
        
        return True
    
    def _get_live_url(self, retry_count: int = 0) -> Optional[str]:
        """Récupère l'URL du flux audio avec gestion des erreurs et retry."""
        if self._stop_event.is_set():
            return None
        
        logger.info(f"Tentative #{retry_count + 1} - Room ID: {self.room_id}")
        api_url = f"https://webcast.tiktok.com/webcast/room/info/?aid=1988&room_id={self.room_id}"
        
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            json_response = response.json()
            
            if json_response.get('status_code') == 0:
                try:
                    pull_data_block = json_response['data']['stream_url']['live_core_sdk_data']['pull_data']
                    stream_data_json_string = pull_data_block['stream_data']
                    stream_data = json.loads(stream_data_json_string)
                    stream_url = stream_data['data']['ao']['main']['flv']
                    
                    logger.info("URL du live récupérée.")
                    return stream_url
                    
                except (KeyError, IndexError) as e:
                    logger.error(f"Structure JSON inattendue: {e}")
                    if self.on_error:
                        self.on_error(f"Erreur de parsing JSON: {e}")
                    return None
            else:
                status_code = json_response.get('status_code')
                reason = json_response.get('data', {}).get('reason', 'Statut non disponible.')
                
                if status_code == 4001:
                    logger.warning(f"Le live est terminé.")
                    return None
                
                if status_code in [5000, 4003] and retry_count < API_MAX_RETRIES:
                    logger.warning(f"Erreur temporaire. Nouvelle tentative dans {API_RETRY_DELAY}s...")
                    time.sleep(API_RETRY_DELAY)
                    return self._get_live_url(retry_count + 1)
                
                logger.error(f"Erreur API TikTok (Code: {status_code}). Raison: {reason}")
                if self.on_error:
                    self.on_error(f"Erreur API TikTok: {reason}")
                return None
        
        except requests.exceptions.Timeout:
            logger.error("Timeout de connexion.")
            if self.on_error:
                self.on_error("Timeout de connexion à l'API TikTok")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur réseau: {e}")
            if self.on_error:
                self.on_error(f"Erreur réseau: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Erreur décodage JSON: {e}")
            if self.on_error:
                self.on_error(f"Erreur JSON: {e}")
            return None
    
    def _transcribe_segment(self, filename: str, segment_number: int) -> None:
        """Transcription d'un segment audio."""
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            logger.warning(f"(T{segment_number}) Fichier vide ou inexistant.")
            self.metrics.record_segment(success=True, silent=True)
            if os.path.exists(filename):
                os.remove(filename)
            return
        
        logger.info(f"(T{segment_number}) Transcription en cours...")
        
        try:
            result = self.model.transcribe(filename, language=self.language)
            transcription = result["text"].strip()
            
            if transcription:
                logger.info(f"(T{segment_number}) {transcription}")
                self.metrics.record_segment(success=True)
                
                if self.on_transcription:
                    try:
                        self.on_transcription(transcription, segment_number)
                    except Exception as e:
                        logger.error(f"Erreur dans le callback on_transcription: {e}")
                        if self.on_error:
                            self.on_error(f"Erreur callback: {e}")
            else:
                logger.debug(f"(T{segment_number}) Silence détecté.")
                self.metrics.record_segment(success=True, silent=True)
        
        except Exception as e:
            logger.error(f"(T{segment_number}) Erreur de transcription: {e}")
            self.metrics.record_segment(success=False)
            if self.on_error:
                self.on_error(f"Erreur transcription segment {segment_number}: {e}")
        
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def _record_segment(self, stream_url: str, output_filename: str, segment_number: int) -> bool:
        """Enregistre un segment audio avec timeout."""
        if self._stop_event.is_set():
            return False
        
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-t", str(self.duration),
            "-i", stream_url,
            "-loglevel", "error",
            output_filename
        ]
        
        logger.info(f"Segment #{segment_number}: Enregistrement ({self.duration}s)...")
        
        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.ffmpeg_timeout)
                
                if process.returncode != 0:
                    logger.error(f"FFmpeg erreur (code {process.returncode})")
                    if self.on_error:
                        self.on_error(f"FFmpeg error: {stderr.decode() if stderr else 'Unknown'}")
                    return False
                
                return True
            
            except subprocess.TimeoutExpired:
                process.kill()
                logger.error(f"Timeout FFmpeg après {self.ffmpeg_timeout}s.")
                if self.on_error:
                    self.on_error(f"FFmpeg timeout après {self.ffmpeg_timeout}s")
                return False
        
        except FileNotFoundError:
            logger.error("FFmpeg non trouvé.")
            if self.on_error:
                self.on_error("FFmpeg n'est pas installé")
            return False
        except Exception as e:
            logger.error(f"Erreur FFmpeg: {e}")
            if self.on_error:
                self.on_error(f"Erreur FFmpeg: {e}")
            return False
    
    def _refresh_stream_url(self, current_url: str) -> Optional[str]:
        """Tente de récupérer une nouvelle URL si l'ancienne expire."""
        logger.warning("Tentative de rafraîchissement de l'URL...")
        new_url = self._get_live_url()
        return new_url if new_url and new_url != current_url else None
    
    def _transcription_loop(self, stream_url: str) -> None:
        """Boucle principale de transcription."""
        segment_count = 0
        current_thread = None
        url_refresh_attempts = 0
        
        try:
            while self.is_running and not self._stop_event.is_set():
                segment_count += 1
                OUTPUT_FILENAME = FILE_A if segment_count % 2 != 0 else FILE_B
                
                # Vérifier que le thread précédent est terminé
                if current_thread is not None and current_thread.is_alive():
                    logger.debug(f"Attente du thread de transcription précédent...")
                    current_thread.join(timeout=30)
                
                # Enregistrement du segment
                if not self._record_segment(stream_url, OUTPUT_FILENAME, segment_count):
                    new_url = self._refresh_stream_url(stream_url)
                    
                    if new_url:
                        logger.info("URL rafraîchie.")
                        stream_url = new_url
                        url_refresh_attempts = 0
                        continue
                    else:
                        url_refresh_attempts += 1
                        
                        if url_refresh_attempts >= 3:
                            logger.error("Impossible de récupérer une URL valide.")
                            break
                        else:
                            logger.warning(f"Nouvelle tentative ({url_refresh_attempts}/3)...")
                            time.sleep(API_RETRY_DELAY)
                            continue
                
                url_refresh_attempts = 0
                
                # Lancer la transcription en arrière-plan
                current_thread = threading.Thread(
                    target=self._transcribe_segment,
                    args=(OUTPUT_FILENAME, segment_count),
                    daemon=False
                )
                current_thread.start()
        
        except Exception as e:
            logger.error(f"Erreur dans la boucle: {e}")
            if self.on_error:
                self.on_error(f"Erreur boucle: {e}")
        
        finally:
            self._cleanup(current_thread)
    
    def _cleanup(self, current_thread: Optional[threading.Thread]) -> None:
        """Nettoie les ressources."""
        logger.info("Nettoyage en cours...")
        
        if current_thread and current_thread.is_alive():
            logger.info("Attente de la dernière transcription...")
            current_thread.join(timeout=15)
        
        for f in [FILE_A, FILE_B]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning(f"Impossible de supprimer {f}: {e}")
        
        stats = self.metrics.get_stats()
        logger.info("="*60)
        logger.info("STATISTIQUES FINALES")
        logger.info("="*60)
        logger.info(f"Segments traités: {stats['total_segments']}")
        logger.info(f"Transcriptions réussies: {stats['successful']}")
        logger.info(f"Transcriptions échouées: {stats['failed']}")
        logger.info(f"Segments silencieux: {stats['silent']}")
        logger.info(f"Taux de succès: {stats['success_rate']}")
        logger.info(f"Durée totale: {stats['elapsed_time']}s")
        logger.info("="*60)
        
        if self.on_complete:
            try:
                self.on_complete(stats)
            except Exception as e:
                logger.error(f"Erreur dans le callback on_complete: {e}")
        
        self.is_running = False
    
    def start(self) -> bool:
        """Démarre la transcription. Retourne True si succès, False sinon."""
        if self.is_running:
            logger.warning("Transcription déjà en cours.")
            return False
        
        if not self._validate_room_id(self.room_id):
            return False
        
        try:
            logger.info(f"Chargement du modèle Whisper '{self.model_name}'...")
            self.model = whisper.load_model(self.model_name)
            logger.info("Modèle chargé.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de Whisper: {e}")
            if self.on_error:
                self.on_error(f"Erreur Whisper: {e}")
            return False
        
        stream_url = self._get_live_url()
        if not stream_url:
            return False
        
        self.is_running = True
        self._stop_event.clear()
        self.metrics.reset()
        self.metrics.start_time = time.time()
        
        logger.info(f"Début de la transcription (Room ID: {self.room_id})")
        logger.info(f"Logs: {LOG_DIR}")
        
        # Lancer la boucle dans un thread séparé
        self._transcription_thread = threading.Thread(
            target=self._transcription_loop,
            args=(stream_url,),
            daemon=False,
            name="TikTokTranscriptionLoop"
        )
        self._transcription_thread.start()
        
        return True
    
    def stop(self) -> None:
        """Arrête la transcription."""
        logger.info("Arrêt demandé...")
        self.is_running = False
        self._stop_event.set()
        
        if self._transcription_thread and self._transcription_thread.is_alive():
            logger.info("Attente de l'arrêt du thread principal...")
            self._transcription_thread.join(timeout=30)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques actuelles."""
        return self.metrics.get_stats()
    
    def wait_until_complete(self, timeout: Optional[float] = None) -> bool:
        """
        Attend que la transcription soit complète.
        
        Args:
            timeout: Timeout en secondes (None = pas de timeout)
            
        Returns:
            True si complète, False si timeout
        """
        if self._transcription_thread is None:
            return True
        
        self._transcription_thread.join(timeout=timeout)
        return not self._transcription_thread.is_alive()


# --- Fonctions utilitaires pour compatibilité rétroactive ---
def live_transcriber(
    room_id: str,
    model: str = MODEL_NAME,
    language: str = LANGUAGE,
    on_transcription: Optional[Callable[[str, int], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[Dict], None]] = None
) -> TikTokLiveTranscriber:
    """
    Interface simple pour créer et démarrer un transcripteur.
    
    Args:
        room_id: ID de la room TikTok
        model: Modèle Whisper (tiny, base, small, medium, large)
        language: Code de langue
        on_transcription: Callback pour chaque transcription
        on_error: Callback pour les erreurs
        on_complete: Callback à la fin
        
    Returns:
        Instance de TikTokLiveTranscriber
        
    Raises:
        RuntimeError: Si impossible de démarrer
    """
    transcriber = TikTokLiveTranscriber(
        room_id=room_id,
        model=model,
        language=language,
        on_transcription=on_transcription,
        on_error=on_error,
        on_complete=on_complete
    )
    
    if transcriber.start():
        return transcriber
    else:
        raise RuntimeError(f"Impossible de démarrer la transcription pour room_id: {room_id}")


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    import sys
    
    def handle_transcription(text: str, segment: int):
        print(f"\n[{segment}] {text}\n")
    
    def handle_error(error: str):
        print(f"\nErreur: {error}\n")
    
    def handle_complete(stats: Dict):
        print(f"\nTerminé avec stats: {stats}\n")
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <room_id> [model]")
        print("Modèles: tiny, base, small, medium, large")
        sys.exit(1)
    
    room_id = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL_NAME
    
    try:
        transcriber = live_transcriber(
            room_id=room_id,
            model=model,
            on_transcription=handle_transcription,
            on_error=handle_error,
            on_complete=handle_complete
        )
        
        # Attendre indéfiniment ou jusqu'à Ctrl+C
        transcriber.wait_until_complete()
        
    except RuntimeError as e:
        print(f"{e}")
    except KeyboardInterrupt:
        print("\nArrêt...")
        transcriber.stop()
        transcriber.wait_until_complete(timeout=10)