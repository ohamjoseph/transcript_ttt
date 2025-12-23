import ssl
import certifi
import os
from TikTokLive import TikTokLiveClient
from TikTokLive.events import ConnectEvent, CommentEvent
from transcript.transcript import TikTokLiveTranscriber



ssl._create_default_https_context = ssl._create_unverified_context

os.environ["SSL_CERT_FILE"] = certifi.where()
# print(ssl.get_default_verify_paths())
# print(certifi.where())

# Create the client
client: TikTokLiveClient = TikTokLiveClient(
    unique_id="@rahinas6",
    )

client._web.httpx_client.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

def my_callback(text, segment, unique_id):
    print(f"Segment {segment}: {text}")
    print(f"unique_id {unique_id}")
    # Envoyer vers une API, DB, etc.

# Listen to an event with a decorator!
@client.on(ConnectEvent)
async def on_connect(event: ConnectEvent):
    print(f"Connected to @{event.unique_id} (Room ID: {client.room_id}")
    
    transcriber = TikTokLiveTranscriber(
        room_id=f"{client.room_id}",
        model="small",
        unique_id=event.unique_id,
        on_transcription=my_callback,
        on_error=lambda e: print(f"Erreur: {e}"),
        on_complete=lambda s: print(f"Stats: {s}")
    )

    transcriber.start()
    


# Or, add it manually via "client.add_listener()"
async def on_comment(event: CommentEvent) -> None:
    print(f"{event.user.nickname} -> {event.comment}")


client.add_listener(CommentEvent, on_comment)

if __name__ == '__main__':
    # Run the client and block the main thread
    # await client.start() to run non-blocking
    client.run()