import requests
import os

FILE_PATH = "data/metadata.pkl"
URL = "https://pikchau-rag.up.railway.app/upload_chunk"
CHUNK_SIZE = 5 * 1024 * 1024  # Reduced to 5MB to prevent SSL drops

def upload_in_chunks():
    total_size = os.path.getsize(FILE_PATH)
    uploaded = 0
    print(f"Starting chunked upload of {total_size / (1024*1024):.1f}MB to Railway...")

    with open(FILE_PATH, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            
            success = False
            for attempt in range(5):
                try:
                    response = requests.post(URL, data=chunk, headers={'Content-Type': 'application/octet-stream'}, timeout=30)
                    if response.status_code == 200:
                        uploaded += len(chunk)
                        print(f"Uploaded {uploaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({(uploaded/total_size)*100:.1f}%)")
                        success = True
                        break
                    else:
                        print(f"Server rejected chunk: {response.status_code}")
                except Exception as e:
                    print(f"Connection dropped, retrying ({attempt+1}/5)...")
                    import time
                    time.sleep(2)
            
            if not success:
                print("Failed to upload after 5 retries.")
                break
    print("Upload complete!")

if __name__ == "__main__":
    upload_in_chunks()
