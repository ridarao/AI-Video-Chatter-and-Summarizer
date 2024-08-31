from dotenv import load_dotenv
import google.generativeai as genai
import os
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import re
import shutil
import urllib.request
from gtts import gTTS
from moviepy.editor import *
import cv2

load_dotenv()  # Initialize all the environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
openai.api_key = 'YOUR API KEY'

prompt = "You are a YouTube video summarizer. You will summarize the text here concisely within the 50 lines just picking the important points in the video: "
# Flag to indicate if directories have been checked
directories_checked = False

def recreate_directories():
    for directory in ["audio", "images", "videos"]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

def check_and_recreate_directories():
    global directories_checked
    if not directories_checked:
        recreate_directories()
        directories_checked = True

# Recreate directories
check_and_recreate_directories()
# Define the blur_faces function
def blur_faces(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Blur faces in the image
    for (x, y, w, h) in faces:
        # Apply Gaussian blur to the face region
        blurred_face = cv2.GaussianBlur(image[y:y+h, x:x+w], (51, 51), 0)
        image[y:y+h, x:x+w] = blurred_face

    # Return the modified image
    return image

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
        return transcript
    except Exception as e:
        raise e 

def generate_gemini_content(transcript_text, prompt, language="en"):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

def create_video_from_text(text):
    paragraphs = re.split(r"[,.]", text)

    i = 1
    for para in paragraphs[:-1]:
        response = openai.Image.create(
            prompt=para.strip(),
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
        urllib.request.urlretrieve(image_url, f"images/image{i}.jpg")

        # Blur the image
        blurred_image = blur_faces(f"images/image{i}.jpg")
        cv2.imwrite(f"images/image{i}_blurred.jpg", blurred_image)

        tts = gTTS(text=para, lang=language, slow=False)
        tts.save(f"audio/voiceover{i}.mp3")

        audio_clip = AudioFileClip(f"audio/voiceover{i}.mp3")
        audio_duration = audio_clip.duration

        image_clip = ImageClip(f"images/image{i}_blurred.jpg").set_duration(audio_duration)

        text_clip = TextClip(para, fontsize=50, color="white", font="Arial")
        text_clip = text_clip.set_pos('center').set_duration(audio_duration)

        clip = image_clip.set_audio(audio_clip)
        video = CompositeVideoClip([clip, text_clip])

        video = video.write_videofile(f"videos/video{i}.mp4", fps=24)
        i += 1

    clips = []
    l_files = os.listdir("videos")
    for file in l_files:
        clip = VideoFileClip(f"videos/{file}")
        clips.append(clip)

    final_video = concatenate_videoclips(clips, method="compose")
    final_video = final_video.write_videofile("final_video.mp4")

def create_prompt_based_video():
    
    prompt_text = input("Enter the prompt for the video generation: ")
    text = generate_gemini_content(prompt_text, prompt, )  # Use the provided prompt and language
    create_video_from_text(text)

def create_youtube_summary_video():
    youtube_link = input("Enter YouTube video link: ")

    if youtube_link:
        video_id = youtube_link.split("=")[1]
        print(f"Thumbnail URL: http://img.youtube.com/vi/{video_id}/0.jpg")

        transcript_text = extract_transcript_details(youtube_link)
        if transcript_text:
            summary = generate_gemini_content(transcript_text, prompt)
            print("Generated Summary:")
            print(summary)

            # Save the summary to a text file
            with open("summary.txt", "w") as file:
                file.write(summary)

            print("Summary saved to 'summary.txt' file.")

            create_video_from_text(summary)

if __name__ == "__main__":
    choice = input("Enter '1' for prompt-based video generation or '2' for YouTube video summarization: ")
    language = input("Enter the language of the video (e.g., 'en' for English, 'fr' for French, 'es' for Spanish, 'de' for German, 'it' for Italian, 'pt' for Portuguese): "
                "'ru' for Russian, 'zh' for Chinese (Simplified), 'ja' for Japanese, 'ko' for Korean, "
                "and 'hi' for Hindi): ")
    
    if choice == '1':
        create_prompt_based_video()
    elif choice == '2':
        create_youtube_summary_video()
    else:
        print("Invalid choice. Please enter either '1' or '2'.")
