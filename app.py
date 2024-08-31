import streamlit as st
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
openai.api_key = 'Your OPEN API KEY'

# Flag to indicate if directories have been checked
directories_checked = False

def recreate_directories():
    for directory in ["audio", "images", "videos"]:
        if not os.path.exists(directory):
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

# Function to extract transcript details from YouTube video
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

# Function to generate content using Gemini
def generate_gemini_content(transcript_text, prompt, language="en"):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Function to create video from text
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

# Function to create video based on prompt
def create_prompt_based_video(prompt_text, prompt, language):
    text = generate_gemini_content(prompt_text, prompt, language)
    create_video_from_text(text)

# Function to create YouTube summary video
def create_youtube_summary_video(youtube_link):
    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.write(f"Thumbnail URL: http://img.youtube.com/vi/{video_id}/0.jpg")

        transcript_text = extract_transcript_details(youtube_link)
        if transcript_text:
            summary = generate_gemini_content(transcript_text, prompt)
            st.write("Generated Summary:")
            st.write(summary)

            # Save the summary to a text file
            with open("summary.txt", "w") as file:
                file.write(summary)

            st.write("Summary saved to 'summary.txt' file.")

            create_video_from_text(summary)

# Streamlit App
st.title("AI_Video chatter and Summarizer")

choice = st.sidebar.radio("Select an option", ("Prompt-based Video Generation", "YouTube Video Summarization"))

if choice == "Prompt-based Video Generation":
    st.header("Prompt-based Video Generation")
    prompt_text = st.text_input("Enter the prompt for the video generation:")
    language = st.selectbox("Select the language of the video:", ("en", "fr", "es", "de", "it", "pt", "ru", "zh", "ja", "ko", "hi"))
    st.write("Once you're ready, click below to generate the video.")
    if st.button("Generate Video"):
        prompt = "You are a YouTube video summarizer. You will summarize the text here concisely within the 50 lines just picking the important points in the video: "
        create_prompt_based_video(prompt_text, prompt, language)

elif choice == "YouTube Video Summarization":
    st.header("YouTube Video Summarization")
    youtube_link = st.text_input("Enter YouTube video link:")
    language = st.selectbox("Select the language of the video:", ("en", "fr", "es", "de", "it", "pt", "ru", "zh", "ja", "ko", "hi"))
    st.write("Once you're ready, click below to generate the video.")
    if st.button("Summarize Video"):
        prompt = "You are a YouTube video summarizer. You will summarize the text here concisely within the 50 lines just picking the important points in the video: "
        create_youtube_summary_video(youtube_link)
# Set page background color
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add an image
st.image(r"C:\Users\Qurban Niazi\Desktop\ii.jpg", use_column_width=True)

        
