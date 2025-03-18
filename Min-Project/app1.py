from flask import Flask, request, jsonify, render_template, session
from groq import Groq
import pyttsx3
import threading
import markdown
import re
from textblob import TextBlob
import json
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24).hex())

# Store conversation histories (in-memory for now)
conversation_histories = {}

# Initialize text-to-speech engine
try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Warning: TTS initialization failed: {e}")
    engine = None

# Create Groq client with API key from environment variable
client = Groq(
    api_key=os.getenv('GROQ_API_KEY')
)

# Emotional context mapping
EMOTION_CONTEXT = {
    'positive': {
        'tone': 'enthusiastic',
        'rate': 150,
        'volume': 1.0
    },
    'negative': {
        'tone': 'empathetic',
        'rate': 120,
        'volume': 0.8
    },
    'neutral': {
        'tone': 'calm',
        'rate': 130,
        'volume': 0.9
    }
}

def analyze_sentiment(text):
    """
    Analyze the sentiment of the input text using TextBlob.
    Returns a dictionary with polarity and subjectivity scores.
    """
    analysis = TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

def get_emotional_context(sentiment_score):
    """
    Determine emotional context based on sentiment score.
    """
    if sentiment_score > 0.2:
        return 'positive'
    elif sentiment_score < -0.2:
        return 'negative'
    else:
        return 'neutral'

def adjust_voice_parameters(emotion):
    """
    Adjust voice parameters based on emotional context.
    """
    context = EMOTION_CONTEXT.get(emotion, EMOTION_CONTEXT['neutral'])
    engine.setProperty('rate', context['rate'])
    engine.setProperty('volume', context['volume'])

def clean_response(raw_response):
    """
    Cleans the raw response to remove unnecessary symbols like *, _, etc.
    Also ensures structured formatting.
    """
    # Remove unwanted characters (*, _, etc.) while keeping useful content
    cleaned_response = re.sub(r"[*_~`]", "", raw_response)

    # Replace newlines and multiple spaces with a single space
    cleaned_response = re.sub(r"\s+", " ", cleaned_response)

    # Convert the cleaned text to Markdown for proper formatting
    # Use Markdown to HTML, then strip HTML tags before passing it
    formatted_response = markdown.markdown(cleaned_response)
    
    # Strip out HTML tags from the markdown conversion
    plain_text = re.sub(r"<[^>]*>", "", formatted_response)

    # Return the cleaned and formatted response
    return plain_text

def format_response_for_speech(text):
    """
    Clean up the formatted response for text-to-speech, ensuring it sounds natural.
    """
    # Remove HTML tags and extra spaces
    clean_text = re.sub(r"<[^>]*>", "", text)
    return re.sub(r"\n", " ", clean_text).strip()

def create_session_id():
    """Generate a unique session ID for a new user."""
    return str(uuid.uuid4())

def get_conversation_history(session_id):
    """Retrieve conversation history for a given session ID."""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    return conversation_histories[session_id]

def add_to_conversation_history(session_id, role, content):
    """Add a message to the conversation history."""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    conversation_histories[session_id].append({
        "role": role,
        "content": content
    })
    
    # Keep history to a reasonable size (last 20 messages)
    if len(conversation_histories[session_id]) > 20:
        conversation_histories[session_id] = conversation_histories[session_id][-20:]

def get_llama_response(prompt, emotion_context, session_id):
    """Fetch a response from ChatGroq's Llama model with conversation history."""
    try:
        # Create an emotionally aware system message
        system_message = f"""You are an emotionally intelligent AI assistant with memory. The user's message appears to be {emotion_context['tone']}. 
        Please respond in a way that acknowledges their emotional state and provides appropriate support or enthusiasm.
        Maintain a {emotion_context['tone']} tone in your response while being helpful and informative.
        Refer to previous parts of the conversation when relevant to show continuity and memory retention."""

        # Get conversation history
        history = get_conversation_history(session_id)
        
        # Build messages array with system message and conversation history
        messages = [{"role": "system", "content": system_message}]
        messages.extend(history)
        
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})

        # Call the Groq API's chat completion endpoint with emotional context and history
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )

        # Extract and clean the response
        raw_response = completion.choices[0].message.content
        cleaned_response = clean_response(raw_response)
        
        # Add user message and bot response to history
        add_to_conversation_history(session_id, "user", prompt)
        add_to_conversation_history(session_id, "assistant", raw_response)

        return cleaned_response
    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p>"

def speak_response(text):
    """Function to handle text-to-speech asynchronously."""
    try:
        if engine is None:
            print("Text-to-speech engine is not available")
            return

        # Clean the response before passing it to the text-to-speech engine
        clean_text = format_response_for_speech(text)
        print(f"Speaking: {clean_text[:50]}...")  # Log first 50 chars

        # Pass the cleaned text to the TTS engine
        engine.say(clean_text)
        engine.runAndWait()
    except Exception as e:
        import traceback
        print(f"Error in text-to-speech: {str(e)}")
        print(traceback.format_exc())

@app.route("/")
def home():
    # Generate a new session ID if one doesn't exist
    if 'session_id' not in session:
        session['session_id'] = create_session_id()
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    
    # Ensure session ID exists
    if 'session_id' not in session:
        session['session_id'] = create_session_id()
    
    session_id = session['session_id']

    if user_message.lower() == "stop":
        if engine:
            engine.stop()  # Stop any ongoing speech synthesis
        return jsonify({"reply": "Speaking functionality has been stopped."})
    
    if user_message.lower() == "clear memory":
        # Clear conversation history for this session
        conversation_histories[session_id] = []
        return jsonify({"reply": "Memory cleared. I've forgotten our previous conversation."})

    # Analyze user's emotional state
    sentiment = analyze_sentiment(user_message)
    emotion_context = get_emotional_context(sentiment['polarity'])
    
    # Fetch response from Llama via ChatGroq with emotional context and session history
    bot_reply = get_llama_response(user_message, EMOTION_CONTEXT[emotion_context], session_id)

    # NOTE: Removed the server-side speech synthesis to prevent double speech
    # We now rely only on the browser's speech synthesis

    return jsonify({
        "reply": bot_reply,
        "emotion": {
            "context": emotion_context,
            "polarity": sentiment['polarity'],
            "subjectivity": sentiment['subjectivity']
        }
    })

@app.route("/history", methods=["GET"])
def get_history():
    """Return the conversation history for the current session."""
    if 'session_id' not in session:
        return jsonify({"history": []})
    
    history = get_conversation_history(session['session_id'])
    return jsonify({"history": history})

if __name__ == "__main__":
    app.run(debug=True)
