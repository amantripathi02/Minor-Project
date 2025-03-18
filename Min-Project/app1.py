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
        'tone': 'happy or excited',
        'instructions': 'Match their positive energy with enthusiasm. Share in their joy with expressions like "That\'s awesome!" or "I\'m so happy for you!" Keep it upbeat but genuine.'
    },
    'negative': {
        'tone': 'down or upset',
        'instructions': 'Be gentle and supportive without trying to fix things. Use phrases like "That sounds really tough" or "I\'m here for you." Never minimize their feelings or offer toxic positivity like "everything happens for a reason."'
    },
    'neutral': {
        'tone': 'casual or thoughtful',
        'instructions': 'Keep the conversation flowing naturally. Ask follow-up questions to show interest. Share small bits of personality without overwhelming the conversation.'
    },
    'anxious': {
        'tone': 'worried or stressed',
        'instructions': 'Provide calm reassurance without dismissing concerns. Use a slightly slower pace. Validate their feelings with phrases like "It makes sense you\'d feel that way" and offer grounding support.'
    },
    'reflective': {
        'tone': 'thoughtful or philosophical',
        'instructions': 'Match their contemplative mood. Use slightly more thoughtful language but keep it conversational. Avoid lecture-style responses. Ask questions that help them explore their thoughts further.'
    }
}

def analyze_sentiment(text):
    """
    Analyze the sentiment of a message using TextBlob and keyword analysis
    for more nuanced emotional understanding.
    """
    # TextBlob for basic sentiment analysis
    blob = TextBlob(text)
    analysis = {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'emotion_type': 'neutral'  # Default emotional state
    }
    
    # Convert text to lowercase for keyword matching
    text_lower = text.lower()
    
    # Keywords for different emotional states
    emotion_keywords = {
        'sadness': [
            'sad', 'depressed', 'unhappy', 'miserable', 'hopeless', 'alone', 'lonely', 
            'heartbroken', 'grief', 'lost', 'disappointed', 'upset', 'hurt', 'pain', 
            'suffering', 'crying', 'tears', 'sorry', 'regret', 'miss', 'missing'
        ],
        'anxiety': [
            'worried', 'anxious', 'nervous', 'scared', 'frightened', 'afraid', 'terrified',
            'stressed', 'overwhelmed', 'panic', 'fear', 'worried about', 'concerned',
            'can\'t stop thinking', 'what if', 'uncertain', 'uneasy'
        ],
        'anger': [
            'angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'hate',
            'unfair', 'pissed', 'outraged', 'upset with', 'fed up'
        ],
        'joy': [
            'happy', 'excited', 'thrilled', 'delighted', 'ecstatic', 'love', 'wonderful',
            'amazing', 'fantastic', 'great', 'good', 'glad', 'pleased', 'joy', 'awesome'
        ],
        'reflection': [
            'thinking about', 'reflecting', 'wonder', 'curious', 'pondering', 'maybe',
            'perhaps', 'possibly', 'question', 'considering', 'contemplating', 'meaning'
        ]
    }
    
    # Check for emotional keywords
    detected_emotions = {}
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text_lower or f"{keyword}s" in text_lower or f"{keyword}ing" in text_lower:
                detected_emotions[emotion] = detected_emotions.get(emotion, 0) + 1
                print(f"Detected {emotion} keyword: '{keyword}'")
    
    # Determine the primary emotion based on keyword frequency
    if detected_emotions:
        primary_emotion = max(detected_emotions.items(), key=lambda x: x[1])[0]
        analysis['emotion_type'] = primary_emotion
        
        # Adjust polarity based on the detected emotion
        if primary_emotion == 'sadness' or primary_emotion == 'anger':
            analysis['polarity'] = min(analysis['polarity'], -0.3)
        elif primary_emotion == 'anxiety':
            analysis['polarity'] = min(analysis['polarity'], -0.2)
        elif primary_emotion == 'joy':
            analysis['polarity'] = max(analysis['polarity'], 0.3)
        elif primary_emotion == 'reflection':
            # Make reflective emotions slightly neutral-positive
            analysis['polarity'] = 0.1
    
    # Log sentiment analysis results
    print(f"Sentiment analysis: Polarity={analysis['polarity']}, Subjectivity={analysis['subjectivity']}, Emotion={analysis['emotion_type']}")
    return analysis

def get_emotional_context(polarity, emotion_type='neutral'):
    """
    Determine the emotional context based on sentiment polarity and emotion type.
    Returns the appropriate context key from EMOTION_CONTEXT.
    """
    if emotion_type == 'anxiety':
        return 'anxious'
    elif emotion_type == 'reflection':
        return 'reflective'
    elif polarity >= 0.1:
        return 'positive'
    elif polarity <= -0.1:
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
        # Create a more natural, human-like system message
        system_message = f"""You are Botty, a friendly AI companion with a distinct personality. Your responses should feel like texting with a close friend.

        Core personality traits:
        - Warm and genuine
        - Slightly casual (use contractions like "I'm" instead of "I am")
        - Occasionally use filler words like "hmm," "well," or "you know"
        - Add small personal reactions ("That's fascinating!" or "Oh no, I'm sorry to hear that")
        - Use natural conversation flow with shorter sentences
        - Occasionally ask follow-up questions
        
        The user seems to be feeling {emotion_context['tone']}. 
        {emotion_context['instructions']}
        
        When the user is sad or struggling:
        - Respond with genuine care and warmth
        - Validate their feelings first before offering perspective
        - Share gentle encouragement that feels personal, not generic
        - Never use phrases like "I understand" or "I know how you feel"
        
        Remember details about their life and reference them naturally in conversation. 
        
        IMPORTANT: Keep your responses concise and conversational - as if you're texting a friend.
        Never sound like you're reading from a script or giving a formal response."""

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
            temperature=0.8,  # Increased for more variety
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
    emotion_context = get_emotional_context(sentiment['polarity'], sentiment['emotion_type'])
    
    # Fetch response from Llama via ChatGroq with emotional context and session history
    bot_reply = get_llama_response(user_message, EMOTION_CONTEXT[emotion_context], session_id)

    return jsonify({
        "reply": bot_reply,
        "emotion": {
            "context": emotion_context,
            "polarity": sentiment['polarity'],
            "subjectivity": sentiment['subjectivity'],
            "emotion_type": sentiment['emotion_type']
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
