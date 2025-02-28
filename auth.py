import streamlit as st
from pymongo import MongoClient
import bcrypt
import jwt
from datetime import datetime, timedelta

# MongoDB connection
def get_mongo_client():
    return MongoClient("mongodb+srv://krishna:1407@mern.jmk1x.mongodb.net/chat_db?retryWrites=true&w=majority&appName=Mern")

client = get_mongo_client()
db = client["chat_db"]
users_collection = db["users"]

# JWT settings
SECRET_KEY = 'your-secret-key'  # Change this to a secure secret key
TOKEN_EXPIRATION = 24  # hours

def generate_token(user_data):
    """Generate JWT token for user"""
    expiration = datetime.now() + timedelta(hours=TOKEN_EXPIRATION)
    token = jwt.encode(
        {
            'user_id': str(user_data['_id']),
            'email': user_data['email'],
            'exp': expiration
        },
        SECRET_KEY,
        algorithm='HS256'
    )
    return token

def verify_token(token):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_logged_in_user():
    """Get current logged in user data from session"""
    if 'token' not in st.session_state:
        return None
    
    user_data = verify_token(st.session_state['token'])
    if not user_data:
        del st.session_state['token']
        return None
    
    return user_data

def login_user(email, password):
    """Login user and return success status"""
    user = users_collection.find_one({'email': email})
    
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        token = generate_token(user)
        st.session_state['token'] = token
        return True
    return False

def register_user(email, password, role='user'):
    """Register new user and return success status"""
    # Check if user already exists
    if users_collection.find_one({'email': email}):
        st.error('Email already registered')
        return False
    
    # Hash password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Create user document
    user = {
        'email': email,
        'password': hashed_password,
        'role': role,
        'created_at': datetime.now()
    }
    
    try:
        users_collection.insert_one(user)
        return True
    except Exception as e:
        st.error(f'Error registering user: {str(e)}')
        return False

def logout():
    """Logout user by clearing session"""
    if 'token' in st.session_state:
        del st.session_state['token']