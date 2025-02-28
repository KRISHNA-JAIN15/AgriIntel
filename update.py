from pymongo import MongoClient
from datetime import datetime
import json
from bson import ObjectId
import streamlit as st

# MongoDB connection
def get_mongo_client():
    return MongoClient("mongodb+srv://krishna:1407@mern.jmk1x.mongodb.net/chat_db?retryWrites=true&w=majority&appName=Mern")

client = get_mongo_client()
db = client['crop_assistant']
soil_collection = db['soil_data']

# Schema for soil data
soil_schema = {
    'user_id': str,  # Reference to user
    'land_details': {
        'total_acres': float,
        'location': str,
        'soil_type': str,
    },
    'soil_parameters': {
        'nitrogen': float,  # mg/kg
        'phosphorus': float,  # mg/kg
        'potassium': float,  # mg/kg
        'ph': float,
        'moisture': float,  # percentage
    },
    'crop_details': {
        'crops_planted': [
            {
                'crop_name': str,
                'acres_allocated': float,
                'planting_date': datetime
            }
        ]
    },
    'last_updated': datetime,
    'created_at': datetime
}

def create_soil_record(user_id, data):
    """Create a new soil record"""
    try:
        # Add metadata
        data['user_id'] = user_id
        data['created_at'] = datetime.now()
        data['last_updated'] = datetime.now()
        
        # Insert into database
        result = soil_collection.insert_one(data)
        return str(result.inserted_id)
    except Exception as e:
        st.error(f"Error creating soil record: {str(e)}")
        return None

def update_soil_record(user_id, data):
    """Update existing soil record"""
    try:
        # Find existing record
        existing_record = soil_collection.find_one({'user_id': user_id})
        
        if not existing_record:
            return create_soil_record(user_id, data)
        
        # Update only provided fields
        update_data = {}
        
        # Update land details if provided
        if 'land_details' in data:
            for key, value in data['land_details'].items():
                if value is not None:
                    update_data[f'land_details.{key}'] = value
        
        # Update soil parameters if provided
        if 'soil_parameters' in data:
            for key, value in data['soil_parameters'].items():
                if value is not None:
                    update_data[f'soil_parameters.{key}'] = value
        
        # Update crop details if provided
        if 'crop_details' in data and 'crops_planted' in data['crop_details']:
            update_data['crop_details.crops_planted'] = data['crop_details']['crops_planted']
        
        # Add last updated timestamp
        update_data['last_updated'] = datetime.now()
        
        # Perform update
        soil_collection.update_one(
            {'user_id': user_id},
            {'$set': update_data}
        )
        return str(existing_record['_id'])
    except Exception as e:
        st.error(f"Error updating soil record: {str(e)}")
        return None

def get_soil_record(user_id):
    """Get soil record for a user"""
    try:
        record = soil_collection.find_one({'user_id': user_id})
        if record:
            record['_id'] = str(record['_id'])
        return record
    except Exception as e:
        st.error(f"Error retrieving soil record: {str(e)}")
        return None

def process_json_upload(user_id, json_file):
    """Process uploaded JSON file and update soil data"""
    try:
        # Read and parse JSON file
        content = json_file.read()
        data = json.loads(content)
        
        # Validate data structure
        if not isinstance(data, dict):
            st.error("Invalid JSON format. Please provide a valid object.")
            return None
        
        # Update soil record with JSON data
        return update_soil_record(user_id, data)
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please check the file format.")
        return None
    except Exception as e:
        st.error(f"Error processing JSON file: {str(e)}")
        return None

def format_soil_data_for_update(form_data):
    """Format form data according to schema"""
    soil_data = {
        'land_details': {},
        'soil_parameters': {},
        'crop_details': {'crops_planted': []}
    }
    
    # Process land details
    if 'total_acres' in form_data:
        soil_data['land_details']['total_acres'] = form_data['total_acres']
    if 'location' in form_data:
        soil_data['land_details']['location'] = form_data['location']
    if 'soil_type' in form_data:
        soil_data['land_details']['soil_type'] = form_data['soil_type']
    
    # Process soil parameters
    soil_params = ['nitrogen', 'phosphorus', 'potassium', 'ph', 'moisture']
    for param in soil_params:
        if param in form_data:
            soil_data['soil_parameters'][param] = form_data[param]
    
    # Process crop details
    if 'crops' in form_data:
        for crop in form_data['crops']:
            if crop.get('crop_name'):
                crop_entry = {
                    'crop_name': crop['crop_name'],
                    'acres_allocated': crop.get('acres_allocated', 0),
                    'planting_date': datetime.strptime(crop['planting_date'], '%Y-%m-%d') if 'planting_date' in crop else datetime.now()
                }
                soil_data['crop_details']['crops_planted'].append(crop_entry)
    
    return soil_data