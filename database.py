from pymongo import MongoClient

def get_mongo_client():
    return MongoClient("mongodb+srv://krishna:1407@mern.jmk1x.mongodb.net/chat_db?retryWrites=true&w=majority&appName=Mern")

def get_soil_parameters(user_email):
    """Fetch soil parameters for a specific user"""
    try:
        client = get_mongo_client()
        db = client['crop_assistant']
        soil_collection = db['soil_data']
        
        # Find soil data for the user
        soil_data = soil_collection.find_one({"email": user_email})
        
        if soil_data:
            return {
                "soil_type": soil_data.get("soil_type", "sandy"),
                "nitrogen": soil_data.get("nitrogen", 20),
                "phosphorus": soil_data.get("phosphorus", 40),
                "potassium": soil_data.get("potassium", 60),
                "ph": soil_data.get("ph", 6.5)
            }
        return None
        
    except Exception as e:
        print(f"Error fetching soil parameters: {str(e)}")
        return None
    finally:
        client.close()

