"""Main application entry point for savey_llm"""
from routes.message_handler import app

if __name__ == "__main__":
    # Run FastStream application
    app.run()
