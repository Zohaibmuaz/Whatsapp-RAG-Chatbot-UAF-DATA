import json
import os
from typing import List, Dict, Any
from fastapi import FastAPI, Form
from fastapi.responses import Response
import google.generativeai as genai
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="UAF WhatsApp Admissions Assistant", version="1.0.0")

# Load environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate required environment variables
if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, GOOGLE_API_KEY]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# Initialize Gemini client
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Global variable to store university programs data
university_programs: List[Dict[str, Any]] = []

def load_university_data():
    """Load university program data from data.json file"""
    global university_programs
    try:
        with open('data.json', 'r', encoding='utf-8') as file:
            university_programs = json.load(file)
        logger.info(f"Successfully loaded {len(university_programs)} university programs")
    except FileNotFoundError:
        logger.error("data.json file not found")
        university_programs = []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing data.json: {e}")
        university_programs = []

def search_programs(user_message: str) -> List[Dict[str, Any]]:
    """
    Search for relevant programs based on user message (Original Localhost Logic).
    Implements simple keyword-based search for program names.
    """
    user_message_lower = user_message.lower()
    relevant_programs = []
    
    for program in university_programs:
        # Safely get program name and faculty, handling potential None values
        program_name = str(program.get('program_name', '')).lower()
        faculty = str(program.get('faculty_or_college', '')).lower()
        
        # Check if program name is mentioned in user message
        if any(keyword in user_message_lower for keyword in program_name.split()):
            relevant_programs.append(program)
        # Check if faculty is mentioned
        elif any(keyword in user_message_lower for keyword in faculty.split()):
            relevant_programs.append(program)
    
    # If no specific matches, return some general programs for context
    if not relevant_programs and university_programs:
        relevant_programs = university_programs[:3]  # Return first 3 programs as general context
    
    return relevant_programs

def format_program_context(programs: List[Dict[str, Any]]) -> str:
    """Format program information into readable context string (Original Localhost Logic)"""
    if not programs:
        return "No specific program information available."
    
    context_parts = []
    
    for i, program in enumerate(programs):
        context_parts.append(f"Program {i+1}:")
        context_parts.append(f"Program Name: {program.get('program_name', 'N/A')}")
        context_parts.append(f"Faculty/College: {program.get('faculty_or_college', 'N/A')}")
        context_parts.append(f"Schedule: {program.get('program_schedule', 'N/A')}")
        context_parts.append(f"Eligibility Criteria: {program.get('eligibility_criteria', 'N/A')}")
        
        if program.get('additional_requirements'):
            context_parts.append(f"Additional Requirements: {program.get('additional_requirements')}")
        
        entry_test_streams = program.get('entry_test_streams', [])
        if entry_test_streams:
            context_parts.append(f"Entry Test Streams: {', '.join(entry_test_streams)}")
        
        if program.get('notes'):
            context_parts.append(f"Notes: {program.get('notes')}")
        
        context_parts.append("")  # Empty line between programs
    
    return "\n".join(context_parts)

def generate_response_with_gemini(user_question: str, context: str) -> str:
    """Generate response using Gemini API with RAG approach"""
    try:
        system_prompt = """You are a friendly and helpful university admissions assistant for the University of Agriculture, Faisalabad. Your task is to answer the user's question based only on the context provided. Do not add any information that is not in the context. If the information is not available in the context, say that you do not have that information.

Please provide clear, helpful, and accurate information based on the context. Be conversational and welcoming, as this is a WhatsApp conversation."""
        
        full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser Question: {user_question}\n\nPlease provide a helpful response based on the context above."
        
        response = gemini_model.generate_content(full_prompt)
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error generating response with Gemini: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again later or contact the university directly for assistance."

@app.on_event("startup")
async def startup_event():
    """Load university data when the application starts"""
    load_university_data()

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"message": "UAF WhatsApp Admissions Assistant is running", "status": "healthy"}

@app.post("/whatsapp")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    """
    Twilio WhatsApp webhook using the reliable TwiML response pattern.
    """
    logger.info(f"Received message from {From}: {Body}")
    response_message_text = ""
    twiml_response = MessagingResponse()

    try:
        # Step A: Retrieval (Your preferred local logic)
        relevant_programs = search_programs(Body)
        logger.info(f"Found {len(relevant_programs)} relevant programs")
        
        # Step B: Augmentation (Your preferred local logic)
        context = format_program_context(relevant_programs)
        
        # Step C: Generation
        generated_response = generate_response_with_gemini(Body, context)
        response_message_text = generated_response

    except Exception as e:
        logger.error(f"Error processing WhatsApp message: {e}")
        response_message_text = "I'm sorry, an unexpected error occurred. Please try your question again."
    
    # Create the TwiML response
    twiml_response.message(response_message_text)
    
    # Return the TwiML as an XML response, which is the standard for webhooks
    return Response(content=str(twiml_response), media_type="application/xml")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "programs_loaded": len(university_programs) > 0,
        "gemini_configured": bool(GOOGLE_API_KEY)
    }

