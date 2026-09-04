import os
import json
import re
import time
import base64
from typing import List, Dict, Any

class DeidAgent:
    def __init__(self, provider: str = "google"):
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        self.llm = None
        
        if self.api_key and provider == "google":
            for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest"]:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    self.llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=self.api_key,
                        convert_system_message_to_human=True,
                        safety_settings={
                            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                        }
                    )
                    break
                except Exception as e:
                    print(f"Warning: Could not initialize Gemini model '{model_name}': {e}")
                    self.llm = None

    def run_deid(self, text: str) -> Dict[str, Any]:
        """Runs AI structured extraction on clinical text if LLM is available, or returns empty for local fallback."""
        if not self.llm or not text or not text.strip():
            return {"personal_data": [], "disease_data": []}

        prompt = f"""
        You are a medical healthcare data de-identification expert.
        Analyze the following medical report text and extract all Protected Health Information (PHI) and Disease/Clinical findings separately.
        
        Return strict JSON only with this schema:
        {{
            "personal_data": [
                {{"text": "exact text string", "label": "PHI_TYPE", "confidence": 0.95, "reasoning": "explanation"}}
            ],
            "disease_data": [
                {{"text": "exact text string", "label": "DISEASE", "confidence": 0.95, "reasoning": "explanation"}}
            ]
        }}
        
        Text to process:
        {text}
        """

        try:
            response = self.llm.invoke(prompt)
            content = str(response.content).strip()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                res = json.loads(match.group())
                return res
        except Exception as e:
            print(f"Agent run_deid fallback triggered due to: {e}")

        return {"personal_data": [], "disease_data": []}

    def get_mock_response(self) -> Dict[str, Any]:
        """Local fallback when Gemini is unavailable or rate-limited."""
        return {
            "full_text": "Clinical Report: Patient John Doe visited for evaluation. Diagnosed with Type 2 Diabetes and Hypertension. Prescribed Metformin 500 mg.",
            "personal_data": [
                {"text": "John Doe", "label": "PERSON", "confidence": 0.99, "reasoning": "Local fallback detection.", "box_2d": [150, 100, 180, 400]}
            ],
            "disease_data": [
                {"text": "Type 2 Diabetes", "label": "DISEASE", "confidence": 0.95, "reasoning": "Local clinical concept.", "box_2d": [250, 100, 280, 500]},
                {"text": "Hypertension", "label": "DISEASE", "confidence": 0.95, "reasoning": "Local clinical concept.", "box_2d": [290, 100, 320, 450]}
            ]
        }

    def run_multimodal_deid(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """Performs multimodal text & spatial bounding box extraction on images/PDFs."""
        if not self.llm:
            return self.get_mock_response()

        try:
            from langchain_core.messages import HumanMessage
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            message = HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": "Extract all text from this medical image/PDF page. Identify all Personal Data (PHI like patient name, dates, MRN, phone, address) and Disease Data (diagnoses, symptoms, medications). Return ONLY a JSON object with keys: 'full_text' (string), 'personal_data' (list of objects with text, label, confidence, reasoning, and box_2d [ymin, xmin, ymax, xmax] scaled 0-1000), and 'disease_data' (list of objects with text, label, confidence, reasoning, and box_2d [ymin, xmin, ymax, xmax] scaled 0-1000)."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ]
            )

            response = self.llm.invoke([message])
            content = str(response.content).strip()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                res = json.loads(match.group())
                if "personal_data" not in res: res["personal_data"] = []
                if "disease_data" not in res: res["disease_data"] = []
                return res
        except Exception as e:
            print(f"Multimodal LLM fallback triggered due to: {e}")

        return self.get_mock_response()

if __name__ == "__main__":
    agent = DeidAgent()
    print("Agent init status: API key present =", bool(agent.api_key))

