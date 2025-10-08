import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "deepseek-v3"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model
        
    def generate_response(self, prompt: str, 
                         temperature: float = 0.7, max_tokens: int = 8192,
                         **kwargs) -> str:
        
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API failed: {e}")
            raise
    
if __name__ == '__main__':
    client = OpenAIClient()
    print(client.generate_response('hello'))