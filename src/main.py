import os
import pandas as pd
import numpy as np
import subprocess
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_FILE = os.path.join(BASE_DIR, "data", "collected.csv")
EMBEDDINGS_FILE_PATH = os.path.join(BASE_DIR, "models", "embeddings.pt")

# إعدادات Ollama
OLLAMA_PATH = "ollama" 
MODEL = "gemma3:4b"

class IslamBotProduction:
    def __init__(self):
        print("جاري تحميل المحرك المحلي (CAMeL-Lab)... يرجى الانتظار.")
        # تغيير الموديل للموديل العربي الذي استخدمته في الـ Notebook
        self.embedder = SentenceTransformer("CAMeL-Lab/bert-base-arabic-camelbert-msa")
        self.df = None
        self.embeddings = None
        self.load_resources()

    def load_resources(self):
        """تحميل البيانات والمتجهات الجاهزة من نوع .pt"""
        # التأكد من وجود ملف الـ CSV
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"لم يتم العثور على ملف البيانات: {DATA_FILE}")
        self.df = pd.read_csv(DATA_FILE)

        # التأكد من وجود ملف المتجهات وتحميله
        if os.path.exists(EMBEDDINGS_FILE_PATH):
            print(f"جاري تحميل المتجهات من: {EMBEDDINGS_FILE_PATH}")
            # تحميل المتجهات وتحويلها لـ Numpy لتناسب sklearn
            self.embeddings = torch.load(EMBEDDINGS_FILE_PATH, map_location=torch.device('cpu'))
            if isinstance(self.embeddings, torch.Tensor):
                self.embeddings = self.embeddings.numpy()
            print("تم تحميل المتجهات بنجاح.")
        else:
            raise FileNotFoundError(f"ملف المتجهات غير موجود في: {EMBEDDINGS_FILE_PATH}")

    def get_context(self, query, top_k=5):
        """البحث عن أفضل k نتائج ودمجها في سياق واحد"""
        query_vec = self.embedder.encode([query])
        similarities = cosine_similarity(query_vec, self.embeddings).flatten()
        # جلب ترتيب أفضل k نتائج
        best_indices = similarities.argsort()[-top_k:][::-1]
        top_results = self.df.iloc[best_indices]
        top_scores = similarities[best_indices]
        
        # دمج الأسئلة والإجابات في سياق واحد
        context_parts = []
        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            context_parts.append(f"السؤال {idx}: {row['question']}\nالإجابة {idx}: {row['answer']}")
        
        context_string = "\n\n".join(context_parts)
        
        return context_string, top_results, top_scores

    def generate_ai_response(self, context, question):
        """استخدام Ollama المحلي للصياغة"""
        prompt = f"""أنت مفتي وخبير شرعي. أجب على سؤال المستخدم بناءً على السياق المقدم لك من قاعدة البيانات. إذا كان السؤال يتعلق بأمور دينية أساسية (مثل أركان الإسلام أو عدد الصلوات اليومية) ولم تجدها في السياق، أجب عليها بوضوح وإيجاز بناءً على علمك الديني العام. لا تكرر الأسئلة من السياق؛ استخرج وقدم الإجابة النهائية فقط.

السؤال: {question}

السياق من قاعدة البيانات:
{context}

الإجابة:"""
        try:
            result = subprocess.run(
                [OLLAMA_PATH, "run", MODEL],
                input=prompt,
                text=True,
                capture_output=True,
                encoding='utf-8'
            )
            return result.stdout.strip()
        except Exception as e:
            return f"خطأ في الاتصال بـ Ollama: {e}"

    def ask(self, user_query):
        """تنفيذ عملية البحث ثم الصياغة"""
        context, top_results, top_scores = self.get_context(user_query)
        ai_answer = self.generate_ai_response(context, user_query)
        
        # الحصول على أفضل نتيجة (Top 1) للمصدر
        top_result = top_results.iloc[0]
        top_score = top_scores[0]
        
        return {
            "answer": ai_answer,
            "original_answer": top_result['answer'],
            "source_url": top_result['URL'],
            "original_question": top_result['question'],
            "confidence": round(float(top_score), 2)
        }

# --- تشغيل البرنامج ---
if __name__ == "__main__":
    try:
        bot = IslamBotProduction()
        print("البوت جاهز للعمل أوفلاين.")
        
        while True:
            user_input = input("\nاسأل سؤالك الشرعي (أو 'exit' للخروج): ")
            if user_input.lower() == 'exit': break
            
            response = bot.ask(user_input)
            
            print("\n" + "="*50)
            print(f"الإجابة:\n{response['answer']}")
            print(f"\nرابط المصدر: {response['source_url']}")
            print("="*50)
    except Exception as e:
        print(f"حدث خطأ أثناء التشغيل: {e}")