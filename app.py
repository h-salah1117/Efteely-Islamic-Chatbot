from flask import Flask, render_template, request, jsonify
from src.main import IslamBotProduction
import os

app = Flask(__name__)

# Initialize the bot
bot = None

def init_bot():
    """Initialize the bot when the app starts"""
    global bot
    try:
        bot = IslamBotProduction()
        print("Bot initialized successfully")
    except Exception as e:
        print(f"Error initializing bot: {e}")
        bot = None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question submission"""
    if bot is None:
        return jsonify({
            'error': 'البوت غير جاهز. يرجى المحاولة لاحقاً.'
        }), 500
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'error': 'يرجى إدخال سؤال'
            }), 400
        
        # Get response from bot
        response = bot.ask(question)
        
        return jsonify({
            'success': True,
            'ai_answer': response['answer'],
            'original_answer': response['original_answer'],
            'source_url': response.get('source_url', ''),
            'original_question': response.get('original_question', ''),
            'confidence': response.get('confidence', 0)
        })
    except Exception as e:
        return jsonify({
            'error': f'حدث خطأ أثناء معالجة السؤال: {str(e)}'
        }), 500

if __name__ == '__main__':
    init_bot()
    app.run(debug=True, host='0.0.0.0', port=5000)

