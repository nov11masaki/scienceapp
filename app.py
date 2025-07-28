from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import csv
import time
import hashlib
import ssl
import certifi
import urllib3
import re
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

# 環境変数を読み込み
load_dotenv()

# SSL設定の改善
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# SSL証明書の設定
ssl_context = ssl.create_default_context(cafile=certifi.where())

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 本番環境では安全なキーに変更

# ファイルアップロード設定
UPLOAD_FOLDER = 'lesson_plans'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB制限

# アップロードディレクトリが存在しない場合は作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 教員認証情報（実際の運用では環境変数やデータベースに保存）
TEACHER_CREDENTIALS = {
    'teacher': 'science2025',  # ID: teacher, パスワード: science2025
    'admin': 'admin123'       # ID: admin, パスワード: admin123
}

# 認証チェック用デコレータ
def require_teacher_auth(f):
    def decorated_function(*args, **kwargs):
        if not session.get('teacher_authenticated'):
            return redirect(url_for('teacher_login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Gemini APIの設定
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("警告: GEMINI_API_KEYが設定されていません")
else:
    print(f"APIキー設定確認: {api_key[:10]}...{api_key[-4:]}")
    
try:
    # 設定オプションを追加してAPIクライアントを初期化
    genai.configure(
        api_key=api_key,
        transport='rest'  # gRPCではなくRESTを使用
    )
    print("Gemini API設定完了（REST使用）")
except Exception as e:
    print(f"Gemini API設定エラー: {e}")
    try:
        # フォールバック: デフォルト設定
        genai.configure(api_key=api_key)
        print("Gemini API設定完了（デフォルト）")
    except Exception as e2:
        print(f"Gemini APIフォールバック設定エラー: {e2}")

# モデル設定（JSON出力に最適化）
def create_model():
    try:
        return genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # 一貫性のためにより低い温度
                max_output_tokens=2000,
                top_p=0.8,
                top_k=20,
                candidate_count=1,
                response_mime_type="application/json"  # JSON形式を指定
            )
        )
    except Exception as e:
        print(f"モデル作成エラー: {e}")
        # フォールバックモデルを試行（JSON指定なし）
        try:
            print("フォールバックモデル gemini-1.5-flash を試行")
            return genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2000,
                )
            )
        except Exception as e2:
            print(f"フォールバックモデル作成エラー: {e2}")
            try:
                print("フォールバックモデル gemini-pro を試行")
                return genai.GenerativeModel('gemini-pro')
            except Exception as e3:
                print(f"gemini-pro作成エラー: {e3}")
                return None

model = create_model()
if model is None:
    print("警告: Geminiモデルの初期化に失敗しました")

# マークダウン記法を除去する関数
def remove_markdown_formatting(text):
    """AIの応答からマークダウン記法を除去する"""
    import re
    
    # 太字 **text** や __text__ を通常のテキストに
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    
    # 斜体 *text* や _text_ を通常のテキストに
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # 箇条書きの記号を除去
    text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*-\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # 見出し記号 ### text を除去
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # コードブロック ```text``` を除去
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # 引用記号 > を除去
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    
    # その他の記号の重複を整理
    text = re.sub(r'\s+', ' ', text)  # 複数の空白を1つに
    text = re.sub(r'\n\s*\n', '\n', text)  # 複数の改行を1つに
    
    return text.strip()

def extract_message_from_json_response(response):
    """JSON形式のレスポンスから純粋なメッセージを抽出する"""
    try:
        # JSON形式かどうか確認
        if response.strip().startswith('{') and response.strip().endswith('}'):
            import json
            parsed = json.loads(response)
            
            # responseフィールドがある場合
            if 'response' in parsed:
                return parsed['response']
            # summaryフィールドがある場合
            elif 'summary' in parsed:
                return parsed['summary']
            # messageフィールドがある場合
            elif 'message' in parsed:
                return parsed['message']
            else:
                # JSONだが適切なフィールドがない場合はそのまま返す
                return response
                
        # リスト形式の場合の処理
        elif response.strip().startswith('[') and response.strip().endswith(']'):
            import json
            parsed = json.loads(response)
            if isinstance(parsed, list) and len(parsed) > 0:
                item = parsed[0]
                if isinstance(item, dict):
                    if 'response' in item:
                        return item['response']
                    elif 'summary' in item:
                        return item['summary']
                    elif 'message' in item:
                        return item['message']
            return response
            
        # JSON形式でない場合はそのまま返す
        else:
            return response
            
    except (json.JSONDecodeError, Exception) as e:
        print(f"JSON解析エラー: {e}, 元のレスポンスを返します")
        return response

def extract_text_from_pdf(pdf_path):
    """PDFファイルからテキストを抽出する"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"PDF読み込みエラー: {e}")
        return None

def save_lesson_plan_info(unit, filename, content):
    """指導案情報をJSONファイルに保存"""
    lesson_plans_file = "lesson_plans/lesson_plans_index.json"
    
    # 既存の指導案情報を読み込み
    lesson_plans = {}
    if os.path.exists(lesson_plans_file):
        try:
            with open(lesson_plans_file, 'r', encoding='utf-8') as f:
                lesson_plans = json.load(f)
        except (json.JSONDecodeError, Exception):
            lesson_plans = {}
    
    # 新しい指導案情報を追加
    lesson_plans[unit] = {
        'filename': filename,
        'upload_date': datetime.now().isoformat(),
        'content_preview': content[:500] if content else "",  # 最初の500文字のプレビュー
        'content_length': len(content) if content else 0
    }
    
    # ファイルに保存
    with open(lesson_plans_file, 'w', encoding='utf-8') as f:
        json.dump(lesson_plans, f, ensure_ascii=False, indent=2)

def load_lesson_plan_content(unit):
    """指定された単元の指導案内容を読み込む"""
    lesson_plans_file = "lesson_plans/lesson_plans_index.json"
    
    if not os.path.exists(lesson_plans_file):
        return None
    
    try:
        with open(lesson_plans_file, 'r', encoding='utf-8') as f:
            lesson_plans = json.load(f)
        
        if unit not in lesson_plans:
            return None
        
        # PDFファイルからテキストを再読み込み
        pdf_path = os.path.join(UPLOAD_FOLDER, lesson_plans[unit]['filename'])
        if os.path.exists(pdf_path):
            return extract_text_from_pdf(pdf_path)
        else:
            return None
            
    except (json.JSONDecodeError, Exception) as e:
        print(f"指導案読み込みエラー: {e}")
        return None

def get_lesson_plans_list():
    """アップロード済みの指導案一覧を取得"""
    lesson_plans_file = "lesson_plans/lesson_plans_index.json"
    
    if not os.path.exists(lesson_plans_file):
        return {}
    
    try:
        with open(lesson_plans_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return {}

def analyze_student_language_level(conversation):
    """児童の言語レベルを分析する"""
    if not conversation:
        return "基本"
    
    user_messages = [msg['content'] for msg in conversation if msg['role'] == 'user']
    
    if not user_messages:
        return "基本"
    
    # 言語レベルの判定基準
    total_chars = sum(len(msg) for msg in user_messages)
    avg_length = total_chars / len(user_messages) if user_messages else 0
    
    # 複雑な表現のチェック
    complex_patterns = ['なぜなら', 'だから', 'しかし', 'でも', 'つまり', 'ということは']
    complex_count = sum(1 for msg in user_messages for pattern in complex_patterns if pattern in msg)
    
    # 科学用語のチェック
    science_terms = ['温度', '体積', '分子', '原子', '熱', '実験', '観察', '予想', '結果']
    science_count = sum(1 for msg in user_messages for term in science_terms if term in msg)
    
    # レベル判定
    if avg_length > 20 and complex_count > 0 and science_count > 1:
        return "高級"
    elif avg_length > 10 and (complex_count > 0 or science_count > 0):
        return "中級"
    else:
        return "基本"

def get_language_style_instruction(level):
    """言語レベルに応じた会話スタイルの指示を返す"""
    styles = {
        "基本": """
言語スタイル指示（基本レベル）:
- ひらがなを多く使い、漢字は最小限にする
- 短い文で話す（15文字以内を心がける）
- 「ね」「よ」「かな」などの親しみやすい語尾を使う
- 身近な例え話を使う（家族、ペット、遊びなど）
- 一度に一つの質問だけする
例: "あたたかくなると、どうなるかな？"
""",
        "中級": """
言語スタイル指示（中級レベル）:
- 適度に漢字を使い、読みやすい文にする
- 中程度の文の長さ（20文字程度）
- 理由を聞く質問を含める
- 学校生活に関連した例を使う
- 因果関係を意識した質問をする
例: "温度が上がると体積が変わると思った理由は何かな？"
""",
        "高級": """
言語スタイル指示（高級レベル）:
- 科学用語を適切に使用する
- やや長めの文で詳しく説明する
- 論理的思考を促す質問をする
- 複数の観点から考えさせる
- 既習事項との関連付けを促す
例: "これまでの実験結果と比較して、どのような共通点や違いが見つかりますか？"
"""
    }
    return styles.get(level, styles["基本"])

def generate_contextual_suggestions(conversation, unit, latest_ai_response, is_regenerate=False):
    """文脈に応じた選択肢を生成する"""
    
    # 児童の言語レベルを分析
    language_level = analyze_student_language_level(conversation)
    
    # 対話の段階を分析
    conversation_stage = "初期"
    if len(conversation) >= 6:
        conversation_stage = "深化"
    elif len(conversation) >= 3:
        conversation_stage = "展開"
    
    # 最近の話題を分析
    user_messages = [msg['content'] for msg in conversation if msg['role'] == 'user'][-3:]
    topics = []
    
    # 経験関連の話題
    experience_keywords = ['見た', '聞いた', '触った', '感じた', '体験', '経験']
    if any(keyword in ' '.join(user_messages) for keyword in experience_keywords):
        topics.append("経験")
    
    # 家族・友達関連の話題
    social_keywords = ['お母さん', 'お父さん', '家族', '友達', '先生', '兄弟']
    if any(keyword in ' '.join(user_messages) for keyword in social_keywords):
        topics.append("社会的")
    
    # 感情・感覚関連の話題
    emotion_keywords = ['楽しい', '面白い', '不思議', 'びっくり', '驚く']
    if any(keyword in ' '.join(user_messages) for keyword in emotion_keywords):
        topics.append("感情")
    
    return {
        'language_level': language_level,
        'conversation_stage': conversation_stage,
        'topics': topics
    }

# APIコール用のリトライ関数
def call_gemini_with_retry(prompt, max_retries=3, delay=2):
    """Gemini APIを呼び出し、エラー時はリトライする"""
    if model is None:
        return "AI システムの初期化に問題があります。管理者に連絡してください。"
    
    for attempt in range(max_retries):
        try:
            print(f"Gemini API呼び出し試行 {attempt + 1}/{max_retries}")
            
            # タイムアウト設定を短くして早期に失敗検出
            import time
            start_time = time.time()
            
            # REST APIでリクエストを送信
            response = model.generate_content(
                prompt,
                request_options={'timeout': 30}  # 30秒タイムアウト
            )
            
            elapsed_time = time.time() - start_time
            print(f"API呼び出し所要時間: {elapsed_time:.2f}秒")
            
            if response.text:
                print(f"API呼び出し成功: {len(response.text)}文字の応答")
                # マークダウン記法を除去してから返す
                cleaned_response = remove_markdown_formatting(response.text)
                return cleaned_response
            else:
                print("空の応答が返されました")
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        print(f"終了理由: {candidate.finish_reason}")
                    if hasattr(candidate, 'safety_ratings'):
                        print(f"安全性評価: {candidate.safety_ratings}")
                print(f"応答全体: {response}")
                raise Exception("空の応答が返されました")
                
        except Exception as e:
            error_msg = str(e)
            print(f"APIコール試行 {attempt + 1}/{max_retries} でエラー: {error_msg}")
            
            # エラーの種類に応じた処理
            if "API_KEY" in error_msg.upper():
                return "APIキーの設定に問題があります。管理者に連絡してください。"
            elif "QUOTA" in error_msg.upper() or "LIMIT" in error_msg.upper():
                return "API利用制限に達しました。しばらく待ってから再度お試しください。"
            elif "TIMEOUT" in error_msg.upper() or "DNS" in error_msg.upper() or "503" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = delay * (attempt + 1)
                    print(f"ネットワークエラー、{wait_time}秒後に再試行...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "ネットワーク接続に問題があります。インターネット接続を確認してください。"
            elif "400" in error_msg or "INVALID" in error_msg.upper():
                return "リクエストの形式に問題があります。管理者に連絡してください。"
            elif "403" in error_msg or "PERMISSION" in error_msg.upper():
                return "APIの利用権限に問題があります。管理者に連絡してください。"
            else:
                if attempt < max_retries - 1:
                    wait_time = delay * (attempt + 1)
                    print(f"その他のエラー、{wait_time}秒後に再試行...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"予期しないエラーが発生しました: {error_msg[:100]}..."
                    
    return "複数回の試行後もAPIに接続できませんでした。しばらく待ってから再度お試しください。"

# 学習単元のデータ
UNITS = [
    "空気の温度と体積",
    "水の温度と体積", 
    "金属の温度と体積",
    "金属のあたたまり方",
    "水のあたたまり方",
    "空気のあたたまり方",
    "ふっとうした時の泡の正体",
    "水を熱し続けた時の温度と様子",
    "冷やした時の水の温度と様子"
]

# 課題文を読み込む関数
def load_task_content(unit_name):
    try:
        with open(f'tasks/{unit_name}.txt', 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return f"{unit_name}について実験を行います。どのような結果になると予想しますか？"

# 学習ログを保存する関数
def save_learning_log(student_number, unit, log_type, data):
    """学習ログをJSONファイルに保存"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'student_number': student_number,
        'unit': unit,
        'log_type': log_type,  # 'prediction_chat', 'prediction_summary', 'reflection_chat', 'final_summary'
        'data': data
    }
    
    # ログディレクトリが存在しない場合は作成
    os.makedirs('logs', exist_ok=True)
    
    # ログファイル名（日付別）
    log_file = f"logs/learning_log_{datetime.now().strftime('%Y%m%d')}.json"
    
    # 既存のログを読み込み
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logs = []
    
    # 新しいログを追加
    logs.append(log_entry)
    
    # ファイルに保存
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

# 学習ログを読み込む関数
def load_learning_logs(date=None):
    """指定日の学習ログを読み込み"""
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    log_file = f"logs/learning_log_{date}.json"
    
    if not os.path.exists(log_file):
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

@app.route('/api/test')
def api_test():
    """API接続テスト"""
    try:
        test_prompt = "こんにちは。短い挨拶をお願いします。"
        response = call_gemini_with_retry(test_prompt, max_retries=1)
        return jsonify({
            'status': 'success',
            'message': 'API接続テスト成功',
            'response': response
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'API接続テスト失敗: {str(e)}'
        }), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_number')
def select_number():
    return render_template('select_number.html')

@app.route('/select_unit')
def select_unit():
    student_number = request.args.get('number')
    session['student_number'] = student_number
    return render_template('select_unit.html', units=UNITS)

@app.route('/prediction')
def prediction():
    unit = request.args.get('unit')
    session['unit'] = unit
    session['conversation'] = []
    
    task_content = load_task_content(unit)
    session['task_content'] = task_content
    
    return render_template('prediction.html', unit=unit, task_content=task_content)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    used_suggestion = request.json.get('used_suggestion', False)  # 選択肢使用フラグ
    suggestion_index = request.json.get('suggestion_index', None)  # 選択肢のインデックス
    conversation = session.get('conversation', [])
    unit = session.get('unit')
    task_content = session.get('task_content')
    
    # 対話履歴に追加
    conversation.append({'role': 'user', 'content': user_message})
    
    # 児童の言語レベルと対話コンテキストを分析
    context_analysis = generate_contextual_suggestions(conversation, unit, "", False)
    language_level = context_analysis['language_level']
    conversation_stage = context_analysis['conversation_stage']
    topics = context_analysis['topics']
    
    # 言語スタイルの指示を取得
    language_style = get_language_style_instruction(language_level)
    
    # Gemini APIへのプロンプト作成（言語レベル適応型）
    system_prompt = f"""
あなたは小学生の理科学習を支援するAIです。産婆法のような対話を通して、学習者の経験や既習事項を引き出し、根拠のある予想を立てさせることが目的です。

現在の学習単元: {unit}
課題: {task_content}

児童の特徴分析:
- 言語レベル: {language_level}
- 対話段階: {conversation_stage}
- 関心のある話題: {', '.join(topics) if topics else '探索中'}

{language_style}

対話ガイドライン:
1. 学習者の答えを否定せず、「なぜそう思うのか」を聞いて根拠を探る
2. 日常経験や既習事項と関連付けさせる質問をする
3. 誘導的な質問は避け、学習者自身の考えを引き出す
4. 3-4回の対話で予想をまとめられるようにする
5. **応答は必ず2-3文以内**で簡潔に返答する
6. **重要**：マークダウン記法（*、**、#、-など）は一切使用禁止。普通の文章のみで回答してください。
7. **重要**：具体例や答えの例示は絶対に行わない。学習者自身に考えさせる

対話回数: {len(conversation)}回目

特別な配慮:
- {conversation_stage}段階なので、{"基本的な質問から始める" if conversation_stage == "初期" else "より深い理解を促す" if conversation_stage == "展開" else "総合的な思考を促す"}
- {"経験談を大切にする" if "経験" in topics else ""}
- {"社会的な関係性を活用する" if "社会的" in topics else ""}
- {"感情面にも配慮する" if "感情" in topics else ""}
"""
    
    # 対話履歴を含めてプロンプト作成
    full_prompt = system_prompt + "\n\n対話履歴:\n"
    for msg in conversation:
        role = "学習者" if msg['role'] == 'user' else "AI"
        full_prompt += f"{role}: {msg['content']}\n"
    
    try:
        ai_response = call_gemini_with_retry(full_prompt)
        
        # JSON形式のレスポンスの場合は解析して純粋なメッセージを抽出
        ai_message = extract_message_from_json_response(ai_response)
        
        # マークダウン記法を除去
        ai_message = remove_markdown_formatting(ai_message)
        
        conversation.append({'role': 'assistant', 'content': ai_message})
        session['conversation'] = conversation
        
        # 学習ログを保存（選択肢使用情報を含む）
        save_learning_log(
            student_number=session.get('student_number'),
            unit=unit,
            log_type='prediction_chat',
            data={
                'user_message': user_message,
                'ai_response': ai_message,
                'conversation_count': len(conversation) // 2,
                'used_suggestion': used_suggestion,
                'suggestion_index': suggestion_index
            }
        )
        
        # 対話が3回以上の場合、予想のまとめを提案
        suggest_summary = len(conversation) >= 6  # user + AI で1セット
        
        return jsonify({
            'response': ai_message,
            'suggest_summary': suggest_summary
        })
        
    except Exception as e:
        print(f"チャットエラー: {str(e)}")
        return jsonify({'error': f'AI接続エラーが発生しました。しばらく待ってから再度お試しください。'}), 500

@app.route('/summary', methods=['POST'])
def summary():
    conversation = session.get('conversation', [])
    unit = session.get('unit')
    
    # 課題文を読み込み
    task_content = load_task_content(unit)
    
    # 予想のまとめを生成（子供が書けるようなシンプルなまとめ）
    summary_prompt = f"""
以下の対話内容を基に、子供自身が書けるような予想文を作成してください。

課題文: {task_content}

要求：
1. 子供が実際に理科ノートに書くような文体にする
2. 1-2文の短い予想文にまとめる
3. 「〜と思います。」「〜だと予想します。」の形で終わる
4. 対話で出てきた根拠（経験）を簡潔に含める
5. 難しい言葉は使わず、小学生が書くような表現にする
6. マークダウン記法（*、**、#、-など）は一切使用しない

例：
「温めると空気は大きくなると思います。夏にタイヤがパンパンになるのを見たことがあるからです。」
「空気を冷やすと小さくなると予想します。寒い日にボールがしぼんでいたからです。」

単元: {unit}

対話履歴:
"""
    for msg in conversation:
        role = "学習者" if msg['role'] == 'user' else "AI"
        summary_prompt += f"{role}: {msg['content']}\n"
    
    try:
        summary_response = call_gemini_with_retry(summary_prompt)
        
        # JSON形式のレスポンスの場合は解析して純粋なメッセージを抽出
        summary_text = extract_message_from_json_response(summary_response)
        
        session['prediction_summary'] = summary_text
        
        # 予想まとめのログを保存
        save_learning_log(
            student_number=session.get('student_number'),
            unit=unit,
            log_type='prediction_summary',
            data={
                'summary': summary_text,
                'conversation': conversation
            }
        )
        
        return jsonify({'summary': summary_text})
    except Exception as e:
        print(f"まとめエラー: {str(e)}")
        return jsonify({'error': f'まとめ生成中にエラーが発生しました。'}), 500

@app.route('/experiment')
def experiment():
    return render_template('experiment.html')

@app.route('/reflection')
def reflection():
    return render_template('reflection.html', 
                         unit=session.get('unit'),
                         prediction_summary=session.get('prediction_summary'))

@app.route('/reflect_chat', methods=['POST'])
def reflect_chat():
    user_message = request.json.get('message')
    used_suggestion = request.json.get('used_suggestion', False)  # 選択肢使用フラグ
    suggestion_index = request.json.get('suggestion_index', None)  # 選択肢のインデックス
    reflection_conversation = session.get('reflection_conversation', [])
    unit = session.get('unit')
    prediction_summary = session.get('prediction_summary', '')
    
    # 反省対話履歴に追加
    reflection_conversation.append({'role': 'user', 'content': user_message})
    
    # 考察支援プロンプト
    system_prompt = f"""
あなたは小学生の理科学習における考察を支援するAIです。実験結果と予想を比較し、日常生活と関連付けた考察を作らせることが目的です。

学習単元: {unit}
予想: {prediction_summary}

以下のガイドラインに従って対話してください：
1. 実験結果を具体的に言語化させる
2. 予想と結果の違いを明確にさせる
3. 日常生活での経験と関連付けさせる
4. 最終的に「(結果)という結果であった。(予想)と予想していたが、(合っていた/誤っていた)。このことから(経験や既習事項)は~と考えた」の形でまとめられるようにする
5. **応答は必ず2-3文以内**で簡潔に返答する
6. マークダウン記法は一切使用禁止
7. **重要**：具体例や答えの例示は絶対に行わない。学習者自身に考えさせる

対話回数: {len(reflection_conversation)}回目
"""
    
    # 対話履歴を含めてプロンプト作成
    full_prompt = system_prompt + "\n\n対話履歴:\n"
    for msg in reflection_conversation:
        role = "学習者" if msg['role'] == 'user' else "AI"
        full_prompt += f"{role}: {msg['content']}\n"
    
    try:
        ai_response = call_gemini_with_retry(full_prompt)
        
        # JSON形式のレスポンスの場合は解析して純粋なメッセージを抽出
        ai_message = extract_message_from_json_response(ai_response)
        
        # マークダウン記法を除去
        ai_message = remove_markdown_formatting(ai_message)
        
        reflection_conversation.append({'role': 'assistant', 'content': ai_message})
        session['reflection_conversation'] = reflection_conversation
        
        # 考察チャットのログを保存（選択肢使用情報を含む）
        save_learning_log(
            student_number=session.get('student_number'),
            unit=unit,
            log_type='reflection_chat',
            data={
                'user_message': user_message,
                'ai_response': ai_message,
                'conversation_count': len(reflection_conversation) // 2,
                'used_suggestion': used_suggestion,
                'suggestion_index': suggestion_index
            }
        )
        
        return jsonify({'response': ai_message})
        
    except Exception as e:
        print(f"考察チャットエラー: {str(e)}")
        return jsonify({'error': f'AI接続エラーが発生しました。しばらく待ってから再度お試しください。'}), 500

@app.route('/final_summary', methods=['POST'])
def final_summary():
    reflection_conversation = session.get('reflection_conversation', [])
    prediction_summary = session.get('prediction_summary', '')
    
    # 最終まとめを生成
    final_prompt = f"""
以下の対話を基に、定型文に従って考察を自然な日本語でまとめてください。

要求：
1. 定型文の形式を守る：「(結果)という結果であった。(予想)と予想していたが、(合っていた/誤っていた)。このことから(経験や既習事項)は~と考えた」
2. マークダウン記法（*、**、#、-など）は一切使用しない
3. 普通の文章のみで書く
4. 箇条書きや記号は使わない

予想: {prediction_summary}

対話履歴:
"""
    for msg in reflection_conversation:
        role = "学習者" if msg['role'] == 'user' else "AI"
        final_prompt += f"{role}: {msg['content']}\n"
    
    try:
        final_summary_response = call_gemini_with_retry(final_prompt)
        
        # JSON形式のレスポンスの場合は解析して純粋なメッセージを抽出
        final_summary_text = extract_message_from_json_response(final_summary_response)
        
        # マークダウン記法を除去
        final_summary_text = remove_markdown_formatting(final_summary_text)
        
        # 最終考察のログを保存
        save_learning_log(
            student_number=session.get('student_number'),
            unit=session.get('unit'),
            log_type='final_summary',
            data={
                'final_summary': final_summary_text,
                'prediction_summary': prediction_summary,
                'reflection_conversation': reflection_conversation
            }
        )
        
        return jsonify({'summary': final_summary_text})
    except Exception as e:
        print(f"最終まとめエラー: {str(e)}")
        return jsonify({'error': f'最終まとめ生成中にエラーが発生しました。'}), 500

# 教員用ルート
@app.route('/teacher/login', methods=['GET', 'POST'])
def teacher_login():
    """教員ログインページ"""
    if request.method == 'POST':
        teacher_id = request.form.get('teacher_id')
        password = request.form.get('password')
        
        # 認証チェック
        if teacher_id in TEACHER_CREDENTIALS and TEACHER_CREDENTIALS[teacher_id] == password:
            session['teacher_authenticated'] = True
            session['teacher_id'] = teacher_id
            flash('ログインしました', 'success')
            return redirect(url_for('teacher'))
        else:
            flash('IDまたはパスワードが正しくありません', 'error')
    
    return render_template('teacher/login.html')

@app.route('/teacher/logout')
def teacher_logout():
    """教員ログアウト"""
    session.pop('teacher_authenticated', None)
    session.pop('teacher_id', None)
    flash('ログアウトしました', 'info')
    return redirect(url_for('index'))

@app.route('/teacher')
@require_teacher_auth
def teacher():
    """教員用ダッシュボード"""
    # 指導案一覧も含めて表示
    lesson_plans = get_lesson_plans_list()
    return render_template('teacher/dashboard.html', 
                         units=UNITS, 
                         teacher_id=session.get('teacher_id'),
                         lesson_plans=lesson_plans)

@app.route('/teacher/lesson_plans')
@require_teacher_auth
def teacher_lesson_plans():
    """指導案管理ページ"""
    lesson_plans = get_lesson_plans_list()
    return render_template('teacher/lesson_plans.html', 
                         units=UNITS, 
                         lesson_plans=lesson_plans,
                         teacher_id=session.get('teacher_id'))

@app.route('/teacher/lesson_plans/upload', methods=['POST'])
@require_teacher_auth
def upload_lesson_plan():
    """指導案PDFのアップロード"""
    try:
        unit = request.form.get('unit')
        
        # 単元の検証
        if unit not in UNITS:
            flash('無効な単元が選択されました', 'error')
            return redirect(url_for('teacher_lesson_plans'))
        
        # ファイルの確認
        if 'file' not in request.files:
            flash('ファイルが選択されていません', 'error')
            return redirect(url_for('teacher_lesson_plans'))
        
        file = request.files['file']
        if file.filename == '':
            flash('ファイルが選択されていません', 'error')
            return redirect(url_for('teacher_lesson_plans'))
        
        if file and allowed_file(file.filename):
            # ファイル名を安全にする（単元名を含める）
            filename = secure_filename(f"{unit}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 既存ファイルがあれば削除
            lesson_plans = get_lesson_plans_list()
            if unit in lesson_plans:
                old_file = os.path.join(app.config['UPLOAD_FOLDER'], lesson_plans[unit]['filename'])
                if os.path.exists(old_file):
                    os.remove(old_file)
            
            # ファイルを保存
            file.save(file_path)
            
            # PDFからテキストを抽出
            extracted_text = extract_text_from_pdf(file_path)
            
            if extracted_text:
                # 指導案情報を保存
                save_lesson_plan_info(unit, filename, extracted_text)
                flash(f'{unit}の指導案がアップロードされました', 'success')
            else:
                flash('PDFからテキストを抽出できませんでした', 'error')
                os.remove(file_path)  # 失敗した場合はファイルを削除
        else:
            flash('PDFファイルのみアップロード可能です', 'error')
            
    except Exception as e:
        flash(f'アップロード中にエラーが発生しました: {str(e)}', 'error')
    
    return redirect(url_for('teacher_lesson_plans'))

@app.route('/teacher/lesson_plans/delete/<unit>')
@require_teacher_auth
def delete_lesson_plan(unit):
    """指導案の削除"""
    try:
        lesson_plans = get_lesson_plans_list()
        if unit in lesson_plans:
            # ファイルを削除
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], lesson_plans[unit]['filename'])
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # インデックスから削除
            del lesson_plans[unit]
            lesson_plans_file = "lesson_plans/lesson_plans_index.json"
            with open(lesson_plans_file, 'w', encoding='utf-8') as f:
                json.dump(lesson_plans, f, ensure_ascii=False, indent=2)
            
            flash(f'{unit}の指導案が削除されました', 'success')
        else:
            flash('指導案が見つかりません', 'error')
    except Exception as e:
        flash(f'削除中にエラーが発生しました: {str(e)}', 'error')
    
    return redirect(url_for('teacher_lesson_plans'))

@app.route('/teacher/logs')
@require_teacher_auth
def teacher_logs():
    """学習ログ一覧"""
    # デフォルト日付を最新のログがある日付に設定
    available_dates = get_available_log_dates()
    default_date = available_dates[0]['raw'] if available_dates else datetime.now().strftime('%Y%m%d')
    
    date = request.args.get('date', default_date)
    unit = request.args.get('unit', '')
    student = request.args.get('student', '')
    
    logs = load_learning_logs(date)
    print(f"ログ読み込み - 対象日付: {date}, 読み込んだログ数: {len(logs)}")
    
    # フィルタリング
    if unit:
        logs = [log for log in logs if log.get('unit') == unit]
        print(f"単元フィルタ適用後: {len(logs)}件")
    if student:
        logs = [log for log in logs if log.get('student_number') == student]
        print(f"学生フィルタ適用後: {len(logs)}件")
    
    # 学生ごとにグループ化
    students_data = {}
    for log in logs:
        student_num = log.get('student_number')
        if student_num not in students_data:
            students_data[student_num] = {
                'student_number': student_num,
                'units': {}
            }
        
        unit_name = log.get('unit')
        if unit_name not in students_data[student_num]['units']:
            students_data[student_num]['units'][unit_name] = {
                'prediction_chats': [],
                'prediction_summary': None,
                'reflection_chats': [],
                'final_summary': None
            }
        
        log_type = log.get('log_type')
        if log_type == 'prediction_chat':
            students_data[student_num]['units'][unit_name]['prediction_chats'].append(log)
        elif log_type == 'prediction_summary':
            students_data[student_num]['units'][unit_name]['prediction_summary'] = log
        elif log_type == 'reflection_chat':
            students_data[student_num]['units'][unit_name]['reflection_chats'].append(log)
        elif log_type == 'final_summary':
            students_data[student_num]['units'][unit_name]['final_summary'] = log
    
    return render_template('teacher/logs.html', 
                         students_data=students_data, 
                         units=UNITS,
                         current_date=date,
                         current_unit=unit,
                         current_student=student,
                         available_dates=available_dates,
                         teacher_id=session.get('teacher_id'))

@app.route('/teacher/student/<student_number>')
@require_teacher_auth
def teacher_student_detail(student_number):
    """特定の学生の詳細"""
    unit = request.args.get('unit')
    
    # デフォルト日付を最新のログがある日付に設定
    available_dates = get_available_log_dates()
    default_date = available_dates[0]['raw'] if available_dates else datetime.now().strftime('%Y%m%d')
    
    date = request.args.get('date', default_date)
    
    print(f"学生詳細ページ - 学生番号: {student_number}, 単元: {unit}, 日付: {date}")
    
    logs = load_learning_logs(date)
    print(f"読み込んだログ数: {len(logs)}")
    
    # unitパラメータがある場合のみフィルタリング
    if unit:
        student_logs = [log for log in logs 
                       if log.get('student_number') == student_number 
                       and log.get('unit') == unit]
        print(f"単元フィルタ後のログ数: {len(student_logs)}")
    else:
        # unitが指定されていない場合は、その学生の全ログを表示
        student_logs = [log for log in logs 
                       if log.get('student_number') == student_number]
        print(f"学生フィルタ後のログ数: {len(student_logs)}")
    
    # ログを時系列で並べる
    student_logs.sort(key=lambda x: x.get('timestamp', ''))
    
    # デバッグ: 最初の数件のログ情報を出力
    for i, log in enumerate(student_logs[:3]):
        print(f"ログ{i+1}: {log.get('log_type')} - {log.get('unit')} - {log.get('timestamp')}")
    
    # 利用可能なログ日付を取得
    available_dates = get_available_log_dates()
    
    return render_template('teacher/student_detail.html',
                         student_number=student_number,
                         unit=unit,
                         logs=student_logs,
                         date=date,
                         available_dates=available_dates,
                         teacher_id=session.get('teacher_id'))

@app.route('/teacher/export')
@require_teacher_auth
def teacher_export():
    """ログをCSVでエクスポート"""
    date = request.args.get('date', datetime.now().strftime('%Y%m%d'))
    logs = load_learning_logs(date)
    
    # CSVファイルを作成
    output_file = f"export_{date}.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'student_number', 'unit', 'log_type', 'content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for log in logs:
            content = ""
            if log.get('log_type') == 'prediction_chat':
                content = f"質問: {log['data'].get('user_message', '')} / 回答: {log['data'].get('ai_response', '')}"
            elif log.get('log_type') == 'prediction_summary':
                content = log['data'].get('summary', '')
            elif log.get('log_type') == 'reflection_chat':
                content = f"質問: {log['data'].get('user_message', '')} / 回答: {log['data'].get('ai_response', '')}"
            elif log.get('log_type') == 'final_summary':
                content = log['data'].get('final_summary', '')
            
            writer.writerow({
                'timestamp': log.get('timestamp', ''),
                'student_number': log.get('student_number', ''),
                'unit': log.get('unit', ''),
                'log_type': log.get('log_type', ''),
                'content': content
            })
    
    return jsonify({'message': f'エクスポートが完了しました: {output_file}'})

@app.route('/chat/suggestions', methods=['POST'])
def get_chat_suggestions():
    """対話の文脈に応じた選択肢を生成する"""
    try:
        # リクエストデータの確認
        data = request.get_json() or {}
        conversation = session.get('conversation', [])
        unit = session.get('unit', '理科')
        is_regenerate = data.get('regenerate', False)  # 再生成フラグ
        
        # 最新のAI応答を取得
        latest_ai_response = ""
        if conversation and len(conversation) > 0:
            latest_ai_response = conversation[-1].get('content', '') if conversation[-1].get('role') == 'assistant' else ""
        
        # 対話が少ない場合は基本的な選択肢を返す
        if len(conversation) < 2:
            default_suggestions = [
                "どのような結果になると思いますか？",
                "理由も教えてください",
                "他にも考えられることはありますか？"
            ]
            return jsonify({
                'suggestions': default_suggestions
            })
        
        # 文脈とレベル分析
        context_analysis = generate_contextual_suggestions(conversation, unit, latest_ai_response, is_regenerate)
        language_level = context_analysis['language_level']
        conversation_stage = context_analysis['conversation_stage']
        topics = context_analysis['topics']
        
        # AIの質問タイプを分析
        question_type = "一般"
        if any(word in latest_ai_response for word in ["どうして", "なぜ", "理由"]):
            question_type = "理由"
        elif any(word in latest_ai_response for word in ["どんな", "どのような"]):
            question_type = "描写"
        elif any(word in latest_ai_response for word in ["他に", "ほか", "別"]):
            question_type = "追加"
        elif any(word in latest_ai_response for word in ["いつ", "どこで"]):
            question_type = "状況"
        
        # 課題文を読み込み
        task_content = load_task_content(unit)
        
        # 学習者の話し方パターンを分析
        user_messages = [msg['content'] for msg in conversation if msg['role'] == 'user']
        user_style = ""
        if user_messages:
            # 最近のユーザーメッセージから話し方を分析
            recent_user_messages = user_messages[-3:] if len(user_messages) > 3 else user_messages
            user_style = " ".join(recent_user_messages)
        
        # レベル別選択肢生成プロンプト（課題文と学習者の話し方を考慮）
        suggestions_prompt = f"""
あなたは小学校理科の指導者です。学習者の実際の話し方パターンを分析して、その子が使いそうな自然な選択肢を3つ生成してください。

課題文: {task_content}

学習者の実際の話し方パターン（これらの表現スタイルに合わせてください）:
{user_style}

直前のAI応答: {latest_ai_response}

最近の対話履歴:
"""
        
        # 最新の対話を含める
        recent_conversation = conversation[-6:] if len(conversation) > 6 else conversation
        for msg in recent_conversation:
            role = "学習者" if msg['role'] == 'user' else "AI"
            suggestions_prompt += f"{role}: {msg['content']}\n"
        
        suggestions_prompt += f"""
選択肢生成の要求:
1. 上記の「学習者の話し方パターン」の文体・語彙・表現レベルに完全に合わせる
2. その学習者が実際に言いそうな表現を使用する
3. 課題文「{task_content}」の内容に関連した選択肢
4. 直前のAI応答「{latest_ai_response}」に対する自然な回答選択肢
5. 各選択肢は30文字以内で、その子らしい表現で作成
6. 言語レベルの概念ではなく、実際の話し方パターンを重視

重要：
- レベル分けの概念は無視し、実際の学習者の表現に忠実に従う
- その子が使った単語や表現の仕方を参考にする
- その子の語尾や助詞の使い方も参考にする

出力形式:
選択肢1: [その学習者が実際に言いそうな表現]
選択肢2: [その学習者が実際に言いそうな表現]  
選択肢3: [その学習者が実際に言いそうな表現]
"""
        
        suggestions_response = call_gemini_with_retry(suggestions_prompt)
        
        # JSON形式のレスポンスの場合は解析
        suggestions_text = extract_message_from_json_response(suggestions_response)
        
        # レスポンスから選択肢を抽出
        suggestions = []
        lines = suggestions_text.split('\n')
        for line in lines:
            if line.startswith('選択肢'):
                # "選択肢1: " の部分を除去
                suggestion = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                if suggestion and len(suggestion) <= 50:  # 長すぎる選択肢を除外
                    suggestions.append(suggestion)
        
        # 選択肢が3つ未満の場合は学習者の話し方に合わせたデフォルトを追加
        if len(suggestions) < 3:
            # 学習者の話し方パターンを基に、シンプルなデフォルト選択肢を作成
            if user_style:
                # 学習者が使いそうな表現を推測
                simple_defaults = ["そう思う", "よく分からない", "もう少し考えたい"]
            else:
                # 話し方パターンが不明な場合は汎用的な選択肢
                simple_defaults = ["はい", "分からない", "聞いてみたい"]
            
            for i, default in enumerate(simple_defaults):
                if len(suggestions) <= i:
                    suggestions.append(default)
        
        return jsonify({
            'suggestions': suggestions[:3],  # 最大3つまで
            'regenerated': is_regenerate,
            'language_level': language_level,
            'context': {
                'stage': conversation_stage,
                'topics': topics,
                'question_type': question_type
            }
        })
        
    except Exception as e:
        print(f"選択肢生成エラー: {str(e)}")
        # エラー時はデフォルトの選択肢を返す
        default_suggestions = [
            "そう思う",
            "よくわからない",
            "もう少し教えて"
        ]
        return jsonify({
            'suggestions': default_suggestions
        })

@app.route('/reflection/suggestions', methods=['POST'])
def get_reflection_suggestions():
    """考察対話の文脈に応じた選択肢を生成する"""
    try:
        # リクエストデータの確認
        data = request.get_json() or {}
        reflection_conversation = session.get('reflection_conversation', [])
        unit = session.get('unit', '理科')
        is_regenerate = data.get('regenerate', False)  # 再生成フラグ
        
        # 最新のAI応答を取得
        latest_ai_response = ""
        if reflection_conversation and len(reflection_conversation) > 0:
            latest_ai_response = reflection_conversation[-1].get('content', '') if reflection_conversation[-1].get('role') == 'assistant' else ""
        
        # 対話が少ない場合は基本的な選択肢を返す
        if len(reflection_conversation) < 2:
            default_suggestions = [
                "予想通りの結果でした",
                "予想と違う結果でした",
                "よくわからない結果でした"
            ]
            return jsonify({
                'suggestions': default_suggestions
            })
        
        # 全体の対話履歴から言語レベルを分析（予想段階+考察段階）
        conversation = session.get('conversation', [])
        all_conversation = conversation + reflection_conversation
        context_analysis = generate_contextual_suggestions(all_conversation, unit, latest_ai_response, is_regenerate)
        language_level = context_analysis['language_level']
        conversation_stage = "考察段階"
        topics = context_analysis['topics']
        
        # AIの質問タイプを分析（考察段階用）
        question_type = "一般"
        if any(word in latest_ai_response for word in ["結果", "実験"]):
            question_type = "結果確認"
        elif any(word in latest_ai_response for word in ["予想", "思っていた"]):
            question_type = "予想比較"
        elif any(word in latest_ai_response for word in ["どうして", "なぜ", "理由"]):
            question_type = "理由"
        elif any(word in latest_ai_response for word in ["感じ", "思う"]):
            question_type = "感想"
        
        # 課題文と予想を読み込み
        task_content = load_task_content(unit)
        prediction_summary = session.get('prediction_summary', '')
        
        # 学習者の話し方パターンを分析（予想段階も含む）
        all_user_messages = [msg['content'] for msg in all_conversation if msg['role'] == 'user']
        user_style = ""
        if all_user_messages:
            recent_user_messages = all_user_messages[-3:] if len(all_user_messages) > 3 else all_user_messages
            user_style = " ".join(recent_user_messages)
        
        # 再生成の場合は異なるアプローチで選択肢を生成
        if is_regenerate:
            suggestions_prompt = f"""
あなたは小学校理科の指導者です。考察段階で、先ほどとは違う視点から学習者が答えやすい選択肢を3つ生成してください。

課題文: {task_content}
学習者の予想: {prediction_summary}
学習者の話し方例: {user_style}

言語レベル: {language_level}
単元: {unit}

直前のAI応答: {latest_ai_response}

考察対話履歴:
"""
            # 最新の4往復分の対話を含める（前回より多く）
            recent_conversation = reflection_conversation[-8:] if len(reflection_conversation) > 8 else reflection_conversation
            for msg in recent_conversation:
                role = "学習者" if msg['role'] == 'user' else "AI"
                suggestions_prompt += f"{role}: {msg['content']}\n"
            
            suggestions_prompt += """
再生成要求（考察段階）:
1. 前回とは異なる角度から考察用選択肢を作成
2. 実験結果と予想の比較に関する具体的な選択肢
3. 感想や驚きを表現できる選択肢も含める
4. 日常生活との関連を考えられる選択肢
5. 各選択肢は30文字以内で分かりやすく
6. 以下の形式で出力してください：

選択肢1: [内容]
選択肢2: [内容]  
選択肢3: [内容]

例（再生成版）：
→ 選択肢1: 思っていたよりもはっきりした変化でした
→ 選択肢2: 予想していた通りで嬉しかったです  
→ 選択肢3: 普段の生活でも似たことがありそうです
"""
        else:
            # 通常の考察用選択肢生成プロンプト
            suggestions_prompt = f"""
あなたは小学校理科の指導者です。考察段階で、直前のAI応答に対して学習者が答えやすい選択肢を3つ生成してください。

課題文: {task_content}
学習者の予想: {prediction_summary}
学習者の話し方例: {user_style}

言語レベル: {language_level}
単元: {unit}

直前のAI応答: {latest_ai_response}

考察対話履歴:
"""
            
            # 最新の3往復分の対話を含める
            recent_conversation = reflection_conversation[-6:] if len(reflection_conversation) > 6 else reflection_conversation
            for msg in recent_conversation:
                role = "学習者" if msg['role'] == 'user' else "AI"
                suggestions_prompt += f"{role}: {msg['content']}\n"
            
            suggestions_prompt += f"""
要求（考察段階 - 学習者の話し方に合わせて）:
1. 課題文「{task_content}」に対する実験結果の考察に関連した選択肢
2. 予想「{prediction_summary}」と実験結果の比較ができる選択肢
3. 学習者の話し方レベル（{language_level}）に合わせた表現を使用
4. 各選択肢は25文字以内で、学習者が実際に言いそうな表現
5. 質問タイプ「{question_type}」に適した回答選択肢

選択肢作成のガイドライン:
- 結果確認型の質問：「〜という結果になった」「〜が観察できた」「〜の変化があった」
- 予想比較型の質問：「予想通りだった」「予想と違った」「一部だけ合っていた」
- 理由型の質問：「〜だから」「〜が原因で」「〜のおかげで」
- 感想型の質問：「驚いた」「面白かった」「もっと知りたい」

出力形式:
選択肢1: [内容]
選択肢2: [内容]  
選択肢3: [内容]
"""
        
        suggestions_response = call_gemini_with_retry(suggestions_prompt)
        
        # レスポンスから選択肢を抽出
        suggestions = []
        lines = suggestions_response.split('\n')
        for line in lines:
            if line.startswith('選択肢'):
                suggestion = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                if suggestion and len(suggestion) <= 50:
                    suggestions.append(suggestion)
        
        # 選択肢が3つ未満の場合は学習者の話し方に合わせたデフォルトを追加
        if len(suggestions) < 3:
            # 学習者の話し方パターンを基に、考察段階に合ったデフォルト選択肢を作成
            if user_style:
                # 学習者が使いそうな表現を推測（考察段階用）
                reflection_defaults = ["予想通りでした", "少し違いました", "よく分からない"]
            else:
                # 話し方パターンが不明な場合は汎用的な考察選択肢
                reflection_defaults = ["そう思う", "違うと思う", "分からない"]
            
            for i, default in enumerate(reflection_defaults):
                if len(suggestions) <= i:
                    suggestions.append(default)
        
        return jsonify({
            'suggestions': suggestions[:3],
            'regenerated': is_regenerate
        })
        
    except Exception as e:
        print(f"考察選択肢生成エラー: {str(e)}")
        default_suggestions = [
            "予想通りでした",
            "予想と違いました", 
            "よくわかりませんでした"
        ]
        return jsonify({
            'suggestions': default_suggestions
        })
        
        # 選択肢生成用のプロンプト（文脈重視）
        suggestions_prompt = f"""
あなたは小学校理科の指導者です。直前のAI応答に対して、学習者（小学生）が次に答えやすい具体的な選択肢を3つ生成してください。

単元: {unit}

直前のAI応答:
{latest_ai_response}

最近の考察対話履歴:
"""
        
        # 最新の3往復分の対話を含める
        recent_conversation = reflection_conversation[-6:] if len(reflection_conversation) > 6 else reflection_conversation
        for msg in recent_conversation:
            role = "学習者" if msg['role'] == 'user' else "AI"
            suggestions_prompt += f"{role}: {msg['content']}\n"
        
        suggestions_prompt += """
要求:
1. 直前のAI応答の内容に直接関連した選択肢を作成
2. 実験結果や考察に関する具体的な答えを選択肢にする
3. 各選択肢は25文字以内で簡潔に
4. 学習者が答えやすい具体的な内容にする
5. 「予想と結果の比較」「理由の説明」「日常生活との関連」を意識
6. 以下の形式で出力してください：

選択肢1: [内容]
選択肢2: [内容]  
選択肢3: [内容]

例：
AIが「予想と同じでしたか？」と聞いた場合
→ 選択肢1: 予想通りでした
→ 選択肢2: 予想と違いました
→ 選択肢3: 少し違いました
"""
        
        suggestions_response = call_gemini_with_retry(suggestions_prompt)
        
        # レスポンスから選択肢を抽出
        suggestions = []
        lines = suggestions_response.split('\n')
        for line in lines:
            if line.startswith('選択肢'):
                # "選択肢1: " の部分を除去
                suggestion = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                if suggestion and len(suggestion) <= 50:  # 長すぎる選択肢を除外
                    suggestions.append(suggestion)
        
        # 選択肢が3つ未満の場合は文脈に応じたデフォルトを追加
        if len(suggestions) < 3:
            # 直前のAI応答に基づいたデフォルト選択肢
            if "予想" in latest_ai_response and ("同じ" in latest_ai_response or "結果" in latest_ai_response):
                context_defaults = [
                    "予想通りでした",
                    "予想と違いました",
                    "よくわからない"
                ]
            elif "どんな" in latest_ai_response or "詳しく" in latest_ai_response:
                context_defaults = [
                    "大きくなりました",
                    "小さくなりました",
                    "変わりませんでした"
                ]
            elif "日常" in latest_ai_response or "生活" in latest_ai_response:
                context_defaults = [
                    "タイヤの変化と似ている",
                    "風船の変化と似ている",
                    "よくわからない"
                ]
            else:
                context_defaults = [
                    "そう思います",
                    "よくわかりません",
                    "もう少し考えます"
                ]
            
            for i, default in enumerate(context_defaults):
                if len(suggestions) <= i:
                    suggestions.append(default)
        
        return jsonify({
            'suggestions': suggestions[:3]  # 最大3つまで
        })
        
    except Exception as e:
        print(f"考察選択肢生成エラー: {str(e)}")
        # エラー時はデフォルトの選択肢を返す
        default_suggestions = [
            "予想通りの結果でした",
            "予想と違う結果でした",
            "よくわからない"
        ]
        return jsonify({
            'suggestions': default_suggestions
        })

@app.route('/resume_session', methods=['GET', 'POST'])
def resume_session():
    """セッション復帰機能"""
    if request.method == 'POST':
        data = request.get_json()
        student_number = data.get('student_number')
        unit = data.get('unit')
        
        if student_number and unit:
            # セッション情報を復元
            session['student_number'] = student_number
            session['unit'] = unit
            
            # 最新の学習ログを読み込んで対話履歴を復元
            try:
                today = datetime.now().strftime('%Y%m%d')
                log_file = f'logs/learning_log_{today}.json'
                
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8') as f:
                        all_logs = json.load(f)
                    
                    # 該当する学生の今日のログを取得
                    student_logs = [log for log in all_logs if 
                                  log.get('student_number') == student_number and 
                                  log.get('unit') == unit]
                    
                    if student_logs:
                        # 最新のログ状態に基づいて復帰先を判定
                        latest_log = student_logs[-1]
                        log_type = latest_log.get('log_type')
                        
                        # 対話履歴を復元
                        conversation = []
                        reflection_conversation = []
                        prediction_summary = ""
                        final_summary = ""
                        
                        for log in student_logs:
                            if log.get('log_type') == 'prediction_chat':
                                data_content = log.get('data', {})
                                user_msg = data_content.get('user_message')
                                ai_msg = data_content.get('ai_response')
                                if user_msg and ai_msg:
                                    conversation.append({'role': 'user', 'content': user_msg})
                                    conversation.append({'role': 'assistant', 'content': ai_msg})
                            elif log.get('log_type') == 'reflection_chat':
                                data_content = log.get('data', {})
                                user_msg = data_content.get('user_message')
                                ai_msg = data_content.get('ai_response')
                                if user_msg and ai_msg:
                                    reflection_conversation.append({'role': 'user', 'content': user_msg})
                                    reflection_conversation.append({'role': 'assistant', 'content': ai_msg})
                            elif log.get('log_type') == 'prediction_summary':
                                session['prediction_summary'] = log.get('data', {}).get('summary', '')
                            elif log.get('log_type') == 'final_summary':
                                session['final_summary'] = log.get('data', {}).get('summary', '')
                        
                        session['conversation'] = conversation
                        session['reflection_conversation'] = reflection_conversation
                        
                        # 最適な復帰先を判定
                        if log_type == 'final_summary':
                            return jsonify({'redirect': '/', 'message': '学習が完了しています'})
                        elif log_type == 'reflection_chat' or session.get('prediction_summary'):
                            return jsonify({'redirect': '/reflection', 'message': '考察画面から再開します'})
                        elif log_type == 'prediction_summary':
                            return jsonify({'redirect': '/experiment', 'message': '実験画面から再開します'})
                        elif log_type == 'prediction_chat' or conversation:
                            return jsonify({'redirect': '/prediction', 'message': '予想画面から再開します'})
                        else:
                            return jsonify({'redirect': '/prediction', 'message': '予想画面から開始します'})
                    else:
                        # ログがない場合は新規開始
                        return jsonify({'redirect': '/prediction', 'message': '新しく学習を開始します'})
                else:
                    # ログファイルがない場合は新規開始
                    return jsonify({'redirect': '/prediction', 'message': '新しく学習を開始します'})
                    
            except Exception as e:
                print(f"セッション復帰エラー: {str(e)}")
                return jsonify({'error': 'セッション復帰に失敗しました'})
        
        return jsonify({'error': '出席番号と単元を指定してください'})
    
    # GET リクエストの場合は復帰画面を表示
    return render_template('resume_session.html')

# ログ分析機能
def analyze_student_learning(student_number, unit, logs):
    """特定の学生・単元の学習過程を言語活動支援の観点でGemini分析（指導案考慮）"""
    print(f"分析開始 - 学生: {student_number}, 単元: {unit}")
    
    # 該当する学生のログを抽出
    student_logs = [log for log in logs if 
                   log.get('student_number') == student_number and 
                   log.get('unit') == unit]
    
    print(f"該当ログ数: {len(student_logs)}")
    
    if not student_logs:
        return {
            'evaluation': '学習データがありません',
            'language_support_needed': ['学習活動への参加が必要です'],
            'prediction_analysis': {
                'experience_connection': 'データなし',
                'prior_knowledge_use': 'データなし'
            },
            'reflection_analysis': {
                'result_verbalization': 'データなし',
                'prediction_comparison': 'データなし',
                'daily_life_connection': 'データなし'
            }
        }
    
    # 指導案の内容を取得
    lesson_plan_content = load_lesson_plan_content(unit)
    lesson_plan_context = ""
    
    if lesson_plan_content:
        # 指導案から重要な部分を抽出（最初の1000文字程度）
        lesson_plan_preview = lesson_plan_content[:1000]
        lesson_plan_context = f"""
指導案に基づく評価基準:
{lesson_plan_preview}

[指導案の内容を踏まえて]
- 学習目標の達成度
- 指導計画に沿った思考過程
- 授業で重視される観点
を含めて分析してください。
"""
    else:
        lesson_plan_context = "※指導案が設定されていないため、一般的な理科学習の観点で分析します。"
    
    # 対話履歴を整理
    prediction_chats = []
    reflection_chats = []
    prediction_summary = ""
    final_summary = ""
    
    for log in student_logs:
        log_type = log.get('log_type')
        data = log.get('data', {})
        
        if log_type == 'prediction_chat':
            prediction_chats.append({
                'user': data.get('user_message', ''),
                'ai': data.get('ai_response', '')
            })
        elif log_type == 'reflection_chat':
            reflection_chats.append({
                'user': data.get('user_message', ''),
                'ai': data.get('ai_response', '')
            })
        elif log_type == 'prediction_summary':
            prediction_summary = data.get('summary', '')
        elif log_type == 'final_summary':
            final_summary = data.get('final_summary', '')
    
    print(f"予想対話数: {len(prediction_chats)}, 考察対話数: {len(reflection_chats)}")
    
    # 分析プロンプト作成（指導案を考慮した教育的観点）
    analysis_prompt = f"""
小学生の理科学習記録を詳細に評価してください。

学習内容: {unit}
学習者ID: {student_number}

{lesson_plan_context}

【予想段階の記録】
"""
    
    # 予想段階の記録（詳細に）
    for i, chat in enumerate(prediction_chats, 1):
        user_msg = chat['user']
        ai_msg = chat['ai'][:100] + "..." if len(chat['ai']) > 100 else chat['ai']
        analysis_prompt += f"予想対話{i}:\n"
        analysis_prompt += f"  学習者: {user_msg}\n"
        analysis_prompt += f"  AI応答: {ai_msg}\n"
    
    if prediction_summary:
        analysis_prompt += f"\n予想まとめ: {prediction_summary}\n"
    
    # 考察段階の記録（詳細に）
    analysis_prompt += f"\n【考察段階の記録】\n"
    for i, chat in enumerate(reflection_chats, 1):
        user_msg = chat['user']
        ai_msg = chat['ai'][:100] + "..." if len(chat['ai']) > 100 else chat['ai']
        analysis_prompt += f"考察対話{i}:\n"
        analysis_prompt += f"  学習者: {user_msg}\n"
        analysis_prompt += f"  AI応答: {ai_msg}\n"
    
    if final_summary:
        analysis_prompt += f"\n最終考察: {final_summary}\n"
    
    analysis_prompt += """
【言語活動支援の分析観点】
生成AIによる言語活動支援の効果を以下の観点で分析してください：

1. 予想段階での言語化支援
   - 日常経験や既習事項を言語として引き出せているか
   - 根拠と予想を関連付けて表現できているか
   - 児童の固有の経験が予想に活かされているか

2. 考察段階での言語化支援
   - 実験結果を自分の言葉で表現できているか
   - 予想との差異について言語化できているか
   - 日常生活や既習事項との関連を言葉で説明できているか

3. 言語活動の深化
   - 対話を通じて思考が深まっているか
   - 「書くことを通して考える」プロセスが見られるか
   - AIの問いかけに応じて自分の言葉で説明しようと試みているか

【出力形式】
以下の形式で分析結果をJSON形式で出力してください：

{
  "evaluation": "言語活動支援の観点からの総合評価",
  "language_support_needed": ["今後の言語化支援のポイント1", "ポイント2", "ポイント3"],
  "prediction_analysis": {
    "experience_connection": "経験の言語化状況",
    "prior_knowledge_use": "既習事項の活用と言語化"
  },
  "reflection_analysis": {
    "result_verbalization": "結果の言語化状況",
    "prediction_comparison": "予想との比較の言語化",
    "daily_life_connection": "日常生活との関連付けの言語化"
  },
  "language_development": "言語活動の変化と成長"
}
"""
    
    try:
        print("Gemini分析開始...")
        response = call_gemini_with_retry(analysis_prompt)
        print(f"Gemini応答（前500文字）: {repr(response[:500])}")
        print(f"Gemini応答（後500文字）: {repr(response[-500:])}")
        
        # 複数の方法でJSONを抽出
        result = None
        
        # 方法1: 通常の正規表現
        import re
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            print(f"抽出されたJSON: {json_str}")
            try:
                result = json.loads(json_str)
                print("方法1でJSON解析成功")
            except json.JSONDecodeError:
                print("方法1でJSON解析失敗")
        
        # 方法2: 複数行にわたるJSONを抽出
        if not result:
            lines = response.split('\n')
            json_lines = []
            in_json = False
            brace_count = 0
            
            for line in lines:
                if '{' in line and not in_json:
                    in_json = True
                    brace_count = line.count('{') - line.count('}')
                    json_lines = [line]
                elif in_json:
                    json_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0:
                        break
            
            if json_lines:
                json_str = '\n'.join(json_lines)
                print(f"方法2で抽出されたJSON: {json_str}")
                try:
                    result = json.loads(json_str)
                    print("方法2でJSON解析成功")
                except json.JSONDecodeError:
                    print("方法2でJSON解析失敗")
        
        # 成功した場合は結果を返す
        if result:
            print("分析完了")
            return result
        
        # 全て失敗した場合はフォールバック
        print("JSON抽出に失敗、フォールバックを使用")
        return {
            'evaluation': '言語活動の記録から対話への取り組み姿勢が確認できます',
            'language_support_needed': ['経験の言語化支援', '既習事項との関連付け支援', '結果の表現力向上支援'],
            'prediction_analysis': {
                'experience_connection': '日常経験の引き出しを継続的に支援',
                'prior_knowledge_use': '既習事項との関連付けを意識させる対話が必要'
            },
            'reflection_analysis': {
                'result_verbalization': '実験結果を自分の言葉で表現する練習が必要',
                'prediction_comparison': '予想との比較を言語化する支援が効果的',
                'daily_life_connection': '日常生活との関連を言葉で説明する機会を増やす'
            },
            'language_development': '対話を通じて徐々に言語化能力が向上しています'
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON解析エラー: {e}")
        # 言語活動支援観点のフォールバック応答
        return {
            'evaluation': '分析処理でエラーが発生しましたが、言語活動への取り組みは確認できます',
            'language_support_needed': ['システム安定化後の詳細な言語化支援', '個別対話支援の継続', '表現力向上のための指導'],
            'prediction_analysis': {
                'experience_connection': '経験の言語化について再評価が必要',
                'prior_knowledge_use': '既習事項の活用状況を確認中'
            },
            'reflection_analysis': {
                'result_verbalization': '結果の言語化について評価中',
                'prediction_comparison': '予想との比較の言語化について分析中',
                'daily_life_connection': '日常生活との関連付けについて評価予定'
            },
            'language_development': 'システム復旧後に言語活動の成長を詳細分析予定'
        }
    except Exception as e:
        print(f"分析エラー: {e}")
        return {
            'evaluation': f'システムエラーが発生しましたが、言語活動の記録は保存されています',
            'language_support_needed': ['システム調整後の分析再実施', '継続的な言語化支援', '個別対話指導の継続'],
            'prediction_analysis': {
                'experience_connection': f'エラー詳細: {str(e)[:30]}...',
                'prior_knowledge_use': 'データ解析後に詳細確認'
            },
            'reflection_analysis': {
                'result_verbalization': 'システム復旧後に評価実施',
                'prediction_comparison': '後日詳細分析予定',
                'daily_life_connection': '包括的評価を後日実施'
            },
            'language_development': 'システム安定後に言語活動の変化を分析'
        }
    
    try:
        analysis_result = call_gemini_with_retry(analysis_prompt)
        
        # JSONパースを試行
        try:
            # JSONの前後の余分なテキストを除去
            start_idx = analysis_result.find('{')
            end_idx = analysis_result.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = analysis_result[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                raise ValueError("JSON形式が見つかりません")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON解析エラー: {e}")
            print(f"レスポンス: {analysis_result}")
            return {
                'evaluation': '分析中にエラーが発生しました',
                'strengths': ['分析データを確認中'],
                'improvements': ['システム側で調整が必要'],
                'score': 5,
                'thinking_process': '評価中',
                'engagement': '評価中',
                'scientific_understanding': '評価中'
            }
    except Exception as e:
        print(f"分析エラー: {e}")
        return {
            'evaluation': 'AI分析でエラーが発生しました',
            'strengths': ['学習に取り組んでいます'],
            'improvements': ['継続的な学習'],
            'score': 5,
            'thinking_process': 'システムエラー',
            'engagement': 'システムエラー', 
            'scientific_understanding': 'システムエラー'
        }

def analyze_class_trends(logs, unit=None):
    """クラス全体の学習傾向をGeminiで分析（指導案考慮）"""
    if unit:
        # 特定単元の分析
        unit_logs = [log for log in logs if log.get('unit') == unit]
        students = set(log.get('student_number') for log in unit_logs)
        analysis_unit = unit
    else:
        # 全体の分析
        unit_logs = logs
        students = set(log.get('student_number') for log in logs)
        analysis_unit = "全単元"
    
    if not unit_logs or len(students) == 0:
        return {
            'overall_trend': '分析対象のデータがありません',
            'common_misconceptions': [],
            'effective_approaches': [],
            'recommendations': []
        }
    
    # 指導案の内容を取得（特定単元の場合）
    lesson_plan_context = ""
    if unit:
        lesson_plan_content = load_lesson_plan_content(unit)
        if lesson_plan_content:
            lesson_plan_preview = lesson_plan_content[:800]
            lesson_plan_context = f"""
指導案情報:
{lesson_plan_preview}

[指導案に基づく分析観点]
- 指導目標の達成状況
- 予想されていた課題や誤解の出現
- 指導計画との整合性
- 次回授業への示唆
"""
        else:
            lesson_plan_context = "※この単元の指導案は設定されていません。"
    
    # 学習データを要約
    summary_data = {}
    for student in students:
        student_logs = [log for log in unit_logs if log.get('student_number') == student]
        summary_data[student] = {
            'prediction_count': len([log for log in student_logs if log.get('log_type') == 'prediction_chat']),
            'reflection_count': len([log for log in student_logs if log.get('log_type') == 'reflection_chat']),
            'has_prediction': any(log.get('log_type') == 'prediction_summary' for log in student_logs),
            'has_final': any(log.get('log_type') == 'final_summary' for log in student_logs)
        }
    
    # よくある予想や考察のパターンを抽出
    predictions = []
    reflections = []
    
    for log in unit_logs:
        if log.get('log_type') == 'prediction_summary':
            predictions.append(log.get('data', {}).get('summary', ''))
        elif log.get('log_type') == 'final_summary':
            reflections.append(log.get('data', {}).get('final_summary', ''))
    
    analysis_prompt = f"""
クラス全体の学習状況を分析してください。

対象単元: {analysis_unit}
学習者数: {len(students)}人

{lesson_plan_context}

各学習者の状況:
"""
    
    for student, data in summary_data.items():
        analysis_prompt += f"学習者{student}: 予想{data['prediction_count']}回 考察{data['reflection_count']}回 "
        analysis_prompt += f"予想完了{'○' if data['has_prediction'] else '×'} 考察完了{'○' if data['has_final'] else '×'}\n"
    
    analysis_prompt += f"\n主な予想:\n"
    for i, pred in enumerate(predictions[:3], 1):  # 最大3つまで
        analysis_prompt += f"{i}. {pred[:50]}...\n"
    
    analysis_prompt += f"\n主な考察:\n"
    for i, ref in enumerate(reflections[:3], 1):  # 最大3つまで
        analysis_prompt += f"{i}. {ref[:50]}...\n"
    
    analysis_prompt += """
言語活動支援の観点からクラス全体の状況を分析してください。

【分析項目】
- overall_trend: クラス全体の言語活動の傾向（100文字程度）
- language_challenges: 児童が共通して抱える言語化の課題を3つ
- verbalization_level: 言語化能力のレベル（発展中/安定/要支援）
- dialogue_engagement: 対話への参加状況
- expression_growth: 表現力の成長状況を2つ

JSON形式で回答してください。
"""
    
    analysis_prompt += """
この学習状況について、以下の形式で分析結果をJSON形式で出力してください。

{
  "overall_trend": "クラス全体で言語活動に意欲的に取り組んでいます",
  "language_challenges": ["経験の言語化", "既習事項との関連付け", "結果の表現"],
  "verbalization_level": "発展中",
  "dialogue_engagement": "積極的に対話に参加しています",
  "expression_growth": ["自分の言葉での表現", "論理的な説明の向上"]
}
"""
    
    try:
        print("クラス分析開始...")
        analysis_result = call_gemini_with_retry(analysis_prompt)
        print(f"クラス分析応答（前500文字）: {repr(analysis_result[:500])}")
        print(f"クラス分析応答（後500文字）: {repr(analysis_result[-500:])}")
        
        # 複数の方法でJSONを抽出
        result = None
        
        # 方法1: 通常の正規表現
        import re
        json_match = re.search(r'\{.*?\}', analysis_result, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            print(f"クラス分析抽出JSON: {json_str}")
            try:
                result = json.loads(json_str)
                print("クラス分析方法1でJSON解析成功")
            except json.JSONDecodeError:
                print("クラス分析方法1でJSON解析失敗")
        
        # 方法2: 複数行JSON抽出
        if not result:
            lines = analysis_result.split('\n')
            json_lines = []
            in_json = False
            brace_count = 0
            
            for line in lines:
                if '{' in line and not in_json:
                    in_json = True
                    brace_count = line.count('{') - line.count('}')
                    json_lines = [line]
                elif in_json:
                    json_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0:
                        break
            
            if json_lines:
                json_str = '\n'.join(json_lines)
                print(f"クラス分析方法2で抽出されたJSON: {json_str}")
                try:
                    result = json.loads(json_str)
                    print("クラス分析方法2でJSON解析成功")
                except json.JSONDecodeError:
                    print("クラス分析方法2でJSON解析失敗")
        
        # 成功した場合は結果を返す
        if result:
            print("クラス分析完了")
            return result
            
        # 全て失敗した場合はフォールバック
        print("クラス分析JSON抽出に失敗、フォールバックを使用")
        return {
            'overall_trend': 'クラス全体として言語活動に意欲的に取り組んでいます',
            'language_challenges': ['経験の言語化', '既習事項との関連付け', '結果の表現力'],
            'verbalization_level': '発展中',
            'dialogue_engagement': '積極的に対話に参加している状況です',
            'expression_growth': ['自分の言葉での表現向上', '思考の言語化進展']
        }
    except Exception as e:
        print(f"クラス分析エラー: {e}")
        return {
            'overall_trend': '言語活動の分析でエラーが発生しました',
            'language_challenges': ['分析データ不足'],
            'verbalization_level': 'システムエラー',
            'dialogue_engagement': 'システムエラー',
            'expression_growth': ['システム調整']
        }

@app.route('/teacher/analysis')
@require_teacher_auth
def teacher_analysis():
    """学習分析ダッシュボード"""
    # デフォルト日付を最新のログがある日付に設定
    available_dates = get_available_log_dates()
    default_date = available_dates[0]['raw'] if available_dates else datetime.now().strftime('%Y%m%d')
    
    date = request.args.get('date', default_date)
    unit = request.args.get('unit', '')
    
    logs = load_learning_logs(date)
    
    # クラス全体の傾向分析
    class_analysis = analyze_class_trends(logs, unit if unit else None)
    
    # 単元別の学習者リスト
    unit_students = {}
    for log in logs:
        log_unit = log.get('unit')
        student = log.get('student_number')
        if log_unit and student:
            if log_unit not in unit_students:
                unit_students[log_unit] = set()
            unit_students[log_unit].add(student)
    
    # 各単元の学習者を配列に変換
    for unit_name in unit_students:
        unit_students[unit_name] = sorted(list(unit_students[unit_name]))
    
    return render_template('teacher/analysis.html',
                         class_analysis=class_analysis,
                         unit_students=unit_students,
                         units=UNITS,
                         current_date=date,
                         current_unit=unit,
                         available_dates=available_dates,
                         teacher_id=session.get('teacher_id'))

@app.route('/teacher/analysis/student')
@require_teacher_auth
def teacher_student_analysis():
    """個別学生の詳細分析"""
    student_number = request.args.get('student') or request.args.get('student_number')
    unit = request.args.get('unit')
    
    # デフォルト日付を最新のログがある日付に設定
    available_dates = get_available_log_dates()
    default_date = available_dates[0]['raw'] if available_dates else datetime.now().strftime('%Y%m%d')
    
    date = request.args.get('date', default_date)
    
    print(f"個別学生分析 - 学生番号: {student_number}, 単元: {unit}, 日付: {date}")
    
    if not student_number:
        flash('学生番号を指定してください', 'error')
        return redirect(url_for('teacher_analysis'))
    
    logs = load_learning_logs(date)
    print(f"読み込んだログ数: {len(logs)}")
    
    # unitが指定されていない場合は、最初に見つかった単元を使用
    if not unit:
        student_logs_all = [log for log in logs if log.get('student_number') == student_number]
        if student_logs_all:
            unit = student_logs_all[0].get('unit')
            print(f"単元が指定されていないため、最初の単元を使用: {unit}")
        else:
            flash('指定された学生の学習データが見つかりません', 'error')
            return redirect(url_for('teacher_analysis'))
    
    # 個別学生分析
    student_analysis = analyze_student_learning(student_number, unit, logs)
    
    # 該当する学生のログも取得
    student_logs = [log for log in logs 
                   if log.get('student_number') == student_number 
                   and log.get('unit') == unit]
    
    # ログを時系列で並べる
    student_logs.sort(key=lambda x: x.get('timestamp', ''))
    
    return render_template('teacher/student_analysis.html',
                         student_analysis=student_analysis,
                         student_number=student_number,
                         unit=unit,
                         logs=student_logs,
                         date=date,
                         available_dates=available_dates,
                         teacher_id=session.get('teacher_id'))

@app.route('/teacher/analysis/api/student', methods=['POST'])
@require_teacher_auth
def api_student_analysis():
    """学生分析のAPI（AJAX用）"""
    data = request.get_json()
    student_number = data.get('student_number')
    unit = data.get('unit')
    
    # デフォルト日付を最新のログがある日付に設定
    available_dates = get_available_log_dates()
    default_date = available_dates[0]['raw'] if available_dates else datetime.now().strftime('%Y%m%d')
    
    date = data.get('date', default_date)
    
    logs = load_learning_logs(date)
    analysis = analyze_student_learning(student_number, unit, logs)
    
    return jsonify(analysis)

@app.route('/teacher/analysis/api/class', methods=['POST'])
@require_teacher_auth
def api_class_analysis():
    """クラス分析のAPI（AJAX用）"""
    data = request.get_json()
    unit = data.get('unit')
    
    # デフォルト日付を最新のログがある日付に設定
    available_dates = get_available_log_dates()
    default_date = available_dates[0]['raw'] if available_dates else datetime.now().strftime('%Y%m%d')
    
    date = data.get('date', default_date)
    
    logs = load_learning_logs(date)
    analysis = analyze_class_trends(logs, unit if unit else None)
    
    return jsonify(analysis)

def get_available_log_dates():
    """利用可能なログファイルの日付一覧を取得"""
    import os
    import glob
    
    log_files = glob.glob("logs/learning_log_*.json")
    dates = []
    
    for file in log_files:
        # ファイル名から日付を抽出
        filename = os.path.basename(file)
        if filename.startswith('learning_log_') and filename.endswith('.json'):
            date_str = filename[13:-5]  # learning_log_YYYYMMDD.json
            if len(date_str) == 8 and date_str.isdigit():
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                dates.append({'raw': date_str, 'formatted': formatted_date})
    
    # 日付でソート（新しい順）
    dates.sort(key=lambda x: x['raw'], reverse=True)
    return dates

if __name__ == '__main__':
    app.run(debug=True, port=5011)