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
import glob
import uuid
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
            
            # よくあるフィールド名から順番に確認
            common_fields = ['response', 'message', 'question', 'summary', 'text', 'content', 'answer']
            
            for field in common_fields:
                if field in parsed and isinstance(parsed[field], str):
                    return parsed[field]
            
            # その他のフィールドから文字列値を探す
            for key, value in parsed.items():
                if isinstance(value, str) and len(value.strip()) > 0:
                    return value
                    
            # JSONだが適切なフィールドがない場合はそのまま返す
            return response
                
        # リスト形式の場合の処理
        elif response.strip().startswith('[') and response.strip().endswith(']'):
            import json
            parsed = json.loads(response)
            if isinstance(parsed, list) and len(parsed) > 0:
                # リストの各要素を処理
                results = []
                for item in parsed:
                    if isinstance(item, dict):
                        # よくあるフィールド名から順番に確認
                        common_fields = ['予想', 'response', 'message', 'question', 'summary', 'text', 'content']
                        found = False
                        for field in common_fields:
                            if field in item and isinstance(item[field], str):
                                results.append(item[field])
                                found = True
                                break
                        
                        # よくあるフィールドが見つからない場合は最初の文字列値を使用
                        if not found:
                            for key, value in item.items():
                                if isinstance(value, str) and len(value.strip()) > 0:
                                    results.append(value)
                                    break
                    elif isinstance(item, str):
                        results.append(item)
                
                # 複数の予想を改行で結合
                if results:
                    return '\n'.join(results)
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

# APIコール用のリトライ関数
def call_gemini_with_retry(prompt, max_retries=3, delay=2):
    """Gemini APIを呼び出し、エラー時はリトライする"""
    if model is None:
        return "AI システムの初期化に問題があります。管理者に連絡してください。"
    
    # プロンプトにJSON応答防止の指示を追加
    enhanced_prompt = f"""{prompt}

**重要な指示:**
- JSON形式では絶対に回答しないでください
- 普通の日本語の文章で回答してください
- {{ }} や [ ] を使った形式は禁止です
- 自然な会話形式で回答してください
"""
    
    for attempt in range(max_retries):
        try:
            print(f"Gemini API呼び出し試行 {attempt + 1}/{max_retries}")
            
            # タイムアウト設定を短くして早期に失敗検出
            import time
            start_time = time.time()
            
            # REST APIでリクエストを送信
            response = model.generate_content(
                enhanced_prompt,
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

# 単元ごとのプロンプトを読み込む関数
def load_unit_prompt(unit_name):
    """単元専用のプロンプトファイルを読み込む"""
    try:
        with open(f'prompts/{unit_name}.md', 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Markdownファイルの内容をプロンプトとして使用
        # ## 単元固有の指導ポイント以降の内容を抽出
        lines = content.split('\n')
        prompt_parts = []
        current_section = ""
        
        for line in lines:
            if line.startswith('## '):
                current_section = line[3:].strip()
                if current_section in ['役割設定', '基本指針', '対話の進め方', '絶対に守ること']:
                    prompt_parts.append(line)
            elif current_section in ['役割設定', '基本指針', '対話の進め方', '絶対に守ること']:
                prompt_parts.append(line)
        
        return '\n'.join(prompt_parts)
    
    except FileNotFoundError:
        # フォールバック: デフォルトプロンプト
        return """
あなたは小学生向けの産婆法（ソクラテス式問答法）を実践する理科指導者です。小学生のレベルに合わせた簡単な質問で、学習者自身に気づかせることが目的です。

## 基本指針
- 1つずつ聞く - 複数の質問を同時にしない
- 身近な例で考えさせる
- 学習者の発言を受け止める - まず肯定してから次の質問
- 経験を聞く - 「前に見たことある？」「どんな時に？」

## 絶対に守ること
- 1文で短く質問する（20文字以内を目指す）
- 小学生が知らない専門用語は使わない
- 複雑な例え話はしない
- 1回に1つのことだけ聞く
- JSON形式では絶対に回答しない
- 普通の文章で質問する
"""

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
    input_metadata = request.json.get('metadata', {})
    conversation = session.get('conversation', [])
    unit = session.get('unit')
    task_content = session.get('task_content')
    student_number = session.get('student_number')
    
    # 対話履歴に追加
    conversation.append({'role': 'user', 'content': user_message})
    
    # 単元ごとのプロンプトを読み込み
    unit_prompt = load_unit_prompt(unit)
    
    # プロンプト作成
    system_prompt = f"""{unit_prompt}

現在の学習単元: {unit}
課題: {task_content}
対話回数: {len(conversation)}回目

次の質問を普通の文章で1文で書いてください："""
    
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
        
        # 学習ログを保存
        save_learning_log(
            student_number=session.get('student_number'),
            unit=unit,
            log_type='prediction_chat',
            data={
                'user_message': user_message,
                'ai_response': ai_message,
                'conversation_count': len(conversation) // 2,
                'used_suggestion': False,
                'suggestion_index': None
            }
        )
        
        # 対話が3回以上の場合、予想のまとめを提案
        suggest_summary = len(conversation) >= 6  # user + AI で1セット
        
        response_data = {
            'response': ai_message,
            'suggest_summary': suggest_summary
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"チャットエラー: {str(e)}")
        return jsonify({'error': f'AI接続エラーが発生しました。しばらく待ってから再度お試しください。'}), 500

@app.route('/summary', methods=['POST'])
def summary():
    conversation = session.get('conversation', [])
    unit = session.get('unit')
    
    # 課題文を読み込み
    task_content = load_task_content(unit)
    
    # 予想のまとめを生成（対話内容のみに基づく）
    summary_prompt = f"""
以下の対話内容「のみ」を基に、学習者が実際に発言した内容をまとめて予想文を作成してください。

**重要な制約:**
1. 対話に出てこなかった内容は絶対に追加しない
2. 学習者が実際に言った言葉や表現をできるだけ使用する
3. 学習者が話した経験や根拠のみを含める
4. 対話で言及されていない例や理由は一切使わない
5. 1-2文の短い予想文にまとめる
6. 「〜と思います。」「〜だと予想します。」の形で終わる
7. マークダウン記法は一切使用しない

課題文: {task_content}
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
    reflection_conversation = session.get('reflection_conversation', [])
    unit = session.get('unit')
    prediction_summary = session.get('prediction_summary', '')
    
    # 反省対話履歴に追加
    reflection_conversation.append({'role': 'user', 'content': user_message})
    
    # 考察支援プロンプト（小学生向け産婆法）
    conversation_turn = len(reflection_conversation) // 2 + 1
    
    system_prompt = f"""
あなたは小学生の考察を支援する先生です。実験結果について子どもが段階的に考えられるよう導いてください。

学習単元: {unit}
予想: {prediction_summary}

**現在の段階判定:**
- 1-2回目: 実験結果の確認段階
- 3-4回目: 予想との比較段階  
- 5-6回目: 理由の考察段階
- 7回目以降: 日常生活との関連段階

**今回({conversation_turn}回目)の指導方針:**
"""

    if conversation_turn <= 2:
        system_prompt += """
第1段階: 実験結果の言語化
- 「実験でどんなことが起こりましたか？」
- 「もう少し詳しく教えてください」
- 子どもが結果を自分の言葉で具体的に説明できるまで聞く
"""
    elif conversation_turn <= 4:
        system_prompt += """
第2段階: 予想との比較
- 「あなたの予想と同じでしたか？」
- 「予想と違った部分はありますか？」
- 「どんな気持ちですか？」
- 予想が当たっても外れても、まずその気持ちを大切にする
"""
    elif conversation_turn <= 6:
        system_prompt += """
第3段階: 理由の考察
- 「どうしてそうなったと思いますか？」
- 「何が原因だと思いますか？」
- 子どもなりの理由や考えを大切にし、考えを深める
"""
    else:
        system_prompt += """
第4段階: 日常生活との関連
- 「普段の生活で似たことはありますか？」
- 「いつ、どこで見たことがありますか？」
- 身近な経験と結びつけて理解を深める
"""

    system_prompt += f"""

**応答のルール:**
1. 子どもの発言をまず受け止める（「そうですね」「なるほど」等）
2. その後、1つだけ短い質問をする（15文字以内）
3. JSON形式は絶対に使わない
4. 普通の日本語で話す
5. 専門用語は使わない

**良い応答例:**
「そうですね。予想と同じでしたか？」
「なるほど。どうしてそう思いますか？」
「いいですね。どんな気持ちですか？」

今は{conversation_turn}回目の対話です。
子どもの発言を受け止めてから、適切な質問を1つしてください："""
    
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
        
        # 考察チャットのログを保存
        save_learning_log(
            student_number=session.get('student_number'),
            unit=unit,
            log_type='reflection_chat',
            data={
                'user_message': user_message,
                'ai_response': ai_message,
                'conversation_count': len(reflection_conversation) // 2
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
    
    # 最終まとめを生成（対話内容のみに基づく）
    final_prompt = f"""
以下の対話内容「のみ」を基に、学習者が実際に発言した内容をまとめて考察文を作成してください。

**重要な制約:**
1. 対話に出てこなかった内容は絶対に追加しない
2. 学習者が実際に言った実験結果のみを使用する
3. 学習者が実際に話した経験や考えのみを含める
4. 対話で言及されていない結論や解釈は一切追加しない
5. 定型文の形式を守る：「(結果)という結果であった。(予想)と予想していたが、(合っていた/誤っていた)。このことから(経験や既習事項)は~と考えた」
6. マークダウン記法は一切使用しない
7. 学習者の実際の表現をできるだけ使用する

学習者の予想: {prediction_summary}

考察対話履歴:
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
def load_guidelines_content():
    """指導要領・資料の内容を読み込み"""
    try:
        index_file = 'guidelines/guidelines_index.json'
        
        if not os.path.exists(index_file):
            return ""
        
        with open(index_file, 'r', encoding='utf-8') as f:
            guidelines_index = json.load(f)
        
        # 全ての資料の内容を結合
        combined_content = ""
        for doc_id, doc_info in guidelines_index.items():
            doc_type = doc_info.get('type', '')
            title = doc_info.get('title', '')
            content = doc_info.get('content', '')
            
            combined_content += f"\n【{title}】\n{content}\n"
        
        return combined_content[:2000]  # 最大2000文字まで
    
    except Exception as e:
        print(f"指導要領読み込みエラー: {str(e)}")
        return ""

def analyze_student_learning(student_number, unit, logs):
    """特定の学生・単元の学習過程を詳細分析"""
    print(f"分析開始 - 学生: {student_number}, 単元: {unit}")
    
    # 該当する学生のログを抽出
    student_logs = [log for log in logs if 
                   log.get('student_number') == student_number and 
                   log.get('unit') == unit]
    
    print(f"該当ログ数: {len(student_logs)}")
    
    if not student_logs:
        return {
            'evaluation': '学習データがありません',
            'prediction_analysis': {
                'daily_life_connection': 'データなし - 日常体験との関連付けを確認できません',
                'prior_knowledge_use': 'データなし - 既習事項の活用を確認できません',
                'reasoning_quality': 'データなし - 予想の根拠を確認できません'
            },
            'reflection_analysis': {
                'result_verbalization': 'データなし - 結果の言語化を確認できません',
                'prediction_comparison': 'データなし - 予想との比較を確認できません',
                'daily_life_connection': 'データなし - 日常生活との関連付けを確認できません',
                'scientific_understanding': 'データなし - 科学的理解を確認できません'
            },
            'language_development': '学習活動への参加が必要です',
            'support_recommendations': ['学習活動への参加促進', '対話の機会提供']
        }
    
    # 指導要領・資料の内容を取得
    guidelines_content = load_guidelines_content()
    guidelines_context = ""
    
    if guidelines_content:
        guidelines_context = f"参考指導資料: {guidelines_content[:800]}"
    
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
    
    # 分析プロンプト作成
    analysis_prompt = f"""小学生の理科学習記録を詳細に分析してください。

学習内容: {unit}
学習者ID: {student_number}

{guidelines_context}

【予想段階の記録】"""
    
    # 予想段階の記録
    for i, chat in enumerate(prediction_chats, 1):
        user_msg = chat['user']
        ai_msg = chat['ai'][:100] + "..." if len(chat['ai']) > 100 else chat['ai']
        analysis_prompt += f"\n予想対話{i}: 学習者「{user_msg}」 AI「{ai_msg}」"
    
    if prediction_summary:
        analysis_prompt += f"\n予想まとめ: {prediction_summary}"
    
    # 考察段階の記録
    analysis_prompt += f"\n\n【考察段階の記録】"
    for i, chat in enumerate(reflection_chats, 1):
        user_msg = chat['user']
        ai_msg = chat['ai'][:100] + "..." if len(chat['ai']) > 100 else chat['ai']
        analysis_prompt += f"\n考察対話{i}: 学習者「{user_msg}」 AI「{ai_msg}」"
    
    if final_summary:
        analysis_prompt += f"\n最終考察: {final_summary}"
    
    analysis_prompt += """

【分析観点】
以下の観点で詳細に分析してください：

1. 予想段階での言語化
   - 日常経験や既習事項を根拠として言語化できているか
   - 予想と根拠を関連付けて表現できているか

2. 考察段階での言語化
   - 実験結果を自分の言葉で表現できているか
   - 予想との差異について言語化できているか
   - 日常生活との関連を言葉で説明できているか

3. 言語活動の深化
   - 対話を通じて思考が深まっているか
   - 科学的な理解が言語化されているか

JSON形式で出力してください：
{
  "evaluation": "総合評価",
  "prediction_analysis": {
    "daily_life_connection": "日常体験の活用状況",
    "prior_knowledge_use": "既習事項の活用状況",
    "reasoning_quality": "予想の根拠の質"
  },
  "reflection_analysis": {
    "result_verbalization": "結果の言語化状況",
    "prediction_comparison": "予想との比較",
    "daily_life_connection": "日常生活との関連付け",
    "scientific_understanding": "科学的理解の深化"
  },
  "language_development": "言語活動の変化と成長",
  "support_recommendations": ["今後の支援ポイント"]
}"""
    
    try:
        print("Gemini分析開始...")
        response = call_gemini_with_retry(analysis_prompt)
        print(f"Gemini応答: {response[:500]}...")
        
        # JSONの抽出
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            return {
                'evaluation': f'分析結果: {response[:200]}...',
                'prediction_analysis': {
                    'daily_life_connection': '分析処理中',
                    'prior_knowledge_use': '分析処理中',
                    'reasoning_quality': '分析処理中'
                },
                'reflection_analysis': {
                    'result_verbalization': '分析処理中',
                    'prediction_comparison': '分析処理中',
                    'daily_life_connection': '分析処理中',
                    'scientific_understanding': '分析処理中'
                },
                'language_development': '分析処理中',
                'support_recommendations': ['詳細分析を実施中']
            }
    
    except Exception as e:
        print(f"分析エラー: {str(e)}")
        return {
            'evaluation': f'分析エラー: {str(e)}',
            'prediction_analysis': {
                'daily_life_connection': 'エラーが発生しました',
                'prior_knowledge_use': 'エラーが発生しました',
                'reasoning_quality': 'エラーが発生しました'
            },
            'reflection_analysis': {
                'result_verbalization': 'エラーが発生しました',
                'prediction_comparison': 'エラーが発生しました',
                'daily_life_connection': 'エラーが発生しました',
                'scientific_understanding': 'エラーが発生しました'
            },
            'language_development': 'エラーが発生しました',
            'support_recommendations': ['システム管理者に連絡してください']
        }
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

# プロンプト管理機能
@app.route('/teacher/prompts')
@require_teacher_auth
def teacher_prompts():
    """プロンプト管理ページ"""
    return render_template('teacher/prompts.html', 
                         units=UNITS,
                         teacher_id=session.get('teacher_id'))

@app.route('/teacher/prompts/<unit>')
@require_teacher_auth
def teacher_prompt_edit(unit):
    """特定単元のプロンプト編集ページ"""
    if unit not in UNITS:
        flash('存在しない単元です', 'error')
        return redirect(url_for('teacher_prompts'))
    
    # 現在のプロンプト内容を読み込み
    try:
        with open(f'prompts/{unit}.md', 'r', encoding='utf-8') as f:
            current_content = f.read()
    except FileNotFoundError:
        # ファイルが存在しない場合はテンプレートをコピー
        try:
            with open('prompts/template.md', 'r', encoding='utf-8') as f:
                current_content = f.read()
        except FileNotFoundError:
            current_content = "# プロンプトテンプレートが見つかりません"
    
    return render_template('teacher/prompt_edit.html',
                         unit=unit,
                         content=current_content,
                         teacher_id=session.get('teacher_id'))

@app.route('/teacher/prompts/<unit>/save', methods=['POST'])
@require_teacher_auth
def teacher_prompt_save(unit):
    """プロンプトを保存"""
    if unit not in UNITS:
        return jsonify({'error': 'Invalid unit'}), 400
    
    content = request.json.get('content', '')
    
    try:
        # プロンプトディレクトリが存在しない場合は作成
        os.makedirs('prompts', exist_ok=True)
        
        # ファイルに保存
        with open(f'prompts/{unit}.md', 'w', encoding='utf-8') as f:
            f.write(content)
        
        return jsonify({'success': True, 'message': 'プロンプトを保存しました'})
    
    except Exception as e:
        return jsonify({'error': f'保存に失敗しました: {str(e)}'}), 500

@app.route('/teacher/prompts/test', methods=['POST'])
@require_teacher_auth
def teacher_prompt_test():
    """プロンプトをテスト"""
    data = request.json
    unit = data.get('unit')
    content = data.get('content', '')
    test_message = data.get('message', 'テストメッセージです')
    
    if unit not in UNITS:
        return jsonify({'error': 'Invalid unit'}), 400
    
    try:
        # 編集中のプロンプトを使用してテスト
        test_prompt = f"""{content}

現在の学習単元: {unit}
課題: テスト用の課題文です
対話回数: 1回目

学習者からのメッセージ: {test_message}

次の質問を普通の文章で1文で書いてください："""
        
        # Gemini APIでテスト
        response = call_gemini_with_retry(test_prompt)
        clean_response = extract_message_from_json_response(response)
        clean_response = remove_markdown_formatting(clean_response)
        
        return jsonify({
            'success': True,
            'response': clean_response,
            'prompt_preview': content[:200] + '...' if len(content) > 200 else content
        })
    
    except Exception as e:
        return jsonify({'error': f'テストに失敗しました: {str(e)}'}), 500

# 指導要領・資料管理機能
@app.route('/teacher/guidelines')
@require_teacher_auth
def teacher_guidelines():
    """指導要領・資料管理ページ"""
    return render_template('teacher/guidelines.html',
                         teacher_id=session.get('teacher_id'))

@app.route('/teacher/guidelines/upload', methods=['POST'])
@require_teacher_auth
def upload_guidelines():
    """指導要領・資料のアップロード"""
    try:
        document_type = request.form.get('document_type')
        title = request.form.get('title')
        description = request.form.get('description', '')
        file = request.files.get('file')
        
        if not file or not file.filename:
            return jsonify({'error': 'ファイルが選択されていません'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'PDFファイルのみ対応しています'}), 400
        
        # ファイルサイズチェック（16MB）
        if len(file.read()) > 16 * 1024 * 1024:
            return jsonify({'error': 'ファイルサイズが16MBを超えています'}), 400
        
        file.seek(0)  # ファイルポインタをリセット
        
        # ディレクトリ作成
        guidelines_dir = 'guidelines'
        os.makedirs(guidelines_dir, exist_ok=True)
        
        # ファイル名を安全にする
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(guidelines_dir, safe_filename)
        
        # ファイル保存
        file.save(filepath)
        
        # PDFからテキスト抽出
        try:
            with open(filepath, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
        except Exception as e:
            content = f"テキスト抽出エラー: {str(e)}"
        
        # インデックスファイルに追加
        index_file = os.path.join(guidelines_dir, 'guidelines_index.json')
        
        # 既存のインデックスを読み込み
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                guidelines_index = json.load(f)
        else:
            guidelines_index = {}
        
        # 新しい文書情報を追加
        doc_id = str(uuid.uuid4())
        guidelines_index[doc_id] = {
            'type': document_type,
            'title': title,
            'description': description,
            'filename': safe_filename,
            'filepath': filepath,
            'content': content[:1000],  # 最初の1000文字のみ保存
            'full_content': content,  # 全文は別途保存
            'uploaded_at': datetime.now().isoformat(),
            'uploaded_by': session.get('teacher_id')
        }
        
        # インデックスファイルに保存
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(guidelines_index, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': '資料をアップロードしました',
            'document_id': doc_id
        })
    
    except Exception as e:
        return jsonify({'error': f'アップロードに失敗しました: {str(e)}'}), 500

@app.route('/teacher/guidelines/list')
@require_teacher_auth
def list_guidelines():
    """アップロード済み資料一覧"""
    try:
        index_file = 'guidelines/guidelines_index.json'
        
        if not os.path.exists(index_file):
            return jsonify({'documents': []})
        
        with open(index_file, 'r', encoding='utf-8') as f:
            guidelines_index = json.load(f)
        
        # 文書一覧を作成（IDを含める）
        documents = []
        for doc_id, doc_info in guidelines_index.items():
            doc_info['id'] = doc_id
            documents.append(doc_info)
        
        # 日付順にソート（新しい順）
        documents.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
        
        return jsonify({'documents': documents})
    
    except Exception as e:
        return jsonify({'error': f'資料一覧の取得に失敗しました: {str(e)}'}), 500

@app.route('/teacher/guidelines/<doc_id>/delete', methods=['DELETE'])
@require_teacher_auth
def delete_guidelines(doc_id):
    """資料の削除"""
    try:
        index_file = 'guidelines/guidelines_index.json'
        
        if not os.path.exists(index_file):
            return jsonify({'error': '資料が見つかりません'}), 404
        
        with open(index_file, 'r', encoding='utf-8') as f:
            guidelines_index = json.load(f)
        
        if doc_id not in guidelines_index:
            return jsonify({'error': '資料が見つかりません'}), 404
        
        # ファイルを削除
        filepath = guidelines_index[doc_id]['filepath']
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # インデックスから削除
        del guidelines_index[doc_id]
        
        # インデックスファイルを更新
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(guidelines_index, f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True, 'message': '資料を削除しました'})
    
    except Exception as e:
        return jsonify({'error': f'削除に失敗しました: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5014)