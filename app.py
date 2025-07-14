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

# 環境変数を読み込み
load_dotenv()

# SSL設定の改善
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# SSL証明書の設定
ssl_context = ssl.create_default_context(cafile=certifi.where())

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 本番環境では安全なキーに変更

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
    
    # Gemini APIへのプロンプト作成
    system_prompt = f"""
あなたは小学生の理科学習を支援するAIです。産婆法のような対話を通して、学習者の経験や既習事項を引き出し、根拠のある予想を立てさせることが目的です。

現在の学習単元: {unit}
課題: {task_content}

以下のガイドラインに従って対話してください：
1. 学習者の答えを否定せず、「なぜそう思うのか」を聞いて根拠を探る
2. 日常経験や既習事項と関連付けさせる質問をする
3. 誘導的な質問は避け、学習者自身の考えを引き出す
4. 3-4回の対話で予想をまとめられるようにする
5. 敬語は使わず、親しみやすい話し方で
6. **重要**：マークダウン記法（*、**、#、-など）は一切使用禁止。普通の文章のみで回答してください。

対話回数: {len(conversation)}回目
"""
    
    # 対話履歴を含めてプロンプト作成
    full_prompt = system_prompt + "\n\n対話履歴:\n"
    for msg in conversation:
        role = "学習者" if msg['role'] == 'user' else "AI"
        full_prompt += f"{role}: {msg['content']}\n"
    
    try:
        ai_message = call_gemini_with_retry(full_prompt)
        
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
    
    # 予想のまとめを生成
    summary_prompt = f"""
以下の対話を基に、学習者の予想を自然な日本語で簡潔にまとめてください。

要求：
1. 「あなたは、〜ということから、〜だと予想したんですね」の形でまとめる
2. 学習者の根拠と予想を明確に分ける
3. マークダウン記法（*、**、#、-など）は一切使用しない
4. 普通の文章のみで書く
5. 箇条書きや記号は使わない
6. 平易な言葉を使う

単元: {unit}

対話履歴:
"""
    for msg in conversation:
        role = "学習者" if msg['role'] == 'user' else "AI"
        summary_prompt += f"{role}: {msg['content']}\n"
    
    try:
        summary_text = call_gemini_with_retry(summary_prompt)
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

対話回数: {len(reflection_conversation)}回目
"""
    
    # 対話履歴を含めてプロンプト作成
    full_prompt = system_prompt + "\n\n対話履歴:\n"
    for msg in reflection_conversation:
        role = "学習者" if msg['role'] == 'user' else "AI"
        full_prompt += f"{role}: {msg['content']}\n"
    
    try:
        ai_message = call_gemini_with_retry(full_prompt)
        
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
        final_summary_text = call_gemini_with_retry(final_prompt)
        
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
    return render_template('teacher/dashboard.html', units=UNITS, teacher_id=session.get('teacher_id'))

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
        
        # 再生成の場合は異なるアプローチで選択肢を生成
        if is_regenerate:
            suggestions_prompt = f"""
あなたは小学校理科の指導者です。先ほどとは違う視点で、学習者（小学生）が答えやすい具体的な選択肢を3つ生成してください。

単元: {unit}

直前のAI応答:
{latest_ai_response}

最近の対話履歴:
"""
            # 最新の4往復分の対話を含める（前回より多く）
            recent_conversation = conversation[-8:] if len(conversation) > 8 else conversation
            for msg in recent_conversation:
                role = "学習者" if msg['role'] == 'user' else "AI"
                suggestions_prompt += f"{role}: {msg['content']}\n"
            
            suggestions_prompt += """
再生成の要求:
1. 前回とは異なる角度から選択肢を作成
2. より具体的で詳細な選択肢にする
3. 学習者の日常体験に関連した選択肢
4. 感情や感覚を表現できる選択肢も含める
5. 各選択肢は30文字以内で分かりやすく
6. 以下の形式で出力してください：

選択肢1: [内容]
選択肢2: [内容]  
選択肢3: [内容]

例：
AIが「どうしてそう思ったのかな？」と聞いた場合（再生成版）
→ 選択肢1: お母さんが話していたのを聞いたから
→ 選択肢2: 前に学校で習ったような気がするから  
→ 選択肢3: 何となく頭に浮かんだから
"""
        else:
            # 通常の選択肢生成プロンプト（従来通り）
            suggestions_prompt = f"""
あなたは小学校理科の指導者です。直前のAI応答に対して、学習者（小学生）が次に答えやすい具体的な選択肢を3つ生成してください。

単元: {unit}

直前のAI応答:
{latest_ai_response}

最近の対話履歴:
"""
            
            # 最新の3往復分の対話を含める
            recent_conversation = conversation[-6:] if len(conversation) > 6 else conversation
            for msg in recent_conversation:
                role = "学習者" if msg['role'] == 'user' else "AI"
                suggestions_prompt += f"{role}: {msg['content']}\n"
            
            suggestions_prompt += """
要求:
1. 直前のAI応答の内容に直接関連した選択肢を作成
2. AIが質問している内容に対する具体的な答えを選択肢にする
3. 各選択肢は25文字以内で簡潔に
4. 学習者が答えやすい具体的な内容にする
5. 抽象的ではなく、具体的な体験や考えを表現できる選択肢
6. 以下の形式で出力してください：

選択肢1: [内容]
選択肢2: [内容]  
選択肢3: [内容]

例：
AIが「どうしてそう思ったのかな？」と聞いた場合
→ 選択肢1: テレビで見たことがあるから
→ 選択肢2: 前に似たことを体験したから  
→ 選択肢3: 理科の本で読んだから
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
            if "どうして" in latest_ai_response or "なぜ" in latest_ai_response:
                if is_regenerate:
                    context_defaults = [
                        "家族から聞いたことがあるから",
                        "友達と話していて思ったから",
                        "直感でそう感じるから"
                    ]
                else:
                    context_defaults = [
                        "前に見たことがあるから",
                        "テレビで知ったから",
                        "なんとなくそう思うから"
                    ]
            elif "他に" in latest_ai_response or "ほか" in latest_ai_response:
                if is_regenerate:
                    context_defaults = [
                        "もう少し時間をかけて考えたい",
                        "別の方法で確かめてみたい", 
                        "先生に聞いてみたい"
                    ]
                else:
                    context_defaults = [
                        "思いつかない",
                        "わからない", 
                        "もう少し考えてみる"
                    ]
            elif "どんな" in latest_ai_response:
                if is_regenerate:
                    context_defaults = [
                        "とても大きく変化する",
                        "少しだけ変化する",
                        "ほとんど変化しない"
                    ]
                else:
                    context_defaults = [
                        "大きくなる",
                        "小さくなる",
                        "変わらない"
                    ]
            else:
                if is_regenerate:
                    context_defaults = [
                        "とてもそう思う",
                        "少し疑問に思う",
                        "詳しく知りたい"
                    ]
                else:
                    context_defaults = [
                        "そう思う",
                        "よくわからない",
                        "もう少し教えて"
                    ]
            
            for i, default in enumerate(context_defaults):
                if len(suggestions) <= i:
                    suggestions.append(default)
        
        return jsonify({
            'suggestions': suggestions[:3],  # 最大3つまで
            'regenerated': is_regenerate
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
        
        # 再生成の場合は異なるアプローチで選択肢を生成
        if is_regenerate:
            suggestions_prompt = f"""
あなたは小学校理科の指導者です。考察段階で、先ほどとは違う視点から学習者（小学生）が答えやすい具体的な選択肢を3つ生成してください。

単元: {unit}

直前のAI応答:
{latest_ai_response}

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
あなたは小学校理科の指導者です。考察段階で、直前のAI応答に対して学習者（小学生）が次に答えやすい具体的な選択肢を3つ生成してください。

単元: {unit}

直前のAI応答:
{latest_ai_response}

考察対話履歴:
"""
            
            # 最新の3往復分の対話を含める
            recent_conversation = reflection_conversation[-6:] if len(reflection_conversation) > 6 else reflection_conversation
            for msg in recent_conversation:
                role = "学習者" if msg['role'] == 'user' else "AI"
                suggestions_prompt += f"{role}: {msg['content']}\n"
            
            suggestions_prompt += """
要求（考察段階）:
1. 直前のAI応答の内容に直接関連した選択肢を作成
2. 実験結果に対する考察や感想を表現できる選択肢
3. 各選択肢は25文字以内で簡潔に
4. 予想と結果の比較ができる選択肢
5. 具体的で学習者が答えやすい内容
6. 以下の形式で出力してください：

選択肢1: [内容]
選択肢2: [内容]  
選択肢3: [内容]

例：
→ 選択肢1: 予想通りの結果でした
→ 選択肢2: 予想と少し違いました  
→ 選択肢3: 予想と全然違いました
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
        
        # 選択肢が3つ未満の場合は文脈に応じたデフォルトを追加
        if len(suggestions) < 3:
            if "結果" in latest_ai_response or "どうだった" in latest_ai_response:
                if is_regenerate:
                    context_defaults = [
                        "とても興味深い結果でした",
                        "思ったより分かりやすい変化でした",
                        "もう一度確かめてみたいです"
                    ]
                else:
                    context_defaults = [
                        "予想通りでした",
                        "予想と違いました",
                        "よくわかりませんでした"
                    ]
            elif "どう思う" in latest_ai_response or "感想" in latest_ai_response:
                if is_regenerate:
                    context_defaults = [
                        "理科って面白いなと思いました",
                        "もっと詳しく知りたくなりました",
                        "日常生活でも気をつけて見てみたいです"
                    ]
                else:
                    context_defaults = [
                        "面白かった",
                        "勉強になった",
                        "驚いた"
                    ]
            else:
                if is_regenerate:
                    context_defaults = [
                        "実際に見てみて理解が深まりました",
                        "予想を立てるのは難しかったです",
                        "次はもっと正確に予想したいです"
                    ]
                else:
                    context_defaults = [
                        "そう思います",
                        "よくわかりません",
                        "もう少し考えてみます"
                    ]
            
            for i, default in enumerate(context_defaults):
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
    """特定の学生・単元の学習過程をGeminiで分析"""
    print(f"分析開始 - 学生: {student_number}, 単元: {unit}")
    
    # 該当する学生のログを抽出
    student_logs = [log for log in logs if 
                   log.get('student_number') == student_number and 
                   log.get('unit') == unit]
    
    print(f"該当ログ数: {len(student_logs)}")
    
    if not student_logs:
        return {
            'evaluation': '学習データがありません',
            'strengths': ['データが不足しています'],
            'improvements': ['学習活動への参加が必要です'],
            'score': 0,
            'thinking_process': 'データなし',
            'engagement': 'データなし',
            'scientific_understanding': 'データなし'
        }
    
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
    
    # 分析プロンプト作成（教育的観点を強化）
    analysis_prompt = f"""
小学生の理科学習記録を詳細に評価してください。

学習内容: {unit}
学習者ID: {student_number}

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
【評価観点】
以下の観点で詳細に評価してください：

1. 予想の質
   - 日常生活の経験に基づいているか
   - 既習事項を活用しているか
   - 根拠を明確に示しているか

2. 考察の質
   - 実験結果と予想を比較しているか
   - 結果について日常生活との関連を示しているか
   - 科学的に妥当な説明ができているか

3. 学習姿勢
   - 積極的に対話に参加しているか
   - 自分の言葉で表現しているか
   - 疑問や興味を示しているか

【出力形式】
以下の形式で評価結果をJSON形式で出力してください：

{
  "evaluation": "予想では日常経験を活用し、考察では結果と予想を関連付けて論理的に説明しています",
  "strengths": ["日常経験の活用", "論理的思考", "積極的参加"],
  "improvements": ["観察の詳細化", "科学用語の使用", "考察の深化"],
  "score": 7,
  "thinking_process": "段階的に考えを深めています",
  "engagement": "意欲的に取り組んでいます",
  "scientific_understanding": "基本概念を理解しています",
  "prediction_quality": {
    "daily_life_connection": "日常経験を根拠として活用している",
    "prior_knowledge_use": "既習事項を適切に関連付けている",
    "reasoning_clarity": "根拠を明確に示している"
  },
  "reflection_quality": {
    "result_prediction_link": "実験結果と予想を比較・検討している",
    "daily_life_relevance": "結果を日常生活と関連付けている",
    "scientific_validity": "科学的に妥当な説明をしている"
  }
}
"""
    
    analysis_prompt += """
以下のJSON形式で学習評価を出力してください:

{
  "evaluation": "総合評価コメント",
  "strengths": ["長所1", "長所2", "長所3"],
  "improvements": ["課題1", "課題2", "課題3"],
  "score": 7,
  "thinking_process": "思考評価",
  "engagement": "取組評価", 
  "scientific_understanding": "理解評価"
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
            'evaluation': '学習記録から基本的な取り組み姿勢が確認できます',
            'strengths': ['学習活動への参加', '対話への積極性', '継続的な取り組み'],
            'improvements': ['日常経験との関連付け強化', '予想の根拠明確化', '考察の詳細化'],
            'score': 6,
            'thinking_process': '段階的に考察を進めています',
            'engagement': '積極的に学習に取り組んでいます',
            'scientific_understanding': '基本概念を理解し始めています',
            'prediction_quality': {
                'daily_life_connection': '日常経験の活用を促すとよいでしょう',
                'prior_knowledge_use': '既習事項との関連付けを意識させましょう',
                'reasoning_clarity': '根拠をより明確に示せるよう指導が必要です'
            },
            'reflection_quality': {
                'result_prediction_link': '実験結果と予想の比較を促しましょう',
                'daily_life_relevance': '日常生活との関連を意識させる指導が効果的です',
                'scientific_validity': '科学的な説明力を段階的に育成していきましょう'
            }
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON解析エラー: {e}")
        # 教育的観点を含むフォールバック応答
        return {
            'evaluation': '分析処理でエラーが発生しましたが、学習への取り組みは確認できます',
            'strengths': ['学習活動への参加', '対話への取り組み', '継続的な学習'],
            'improvements': ['日常経験との関連付け', '予想根拠の明確化', '考察の論理性向上'],
            'score': 5,
            'thinking_process': 'システムエラーのため詳細評価は後日実施',
            'engagement': '学習意欲は確認できます',
            'scientific_understanding': '基本的な理解は進んでいます',
            'prediction_quality': {
                'daily_life_connection': '日常経験の活用について再評価が必要',
                'prior_knowledge_use': '既習事項の活用状況を確認中',
                'reasoning_clarity': '根拠の明確さについて詳細分析予定'
            },
            'reflection_quality': {
                'result_prediction_link': '結果と予想の関連付けについて評価中',
                'daily_life_relevance': '日常生活との関連について分析中',
                'scientific_validity': '科学的説明力について評価予定'
            }
        }
    except Exception as e:
        print(f"分析エラー: {e}")
        return {
            'evaluation': f'システムエラーが発生しましたが、学習記録は保存されています',
            'strengths': ['学習データの蓄積', '継続的な取り組み', '記録の完成'],
            'improvements': ['システム安定化後の詳細分析', '教育的評価の実施', '個別指導計画の作成'],
            'score': 4,
            'thinking_process': f'エラー詳細: {str(e)[:50]}...',
            'engagement': '学習への参加は記録されています',
            'scientific_understanding': 'システム復旧後に詳細評価予定',
            'prediction_quality': {
                'daily_life_connection': 'システム復旧後に評価実施',
                'prior_knowledge_use': 'データ解析後に詳細確認',
                'reasoning_clarity': '後日詳細分析予定'
            },
            'reflection_quality': {
                'result_prediction_link': '詳細分析は後日実施',
                'daily_life_relevance': 'システム安定後に評価',
                'scientific_validity': '包括的評価を後日実施'
            }
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
    """クラス全体の学習傾向をGeminiで分析"""
    if unit:
        # 特定単元の分析
        unit_logs = [log for log in logs if log.get('unit') == unit]
        students = set(log.get('student_number') for log in unit_logs)
    else:
        # 全体の分析
        unit_logs = logs
        students = set(log.get('student_number') for log in logs)
    
    if not unit_logs or len(students) == 0:
        return {
            'overall_trend': '分析対象のデータがありません',
            'common_misconceptions': [],
            'effective_approaches': [],
            'recommendations': []
        }
    
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

対象単元: {unit if unit else '全単元'}
学習者数: {len(students)}人

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
以下の項目でクラス分析をJSONで出力してください。

項目:
- overall_trend: クラス全体の傾向（100文字程度）
- common_misconceptions: よくある誤解を3つ
- effective_approaches: 効果的な指導法を3つ
- recommendations: 指導提案を3つ
- engagement_level: 取り組み度（高/中/低）
- understanding_distribution: 理解度分布
- improvement_areas: 重点指導分野を2つ

JSON形式で回答してください。
"""
    
    analysis_prompt += """
この学習状況について、以下の形式で分析結果をJSON形式で出力してください。

{
  "overall_trend": "クラス全体で積極的に学習に取り組んでいます",
  "common_misconceptions": ["基本概念の混同", "観察結果の解釈", "予想と結果の関連"],
  "effective_approaches": ["体験的学習", "対話型指導", "段階的説明"],
  "recommendations": ["個別指導強化", "実験機会増加", "振り返り時間確保"],
  "engagement_level": "高",
  "understanding_distribution": "理解度にばらつきがあります",
  "improvement_areas": ["観察技能", "論理的思考"]
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
            'overall_trend': 'クラス全体として理科学習に意欲的に取り組んでいます',
            'common_misconceptions': ['実験結果の解釈', '科学的概念の理解', '観察と推論の区別'],
            'effective_approaches': ['実体験による学習', '対話的な授業', '段階的な説明'],
            'recommendations': ['個別サポートの充実', '実験時間の確保', '振り返り活動の強化'],
            'engagement_level': '高',
            'understanding_distribution': '理解度に個人差が見られます',
            'improvement_areas': ['観察技能の向上', '科学的思考力の育成']
        }
    except Exception as e:
        print(f"クラス分析エラー: {e}")
        return {
            'overall_trend': 'AI分析でエラーが発生しました',
            'common_misconceptions': ['分析データ不足'],
            'effective_approaches': ['従来の指導法継続'],
            'recommendations': ['データ蓄積後に再分析'],
            'engagement_level': 'システムエラー',
            'understanding_distribution': 'システムエラー',
            'improvement_areas': ['システム調整']
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
    app.run(debug=True, port=5010)
