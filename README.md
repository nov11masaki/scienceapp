# 小学校理科学習支援システム

小学生の理科学習における予想と考察を支援するFlask Webアプリケーションです。AI（Gemini API）との対話を通じて、産婆法的アプローチにより学習者の既習事項・経験を引き出し、根拠のある予想・考察を作成支援します。

## 🎯 主な機能

### 👨‍🎓 学習者用機能
- **出席番号選択** - 1〜30番からの選択（カード形式UI）
- **学習単元選択** - 9つの理科単元から選択
- **AI対話による予想** - Gemini APIを使用した産婆法的対話
- **AI選択肢機能** - 詰まった時の入力支援（使用状況を記録）
- **セッション復帰** - 中断した学習の続きから再開
- **実験実施確認** - 実験実行の確認
- **AI対話による考察** - 結果と予想の比較・まとめ

### 👩‍🏫 教員用機能
- **認証システム** - 教員専用ログイン（ID: teacher, Pass: science2025）
- **指導案管理** - 単元別PDFファイルのアップロード・管理機能
  - PDF形式の指導案をアップロード可能
  - 指導案内容の自動テキスト抽出
  - 単元別の指導案設定状況確認
- **学習ログ閲覧** - 学生の対話履歴と学習過程の確認
- **学生別データ表示** - 出席番号・単元・日付別のフィルタリング
- **詳細な学習記録** - タイムライン形式での学習過程表示
- **AI学習分析** - Geminiによる自動分析と評価（指導案を考慮）
  - 個別学生分析（10段階評価、優れている点、改善点など）
  - クラス全体分析（学習傾向、よくある誤解、指導提案など）
  - 指導案に基づく学習目標達成度評価
  - 授業計画との整合性分析
- **選択肢使用状況確認** - 学習者のAI選択肢依存度表示
- **データエクスポート** - 学習データのCSV出力

## 📚 対応学習単元

- 空気の温度と体積
- 水の温度と体積
- 金属の温度と体積
- 金属のあたたまり方
- 水のあたたまり方
- 空気のあたたまり方
- ふっとうした時の泡の正体
- 水を熱し続けた時の温度と様子
- 冷やした時の水の温度と様子

## 🛠️ 技術仕様

- **バックエンド**: Flask (Python)
- **AI**: Google Gemini API (gemini-2.0-flash-exp)
- **フロントエンド**: HTML5, CSS3, JavaScript, Bootstrap 5
- **データ保存**: JSON形式のログファイル（logs/learning_log_YYYYMMDD.json）
- **セッション管理**: Flaskセッション
- **PDF処理**: PyPDF2（指導案テキスト抽出）
- **ファイル管理**: Werkzeug（セキュアなファイルアップロード）

## 🚀 セットアップ・起動方法

### 1. 必要なパッケージのインストール
```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定
`.env`ファイルを作成して、Gemini APIキーを設定：
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. アプリケーションの起動
```bash
python app.py
```

### 4. アクセス
- 学習者用: http://localhost:5011
- 教員用: http://localhost:5011/teacher

## 💡 教育的配慮

### 産婆法的アプローチ
- AIが答えを誘導せず、学習者の経験を引き出す
- 「なぜそう思うのか」を問いかけて根拠を探る
- 日常経験や既習事項との関連付けを促進

### 段階的学習プロセス
1. **予想段階**: 既習事項・経験に基づいた根拠のある予想
2. **実験段階**: 実際の観察・実験
3. **考察段階**: 結果と予想の比較、日常生活との関連付け

### AI支援機能
- **選択肢機能**: 詰まった時の入力支援（使用状況を記録し教員が確認可能）
- **プロンプト最適化**: マークダウン記法や特殊記号の自動除去
- **エラーハンドリング**: API呼び出し失敗時の適切な処理

## 🎨 UI/UX特徴

- **直感的操作**: カード形式の選択とチャット形式の対話
- **レスポンシブデザイン**: スマートフォン・タブレット対応
- **現代的なデザイン**: Bootstrap 5とカスタムCSSによる美しいUI
- **アクセシビリティ**: 小学生にも使いやすいインターフェース
- **教員ダッシュボード**: 学習分析結果の視覚的表示

## 📊 学習分析機能（NEW）

### AI自動分析
- **個別分析** (`analyze_student_learning`関数)
  - 学習者一人ひとりの学習過程をGeminiが自動評価
  - 10段階評価システム
  - 優れている点・改善点の具体的指摘
  - 思考過程・学習意欲・科学的理解の詳細評価
  - **予想の質**: 日常経験の活用、既習事項の関連付け、根拠の明確性
  - **考察の質**: 結果と予想の比較、日常生活との関連、科学的妥当性

- **クラス全体分析** (`analyze_class_trends`関数)
  - 全体的な学習傾向の把握
  - よくある誤解の特定
  - 効果的だった指導法の抽出
  - 今後の指導提案
  - 学習意欲レベルの評価
  - 理解度分布の分析

### 指導案連携分析
- 指導案が設定された単元では、AI分析時に指導案の内容が自動考慮
- 学習目標の達成度評価
- 指導計画との整合性分析
- 次回授業への具体的提案

### 学習ログ管理
- 日付別ログファイル自動生成
- 学習者・単元・日付での柔軟なフィルタリング
- セッション情報の保存・復帰機能
- 詳細な学習過程の記録（タイムライン形式表示）

## 📁 ディレクトリ構造

```
science-sup/
├── app.py                      # メインアプリケーション
├── requirements.txt            # 依存関係
├── .env                       # 環境変数（要設定）
├── logs/                      # 学習ログ（自動生成）
│   └── learning_log_YYYYMMDD.json
├── lesson_plans/              # 指導案ファイル（自動生成）
│   ├── lesson_plans_index.json
│   └── *.pdf                  # アップロードされた指導案PDF
├── templates/                 # HTMLテンプレート
│   ├── base.html
│   ├── index.html
│   ├── select_number.html
│   ├── select_unit.html
│   ├── prediction.html
│   ├── experiment.html
│   ├── reflection.html
│   ├── resume_session.html
│   └── teacher/               # 教員用テンプレート
│       ├── login.html
│       ├── dashboard.html
│       ├── logs.html
│       ├── lesson_plans.html
│       ├── student_detail.html
│       ├── analysis.html
│       └── student_analysis.html
├── static/                    # 静的ファイル
│   └── css/
│       └── style.css
└── tasks/                     # 単元別課題文
    ├── 空気の温度と体積.txt
    ├── 水の温度と体積.txt
    └── ...（9つの単元）
```

## 🔐 認証情報

### 教員ログイン
- **ID**: teacher
- **パスワード**: science2025

### 管理者ログイン（予備）
- **ID**: admin
- **パスワード**: admin123

## 📈 データ形式

### 学習ログ構造
```json
{
  "timestamp": "2025-07-15T10:30:00.000000",
  "student_number": "1",
  "unit": "空気の温度と体積",
  "log_type": "prediction_chat",
  "data": {
    "user_message": "学習者の入力",
    "ai_response": "AIの応答",
    "conversation_count": 1,
    "used_suggestion": false,
    "suggestion_index": null
  }
}
```

### ログタイプ
- `prediction_chat`: 予想段階の対話
- `prediction_summary`: 予想のまとめ
- `reflection_chat`: 考察段階の対話
- `final_summary`: 最終考察

### 分析結果データ構造
```json
{
  "evaluation": "総合評価コメント",
  "strengths": ["長所1", "長所2", "長所3"],
  "improvements": ["改善点1", "改善点2", "改善点3"],
  "score": 7,
  "thinking_process": "思考過程の評価",
  "engagement": "学習意欲の評価",
  "scientific_understanding": "科学的理解の評価",
  "prediction_quality": {
    "daily_life_connection": "日常経験の活用状況",
    "prior_knowledge_use": "既習事項の活用状況",
    "reasoning_clarity": "根拠の明確性"
  },
  "reflection_quality": {
    "result_prediction_link": "結果と予想の関連付け",
    "daily_life_relevance": "日常生活との関連性",
    "scientific_validity": "科学的妥当性"
  }
}
```

## 🚨 注意事項

- **APIキー管理**: `.env`ファイルのGemini APIキーは適切に管理してください
- **ポート設定**: デフォルトポートは5011です（app.py最下部で変更可能）
- **データ保存**: 学習ログは`logs/`ディレクトリに日付別で自動保存されます
- **指導案管理**: 指導案PDFは`lesson_plans/`ディレクトリに保存され、同一単元の新規アップロードで上書きされます
- **ファイルサイズ**: PDFアップロードは最大16MBまで対応
- **セッション管理**: Flaskセッションを使用（本番環境では適切なSecret Keyに変更）

## 📖 指導案機能の使用方法

### 指導案のアップロード
1. 教員ログイン後、ダッシュボードから「指導案管理」をクリック
2. 単元を選択し、PDFファイルをアップロード
3. アップロード後、自動でテキスト抽出が実行されます

### 分析への活用
- 指導案が設定された単元では、AI分析時に指導案の内容が考慮されます
- 学習目標の達成度、指導計画との整合性を含む詳細な分析が可能
- 次回授業への具体的な提案も含まれた分析結果が提供されます

## 🎓 開発・カスタマイズ

- **単元追加**: `tasks/`ディレクトリに新しい課題文ファイルを追加
- **スタイル変更**: `static/css/style.css`でデザインをカスタマイズ
- **AI応答調整**: app.py内のプロンプトを編集してAIの応答を調整
- **分析機能拡張**: `analyze_student_learning`関数や`analyze_class_trends`関数をカスタマイズ

## 🆕 最新アップデート（2025年7月15日）

- **分析機能の大幅強化**: GitHubリポジトリの高度な分析機能を統合
- **指導案連携分析**: 指導案を考慮した教育的AI分析を実装
- **詳細評価システム**: 予想の質・考察の質を多角的に評価
- **10段階評価**: 客観的な学習評価システムを導入
- **クラス傾向分析**: 全体的な学習状況の把握とトレンド分析
- **UI/UX改善**: Bootstrap 5によるレスポンシブデザインの実装
- **教員ダッシュボード**: 分析結果の視覚的表示機能を充実

## 🔗 関連リンク

- **GitHub リポジトリ**: https://github.com/nov11masaki/scienceapp
- **Gemini API**: https://ai.google.dev/
- **Flask ドキュメント**: https://flask.palletsprojects.com/
- **Bootstrap 5**: https://getbootstrap.com/

---

このシステムは、小学校理科教育における「予想→実験→考察」の学習サイクルを、AI技術を活用して効果的に支援することを目的としています。産婆法的アプローチにより、学習者の主体的な学びを促進し、教員には詳細な学習分析機能を提供します。
