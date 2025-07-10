<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

このプロジェクトは、小学校理科の予想と考察を支援するFlask Webアプリケーションです。

## プロジェクトの特徴

- Gemini API (gemini-2.0-flash-exp) を使用したAI対話システム
- 産婆法的アプローチによる学習者の既習事項・経験の引き出し
- カード形式のUI（出席番号選択、単元選択）
- チャット形式の対話インターフェース
- レスポンシブデザイン（Bootstrap 5使用）

## コーディング方針

- Pythonコードは適切なエラーハンドリングを含める
- HTMLテンプレートはJinja2テンプレートエンジンを使用
- CSSはカスタムプロパティ（CSS変数）を活用
- JavaScriptは非同期処理（fetch API）を適切に実装
- セッション管理を活用したユーザー状態の保持
- AIとの対話では教育的配慮を重視

## ファイル構成

- `app.py`: Flaskアプリケーション（メインロジック）
- `templates/`: Jinja2テンプレート
- `static/css/style.css`: カスタムスタイルシート
- `tasks/`: 単元別課題文ファイル
- `.env`: 環境変数（Gemini APIキー）
