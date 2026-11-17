# 選出A1024とA2048の完全4勝Match実戦比較（研究設計）

## 目的と学習との分離

ユーザーの「選出が賢い＝実際の勝率」を検証するための**別軸**の研究。教師Top1一致率50%以上の新基準、および3フェーズ同一統合policyのリーグ開始条件とは混同しない。

- [学習Run #44058355205](https://github.com/cueikusuta/p-RL1024/actions/runs/44058355205)のA2048が完走し、選定モデルのSHAと未使用教師1024Matchの一致率確認が完了するまで、こちらの実戦比較は起動しない。
- 凍結A1024 `selection-A-1024-e40-s90253.pt` と、事前に再使用開発データで選定された新A2048を、ソース・エンジン・公開Vocabulary同一か検査したうえで比較する。
- Pythonの凍結A推論重みを研究専用JSONに変換。独立SHA256、全tensor形状、60合法候補順序、公開自軍/相手公開種族のみ、Python↔JavaScript全60 logits数値一致を検査。相手未公開の型や教師targetをモデル推論に入れない。
- Matchの両陣営はドラフト`strategy`、戦闘`basic`で固定する。**選出のみ旧A1024対新A2048**にし、独立seed 2490001..2490512 の各seedで両席交換して**1024完全4勝Match**を権威ゲームエンジンでプレイする。
- 各seedの新モデル勝利数0、1、2から独立512Match-seedクラスターを形成し、両席平均勝率および**10,000回cluster bootstrap近似95%両側CI**を計算する。点推定の50%越えや少数Matchだけで新モデルが強いと判定しない。
- 計測できるのは**固定draftとbattleでの新旧Aの直接対戦勝率**のみ。凍結AQ-Baseに勝てることも、他の強い戦闘AIと組み合わせた実力も示さない。リーグ開始・Champion昇格・PRマージは禁止。

## 実装と検証

- 重み変換：`scripts/rnad/export_selection_a_js.py`
- 推論：`scripts/rnad/selection-a-public-infer.mjs`
- 受理済み完全Match比較：`scripts/rnad/evaluate-a-selection-headtohead.mjs`
- セキュリティ/正本テスト：`test/rnad-selection-a-public-infer.test.js`
- 試験：旧A1024で[Python-JS一致Run #44060832634](https://github.com/cueikusuta/p-RL1024/actions/runs/44060832634) **success**、60 logits最大誤差約`1.11e-6`、非公開情報の不干渉・SHA不正拒否成功。
- 本番比較用研究Workflow：`.github/workflows/rnad-i5-selection-a-headtohead.yml`。起動Manifestは別ファイルに事前登録し、A2048学習・新規教師確認完了後に作成する。

新しいゲームAIを本番利用する機能、ルールエンジン変更、実戦policyの変更は含まない。
