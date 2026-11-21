# R-NaD Issue #5：v5公開情報修正と選出Outcome比較（2026-11-17）

## 目的

模倣率を上げるだけでなく、本物のAQ-Base-001への完全4勝Match勝率を高めること。旧v3-U4は凍結対照とする。新しい特徴量による教師蒸留と、実際の勝敗で選出を評価する小規模介入実験は、**互いに独立した仮説**として検証する。どちらも学習に成功しただけでは強度改善・Champion条件を満たさない。

## 仮説A：v5公開情報特徴の不足修正

- v4の48D/32Dは、公開相手を同タイプの別種族に変えても特徴が変わらない例があった（`test/rnad-v4-public-species-alias.test.js`）。
- v5では**完全に別のモデルschema** `rnad-three-phase-features-v5`（96D状態/48D行動）を用い、v4の全チャネルを前半に保持する。追加48状態次元は相手公開6種族について、公開種族名に基づく安定カテゴリ符号と公開base stats（速度・攻撃値・耐久値・HP）を各8次元で持つ。追加16行動次元は選出先発ごとの6種への速度・攻撃相性、選出3体の速度/メガ等の公開・自席情報を保持する。戦闘/ドラフトの追加領域は0（v4成分はそのまま）。
- カテゴリ符号は4bitハッシュのため衝突可能。完全な種族識別の数学的保証ではない。種族ごとの基礎能力と併用し、公開6種の同タイプ異種族ケースに対して実特徴差を必須テストにする。v5の効果は**実験仮説**であり教師一致・実戦勝率の達成は未証明。
- 相手の未公開型/技/持ち物/特性・本当の選出はActor入力に含めない。既存`projectRnadObservation`の座席投影を通す。研究モデルのv2/v3/v4重み・checkpointと旧teacher dataは不変更。

## 仮説B：選出Outcomeの対照実験

- `scripts/rnad/evaluate-selection-outcome-rank.mjs` は凍結v3のドラフト・戦闘・後続ラウンドを保ち、**第1ラウンドの最初の選出だけ**凍結v3の行動logit順位0〜K-1へ介入する。
- 各seed・両席について同じ権威Matchと本物AQ-Base-001で完全4勝まで実走。最初の選出時の公的観測/60合法候補support/コード・モデルSHAを検証。各介入先の実勝敗、v5公開state+action特徴を保存。
- `scripts/rnad/learn_selection_outcome.py` は上述した保存済み受理Matchのpublic featuresから線形ロジットで勝敗予測を学習し、**別未使用seed**で第1候補（v3基準）と学習による選択を比較する。教師模倣top1ではなく結果依存ランキングの最小診断。
- 注意：介入後に対戦経過が変化するので観測勝敗は高分散であり、真のQ(s,a)の不偏推定や個別選出の実用強度を証明しない。少数seedでの学習/選択性能は過学習の可能性が高い。`selection_policy_deployed=false`, `strength_established=false` を必ず保持し、この試験だけでモデルを採用しない。

## 事前固定のパイロットと成功判定

- パイロットは既存`rnad-i5-phase-iteration.yml`や旧v4 Utility学習と**重複起動しない**単発manifestで起動。GitHub-hosted Ubuntu CPU・1 writer。
- **同一公開AQ教師、同一train seed、4個の受理済み完全Match/1epoch、同一holdout seed、4Match**でv4/v5教師蒸留を比較し、Top1/10、正規化CE、教師utility regretを記録。この標本は正規の目標達成ゲート（最低256 phase判断）より小さく、テスト用screenのみ。
- v5教師から2完全Match NeuRD継続で権威受理/Optimizer更新/モデルschema/checkpointが通るか確認。
- 同一独立4seed・両席のAQ実Matchで、v3全フェーズとv5選出だけの置換を比較。少数で95%CIが0を含む場合は「強度向上未証明」。
- 結果ベース順位の対照は学習用4seedと評価用別4seed、両席、各3候補を実際のAQ完全Matchで評価。値を保守的なseed単位95%CIで示すが、研究のスクリーニングであり独立Champion holdoutに流用しない。
- 関連ファイルのcontract/unit testを必須CIに含める。重いworkflowはmanifest push時/手動起動のみ。前回実験の途中失敗・結果は破棄せずtraceとSHAを照合して原因診断。
- パイロットを通過しても性能改善を断言しない。次段で **同一教師32/128 Match→holdout32 Match→選出のみ固定戦闘のpaired AQ 256Match screen**を一要因ごとに増やす。最終モデル選出後の完全未使用1024+ seed・両席・対応あり95%CI（事前固定一次仮説）と独立ChatGPTレビューはChampion用に別途必要。

## 状態

この文書は、コード契約と実験計画を記録するもの。研究runで得られた実測と、これから起動するpilotの結果は別々に記録する。Issue #5 / Draft PR #6、main未マージ・Champion未昇格を維持。
