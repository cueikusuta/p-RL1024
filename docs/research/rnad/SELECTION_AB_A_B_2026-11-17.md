# I5 選出研究A/B（教師分解スコア／詳細公開情報Attention）

対象：[Issue #5](https://github.com/cueikusuta/p-RL1024/issues/5)／既存[Draft PR #6](https://github.com/cueikusuta/p-RL1024/pull/6)。既存の`structured-hard`・凍結v5・ゲームエンジン・相手非公開情報の扱い・リーグゲートを変更せず、新たな探索用学習器2系統を比較する。

## 前提となる教師構造

教師 `src/ai/team-assessment.js::rankStrategySelections` は相手の公開6種族に対する見積もりと、自己6体の各セットを使って、合法候補ごとに `score = breakdown.core + breakdown.lead` を返す。`core` は選択した**順不同3体（20通り）**に、`lead` は自分の**先発位置（6通り）**に依存する。120個の先発・控え順あり候補を`scripts/rnad/selection-oracle.mjs`で権威60候補に投影しても、両成分は一意に復元できる。今回のトレース輸出では、**各60候補すべての総utilityと20個のcore・6個のleadが数値的に一致することを必須**とし、一致しなければ学習を停止する。

## 新しい入力

`scripts/rnad/export-selection-rich.mjs`は`runRnadAcceptanceSmoke`を通じて**権威が受理した完全4勝Matchだけ**から局面を記録する。教師が観測可能な自軍6体の`speciesId`、`abilityId`、`itemId`、最大4つの`moveId`、タイプ・能力値・コスト・HP・Mega公開情報、および相手の**公開6種族**の種族・タイプ・基礎能力値を保存する。V5の96D state/60×48D actionは**凍結ベースラインとの同一局面比較だけ**に保持する。

固定カタログから作った語彙のサイズとSHA-256を保存。6つの自軍Token(9カテゴリ＋16数値)と6つの相手Token(3カテゴリ＋6数値)をモデルに渡す。**敵の未選択3体、技、持ち物、特性、真の型、対戦の内部状態は保存・使用しない**。教師スコア、分解成分、正解選出は別`targets`に格納され、学習損失・評価だけに使用する。モデルforwardは`actor`テンソルのみを受け取り、教師の意思決定計算を実行しない。受理済み教師と以前の96D/48D/utility/choiceが1 Matchで完全一致する回帰テストを追加した。

## A：教師のcore/leadを個別に回帰

`DecomposedTeacher`モデルは公開自軍／敵Tokenから学習埋め込み、6×6の学習可能なペア相互作用、20個の順不同3体スコアと6個の先発スコアを算出する。教師の最終点数を唯一の教師信号にせず、**core 20個・lead 6個をそれぞれ個別にMSEで教師あり学習**し、合成した60 logitsにutility回帰とTop1 CEも課す。教師utilityや`rankStrategySelections`は推論時には入力・実行しない。モデルは元教師評価式を固定して写すわけではなく、実データから分解関数を学習する。

## B：詳細Token×自己Attention

`DetailedAttention`モデルは公開敵6種族＋自軍6体の**12Token**をTransformer Encoder 2層、幅64、4 headで処理する。各候補について先発・3体平均・交互作用・全体Contextを合成して合法60候補を採点する。訓練は教師Top1 CE主体＋弱い全60候補utility分布のlistwise損失。20/6の分解ヘッドは持たず、Aとは異なる情報統合の仮説をテストする。

## 実験・統計

- 学習：既存の受理済み完全教師256Matchのseed`2471001..2471256`を**同一エンジン・同じteacherで再輸出**。epoch=10、固定学習seed、各armで同数の受理選出判断を利用。
- 開発評価：過去に使用済みの独立128Match seed`2472001..2472128`を再輸出。これは**探索用**であり、正式な未使用性能ではない。A/B両方と元`structured-hard`を同じ局面・同じseedで比較し、Top1、Top10、3体一致、教師utility regret、対応ありseed-cluster差を記録。
- 開発用128Match上で事前固定ルール`Top1最大→regret最小→名前順`によりA/Bから**1モデルだけ選択**。旧structured-hardよりTop1が少なくとも0.1ポイント高い場合に限り、追加の新規未使用256完全教師Match seed`2477001..2477256`を生成・評価する。そうでなければ未使用Matchを消費せず、両方式が現状では不十分と研究報告する。
- 新規未使用確認では**候補を再選択・再学習せず**、完全Match seedを統計の独立単位として同一seedの`structured-hard`と対応ありTop1差／5000回bootstrap近似95%両側信頼区間を測る。点推定だけや少数Matchで優劣を確定しない。
- これは教師の**訪問局面でのTop1一致率**であり、実戦勝率、他の自己対戦局面の性能、教師の上回り、3フェーズ共通方策の昇格を意味しない。正の95%区間下限が確認されない改善は「確認済み」と呼ばない。

CI：`.github/workflows/rnad-i5-selection-ab-preflight.yml`・`.github/workflows/rnad-i5-selection-ab.yml`、コード`scripts/rnad/selection_ab.py`・`selection_ab_confirm.py`、検証`test/rnad-selection-rich.test.js`と`test/rnad_selection_ab_test.py`。結果はGitHub Actions Artifactへ保存する。

## 不変の安全ゲート

リーグ学習開始条件は、**同一統合方策**の新規未使用受理済み教師MatchでDraft Top1>50%、Selection Top1≥80%、Battle Top1≥80%と、3相Bonferroni補正80%片側下限を全て満たすこと。A/Bは選出の研究専用で、判定の代替ではない。既存PR #6はDraft・未マージ、リーグ開始・Champion昇格は行わず、独立AIレビューは実装と別Chatセッションで行う。
