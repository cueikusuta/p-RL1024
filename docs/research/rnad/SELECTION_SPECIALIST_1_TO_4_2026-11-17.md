# Issue #5: 選出模倣率改善 1～4（2026-11-17）

[Issue #5](https://github.com/cueikusuta/p-RL1024/issues/5)／[既存Draft PR #6](https://github.com/cueikusuta/p-RL1024/pull/6)、新研究用の独立モデル。凍結v3・旧v4/v5重み・Champion・既存強度評価には影響しない。

## 目的、達成条件

ユーザー指定のリーグ学習開始条件は **同一統合方策の未使用教師MatchでドラフトTop1 >50%、選出Top1 >=80%、戦闘Top1 >=80%**。同じsource/model/teacher/engine・受理済み完全4勝Match・独立seedクラスター・familywise 80%下限が必要。個別の選出専用研究モデルの評価だけではリーグを開始しない。

既存の学習済みv5 Selection重点モデルは、[Run #44032488678](https://github.com/cueikusuta/p-RL1024/actions/runs/44032488678)で新規256学習／独立128教師Match評価（1,476選出判断）でTop1 **26.49%**、Top10 **79.13%**。以前の18.23%は別モデル／別holdoutであり、直接差分は不正確。

## 1. 選出専用ネットワーク

`scripts/rnad/train_selection_specialist.py` の `SelectionSpecialist` は96D公開・自席stateと60×48Dの合法選出行動のみを入力する。ドラフト・戦闘とモデル重みを共有しない。モデルは別schema `rnad-selection-specialist-96x48-v1`、凍結既存Actorを変更しない。出力は権威合法60候補のscoreのみで、学生の推論に教師utility、相手非公開セット、勝敗ラベルを渡さない。

`scripts/rnad/selection-specialist-infer.mjs` は出力JSON重みのモデルschema・テンソルshape・finite数値・候補順序バッファをfail-closed検証したうえでPyTorchに対応するscoreを推論する。両ネットワーク（flat / structured）のPython-JS logit数値一致・改ざん不受理をテストする。

## 2. 3体＋先発の構造化

本物の `legalPureRlSelections` の合法順序（6先発×残り5体から2控え=60）を使用し、同じunordered3体の組合せを **20通り×3先発** に正準化。`structured` armは
- Team value: 同じ3体を3先発にした各行の特徴量をpoolする
- Lead value: 同一先発を10控えペアにした特徴量をpoolする
- Team×Lead×action相互作用: 60候補の全特徴を使い補正する
の和で60候補を採点する。`flat` armは従来型のstate/action MLPで、構造化の効果を同一条件で切り分ける。固定6スロットへの恣意的なlearnable biasはない。JSテストで正確な合法列挙の順序・20×3・合法候補60個を照合する。

## 3. Top1教師一致 loss

同じ学習256受理完全Match、同じ初期seed・学習6epochで、以下を事前指定した4 armで比較する。

| Arm | 構造 | 損失 |
|---|---|---|
| flat-hard | 通常MLP | 教師Top1 hard CE |
| structured-hard | 3体／先発分離 | 教師Top1 hard CE |
| flat-mixed | 通常MLP | hard CE 70%＋全合法60候補教師utility soft 30% |
| structured-mixed | 3体／先発分離 | hard CE 70%＋全合法60候補教師utility soft 30% |

soft targetは教師utilityの局面内標準偏差に基づく温度で生成し、以前のtop10打切りで生じる任意境界を回避。mix=30%は比較仮説であり性能保証ではない。いずれもモデル保存とseed/weight SHA照合を行う。

## 4. 入力の十分性診断

`test/rnad-selection-specialist.test.js`で以下を検査する。
- 公開相手6種族を変更するとv5入力が変わること（公開情報の受信）
- 相手の非公開技・持ち物を変更してもv5入力が変わらないこと（情報漏洩防止）
- 自軍の知っている素早さ・持ち物・先発を変えると候補特徴が変化すること
- 60候補の列挙・先発／控えの同値が一致すること

さらに同じ実教師holdoutの各60候補について、**同一action特徴量に異なる教師utilityが付く候補群**と、同じ3体で先発を変えても入力特徴が同値になる群をカウントする。教師Top1と2位の相対utility gapも計測する。

**監査結果がゼロでも入力十分性は証明されない**。公開相手の4bit種族hash衝突、候補間相互作用、未表現の自軍セット情報、教師自身の不安定性などは今後の研究仮説として扱う。チェック対象は実装済みの公開可視の座標のみ。

## 実験・厳密性

[初回Run #44038986373](https://github.com/cueikusuta/p-RL1024/actions/runs/44038986373)：preflight成功、compareは**トレースengineが実際には40桁source commitなのに検査器が64桁と決めつけたため**学習開始前にfail-closed。修正を適用し、[再実行Run #44039529263](https://github.com/cueikusuta/p-RL1024/actions/runs/44039529263)を**まったく同じ固定データ・4 arm・6epoch**で単発再実行した。重いデータ再生成はなし。

- 学習：元Run #44032488678の保存済み権威受理256教師Match（seed 2471001..2471256）
- 評価：同Runの128受理済み教師Match（2472001..2472128）。**既に同Runで探索評価済みなので、本実験のholdoutは開発用であり、新規未使用の最終ゲートには使用不可**。
- 保存済みv5選出重点モデルの実重みをBaselineとして、同一128教師MatchでTop1/Top10/normalized CE/teacher utility regretを比較し、元Baseline Top1 26.49%を厳密に再現する。
- 4比較についてseed-cluster bootstrapで多重比較補正した近似80%同時区間を診断表示。ただし**確認的結論ではない**。新方策選択後に完全未使用新seed評価と統合Actorでのテストが必要。
- GitHub-hosted CPU、元学習用source/engine SHAを照合し、コスト節約のため保存済みデータを一度だけダウンロードして使う。
- 勝率評価・リーグ開始・Champion昇格・PR Ready化・mainマージは別ゲートであり、この研究Workflowは一切行わない。

**状態はRunから確認する。** 成功／模倣率改善／選出80%達成をRunの結果なしで宣言しない。
