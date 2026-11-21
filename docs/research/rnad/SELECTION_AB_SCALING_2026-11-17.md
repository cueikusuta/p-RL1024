# I5 選出A/B実学習の追加実験：512 Match、各3 seed、10／20 epoch

リポジトリ：`cueikusuta/p-RL1024`。Goalは既存[Issue #5](https://github.com/cueikusuta/p-RL1024/issues/5)、作業は既存[Draft PR #6](https://github.com/cueikusuta/p-RL1024/pull/6)を継続。

## 今回の変更

前回のA・B（教師評価分解型／公開・自軍12Token Attention型）では各**256 Match、10 epoch**の学習を済ませた。今回は同じ計算を繰り返さず、両方を新規に**512 Match**で最初から学習し直す。乱数seed **90251／90252／90253** で各モデルを学習し、**10 epochと20 epoch**の固定2点でチェックポイントを保存する。研究用合計 **2モデル×3 seed×2 epoch=12重み**。旧モデルも凍結して残す。

教師データは追加生成しない。既存 [Run #44047154334](https://github.com/cueikusuta/p-RL1024/actions/runs/44047154334)の受理済み完全4勝データの**元訓練256 Match（2471001～2471256）**と**以前は独立確認に使った256 Match（2477001～2477256）**を学習集合にする。後者は今回から**既使用の訓練データ**に変更され、独立検証に再利用しない。開発用は以前から使用済みの128 Match（2472001～2472128）。`run/source/engine/vocab`、合法選出60順、全受理済み完全Matchを検証し、異なるsourceや重複seedは拒否する。

## 比較の解釈

旧256 Match×10 epochと新512 Match×10/20 epochでは、増える情報量と学習step数が混在する。**データ量の純粋な因果効果**とは呼ばず、学習条件の統合的な拡張試験とする。学習途中のtrain loss、開発Top1、Top10、3体一致率、教師utility regret、チェックポイントSHA-256を保存する。小標本の単独勝敗やbest-seedだけで優位を宣言しない。

各A/Bについて、開発128 Matchで`Top1最大→regret最小→短いepoch→seed`の規則から**各1重みを事前固定**する。旧同方式の開発Top1を片方でも0.5ポイント超上回れば、**新規未使用512完全教師Match seed 2479001～2479512**を生成し、その後**新規訓練なし**で固定2方式をそれぞれ旧同方式と同一Matchで対応あり評価する。**独立Match seedの95%家族単位誤り率制御**のため、2比較Bonferroni補正として個別に近似**97.5%両側cluster bootstrap CI**を算出する（10,000回）。0をまたぐ場合は改善確認と扱わない。次の結果を見てからモデルやハイパーパラメータを変更して、そのholdoutを再使用しない。

## 安全・統合制約

推論入力は自軍6体の技・特性・持ち物・ステータス、相手公開6種族に限定する。教師core／lead／utilityは学習目的変数のみであり推論入力から除外する。教師scoreを推論側で直接呼ばない。GitHub-hosted `ubuntu-latest` の専用[Workflow](https://github.com/cueikusuta/p-RL1024/blob/feat/5-rnad-stage0-contracts/.github/workflows/rnad-i5-selection-ab-scaling.yml)で実行する。別のR-NaD学習・モデル系列・凍結v5・既存Championを変更しない。

なお本実験は選出単独の教師Top1模倣だけ。**同じ統合policy**におけるドラフトTop1>50%、選出Top1≥80%、戦闘Top1≥80%、3相Bonferroni補正80%片側下限を満たすまでは**リーグ学習を開始しない**。独立Chatセッションのレビュー、最新CIの成功、真のAQ実戦強度を別途確認するまで、Champion昇格・PRマージはしない。
