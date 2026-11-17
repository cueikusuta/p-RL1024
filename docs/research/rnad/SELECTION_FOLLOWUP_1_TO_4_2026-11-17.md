# Issue #5 — 選出AI追加改善 1〜4（研究・安全性）

既存 [Issue #5](https://github.com/cueikusuta/p-RL1024/issues/5) / [Draft PR #6](https://github.com/cueikusuta/p-RL1024/pull/6) の継続。従来の `SELECTION_SPECIALIST_1_TO_4_2026-11-17.md` の次の追加研究であり、凍結v5・旧4-arm重み・Champion・リーグ学習は更新しない。

## ① 局面間の特徴量不足監査

`feature_sufficiency_audit` を従来の同一局面候補48D比較から拡張する。**actorが推論時に実際に見る96D状態 + 順序を保つ60×48D合法候補**をfloat32へ正規化してSHA-256で同一性を比較し、独立した受理済みMatch seed間で入力が完全一致した組と、そのうち教師Top1が不一致の組を検出する。**異なる候補群が同じstate座標でも同じ観測ではない**ので、stateのみで教師選択を比べない。独立Match seedの共通入力数も記録する。

これは**必要条件の診断**である。同じ特徴量で異なるTop1が存在すれば、現入力だけの決定的なネットワークはその両方を同時には再現できない。一方で一致入力が0でも、未表現の公開情報、衝突しない連続特徴、教師の不安定性、非公開情報は排除できない。元データが教師の訪問局面に限られることも維持する。

## ② 教師1位／2位差・3体／先発の切り分け

選出教師utilityの局面内 **`(Top1 utility - Top2 utility) / (max utility - min utility)`** を計算し、事前固定した `<=1%`, `1–5%`, `5–10%`, `>10%` の4区分で研究モデルのTop1精度・3体unordered一致率・3体正解時の先発一致率・教師regretを出す。bucketごとの独立Match seedクラスタ数を併記し、少数区分の違いを有意な優劣とは判断しない。

Top1一致率と教師utility regretは別指標。教師が僅差を返した局面では0/1 Top1指標だけで性能の善し悪しを断定しない。

## ③ 学習データ量とモデル幅の統制比較

保存済み `Run #44032488678` から受理済み**256教師Match**と、**過去に研究選別で使用済みの128評価Match**を再利用する。トレース/ゲームを再生成しない。学習Matchを先頭128と全256、MLP幅を64/96/128、損失を従来の`structured-hard`、epochを6、seedを57720に固定し、**2×3=6 arm** でTop1・Top10・教師regretを比較する。

256×96 armは前回[Run #44039529263](https://github.com/cueikusuta/p-RL1024/actions/runs/44039529263) の`structured-hard.json`保存済みテンソルとの**完全一致を必須**とする。異なる場合は継続比較をfail-closedで停止する。6 armの研究重みは`.pt`とSHA-256で研究Artifactへ保存し、96幅以外の重みは旧JS `rnad-selection-specialist-96x48-v1` と誤認させない。

**この既使用128Match評価は探索専用であり、新たなheld-out性能・独立有意差・実戦強度の証明には使えない。** 研究結果から固定候補を選んだ後に、新規・未使用教師Matchで独立Matchクラスタ統計と事前決定の必要数を使って再検証する。

## ④ JS推論重みのSHA-256照合

`scripts/rnad/selection-specialist-infer.mjs` のCLIで、重みJSONとは**別経路で信頼したダイジェスト**を`--sha256`で必須指定し、実ファイルbytesのSHA-256一致後にschema/全テンソル/60候補順をチェックする。モデルJSON自身に記載されたchecksumを信用してはならない。呼び出し側APIも、照合していないモデルの推論を拒否する。

使用例：

```sh
node scripts/rnad/selection-specialist-infer.mjs --model structured-hard.json --sha256 1b5f39cee6f00d1fe5c095442996fb706086dd4456d84ce5974a861f9e4e81cd --input state-actions.json
```

※上記digestは元の[4arm学習Artifact](https://github.com/cueikusuta/p-RL1024/actions/runs/44039529263/artifacts/14307824700)に対応する固定値であり、研究レポート/信頼済みマニフェストに保管された値を使う。GitHubから取得した重みとchecksumを同じ未検証の場所から取り込むだけでは改ざん耐性の根拠にならない。

テストではPyTorch↔JSのscore数値一致、正常shapeかつ有限値への改変、候補順改変、digest未指定を確認する。

## 実行と合否

専用workflow `.github/workflows/rnad-i5-selection-followup.yml` の`preflight`成功後のみ`compare`を実行する。GitHub-hosted `ubuntu-latest`を使い、1つの保全済み教師Datasetを再利用して重複収集しない。結果は`selection-followup-report.json`、`FOLLOWUP_REPORT.md`およびモデル重みArtifactに格納する。

現行リーグゲート：**統一した同一policyの新規未使用MatchでドラフトTop1 >50%、選出Top1 ≥80%、戦闘Top1 ≥80%、3相Bonferroni補正80%片側下限**。この6 armは選出単独研究であって、ゲート適格でない。未通過の間はリーグ開始・Champion昇格・mainマージなし。既存PRはDraft維持。別Chatでの独立レビューが必要。
