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


## 追加：新規未使用256教師Matchでの1候補独立検証

研究用6armの既使用128Match開発データから、**新データを見る前に**Top1最大→regret最小→width最小→名前順という固定規則で**1モデルだけ**選ぶ。研究Run [#44042088126](https://github.com/cueikusuta/p-RL1024/actions/runs/44042088126) はpreflight/compareとも成功。参考Top1は幅64/96/128で、教師128Match=27.85/28.12/28.39%、教師256Match=28.05/28.86/28.79%。最大は`structured-hard-m256-w96`、従来のstructured-hardと重み完全一致。これは**以前から再利用している開発用holdoutの数値**である。

独立確認には元教師source commit `022aa9cd5b4ac37514df1d845ea00c11c3c13a5c`をcheckoutして、新たなseed `2473001..2473256` の**256完全4勝受理済みMatch**を生成する。以前の学習`2471001..2471256`と開発`2472001..2472128`のseedとは重複しない。ゲームエンジン・教師モデルID・feature schemaを一致させ、生成ファイルのSHA-256一覧をartifactに保全する。

比較相手は前回の構造化モデル**自身ではなく**、凍結v5選出重点の**実重み** `6580738ace427364d89e83b94b465ea4f7febead1cb4e18f578d7fb8cb79f5f6`。元v5のドラフト・選出・戦闘間のrecurrent memoryを完全Matchで継続し、同一新seedの選出に対して対応ありTop1差とseedクラスタの**近似95%両側bootstrap CI**を出す。50%のリーグ開始ゲートとは別の独立検証であり、区間に0が含まれる場合は性能優位を宣言しない。

`.github/workflows/rnad-i5-selection-fresh-256.yml`は `preflight → gather → evaluate` の順。先行6arm成功artifactに依存するが、初回研究を再学習しない。新MatchはGitHub-hosted Ubuntuで一度だけ生成し、十分な証拠が得られなければ追加試行を勝手に「合格」に換算しない。


## 独立256教師Matchの検証結果（Run #44042810450 success）

[Run #44042810450](https://github.com/cueikusuta/p-RL1024/actions/runs/44042810450) は **preflight / gather / evaluate すべてsuccess**。未使用seed `2473001..2473256` の256受理済み完全4勝教師Matchを元`022aa9cd5b4ac37514df1d845ea00c11c3c13a5c`で生成し、全トレースのaccepted/completed/seed/engine/sourceおよびSHA一覧を検証済み（engine `ea81165993110ee9fb3a09dcb97c6708c0d7397b`）。

教師Matchデータ：[Artifact 14309894143](https://github.com/cueikusuta/p-RL1024/actions/runs/44042810450/artifacts/14309894143)。
研究比較結果：[Artifact 14310228945](https://github.com/cueikusuta/p-RL1024/actions/runs/44042810450/artifacts/14310228945)。

| 未使用教師256完全Match・同一局面 | Top1 |
|---|---:|
| 凍結v5選出重点（完全Matchを通じたrecurrent memory） | **25.12%** |
| 既用開発データから事前固定した`structured-hard-m256-w96` | **27.43%** |

- 独立Match seedクラスタ平均の**対応あり差 +2.36ポイント**
- 5000回seed-cluster bootstrapの近似**95%両側区間 [+0.96, +3.77]ポイント**
- 区間下限が0を超え、凍結v5に対する教師Top1一致の**独立データ上の改善を支持**。ただしBootstrap区間は有限標本で厳密な区間ではない。
- 既存のstructured-hardの新seed参考Top1も27.43%。初期開発用holdout（28.86%）の値をそのまま一般化精度として引用しない。
- これは**選出単独教師模倣の確認**。教師局面外での自己対戦強度や、同じ統合policyでのDraft>50%/Selection≥50%/Battle≥80%を通していない。リーグ・Champion・PR統合は依然禁止。

## 自軍の仮想変更に対するv5特徴量の感度（Run #44043961072 success）

[公開／自軍情報の診断Run #44043961072](https://github.com/cueikusuta/p-RL1024/actions/runs/44043961072) は成功。ある固定した6体で80通りの`abilityId`代替では**75通り**が同じ入力、80通りの`itemId`代替では**77通り**が同じ入力、`moveId`代替では1通りが同じ入力だった。技では79通り、特性では5通り、持ち物では3通りが特徴量を変えた。公開相手6種族と自軍素早さの変更は反映され、敵非公開セット変更には非干渉だった。

仮想代替は合法なポケモン型と限らず、この不感度は本物の教師utility矛盾を証明しない。しかしv5で特性・持ち物の意味的な影響が十分区別されない可能性があるため、次の独立した特徴量研究での改善仮説とする。既存スキーマv5と凍結重みは変更していない。
