# RNAD Issue #5 — 実験エビデンス横断監査（2026-11-16 JST）

**目的**：方針を変更する前に、「学習工程が成立した」「実際の勝率改善が実証された」「試合途中で止まった」「CI/パッケージの不具合で評価未成立」を明確に区別し、モデル・ソース・seed・対戦相手・推論モード別にデータを集約する。既存 [Issue #5](https://github.com/cueikusuta/p-RL1024/issues/5) / [Draft PR #6](https://github.com/cueikusuta/p-RL1024/pull/6) に対応する。

**監査基準**：GitHub Actionsの実際のjob結論／stdoutで出力された結果JSON／sourceとengine/model SHA／独立seed×両席の実際に4勝到達したMatchのみを性能証拠とみなす。単なるworkflow `success` や保存された重み変更は強化の証拠にしない。4勝未達・決定回数上限で中断したMatchは勝敗へ補完しない。**対Basic、対Stage0、対AQ teacher、対旧モデルの勝率は混ぜてランキングしない**。異なるholdout seedブロック間で単純な勝率差を因果効果とみなさない。128 Match（64 seed×両席）の二席は独立128 seedではなく、同じseed内で相関する。同一holdoutで多数のモデルを見て選んだ候補には選択バイアスがある。

## 0. 系列と主要モデルの識別子

| 名称 | 意味／モデルSHA256 |
| --- | --- |
| 普通のbase128 | v2の原系列、128**完成4勝Match**時点の重みSHA `8e8c2b5387ab980d3e7bc823537babe384dea040070d14dec694bee96765938f` |
| 普通のbase256 | **同一元系列の完全optimizer/magnet/RNG checkpointから継続**し256完成Match時点のSHA `59ca3d8d952ad61cae93ba420ad48dbae3204e456d31ddcf082ccbd1a3de9c1c` |
| ガチャc005 | base128由来100変異候補から選抜し**新しいoptimizerで16完成Match**学習した個体、SHA `c20a92ad345170896b721e02f5fae558afb2c53ca424be104a2ec2467c1654c0`。元base128の完全optimizer継続とは異なる |
| ガチャc019 | 別の100候補生存個体、SHA `5f6122856068b9f7cd5183b8628928fe75fd1642bca7fb8b2251be8bb8c10ab2` |
| 凍結元のコード | v2モデルの原研究Source `f08a9587d6a7033546eab00ffa6479399ca71c2f`、engine `ea81165993110ee9fb3a09dcb97c6708c0d7397b` |
| guard付き再ウォームスタート | `anti-loop-v1`、新しいoptimizer、実学習merge-source `571bac81d62373ab063c2e1508b0647122786e80`。元モデル重みを使うが原optimizerを再開しない |
| 交代機会損失付き系列 | `anti-loop-v1 + opportunity-v1`、初回は凍結元重みから**新しいoptimizer**、実学習merge-source `03bbd42b9aa393aa8af3ede9bdb2b3ba87fdc9ea`。同系列内64→128→256は完全optimizerを再開する |

## 1. 成功／改善が実証されたもの（範囲限定）

| 実験 | 実測・比較条件 | 判定 | 原記録 |
| --- | --- | --- | --- |
| **同一v2モデルの推論方式 sampled→greedy** | Basicに対し、同じ128 seed×両席の256完成Match。sampled **16/256 (6.3%)**→greedy **123/256 (48.0%)**。独立paired seedに対する保守的Hoeffding差区間 **+17.8〜+65.8pt** | **greedy改善の根拠あり**。ただしBasicに勝ち越したわけではなく、Championではない | [v1/v2モード比較 #43744199284](https://github.com/cueikusuta/p-RL1024/actions/runs/43744199284)、[正式差分チェック #43747506291](https://github.com/cueikusuta/p-RL1024/actions/runs/43747506291) |
| v1→v2表現＋学習パイプライン変更 | 同じBasic 256 Match、greedy v1 **27/256**、greedy v2 **123/256**。sampled v1 **3/256**、sampled v2 **16/256** | **大幅な成績改善を観測**。同時に複数の仕様・モデルが変わっており、「個々の特徴が原因」とは分解できない | [#43744199284](https://github.com/cueikusuta/p-RL1024/actions/runs/43744199284) |
| 学習と完全状態再開の技術検証 | 原系列128→256、機会損失系列で標準64→128、c005 128→256まで、各Matchの受理済み軌跡・source/engine/weights・seed・optimizer/magnet/RNG・checkpoint SHA検証が成功 | **学習基盤は動作**。勝率改善の証明とは別 | [原系256 #43752012559](https://github.com/cueikusuta/p-RL1024/actions/runs/43752012559)、[交代コスト継続 #43789378463](https://github.com/cueikusuta/p-RL1024/actions/runs/43789378463) |
| **交代ループの再現・安全対策** | seed `1720101`第7ラウンドでc005とbase128が各1,501回連続自主交代・技0回・直前逆戻り各1,500回。元の3,000判断上限到達。`anti-loop-v1`で当該seedの両席4勝Matchが**2/2完走** | **限定された安全・進行改善を実証**。強さの向上とは別 | [原因再現 #43778153444](https://github.com/cueikusuta/p-RL1024/actions/runs/43778153444)、[guard #43779235844](https://github.com/cueikusuta/p-RL1024/actions/runs/43779235844) |

> **重要な数値訂正**：過去の時系列説明には「v2 sampled 27/256」とも書かれていたが、実際の原評価JSON schema `rnad-003-v1-v2-greedy-comparison-v1` は **v2 sampled 16/256、v2 greedy 123/256、v1 sampled 3/256、v1 greedy 27/256**。ここでは原Jobの出力を採用する。別runの数字の混同を避ける。

## 2. 実行は成立、**独立評価で強化の確証なし**

### 原系列：16→128→256

| 訓練/比較 | 実測 | 論点 |
| --- | --- | --- |
| v2 teacher初期化後 R-NaD 16完成Match | 同一Basic holdout：teacher **136/256**、R-NaD16 **134/256**。直接R-NaD16対teacher **66/128** | 受理済み軌跡での学習と更新は成功、勝率改善なし。[#43746142851](https://github.com/cueikusuta/p-RL1024/actions/runs/43746142851) |
| 16→128完全状態継続 | 同一Basic holdout：teacher **129/256**、128世代 **130/256**。直接128対teacher **65/128** | ほぼ互角。[#43747882146](https://github.com/cueikusuta/p-RL1024/actions/runs/43747882146) |
| 128→256完全状態継続 | 同一Basic holdout：128世代 **135/256**、256世代 **132/256**。直接256対128 **135/256** | Basicへの成績は3勝低下、直接では256が135勝とわずかに勝ち越し。**単調な強化は証明されず**。[#43752012559](https://github.com/cueikusuta/p-RL1024/actions/runs/43752012559) |
| 旧Warmstart teacher→R-NaD256 | 対Stage0とは別のteacher対Basic **123/256**、teacher+256 R-NaD **115/256**、直接teacherに対してR-NaD **69/128** | Basic相手は8勝減、直接は微差。差の保守的95%区間 `-27.1..+20.9pt`。[#43697864312](https://github.com/cueikusuta/p-RL1024/actions/runs/43697864312) |
| eta/magnet変更 | `eta=.05, magnet_period=16`、64完成Match、対teacher **32/64**、対既存R-NaD256 **27/64** | 有意な改善を確認できず。[#43698077819](https://github.com/cueikusuta/p-RL1024/actions/runs/43698077819) |

### ガチャ100候補→上位個体の独立評価

実行した100候補の事前スクリーニング、上位4個体への**16完成Matchずつの独立ウォームスタート**は成立。[#43751083532](https://github.com/cueikusuta/p-RL1024/actions/runs/43751083532) の小規模finalist holdoutでは base128(Basic) **35/64**、c005(Basic) **39/64**、c005(base128直接) **33/64**であった（サンプル少・選抜バイアスあり）。

続く新規256 seed×両席、**512 Match**のBasic評価は[#43777100106](https://github.com/cueikusuta/p-RL1024/actions/runs/43777100106) の原Shard監査で検証済み：

| 原モデル | 対Basic勝数/512 | 対Basic勝率 |
| --- | ---: | ---: |
| base128 | **276/512** | 53.91% |
| base256 | **267/512** | 52.15% |
| c019 | **250/512** | 48.83% |
| c005 | **284/512** | 55.47% |

同じ新seedのモデル直接対戦：c019対base128 **256/512**、c019対base256 **256/512**、c005対base256 **260/512**。**c005対base128は4つの128-Match shardのうち2つが3,000判断上限による交代ループで中断**。完走した2 shardのみの **122/256** であり512 Match勝率として扱うことは禁じる。対Basicではc005が最多だがbase128との差は8/512 Matchに過ぎず、直接優越の確証はない。

### 交代guardのアブレーション

[archive検証 #43779757849](https://github.com/cueikusuta/p-RL1024/actions/runs/43779757849)：凍結**同じc005重み**、32独立seed×両席、guard付き対guardなしの直接 **33/64**。対Basicは guardなし **34/64**、guardあり **33/64**。**交代の安全対策は成功、強さの明確な増減は確認できず**。実際に停止したseedの完走改善と勝率改善は独立の問い。

### guardのみ・新しいoptimizerの64-Matchウォームスタート（`anti-loop-v1`、機会損失報酬なし）

[復旧・全ジョブ成功 #43782430025](https://github.com/cueikusuta/p-RL1024/actions/runs/43782430025)、holdout direct `1760001...`／Basic `1761001...`、64 seed×両席。

| 凍結元 | 追加前対Basic | 64追加後対Basic | 追加後対凍結元直接 |
| --- | ---: | ---: | ---: |
| base128 | 67/128 | 68/128 | **61/128** |
| base256 | 64/128 | 63/128 | **65/128** |
| c005 | 71/128 | 68/128 | **58/128** |

3系列とも64完成Match/64 update、元モデルとは違うmodel SHA、新 optimizerで学習。「原のoptimizer stateの継続」ではない。いずれもChampion昇格を根拠づける改善ではない。

### 機会損失学習（`anti-loop-v1 + opportunity-v1`）

実験は元重みから新しいoptimizerで開始し、**その同じ系列内部の更新は完全optimizerを継続**。[研究 #43787170970](https://github.com/cueikusuta/p-RL1024/actions/runs/43787170970) でbase128/base256各64完成Match、c005 128完成Match。元runの評価jobは失敗したが学習artifactは全件成功・保存され、[評価だけ復旧 #43789106137](https://github.com/cueikusuta/p-RL1024/actions/runs/43789106137)は3モデルとも成功、source/engine/model SHAと独立両席4勝Matchを監査した。holdout direct `1780001...`、Basic `1781001...`。

| 凍結元 | 元対Basic | +64対Basic | +128対Basic | +64対元直接 | +128対元直接 | +128対+64直接 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base128 | 72/128 | **76/128** | （別seed） | 63/128 | — | — |
| base256 | 67/128 | **66/128** | （別seed） | 66/128 | — | — |
| c005 | 69/128 | 66/128 | **70/128** | **58/128** | **57/128** | **61/128** |

c005の +128 は+64とBasicへの数字を4勝分改善したが、直接では元に57/128、+64に61/128といずれも負け越し。**「学習量128の効果が明確に出た」は支持されない**。同一seedと異なるseedの表を縦横に混ぜないこと。

実際に報酬を与えた自主交代は base128 **21/642件**、base256 **24/481件**、c005 **71/1508件**（上記はそれぞれ64/64/128学習Match内の自主交代）。`opportunity-v1`はHP/タイプ/現在の合法攻撃の代理値により小さなゼロサム報酬を与える**研究用の別目的**であり、真の反事実ターン損失を計算しているわけではなく、潜在関数による方策不変shapingでもない。

さらに[完全状態の継続と新規seed評価 #43789378463](https://github.com/cueikusuta/p-RL1024/actions/runs/43789378463)：base128/base256 **累計128完成Match**、c005 **累計256完成Match**まで学習・元checkpointからoptimizer/RNG等の復元を検証し、保存済み。別holdout seed `1790001...` と `1791001...`（各64seed×両席）の**先行完了した**結果：

| 学習元系列 | 別の新holdout Basicでの比較 | 直接の後続モデル対先行 |
| --- | --- | --- |
| base128 | +64: **67/128**、+128: **67/128** | +128対+64: **61/128** |
| base256 | +64: **63/128**、+128: **70/128** | +128対+64: **64/128** |
| c005 | +128→+192→+256 の完全状態学習・各重み保存済み | **+192/+256新seed評価は当該Runのc005 evaluate job完了待ち**（本監査時点） |

この2つのBasic列は別runと**異なるholdout seed**なので、前段の72→76、67→66と直接差し引きしないこと。新holdoutでbase256のBasicは63→70と数値上増加したが、直接+128 vs +64 は64/128の互角。**「強くなった」とは認定しない**。

### Teacher / 学習重み変更

[4系統完走 #43756892480](https://github.com/cueikusuta/p-RL1024/actions/runs/43756892480)、各24教師Match学習→32独立seed×両席の対Basic検証：

| 3フェーズ教師CE／label smoothing | 対Basic |
| --- | ---: |
| equal 1:1:1、smooth 0 | 25/64 |
| equal 1:1:1、smooth .025 | **29/64** |
| draft-selection重視 2:2:1、smooth 0 | 22/64 |
| draft-selection重視 2:2:1、smooth .025 | 20/64 |

小規模データでのフェーズ重視は正解ラベルの改善を狙えたが**勝率を改善せず**。equal+.025が最高でも29/64は50%未満であり強さの実証ではない。別の強いTeacher導入・模倣精度と勝率の関係は未確定。

## 3. 失敗・未成立の区別

| 分類 | 具体例 | 結論と再実行方針 |
| --- | --- | --- |
| **ゲーム方策の不具合（実際の未完Match）** | `c005-vs-base128`の2つの新seed shardが3,000判断上限により中止。[#43757054212](https://github.com/cueikusuta/p-RL1024/actions/runs/43757054212)、再現seed `1720101`で相互逆戻り各1,500回 | **方策の安定性の失敗**。上限解除・未完データへの勝敗代入は禁止。guardで原因seedの完走は検証済み |
| **CI・復旧環境の不具合（学習そのものは完走）** | 学習artifactを`results/$MODEL`でなく`results`直下へ展開し `ENOENT`。[#43781276450](https://github.com/cueikusuta/p-RL1024/actions/runs/43781276450) | 学習を再実行せず[評価復旧 #43782430025](https://github.com/cueikusuta/p-RL1024/actions/runs/43782430025) |
| **ソース出自混同** | `pull_request`のHEAD SHAとGitHub synthetic merge checkout SHAを混同し評価で拒否。[#43788476047](https://github.com/cueikusuta/p-RL1024/actions/runs/43788476047) | 学習artifactの`source_sha`を正として厳密に元SHA checkout。後続[復旧 #43789106137](https://github.com/cueikusuta/p-RL1024/actions/runs/43789106137)で成功 |
| **Python環境不備** | 復旧用Pythonが`torch`をimportできず失敗。[#43788961276](https://github.com/cueikusuta/p-RL1024/actions/runs/43788961276) | PyTorch CPU等の依存を固定し、評価のみを再実行 |
| **JSON検証/出力バグ** | `json.dumps(...,flush=True)`に誤ってflushを渡す、Basic evaluatorにないトップレベル`seed`を参照、`-0`と`0`のJS配列比較失敗 | **研究の失敗ではなくCI/検証コードの失敗**。原結果が正当ならartifactだけで修正・監査。[#43755359502](https://github.com/cueikusuta/p-RL1024/actions/runs/43755359502)、[#43779235844](https://github.com/cueikusuta/p-RL1024/actions/runs/43779235844)、[#43787068102](https://github.com/cueikusuta/p-RL1024/actions/runs/43787068102) |
| **successだが学習未実施** | `128`継続の旧runでguard判断がskipし、jobは成功と表示。[#43747506276](https://github.com/cueikusuta/p-RL1024/actions/runs/43747506276) | `success`のみで学習完了認定しない。必ず段階・model/checkpoint SHA・完成Match実行ログを照合 |
| **性能改善の根拠不足** | 追加64/128 Matchでの同一seed直接対戦が58〜66/128などに留まる、多数のモデル候補を同一holdoutで確認 | **未証明**。探索上の有望/低迷という記述に留める。別seed・相手多様性・confidence gateが必要 |

## 4. 未解決の検証不足（次の方針決定前に埋めるべきギャップ）

1. **Battle以外のDraft/Selection**の模倣一致率が低い。旧v1 heldoutでDraft約2.2%、Selection約7.1%、Battle約55%；v2 heldoutでDraft約4.4%、Selection約6.6%、Battle約69.5%。基本的なカテゴリ特徴/教師品質の欠落の可能性はあるが原因未確定。学習損失の減少だけでは対戦性能を評価できない。
2. **同一 source・同一seed・同一 opponent を揃えた個体間トーナメント**が不足：通常128 vs256、c005 64/128/192/256、および複数固定教師・Basicを併用して直接比較し、学習途中の性能の山／谷を調べる。既存の異なるseed結果は合算しない。
3. **安全性と強さの別評価**：交代の頻度・往復率・攻撃/交代率・強制/pivot率・4勝完走率・モデル当たりdecision-cap率を同時に見る。交代抑制によって完走したことを勝率向上と同一視しない。
4. **統計設計**：モデル選抜に使用したseedを確証評価へ再利用しない、1 seed両席を単位としたpaired CI、候補・条件の多数試行による選択バイアスを考慮、事前の合格基準に達しない場合は「未証明」とする。
5. **機械学習の実装契約**：`opportunity-v1`が実際に罰する自主交代は対象全体の数%に過ぎない、攻撃圧力の代理値は反事実損失ではない、GRUの内部記憶と明示的な直近受理交代履歴の差は未切り分け。独立の数学・秘密情報・報酬整合性レビューが必要。
6. **研究結果の永続化**：多くのGitHub Actions artifactは30日保管。重要な結果JSON/モデルSHA/checkpoint参照を期限前に長期保全する必要あり。
7. **最新PR品質ゲート**：この監査は各原runのSOURCEを根拠とするが、Draft PR #6 最新HEADの独立ChatGPT Chatレビューを代替しない。Champion、mainへの統合は未実施。

## 5. 暫定分類（研究方針はまだ決めない）

- **成功した実証**：v2におけるgreedy推論方式の変更、技術的な完全状態resume、交代無限往復の安全封じ。前者は実測勝率差のpaired CIで改善あり、後二者は工学的改善のみ。
- **有望だが未確証**：ガチャc005は対Basic 284/512で最多、交代報酬付きbase128 64更新時は76/128対Basic。しかし固定base128との直接優越は未確証。
- **勝率改善を実証しなかった**：原16→128→256の単純延長、100ガチャ個体の選抜、teacher CE重み/スムージング、小さいetaやmagnet変更、guard付き64 Matchの新optimizer学習、交代コスト付き64/128 Matchの追加学習。
- **検証未完**：c005 128→192→256の新seed評価結果（進行run #43789378463）、大規模な独立多相手確証、別ChatGPT Chat独立レビュー、今後のモデル間統一トーナメント。

**ここでは次の学習戦略やChampion選択を確定しない。** まずこの実証レベルの差を基準に、次の仕様・費用・優先順位を検討する。
