# R-NaD 3フェーズの教師一致率ギャップと修正計画（2026-11-17）

対象：Issue #5 / Draft PR #6。これは**検証済み実装の構造監査＋改善仮説**であり、低一致率の唯一の原因を確定した報告ではない。最終目的は教師Top-1ではなく**本物の凍結AQ-Base-001に対する完全4勝Matchの勝率**。AQ模倣用教師（`aq-public-belief`）と、対戦相手としてのオリジナルAQ-Baseは異なる。

## 1. 既存実測（未使用8完全MatchのAQ-public-belief教師、Run #43812943402）

| phase | 受理教師行動 | 候補数平均 | Top-1 / 一様Top-1 | Top-10 / 一様Top-10 | 平均正規化CE |
|---|---:|---:|---:|---:|---:|
| draft | 88 | 1690.69 | 3.41% / 0.10% | 17.05% / 1.02% | 0.856 |
| selection | 88 | 60 | 4.55% / 1.67% | 46.59% / 16.67% | 0.900 |
| battle | 719 | 4.56 | 76.77% / 31.25% | 100% / 100% | 0.480 |

**事実**：ドラフトと選出の候補が圧倒的に多い。ゆえにTop-1をフェーズ間で直接比較するのは不適切。選出は候補数で補正したCEでも一様予測の1に近い。一方、v2/v3の実権威局面での*完全一致action特徴衝突*はドラフト0/391,445、選出8/16,080であり、単純な完全衝突が主因という仮説を支持しない（Run #43811523683）。

## 2. コードで確認した具体的な表現不足

**source**: `src/ml/rnad-phase-features.js`、`src/ml/rnad-phase-features-v2.js`、`src/ml/rnad-phase-features-v3.js`、`scripts/rnad/train_teacher.py`、`scripts/rnad/aq-teacher-policy.mjs`、`src/ai/team-assessment.js`、`src/ai/selection-tactical.js`。v1/v2/v3の重み・checkpointは不変。

1. **draft：教師スコアの非線形要因が入力にない／圧縮される**。`chooseStrategyTeam`は個体の強さだけでなく、能力タグの組み合わせ・画面/補助技と積み役の組み合わせ・トリックルームでの速度役割・メガ機会・種族/持ち物重複制約・予算・チーム多様性を考慮する。v1/v2のdraft actionは16個の集約値（平均能力・タイプ数・平均火力proxy・リード圧力など）で、これらの**ペア相互作用や役割の順序**を直接表現しない。v1の種族IDを連続スカラーhashとして扱うのも類似度に意味がなく、v2では一部hashを代理値に上書き。
2. **selection：相手の個別種族の識別情報が上書きされる**。v1のstate座標24〜26は公開相手種族名のhashだが、v2 `encodeRnadThreePhaseDecisionV2`ではstate 24〜26を`pool.resistance`、`opponent species present`のフラグ、`pool.uniqueCoverage`に置き換える。残る敵タイプの集計は公開種族から抽出した**タイプ対面の平均/最悪値などの粗いproxy**で、相手の個別種族・技候補・素早さ・役割分担を保持しない。v3の追加stateにも相手の公開技/種族の*件数*はあるが、相手種族IDを完全復元する情報はない。
3. **selection：自軍先発の識別が弱い**。v1のaction座標14は先発species hash、15は選出チームspecies hash平均だが、v2で`scores.lead`（平均攻撃圧力）と`scores.coverage`に上書き。v3の追加categoricalは選出3体のspecies/ability/item/movesを**無順序**の4-bucketずつに縮約するため、先発だけが変わる重要差分が明示座標にない。合法候補の大半は完全一致衝突しなくても、異なる対面の価値を区別するには不十分になり得る。
4. **教師の価値関数と学習特徴量の不一致**。draft教師は非線形チーム役割・予算等、selection教師`chooseTacticalSelection`はBasicのtrio得点＋各種役割プラン/速度等を使う。studentは16次元actionと32次元stateの限られたproxyしか見ない。さらに`aq-public-belief`教師は本物のAQ-Baseの完全Match得点・ターン履歴を受け取らない。高一致率が出ないことに複数の理由があり、現状の数値だけで学習則バグと確定しない。
5. **教師データ／目的**。`train_teacher.py`は各accepted decisionに単一教師choiceのCrossEntropyと、終局±1に対する小さいValue MSEを付加。共有GRU・共有policy projectionが三フェーズを担当する。フェーズ均等損失/selection-focusを比較済みだが、8Match程度の短期追加では選出Top-1は両条件同じ。Top-1は近似同等の候補を全て誤答として扱う。多数の教師候補が実際に同等近傍かは、**未測定**なので独立検査が必要。

## 2.5 選出で判明した教師・学生・本物AQの候補集合不一致（重要）

実ソースをさらに比較したところ**同じselectionという名称でも候補数・最適化している先発選択が異なる**ことを確認した。

| 実装 | 合法/探索候補の構成 | 効果 |
|---|---|---|
| R-NaD `legalPureRlSelections` | 6体から3体、先発は3体のいずれか、控え2体の順序は正規化：**60通り** | 学生Actorの合法集合 |
| 教師 `aq-public-belief` 内の `chooseTacticalSelection` → `rankBasicSelectionCandidates` | 6体から3体の組み合わせ**20通り**。先発は常に選ばれた3体のうち**元ロスターで最小インデックスの個体**。先発選び専用の対面スコアはない | 60通りの学生に20通りの限定された教師を模倣させている |
| 対戦相手の本物AQ `chooseStrategySelection` → `rankStrategySelections` | 3体選択×先発×控え順の**120通り**を候補化し、公開相手候補に対する各先発のmatchup・役割相性等をスコア化 | 教師ラベルと**実際に倒したいAQの選出ロジック**が一致していない |

従って、選出Top-1 4.55%が低い理由は、学習不足だけではない。教師の**任意のロスター順序による先発ラベル**と学生の**先発のspecies-IDを消した集約特徴**が整合していない。選出教師が最適化する評価値も、実AQとは違う。ただし、この問題だけで観測された低Top1すべてを説明できるかは未実測。

**最優先修正案（特性v4より前）**：学習に使うselection教師を新versionへ分離し、権威Matchで受理可能な**60候補へ元AQの公開情報・対面スコアを写像して順位と効用差を与える**。120候補のうち控え順だけが逆のものは60候補の正準アクションへ集約する際の意味保存を確認。直接`chooseStrategySelection`の戻り値を学生の合法アクションとして無検証で渡すことは禁止。teacher scoreのtopK/温度付き分布と採択理由、60候補への対応/重複/漏れを契約テストで検証する。必ず既存`aq-public-belief`教師・v1/v2/v3重みとは**別のteacher version/source/seed**で実験し、純RLの優越とは混同しない。

## 3. 修正実施順とゲート

1. **まず評価契約の改善（安価）**：未使用教師データに対し、ドラフト/選出ごとに合法候補数、Top-1/3/10、一様基準、teacher-choice probability、正規化CEに加え、教師に合法な候補**全体の順位・スコア差・正規化regret**と、先発のみ違う候補の差を記録する。教師効用の比較には`scoreStrategyDraftRoster`・`rankTacticalSelectionCandidates`の公開情報入力のみを使用し、擬似的な秘密や実戦勝率をラベルにしない。スコアが実際に近いかを確認後、無意味なTop-1完全一致の強制を避ける。
2. **v4を新しい版として設計（既存重みを上書きしない）**：相手公開6種族の情報を位置・種類・公開タイプ/役割ごとに表現し、相手未公開型はbelief分布でのみ推定。先発を別エンコードし、3体の各個体属性と補完性、トリックルーム速度方向、screen→積み、バトン→受け手等の相互作用、予算・carry・mega実効利用数などをaction特性に入れる。ドラフト候補は順不同のSet構造、選出は「先発＋控え集合」の構造を尊重。次元とモデル容量は実際の比較結果で決める。
3. **秘匿・受理/合法の自動テスト**：相手未公開セット/技/持ち物だけを変えてもactorの特徴・確率が不変、公開相手種族を異なる組み合わせへ変えたとき相性が適切に変化、先発を交代したとき先発固有情報が変化、同じ3体の控え順だけの差には虚偽優劣を付けない。公開値とラベルの真値の混用を禁止。
4. **損失とモデル構造を一要因ごとにアブレーション**：従来v3 vs v4 encoderだけ変更、独立のdraft/selection/battle policy head、soft-label/utility-regret蒸留、教師なしRL等は**別条件**にして、同一teacher seed・初期重み・optimizer budget・正規化CE/価値・実Match勝率で比較。単純なselection loss増量だけで改善しなかった事実を考慮する。
5. **強度ゲート**：固定AQ-Base-001の完全4勝Match、同一seed両席・score=4・評価源とモデルSHAを固定し16Match smoke→256Match screen→事前登録の独立未使用1024+Match・95%CIで判定。候補を選ぶために使ったseedを最終held-outに再使用しない。Champion昇格は独立ChatGPT Chatレビューと最新HEAD必須CIまで保留。

## 4. 同時進行中の最小対照試験

- [R-NaD v3-U4正確な128→256 Match継続学習 #43924729663](https://github.com/cueikusuta/p-RL1024/actions/runs/43924729663)。128 checkpointのoptimizer/magnet/RNG・source・engine・モデルSHAを拘束した自動再開、16Matchごとに境界検証。新規AQ比較seed `1987001..1987064`、128版/256版を両席同条件で評価（実証結果はRun完了後に報告）。
- [ドラフト/選出/戦闘のv2単一フェーズ置換試験 #43925113734](https://github.com/cueikusuta/p-RL1024/actions/runs/43925113734)。v3全phaseを基準に、draftだけ/selectionだけ/battleだけv2に変更、v2全phaseも対照。種 `1990001..1990024`・各48 Match。5モデルの比較結果を強度の確証にせず、次の改善対象選定用screenに限定。
- 二つとも初回投入後は**manual-only**に戻し、定期スケジュールは作っていない。PR #6はDraft継続、main未統合、正式Champion未昇格。
