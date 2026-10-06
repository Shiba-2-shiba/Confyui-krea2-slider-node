# Krea2 Regional Attention 実装計画

作成日: 2026-10-06。状態: Phase A実装候補／CPU確認済み／実機Gate Aの小マスク条件は未達。

**目的:** 1つのKSamplerで男女を生成し、女性の文章とSlider LoRAを指定領域へ割り当てる。小マスクの外側に別の女性や顔が出る問題を、attentionの情報経路から改善する。

**実行方法:** 依存順に実装し、各段階でテストする。実機生成はユーザーが担当する。プロンプト領域制御の実機評価を先に行い、その合格後に領域LoRAを実装・評価する。

**技術:** ComfyUI V3、PyTorch、Krea2 SingleStreamDiT、標準KSampler、既存Qwen系CLIP/VAE。追加Python依存は導入しない。

この文書に要件、設計判断、タスク、合格条件をまとめる。ユーザーの実行指示により、Task 1〜3とTask 5のprompt-only実装を進めた。実行状況と検証証跡は[進捗記録](2026-10-06-krea2-regional-attention.progress.md)を参照する。領域LoRAと最終commit/pushは後述の実機ゲートに従う。

追加指示: ユーザーがPhase A候補のcommit/pushを明示的に依頼したため、実機評価用に`feat/regional-attention`へ先行公開する。実機Gate Aと領域LoRAの実装ゲートは引き続き適用する。

## 1. 根拠と開始地点

対象を`T`、参照元を`R`と表記する。

- T: `Confyui-krea2-slider-node`、HEAD `df1d2f3f18f7b2114b27914b6c227b2597da488b`。
- R: `../ComfyUI-Krea2-Regional`、HEAD `307081f2b954d9f5e683dadafdb53ca971ebdbfd`、pyproject版1.2.0。
- 実機ログにあった本体: ComfyUI `7c8fbc698b3c3dce0f525f6b5044b93d9395c5c9`、RTX A4000 16GB、torch 2.13.0+cu130、DynamicVRAM。次回も同じ版かログで確認する。

| 証拠 | 確認できたこと | 計画への反映 |
| --- | --- | --- |
| `krea2_two_person_region_debug_00001_.png` / `00002_.png` | 同じ小マスク・seedでdeagingだけ2→0。0でも領域外に女性が残る | Hookだけの修正や強度変更を解決策としない |
| `krea2_two_person_regional_area_00001_.png` / `00002_.png` | areaを使うと領域内の人物配置は改善したが、LoRA 0でも上側に余分な頭が残る | area cropは比較対象として保存する |
| `krea2_two_person_regional_empty_left_00001_.png` | DEFAULT文だけ変更しても重複が残る | 背景文章の言い換えを繰り返す段階を終了する |
| 以前の実機診断ログ | 女性条件だけにHook、8回の適用で140件のbackup、解除後は残数0 | 大きな重み残留の証拠はない。ただし件数ログだけで数値的復元まで証明したとはしない |
| `T/krea2_slider_node/native_hooks.py:1,42,73,106` | 現行Hookは重み適用・復元を担当し、文章のattentionを制御しない | 新しいRegional推論経路を追加する |
| `T/krea2_slider_node/region_area.py:13,36` | spatial areaを作り、3D latentで標準AABB経路のエラーを回避する | 今回のattention経路にはarea cropを重ねない |
| `R/krea2_regional.py:434,515,1002` | 文章を連結し、text/image token間のattention許可行列をforwardに渡す | Regional Attentionの参照実装にする |
| `R/krea2_regional.py:199,262,468` | 元Linearのforward後にtokenでマスクしたLoRA差分を加える | INT8領域LoRAの候補にする |
| `R/krea2_regional.py:445,458` | base textは全imageと相互参照し、background imageも領域間の中継になり得る | 参照元をそのまま移すだけで完全隔離とは扱わない |
| `R/test_nodes.py:63,127,159` | CPUの文章分離・LoRA locality・5Dテストがある | テスト設計を参考にする。INT8実証とは区別する |

PNGはワークスペース直下、一部はDownloads直下にある。`ログ.txt`は毎回上書きされているため、過去ログの内容を現在のログと取り違えない。実装開始時に入力ファイルのSHA-256と抽出設定を`test-results/regional-attention/`へ保存する。画像・モデル・ログ本体をGitへ追加しない。

## 2. 要件と範囲

### 初期版で実装する

1. 既存`Krea2 Region Masks`とCLIP Text Encodeの出力を使うRegional推論ノード。
2. base・女性・男性・背景の文章segmentとimage tokenを結ぶattention制御。
3. マスクの補集合を背景領域として扱う。今回の上側の空間を未割当のままにしない。
4. 既存Sliderの`lora_unet_*`＋down/up/alpha形式を読む、token単位の領域LoRA。
5. 4D latentと静止画の5D latent `(B,C,1,H,W)`、CFGのcond/uncond、clone・例外時の状態復元。
6. 1回生成の比較ワークフローと、実機判定用の診断ログ。

### 初期版に含めない

学習処理の変更、既存Hook経路の削除、参照画像、Detailer、別画像の貼り込み、captioner、LoKr、adaptive masks、region lock、UI全面移植。既存矩形エディタを使う。動画`T>1`は明示的に拒否する。

根拠: 既存MASK型は`T/nodes_region_masks.py:7`、LoRA書き出しは`T/krea2_slider_node/lora.py:73`、V3登録は`T/nodes.py:125`。参照元の広い機能は`R/krea2_regional.py:391,773`と`R/krea2_builder.py:289`にあり、今回の問題の切り分けには必須でない。

## 3. 設計判断

### 3.1 追加するV3ノードと接続

| ノードID案 | 入力 | 出力 |
| --- | --- | --- |
| `Krea2RegionalSlider` | LoRA名、strength、任意prev_loras | `KREA2_SLIDER_LORAS` |
| `Krea2RegionalPromptRegion` | CONDITIONING、MASK、任意loras・prev_regions | `KREA2_SLIDER_REGIONS` |
| `Krea2ApplyRegionalAttention` | MODEL、base CONDITIONING、background CONDITIONING、regions、isolation、release_percent、debug_logging | MODEL、統合CONDITIONING |

型名・node IDを参照元と分け、両パッケージを同時に入れても衝突させない。リージョン数は初期版1〜4個。各conditioningは単一entryで、既存mask/area/default/hooks/reference_latentsを含まないものに限定する。入力違反はSampler前に説明付きで停止する。

```text
Load Model → global darkbrush（固定）→ Krea2ApplyRegionalAttention → KSampler
CLIP → base / background / female / male の各Encode ──────────┘
Region Masks → female MASK / male MASK → RegionalPromptRegion chain ─┘
RegionalSlider → female region のみ
```

この経路ではnative `Create Hook LoRA`、`Cond Pair Set Props`、`Krea2 Pair Region Area`を使わない。標準Hookは1つのforward内のトークン別LoRAを表現できないため、RegionalSliderは参照元のactivation差分方式を採用する。学習・従来Hookは独立した既存経路として使える状態を維持する。[根拠: `T/nodes_native_hooks.py:7`、`R/krea2_regional.py:199,862,891,1002`]

### 3.2 初期版のattentionルール

参照元の共有base/backgroundを経由した間接伝播までテスト対象にする。行をquery、列をkeyとして次を構成する。

| query | 初期の`strict`モードで参照できるkey |
| --- | --- |
| base text | base textのみ。画像や他region文章を読み戻して中継しない |
| region i text | 同じregionのtext・image、base text |
| region i image | 同じregionのtext・image、base text |
| background text / image | background内のtext・image、base text |

txtfusionの文章側attentionもsegment間を分離する。全queryに有効なkeyを残し、全遮断行によるNaNを防ぐ。背景MASKはtoken grid上で対象regionの補集合として作る。重複はregion chainの先頭優先とし、画面上の全tokenのownerを1つに決める。

マスク縮小は実Krea2のpatchサイズとlatent H/Wから計算する。初期は硬いtoken mask、feather/growなし。area補間後`>=0.5`をowner候補にし、消滅する極小regionはエラーとする。実効token境界をログに出し、画素矩形と完全一致すると説明しない。[参照: `R/krea2_regional.py:352,361,434,561`]

`balanced`モードはstrictの評価後に追加する。`release_percent=0.5`以降のimage↔image経路だけを開き、文章segmentの制限は維持する。strictは全ステップで制限する。時間窓はstep番号でなくモデルのsigma変換を使う。[参照: `R/krea2_regional.py:610,1045`]

この遮断規則は新規適合作業であり、参照元で検証済みとは扱わない。許可行列の正しさと「余分な人物が出ない」実画像品質は別の合格条件とする。

### 3.3 モデル適用・INT8・実行状態

- ComfyUIのmodel cloneと`WrappersMP.DIFFUSION_MODEL`、`optimized_attention_override`を使う。Krea2本体やグローバルクラスを書き換えない。[`R/krea2_regional.py:515,694,1013,1077`]
- `ContextVar`で呼出し単位のregion状態を管理し、`try/finally`で元の状態へ戻す。参照元のグローバル`CTX`をそのまま移植しない。[`R/krea2_regional.py:48,62,751`]
- cond/uncondを`cond_or_uncond`で判別する。同じtoken長のnegativeでもregional制御を誤適用しない。token長だけの識別に依存しない。[`R/krea2_regional.py:561,578`]
- 既存attention maskと新maskは制限の共通部分を取る。未知の形式や互換性のないoverrideは黙って無制限に戻さず、debug情報付きで停止する。[参照元の要修正箇所: `R/krea2_regional.py:694,729`]
- LoRAは元の量子化Linear.forwardを実行した結果へ低rank差分を加える。INT8 packed weight/scaleを再量子化しない。native Hookの重み置換とは同じモデル経路で併用しない。[`R/krea2_regional.py:199,245`、`T/krea2_slider_node/native_hooks.py:42`]
- 初期対応は今回の学習既定である`blocks.*.attn.*`のstandard LoRA。非空間的なtimestep/modulation層、未知キー、未対応adapterは件数を示して拒否し、黙ってglobal適用・skipしない。0件matchを成功にしない。[`T/krea2_slider_node/lora.py:32,73`、`R/krea2_regional.py:262`]
- 実行キャッシュはモデル適用インスタンス単位とし、geometry/device/dtype/batch/segment構成をキーにする。入力変更・新しいsampling run・例外・clone切替で古いmaskやadapterを使い回さない。[`R/krea2_regional.py:551,621`]

## 4. 実装タスク

各コードタスクは、失敗するテストを先に確認し、最小実装後に合格を確認する。ComfyUI本体がない環境でのskipを統合テスト合格として扱わない。

### Task 1: 参照元と再現条件の固定

**Files:** `licenses/Krea2-Regional-MIT.txt`（追加）、`THIRD_PARTY_NOTICES.md`（更新）、`docs/regional-attention.md`（追加）。

- [ ] 上記T/RのHEADとclean状態を確認し、実装開始地点を記録する。
- [ ] 比較PNGからprompt・seed・LoRA強度・マスク・解像度を抽出し、同じ設定と異なる設定を一覧化する。過去の140件復元ログは履歴上の観測として記録する。
- [ ] 採用する関数と変更理由を記録し、MIT本文をそのまま同梱する。参照元の`Copyright (c) 2026 YOUR_NAME`を勝手に別人名へ置き換えない。[`R/LICENSE:1`]
- [ ] 初期UIは既存の2矩形を再利用し、Builder/JS/サーバールートを移植しないことを確定する。

**完了条件:** 参照commit、比較設定、ライセンス、予定変更ファイルの記録が揃う。

### Task 2: attention許可行列と背景owner

**Files:** `krea2_slider_node/regional_attention.py`、`tests/test_regional_attention.py`（追加）。

**Interfaces:** `build_region_owners(masks, token_hw) -> owners`、`build_attention_masks(segments, owners, isolation) -> (joint_mask, text_mask)`。所有者はregion番号と背景番号、maskはTrue=参照可。

- [ ] 次のassertを持つCPUテストを作る: 女性text→領域外image不可、男性→女性text不可、背景→女性image不可、base text→image不可、全行で1個以上のkeyが有効。
- [ ] 小マスク・上下左右の端・重複・全画面・空マスク・1token未満・異なる画像解像度を検証する。重複は先頭優先、補集合は漏れなく背景へ割り当てる。
- [ ] 2層以上の小attention系で女性embeddingだけを変え、strictでは領域外出力差がfloat32許容誤差`1e-5`以内になることを確認する。女性側には非ゼロ差を要求し、無動作による偽合格を防ぐ。
- [ ] 参照元のbase/background中継規則では失敗する検査を含め、上記strict規則を実装する。

**検証:** `python -B -m pytest -q tests/test_regional_attention.py`。テンソルの許可行列・遮断効果が全件合格。

### Task 3: V3ノード・Krea2 forwardへの接続（LoRAなし）

**Files:** `nodes_regional_attention.py`、`krea2_slider_node/regional_runtime.py`、`tests/test_regional_runtime.py`、`tools/validate_regional_attention.py`（追加）、`nodes.py:125`と`tools/validate_comfy.py:27`（更新）。

**Interfaces:** `RegionSpec(conditioning, mask, loras)`と`RegionalBundle(segments, masks, background, isolation)`を`regional_runtime.py`で定義。`apply_regional_attention(model, base, background, regions, isolation, release_percent, debug_logging)`はclone MODELと単一CONDITIONINGを返す。

- [ ] Region/ApplyノードのV3スキーマ、入力検証、登録、無効化・再接続をテストする。独自型名は上表に固定する。
- [ ] Krea2 forwardをwrapperで包み、4Dと5D静止画、CFG1、CFG>1のmixed batch、同token長negative、他overrideの共存、参照latent拒否を検証する。
- [ ] 本体Krea2の小型モデルを使うCPU統合テストを実行可能にする。2ステップ以上の入力変化で領域外への伝播、例外後のcontext復元、clone A/Bの非干渉を検査する。
- [ ] 通常画像では元のforwardと等価となる1領域・無制限の検算モードをテスト用に設ける。製品UIに不要な検算スイッチは出さない。
- [ ] 対象forward回数を計測し、CFG1のEulerでは1stepあたり1回となることを確認する。参照方式のN領域分samplingへ戻さない。

**検証:** `python -B tools/validate_regional_attention.py /tmp/ComfyUI`。本体の実行版・実行テスト数を出力し、未収集／全skipは失敗にする。

**実機Gate A:** Task 5のprompt-onlyワークフローを先に用意し、ユーザーがTask 6のA評価を実施する。人物重複が残ればTask 4へ進まず、routingログ・背景owner・base文の寄与を切り分ける。

### Task 4: INT8対応のtoken単位Slider

**Files:** `krea2_slider_node/regional_lora.py`、`tests/test_regional_lora.py`、`tests/test_regional_int8.py`（追加）、`nodes_regional_attention.py`と統合検証ツール（更新）。

**Interfaces:** `load_regional_lora(path, strength) -> RegionalLoRASpec`、`attach_regional_loras(model_clone, bundle)`。region nodeの任意lorasへ接続する。

- [ ] 既存exportの`lora_unet_blocks_0_attn_wq.lora_down/up.weight`とalphaを読み、対象moduleへ一意に対応させる。0/正/負強度、rankとshape不一致、未知キー、非空間層を検証する。
- [ ] 元Linearを保持するactivation差分方式を実装し、token maskは現在のContextVarからのみ取得する。量子化weight・scale・state_dictキーが不変であることを確認する。
- [ ] 実ComfyUIのBF16/FP16・INT8・ConvRot層を使い、領域内deltaは有効、領域外への直接deltaは0、元モデルと別cloneは不変、例外後もcontextが空になることを検査する。
- [ ] 実行環境の量子化レジストリを確認してfixtureを選ぶ。`int8_tensorwise`など単一形式名の存在を仮定しない。`--require-int8`指定時に必要形式がないなら失敗として報告する。
- [ ] darkbrushをglobal標準LoRAとして先に適用した構成、地域LoRAなし、強度0、繰返しqueue、モデル切替を検査する。VRAMに別のフルモデルコピーを常駐させない。

**検証:** `python -B tools/validate_regional_attention.py /tmp/ComfyUI --require-int8`。量子化形式ごとに実行数・不変性・layer match数を報告する。実機Gate BはTask 6参照。

### Task 5: ワークフロー・配置説明・診断

**Files:** `workflows/krea2_two_person_attention.json`、`workflows/krea2_two_person_attention_tips.md`（追加）、`krea2_slider_node/diagnostics.py`、`docs/regional-attention.md`、`README.md`（更新）。

- [ ] Task 3直後にprompt-only版を作る。baseは共通の画風・光・全体配置、女性/男性/背景の内容は各region promptへ分ける。全画面を背景owner込みで割り当てる。
- [ ] 女性の「左半分に立つ」という古い固定文を使わず、実際の矩形の位置・サイズから位置説明を組み立てる。参照元`R/krea2_builder.py:173`を参考にする。baseへの人物名を含む位置ヒントはON/OFFで切り分け、実機Gate Aで採用値を決める。
- [ ] 初期値はstrict、LoRAなし、adaptive/feather/growなし、CFG1・8 steps・seed42。maskは過去の`x=0,y=0.367556,w=0.460948,h=0.632444`を含む比較を用意する。
- [ ] Task 4後に女性regionのSlider選択を追加する。実機画像の学習概念比較ではSliderを1種類ずつ使う。男性regionのLoRAは空で開始する。
- [ ] ログにrun ID、ComfyUI/repo版、latent/token grid形状、prompt segment長、owner別token数、非許可edge検査、mask hash、LoRA match数/強度、sampling forward数、peak VRAMを出す。全文prompt・重み本体を出さない。
- [ ] 毎runのsampling状態を新しくし、既存診断と区別できるイベント名を付ける。無効時に再実行を強制しない。未知のattention形式による制御解除を成功扱いしない。

**完了条件:** JSON相互参照・型・実schemaが一致し、背景補集合と入力Slider数をログから判定できる。実機評価に必要な操作は1回のqueueで完結する。

### Task 6: 実機評価と採否

評価画像はPNGのAPIメタデータとrun ID付きログを1組で保存する。以前の画像は原因調査の証拠であり、エンジンが変わる新実装との画素一致基準にはしない。

| Gate | ユーザーの実機操作 | 合格条件 |
| --- | --- | --- |
| A: 文章だけ | 全地域LoRAなし。小マスク／左半分マスクそれぞれseed 42,123,777,2026,31415で生成（計10枚） | 全10枚で女性1人・男性1人。マスク外に余分な女性の顔/頭/身体断片がない。頭と胴体が境界で二重化しない。指定領域からのずれは実効token境界1個分以内を目視記録 |
| B: Slider | Aの5seed×女性Slider強度0/1/2。成人agingまたは成人breastを1種類ずつ評価 | 重複が再発しない。女性に目的概念の変化がある。男性に同じ概念変化が見られず、顔/体格/衣服に明瞭な破綻がない。raw画素差だけを局所性の尺度にしない |
| C: 実行 | RTX A4000 16GB・指定INT8・同じdarkbrushで連続3queue | OOM・dtype・shape例外なし。前run/別cloneのregionが混入しない。CFG1で規定の1forward/step。実行時間・peak allocated/reserved VRAMを報告 |
| D: balanced | A/B合格後、同じseed群でrelease=0.5とstrictを比較 | 継ぎ目改善と重複率を別々に記録。重複が戻る場合strictを既定に残す |

**判定:** CPUルーティング試験だけでは画像品質合格を宣言しない。Aで失敗した場合は、LoRA追加・高強度化・再学習へ進まない。bg/baseの情報経路をログと介入試験で特定する。正しいroutingでも人物重複が残れば、この方式での未達条件として明示し、二段階生成へ勝手に変更しない。

### Task 7: 回帰確認・文書・リリース

**Files:** 既存`tests/`、`tools/validate_comfy.py`、`THIRD_PARTY_NOTICES.md`、`docs/regional-attention.md`。

- [ ] `python -B -m pytest -q`、Python構文検査、利用可能な既存lint/typecheck、`git diff --check`を実行する。未使用ツールは追加installせず未実施を記録する。
- [ ] 既存5ノードのID・入出力・学習式とexport形式、旧ワークフローが変わっていないことを確認する。[`T/nodes.py:125`、`T/tests/test_native_hooks.py:64`]
- [ ] 本体統合テストはskipなしの専用結果を記録する。通常pytestでのskip数と区別する。
- [ ] 現行Hookと新Regional経路を同時に重ねない接続図、ログ、実機で通ったComfyUI commit・量子化形式を文書化する。
- [ ] 移植元ライセンス・commit・改変内容を再点検する。適応マスク、Detailerなど未採用の機能を提供したと記載しない。
- [ ] ユーザーの実機証跡とレビューを取り込んでから、実装結果を報告しcommit/pushする。未達なら実験状態と残件を報告する。

## 5. リスクと検証の焦点

| リスク | 対応・具体的検査 |
| --- | --- |
| base/backgroundを経由した複数層の情報中継 | Task 2の有向maskと2層以上のperturbation検査。Task 3の複数step検査 |
| INT8 class swap・DynamicVRAMとの衝突 | 元forward、packed weight/scale不変、global speed LoRAとの組合せをTask 4で実証 |
| sourceのグローバルCTX、長さだけのcond識別、既存mask無視 | ContextVar、same-length negative、bool/additive maskの合成をTask 3で検査 |
| 解像度・マスク・prompt変更後の古いキャッシュ | 連続runでgeometry・dtype・batch・文章長を変更するテスト |
| attention行列で16GBを超える | 実効token数から必要mask容量を事前計算。初期は総text token数512以下、超過は無言truncateせずエラー。領域数1〜4、同じmaskをstep間で再利用しTask 6で実測 |
| strictで背景・光がタイル状になる | まず重複解消をGate Aで判定、その後balancedを比較。見た目だけで漏れの検査を緩めない |
| VAE復号で境界の画素に差が出る | attentionの数学的遮断と最終画素の完全一致を区別する。合格基準は人物重複・対象概念・構図 |
| 参照実装のコピー範囲が膨張する | 中核のみを分割適合し、Builder/Detailer/サーバー/JSは持ち込まない。MIT表記を保持 |

## 6. 終了条件と引継ぎ

計画の完了条件は、この文書の要件・インターフェース・検証方法が自己矛盾なく定義されていること。製品実装の完了条件はTask 1〜7と実機Gate A〜Cの証跡が揃うこと。balancedは任意の追加評価であり、失敗時はstrictを採用する。

最初の実装単位はTask 1〜3＋Task 5のprompt-only版。既存Hookの重み復元修正へ追加パッチを重ねる作業から始めない。これまでのpromptのみ・LoRA0の失敗を再現条件として、Regional Attentionの効果を先に判断する。
