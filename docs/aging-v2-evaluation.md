# 女性用 aging v2：評価と対策実験

2026-10-03。対象は `生成画像/生成画像３` の学習レポートと3枚のPNG。
目的は女性の顔・首・手の加齢を維持し、同時に登場する男性を保持すること。

## 判定

**今回の立位・seed=1では、v2は男性への加齢の波及を解消していない。**
同条件のv1に対して、性別の選択性が改善したとは判断できない。
強度2では女性の加齢が強くなっている一方、男性にも変化が強く見える。
これは目視評価であり、年齢推定器や画素差分によるスコアではない。

| 強度 | v1：親フォルダー | v2：生成画像３ | 目視所見 |
|---:|---|---|---|
| 0 | `Krea2_turbo_00041_.png` | 未添付 | 親フォルダーの画像を基準参照に使える |
| 1 | `Krea2_turbo_00043_.png` | `Krea2_turbo_00041_.png` | 0との比較で顔立ちや描写が変化。明確な女性限定の加齢とは判定しにくい |
| 2 | `Krea2_turbo_00045_.png` | `Krea2_turbo_00042_.png` | v2は女性の目元・口元・首に加齢。男性にも目元・ほうれい線・首の変化があり、保持に失敗 |
| 3 | `Krea2_turbo_00047_.png` | `Krea2_turbo_00043_.png` | v1/v2とも男女が加齢。襟、シャツの形、髪・輪郭にも変化 |

新画像の手は画面外なので、手の保持・加齢は評価できない。
女性単独、男性単独、座位、左右反転、別seedのv2画像も未添付。
この結果だけで全構図・全seedでv2が劣ると断定しない。

## 同条件比較の確認

`生成画像２` の男女同時画像は座位だが、**親フォルダーには今回と完全に同じ立位プロンプトのv1画像がある**。
上表の強度1/2/3の各組でPNGの `prompt` JSON全体を比較し、差が `71.inputs.lora_name` だけであることを確認した。
親フォルダーの強度0も同じ生成設定・プロンプトで、aging LoRAは0。
モデルの重みのハッシュやComfyUI実行バージョンはPNGから確認できない。同一性の根拠は記録された生成条件である。

| 項目 | 共通条件 |
|---|---|
| ベース | `intorealismAsian_k2JAVFLASHV1.safetensors` |
| Turbo用LoRA | `krea2_darkbrush.safetensors`、0.8 |
| encoder / VAE | `qwen3vl_4b_fp8_scaled.safetensors` / `qwen_image_vae.safetensors` |
| seed / 解像度 | 1 / 1024×1024 |
| steps / CFG | 8 / 1 |
| sampler / scheduler / denoise | euler / simple / 1 |
| negative | ConditioningZeroOut |
| aging LoRA v1 | `krea2_aging_20261002T050755Z_fccfa81a.safetensors` |
| aging LoRA v2 | `krea2_aging_20261002T174434Z_5e6cf2dc.safetensors` |

`krea2_darkbrush` の役割はTurbo用であり、画風LoRAとして扱わない。評価条件でも0.8を維持する。

## 学習レポート

v2の入力は `aging_slider_fullbody_v2.yaml`、8レコード、状態は `training_completed`。
記録されたYAMLハッシュ `80ef5f09cb0f346599261eac099d99b56664afedb0c8f53fea9b6c16f1f781e8`
はローカルv2をLFへ正規化したハッシュと一致する。WindowsのCRLFによるバイト差を別内容と混同しない。
レポートの8レコードと新PNGのLoRA名も照合した。

v1/v2とも400steps、rank/alpha=16/16、attention、LR=0.0001、768×768、trajectory_steps=8、eta=1、teacher_guidance_scale=1、teacher_norm_reference=none、single、anchor_strength=1、seed=42、vary_seed=true、BF16/convrot_int8。

| 項目 | v1 | v2 |
|---|---:|---:|
| 学習レコード | 6 | 8 |
| anchor有効step | 400/400 | 300/400 |
| 各レコードの使用回数 | 66～67 | 50 |
| 最初の100stepのslider_loss平均 | 0.01005436 | 0.01094977 |
| 最後の100stepのslider_loss平均 | 0.00249858 | 0.00435235 |
| 最後の100stepのanchor_loss平均（無効stepの0も含む） | 0.00032000 | 0.00051231 |
| 非ゼロ勾配step | 400 | 400 |
| 最大grad_norm | 0.22618 | 0.30224 |
| 所要時間 | 16010秒 | 15725秒 |

勾配は全stepで有効で、クリッピング上限1にも達していない。学習が実行されなかったという証拠はない。
異なるYAMLのloss平均は画像品質や性別分離の優劣を表さない。
既存男性anchorの更新機会が減り、追加2件ではanchorが無効になる。
この**更新回数の減少は事実**だが、波及の原因であるという主張は未検証。

現行の順番選択から推定すると、混合レコード7の最初/最後の10回のslider_lossは約0.00967/0.00958、レコード8は0.01499/0.00647。
レコード番号はログに保存されておらず、現行の `records[step % len(records)]` を仮定した推定。
seedとtimestepも変わるので、これだけで収束不足や十分な収束を判定しない。
数値、PNGのハッシュ・生成グラフ、比較結果は [metadata-summary.json](aging-v2-evidence/metadata-summary.json) に保存した。

## 原因の切り分け

1. **確定した制約：人物領域の保持損失がない。** [training.py](../krea2_slider_node/training.py) は全予測テンソルのMSEを使う。anchorは別の男性単独テキスト条件でベース予測を保持するが、男女同時の男性領域だけに損失を掛けてはいない。
2. **未検証の有力仮説：教師のpositive予測にも男性の加齢が混ざる。** [slider_loss.py](../krea2_slider_node/slider_loss.py) の式は `teacher = base + eta_eff * (positive - negative)`。今回の `target = negative`、eta_eff=1、正規化なしでは数式上 `teacher = positive`。男性保持の文章があっても、実際の教師予測が守る保証はない。
3. **未検証の補助仮説：anchor更新の減少と構図の一般化不足。** 混合条件は座位2件だけで、今回の立位は未学習。既存anchorは25%減った。どちらが支配的かは今回の資料から判別できない。
4. **確定した評価上の制約：+2/+3は学習した+1からの外挿。** +1の効果と+2での男性保持を別々に判定する。

教師の波及は、実際の学習モデル・量子化・latent/timestepでpositive/negativeの予測差を調べるまで確定しない。
Turbo推論でテキストだけを変える画像比較は安価な一次診断であり、教師テンソルの直接検証ではない。
`source_kind=raw` と空の `source_metadata` だけでは元チェックポイントのRAW区分を独立には確定できない。
ネイティブ学習入力では外付けLoRAのpatchを拒否する実装で、評価のTurbo用LoRAを学習にも掛けたとは扱わない。

Concept Slidersの一次資料にも、加齢と性別などの干渉を避けるため保護する概念を含めた複数条件を使う考え方がある。ただし、Krea2の複数人物の領域保持を実証した資料ではない。[著者の研究ページ](https://sliders.baulab.info/)

## 用意した対策と次の判定

### 再学習前の22件の評価

[evaluation-api-bundle.json](aging-v2-evidence/evaluation-api-bundle.json) に未送信のComfyUI APIリクエストを保存した。
独自のResolutionSelectorとSwitchを除き、標準ノードの明示的な1024×1024設定とモデル接続に置き換えた。
**画面用workflow JSONではない。各 `cases[i].api_request` を1件ずつ `/prompt` に送る形式**。
サーバーのノード・モデル一覧との照合、元PNGの実画像での再現確認は未実施。

- 15件：立位男女、座位男女左右、女性単独、男性単独の5条件 × 強度0/1/2、seed=1。
- 3件：立位男女、seed=2の基準0、v1強度2、v2強度2。
- 4件：v2の混合レコード7/8のtargetとpositiveをaging LoRAなし・Turbo 0.8で生成。テキストによる女性限定の変化がこの推論構成で成立するかを見る。

まず既存画像で不足する条件と4件のテキスト診断を優先する。
座位でも男性が加齢するなら、立位を足すだけでは原因を説明できない。
テキスト診断のpositiveでも男性が加齢するなら、同じpositiveを増やしたり長く学習したりする前に教師側を調べる。

### 男性anchorを全stepに戻す比較用YAML

[aging_slider_fullbody_v2_anchor_ablation.yaml](../prompts/aging_slider_fullbody_v2_anchor_ablation.yaml) を追加した。
v2の8レコードの全Slider本文・順番・既存6件のanchorを維持し、追加2件に**男性単独anchor**だけを付けた。
男性の左右位置、机、髪、シャツ、背景、滑らかな成人の肌を記述する。
男女両方の基準をanchorにして女性の加齢まで抑える変更ではない。

比較はv2と同じ400steps・全設定・入力モデルで初期状態から学習する。全400stepでanchor有効になり、各Slider条件は50回のまま。
これは「混合条件のstepにも男性単独保持を入れる」という対策パッケージの比較であり、頻度とanchor本文の寄与を別々には分離しない。
総step、anchor_strength、eta、rank、LRを同時に変えず、既存v2 LoRAに継ぎ足す比較もしない。

**このanchorも混合latentを男性単独テキストで評価する現行の方式で、画像内の男性を直接保護するマスクではない。**
女性の効果も弱まる可能性があるため、対策候補であって改善済みのv3とは呼ばない。

成功条件は、女性単独と男女同時の女性に加齢が現れ、男性単独と男女同時の男性の顔・首・特徴が基準0に近く保たれること。
男性保持とともに女性の効果も消えた場合は失敗。衣服変化も別項目として記録する。
強度1/2、座位左右と立位、少なくともseed=1/2で判定する。手は可視の座位画像で確認する。

失敗が続く場合は、混合人物の保持対象を直接指定できる教師・領域マスクまたは画像ペアの監督へ設計を進める。
単純な左右半分のマスクは人物や手の重なりに弱く、人物対応の確認なしに本対策にしない。
現行ノードは画像・人物マスクを学習入力に取らないため、その段階は別の設計変更になる。

## 実施範囲

トレーナーは変更せず、比較用YAML、比較条件を守る回帰テスト、評価資料・未送信API条件を追加した。
ローカルの標準modelフォルダーには対象LoRA・ベースの該当ファイルを確認できず、添付レポートの学習環境は別のLinux/NVIDIA環境。
実GPUの再学習と追加生成は未実施。追加モデル検索パスや実行先サーバーも未確認なので、ローカルで実行不可能と断定するものではない。
CPUテストは読込と比較条件の保全の検証であり、男性保持の改善を意味しない。

検証：`python -B -m pytest -q` → **55 passed, 1 skipped, 81 subtests passed**。
スキップは同梱されていない公式参照コードとのモデル一致テスト。
追加テストはYAML追加前に失敗することを確認し、追加後に通過。既存v2と添付レポートの8件の一致、比較版のSlider本文の維持、22件のAPIグラフの参照・固定条件も検証した。
`git diff --check` は通過。ruffはローカルに未インストールで未実施。今回のPython変更は回帰テストのみ。
